import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
import stackstac
import torch.nn as nn
import cv2
from os.path import join
from collections import OrderedDict
from pytorch_lightning import LightningDataModule

from typing import Any, Optional, Tuple, Union, Sequence
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Subset

from typing import Iterator, Optional
from torch.optim import AdamW
import torch
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    SemanticSegmentationLabelSource)

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple, MultiInputCrossValidator,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn,
                                  MergeDataset, show_single_tile_multi, get_label_source_from_merge_dataset, create_scenes_for_city, PolygonWindowGeoDataset)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)

# Helper functions
def check_nan_params(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")

# implements loading gdf
class CustomVectorOutputConfig(Config):
    """Config for vectorized semantic segmentation predictions."""
    class_id: int = Field(
        ...,
        description='The prediction class that is to be turned into vectors.'
    )
    denoise: int = Field(
        8,
        description='Diameter of the circular structural element used to '
        'remove high-frequency signals from the image. Smaller values will '
        'reduce less noise and make vectorization slower and more memory '
        'intensive (especially for large images). Larger values will remove '
        'more noise and make vectorization faster but might also remove '
        'legitimate detections.'
    )
    threshold: Optional[float] = Field(
        None,
        description='Probability threshold for creating the binary mask for '
        'the pixels of this class. Pixels will be considered to belong to '
        'this class if their probability for this class is >= ``threshold``. '
        'Defaults to ``None``, which is equivalent to (1 / num_classes).'
    )

    def vectorize(self, mask: np.ndarray) -> Iterator['Polygon']:
        """Vectorize binary mask representing the target class into polygons."""
        # Apply denoising if necessary
        if self.denoise > 0:
            kernel = np.ones((self.denoise, self.denoise), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to polygons
        for contour in contours:
            if contour.size >= 6:  # Minimum number of points for a valid polygon
                yield Polygon(contour.squeeze())

    def get_uri(self, root: str, class_config: Optional['ClassConfig'] = None) -> str:
        """Get the URI for saving the vector output."""
        if class_config is not None:
            class_name = class_config.get_name(self.class_id)
            uri = join(root, f'class-{self.class_id}-{class_name}.json')
        else:
            uri = join(root, f'class-{self.class_id}.json')
        return uri

# Sentinel Only Models
class SentinelDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (12, 24, 36),
                sched_step_size = 10,
                pos_weight: float = 1.0):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
                
        # Main encoder - 4 sentinel channels
        self.deeplab = deeplabv3_resnet50(pretrained=False, progress=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.deeplab.classifier = DeepLabHead(2048, 1, atrous_rates = (12, 24, 36))
        
        # load pretrained weights
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            state_dict = checkpoint["state_dict"]
                
            # Convert any float64 weights to float32
            for key, value in state_dict.items():
                if value.dtype == torch.float64:
                    state_dict[key] = value.to(torch.float32)

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # Remove the prefix 'segm_model.'
                new_key = key.replace('segm_model.', '')
                # Exclude keys starting with 'aux_classifier'
                if not new_key.startswith('aux_classifier'):
                    new_state_dict[new_key] = value                

            self.deeplab.load_state_dict(new_state_dict, strict=True)
            self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.deeplab(x)['out'].squeeze(dim=1)
        return x

    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        segmentation = self(img)
        groundtruth = groundtruth.float().to(self.device)
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch, seg: {segmentation.shape} vs gt: {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)
        
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_mean_iou', mean_iou)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        test_loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.deeplab.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=self.gamma  
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

# Show predictions function
def create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5,  class_id=1, figsize=(30, 10)):
    """
    Create a plot of smooth predictions, discrete predictions, and ground truth side by side.
    
    Args:
    pred_labels (SemanticSegmentationLabels): Prediction labels
    gt_labels (SemanticSegmentationLabels): Ground truth labels
    class_id (int): Class ID to plot (default is 1, assumed to be informal settlements)
    figsize (tuple): Figure size (default is (30, 10))
    
    Returns:
    tuple: (fig, (ax1, ax2, ax3)) - Figure and axes objects
    """
    # Get scores and create discrete predictions
    scores = pred_labels.get_score_arr(pred_labels.extent)
    pred_array_discrete = (scores > threshold).astype(int)

    # Get ground truth array
    gt_array = gt_labels.get_label_arr(gt_labels.extent)

    # Create a figure with three subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Plot smooth predictions
    scores_class = scores[class_id]
    im1 = ax1.imshow(scores_class, cmap='viridis', vmin=0, vmax=1)
    ax1.axis('off')
    ax1.set_title(f'Smooth Predictions (Class {class_id} Scores)')
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot discrete predictions
    pred_array_discrete_class = pred_array_discrete[class_id]
    im2 = ax2.imshow(pred_array_discrete_class, cmap='viridis', vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title(f'Discrete Predictions (Class {class_id})')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot ground truth
    gt_class = (gt_array == class_id).astype(int)
    im3 = ax3.imshow(gt_class, cmap='viridis', vmin=0, vmax=1)
    ax3.axis('off')
    ax3.set_title(f'Ground Truth (Class {class_id})')
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3)

# Buildings Only Models
class BuildingsDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (12, 24, 36),
                sched_step_size = 10,
                pos_weight: float = 1.0):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
        
        self.deeplab = deeplabv3_resnet50(pretrained=False, progress=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.deeplab.classifier = DeepLabHead(2048, 1, atrous_rates = (12, 24, 36))
        
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            state_dict = checkpoint["state_dict"]
                
            # Convert any float64 weights to float32
            for key, value in state_dict.items():
                if value.dtype == torch.float64:
                    state_dict[key] = value.to(torch.float32)

            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # Remove the prefix 'segm_model.'
                new_key = key.replace('segm_model.', '')
                # Exclude keys starting with 'aux_classifier'
                if not new_key.startswith('aux_classifier'):
                    new_state_dict[new_key] = value                

            conv1_weight = new_state_dict['backbone.conv1.weight']
            new_conv1_weight = conv1_weight[:, :1, :, :].clone()
            new_state_dict['backbone.conv1.weight'] = new_conv1_weight
            
            self.deeplab.load_state_dict(new_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Move data to the device
        x = x.to(self.device)
        x = self.deeplab(x)['out']#.squeeze(dim=1)
        return x

    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        segmentation = self(img)
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)
        
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_mean_iou', mean_iou)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        test_loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, groundtruth)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.deeplab.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=self.gamma  
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class PredictionsIterator:
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, _ = dataset[idx]
                image = image.unsqueeze(0).to(device)

                output = self.model(image)
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = dataset.windows[idx]
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)
    
# Mulitmodal Model and helpers definitions
class MultiModalDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
    
    def setup(self, stage=None):
        pass
    
class MultiResPredictionsIterator:
    def __init__(self, model, merged_dataset, device='cuda'):
        self.model = model
        self.dataset = merged_dataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(merged_dataset)):
                sentinel, buildings = self.get_item(merged_dataset, idx)
                
                sentinel_data = sentinel[0].unsqueeze(0).to(device)
                sentlabels = sentinel[1].unsqueeze(0).to(device)

                buildings_data = buildings[0].unsqueeze(0).to(device)
                labels = buildings[1].unsqueeze(0).to(device)

                output = self.model(((sentinel_data,sentlabels), (buildings_data,labels)))
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = self.get_window(merged_dataset, idx)
                self.predictions.append((window, probabilities))

    def get_item(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_item(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_item(dataset.datasets[dataset_idx], sample_idx)
        elif isinstance(dataset, MergeDataset):
            return dataset[idx]
        else:
            raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    def get_window(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_window(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_window(dataset.datasets[dataset_idx], sample_idx)
        elif isinstance(dataset, MergeDataset):
            # Assume the first dataset in MergeDataset has the windows
            return dataset.datasets[0].windows[idx]
        else:
            raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    def get_concat_dataset_indices(self, concat_dataset, idx):
        for dataset_idx, dataset in enumerate(concat_dataset.datasets):
            if idx < len(dataset):
                return dataset_idx, idx
            idx -= len(dataset)
        raise IndexError('Index out of range')

    def __iter__(self):
        return iter(self.predictions)

class OriginalMultiResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (12, 24, 36),
                sched_step_size = 10,
                buil_channels = 128,
                buil_kernel = 5,
                pos_weight = 1.0,
                buil_out_chan = 4):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
        self.buil_channels = buil_channels
        self.buil_kernel1 = buil_kernel
        self.buil_out_chan = buil_out_chan
        
        # Main encoder - 4 sentinel channels + buil_out_chan
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        self.encoder.backbone.conv1 = nn.Conv2d(4 + self.buil_out_chan, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.classifier = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)

        # Buildings footprint encoder
        self.buildings_encoder = nn.Sequential(
            nn.Conv2d(1, self.buil_channels, kernel_size=self.buil_kernel1, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(self.buil_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.buil_channels, self.buil_out_chan, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )
        
        # load pretrained weights
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            original_state_dict = checkpoint["state_dict"]

            # Convert any float64 weights to float32
            for key, value in original_state_dict.items():
                if value.dtype == torch.float64:
                    original_state_dict[key] = value.to(torch.float32)
                
            # removing prefix
            state_dict = OrderedDict()
            for key, value in original_state_dict.items():
                if key.startswith('segm_model.'):
                    new_key = key[len('segm_model.'):]
                    state_dict[new_key] = value

            # Extract the original weights of the first convolutional layer
            original_conv1_weight = state_dict['backbone.conv1.weight']
            new_conv1_weight = torch.zeros((64, 8, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            nn.init.kaiming_normal_(new_conv1_weight[:, 4:, :, :], mode='fan_out', nonlinearity='relu')
            state_dict['backbone.conv1.weight'] = new_conv1_weight
                
            self.encoder.load_state_dict(state_dict, strict=False)
        
    def forward(self, batch):
        
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        b_out = self.buildings_encoder(buildings_data)

        concatenated = torch.cat([sentinel_data, b_out], dim=1)    

        segmentation = self.encoder(concatenated)['out']

        return segmentation.squeeze(1)
    
    def training_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)
        
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)
                
        return loss
    
    def validation_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('val_mean_iou', mean_iou)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)     
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, sentinel_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)
        
    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,  # adjust step_size to your needs
            gamma=self.gamma      # adjust gamma to your needs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MultiResolution128DeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (12, 24, 36),
                sched_step_size = 10,
                buil_channels = 128,
                buil_kernel = 5,
                pos_weight = 1.0,
                buil_out_chan = 4):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
        self.buil_channels = buil_channels
        self.buil_kernel1 = buil_kernel
        self.buil_out_chan = buil_out_chan
        
        # Main encoder - 4 sentinel channels + buil_out_chan
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        self.encoder.backbone.conv1 = nn.Conv2d(4 + self.buil_out_chan, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.classifier = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)

        # Buildings footprint encoder
        self.buil_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,5), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(128),
            nn.Conv2d(128, 4, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )
                
        # load pretrained weights
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            original_state_dict = checkpoint["state_dict"]

            # Convert any float64 weights to float32
            for key, value in original_state_dict.items():
                if value.dtype == torch.float64:
                    original_state_dict[key] = value.to(torch.float32)
                
            # removing prefix
            state_dict = OrderedDict()
            for key, value in original_state_dict.items():
                if key.startswith('segm_model.'):
                    new_key = key[len('segm_model.'):]
                    state_dict[new_key] = value

            # Extract the original weights of the first convolutional layer
            original_conv1_weight = state_dict['backbone.conv1.weight']
            new_conv1_weight = torch.zeros((64, 8, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            nn.init.kaiming_normal_(new_conv1_weight[:, 4:, :, :], mode='fan_out', nonlinearity='relu')
            state_dict['backbone.conv1.weight'] = new_conv1_weight
                
            self.encoder.load_state_dict(state_dict, strict=False)
        
    def forward(self, batch):
        
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        buildings_data = buildings_data.to(torch.float32)
        
        # Move data to MPS device
        buildings_data = buildings_data.to(self.device)

        sentinel_data, _ = sentinel_batch
        sentinel_data = sentinel_data.to(self.device)
        
        b_out = self.buil_encoder(buildings_data)

        concatenated = torch.cat([sentinel_data, b_out], dim=1)    

        segmentation = self.encoder(concatenated)['out']

        return segmentation.squeeze(1)
    
    def training_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)
        
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)
                
        return loss
    
    def validation_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('val_mean_iou', mean_iou)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)     
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, sentinel_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)
        
    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,  # adjust step_size to your needs
            gamma=self.gamma      # adjust gamma to your needs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MultiResolutionFPN(pl.LightningModule):
    
    def __init__(self,        
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                sched_step_size = 10,
                pos_weight = 1.0):
        super().__init__()
        super(MultiResolutionFPN, self).__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.gamma = gamma
        self.sched_step_size = sched_step_size
        
        # Sentinel encoder
        self.s1 = self._make_layer(4, 128) # 128x128x128
        self.s2 = self._make_layer(128, 256) # 256x36x36
        self.s3 = self._make_layer(256, 512) # 512x18x18
        
        self.s1_mid = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0) # 64x144x144
        
        # Buildings encoder
        self.e1 = self._make_layer(1, 64) # 64x144x144
        self.e2 = self._make_layer(64, 128) # 128x72x72
        self.e3 = self._make_layer(128, 256) # 256x36x36
        self.e4 = self._make_layer(256, 512) # 512x18x18
        
        # Decoder
        self.d1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1) # 
        self.d2 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1) # 
        self.intrpolate1 = F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)
        self.d3 = nn.ConvTranspose2d(512, 128, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.d4 = nn.ConvTranspose2d(256, 1, kernel_size=1, stride=1, padding=0, output_padding=0) # 
        # self.interpolate2 = F.interpolate(scale_factor=2, mode='bilinear', align_corners=False)
        self.feature_maps = {}
        
    def forward(self, batch):
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        # Move data to the device
        sentinel_input = sentinel_data.to(self.device)
        buildings_input = buildings_data.to(self.device)
        buildings_labels = buildings_labels.to(self.device)
        
        # Sentinel encoder
        s1 = self.s1(sentinel_input)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s1_mid = self.s1_mid(sentinel_input)
        
        # Buildings encoder
        e1 = self.e1(buildings_input)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        
        # Lateral connections
        l1 = torch.cat([s1_mid, e1], dim=1)
        # print(f"l1 shape: {l1.shape}")
        l2 = torch.cat([s1, e2], dim=1)
        # print(f"l2 shape: {l2.shape}")
        l3 = torch.cat([s2, e3], dim=1)
        # print(f"l3 shape: {l3.shape}")
        l4 = torch.cat([s3, e4], dim=1)
        
        # Decoder
        d1 = self.d1(l4)
        # print(f"d1 shape: {d1.shape}")
        d2 = self.d2(torch.cat([d1, l3], dim=1))
        # print(f"d2 shape: {d2.shape}")
        upsamp1 = self.intrpolate1(torch.cat([d2, l2], dim=1))
        d3 = self.d3(upsamp1)
        # print(f"d3 shape: {d3.shape}")
        d4 = self.d4(torch.cat([d3, l1], dim=1))
        # print(f"d4 shape: {d4.shape}")
        # out = self.upsample2(d4)
        # print(f"upsample2 shape: {d4.shape}")

        return d4#.squeeze(1)
    
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def training_step(self, batch):
        
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch
        
        buildings_labels = buildings_labels.unsqueeze(1)
        
        segmentation = self.forward(batch)
        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, buildings_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, buildings_labels)
        precision, recall = self.compute_precision_recall(preds, buildings_labels)

        self.log('train_loss', loss)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_mean_iou', mean_iou)
                
        return loss
    
    def validation_step(self, batch):
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch
        buildings_labels = buildings_labels.unsqueeze(1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, buildings_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, buildings_labels)
        precision, recall = self.compute_precision_recall(preds, buildings_labels)

        self.log('val_mean_iou', mean_iou)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)     
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, buildings_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, buildings_labels)
        precision, recall = self.compute_precision_recall(preds, buildings_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        
    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def compute_precision_recall(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        return precision.mean(), recall.mean()
    
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,
            gamma=self.gamma
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def add_hooks(self):
        def hook_fn(module, input, output):
            self.feature_maps[module] = output

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)

class LateralMultiResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                 use_deeplnafrica: bool = True,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-1,
                 gamma: float = 0.1,
                 atrous_rates = (12, 24, 36),
                 sched_step_size = 10,
                 buil_channels = 128,
                 buil_kernel = 5,
                 pos_weight = 1.0,
                 buil_out_chan = 4):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
        self.buil_channels = buil_channels
        self.buil_kernel1 = buil_kernel
        self.buil_out_chan = buil_out_chan

        # Modified buildings encoder
        self.buildings_encoder = nn.ModuleList([
            self._make_buildings_layer(1, 2, stride=2),    # 512x512 -> 256x256
            self._make_buildings_layer(2, 4, stride=4),   # 256x256 -> 64x64
            self._make_buildings_layer(4, 8, stride=2),  # 64x64 -> 32x32
            self._make_buildings_layer(8, 16, stride=1), # 32x32 -> 32x32
        ])

        # Modified main encoder
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)
        self.encoder.backbone.conv1 = nn.Conv2d(4 + 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            original_state_dict = checkpoint["state_dict"]

            # Convert any float64 weights to float32
            for key, value in original_state_dict.items():
                if value.dtype == torch.float64:
                    original_state_dict[key] = value.to(torch.float32)
                
            # removing prefix
            state_dict = OrderedDict()
            for key, value in original_state_dict.items():
                if key.startswith('segm_model.'):
                    new_key = key[len('segm_model.'):]
                    state_dict[new_key] = value

            # Extract the original weights of the first convolutional layer
            original_conv1_weight = state_dict['backbone.conv1.weight']
            new_conv1_weight = torch.zeros((64, 4 + 2, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            nn.init.kaiming_normal_(new_conv1_weight[:, 4:, :, :], mode='fan_out', nonlinearity='relu')
            state_dict['backbone.conv1.weight'] = new_conv1_weight
                
            self.encoder.load_state_dict(state_dict, strict=False)
            
        # Modify ResNet layers to output intermediate features
        self.encoder.backbone.layer1.register_forward_hook(self._get_intermediate_feat)
        self.encoder.backbone.layer2.register_forward_hook(self._get_intermediate_feat)
        self.encoder.backbone.layer3.register_forward_hook(self._get_intermediate_feat)

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 + 4, 256, kernel_size=1),    # After layer1
            nn.Conv2d(512 + 8, 512, kernel_size=1),   # After layer2
            nn.Conv2d(1024 + 16, 1024, kernel_size=1), # After layer3
        ])
        
    def _make_buildings_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _get_intermediate_feat(self, module, input, output):
        setattr(self, f'{module.__class__.__name__}_feat', output)

    def forward(self, batch):
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        buildings_data = buildings_data.to(self.device)
        sentinel_data, _ = sentinel_batch
        sentinel_data = sentinel_data.to(self.device)
        
        input_shape = sentinel_data.shape[-2:]

        # Process buildings data
        b_feats = []
        x = buildings_data
        for layer in self.buildings_encoder:
            x = layer(x)
            # print(f"Shape of buildings features: {x.shape}")
            b_feats.append(x)

        # Process sentinel data
        x = torch.cat([sentinel_data, b_feats[0]], dim=1)
        # print(f"Shape after initial concatenation: {x.shape}")
        
        x = self.encoder.backbone.conv1(x)
        x = self.encoder.backbone.bn1(x)
        x = self.encoder.backbone.relu(x)
        x = self.encoder.backbone.maxpool(x)
        # print(f"Shape after initial main encoder layers: {x.shape}")
        
        # Apply ResNet layers and fuse with buildings features
        x = self.encoder.backbone.layer1(x)
        # print(f"Shape after layer1: {x.shape}")
        x = self.fusion_layers[0](torch.cat([x, b_feats[1]], dim=1))
        # print(f"Shape after fusion1: {x.shape}")

        x = self.encoder.backbone.layer2(x)
        x = self.fusion_layers[1](torch.cat([x, b_feats[2]], dim=1))
        # print(f"Shape after layer2 and fusion2: {x.shape}")

        x = self.encoder.backbone.layer3(x)
        # print(f"Shape after layer3: {x.shape}")
        x = self.fusion_layers[2](torch.cat([x, b_feats[3]], dim=1))
        # print(f"Shape after fusion3: {x.shape}")

        x = self.encoder.backbone.layer4(x)
        # print(f"Shape after layer4: {x.shape}")

        x = self.encoder.classifier(x)
        # print(f"Shape after classifier: {x.shape}")

        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x.squeeze(1)

    def training_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)
        
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)
                
        return loss
    
    def validation_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('val_mean_iou', mean_iou)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)     
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, sentinel_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)
        
    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall
    
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,  # adjust step_size to your needs
            gamma=self.gamma      # adjust gamma to your needs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

class MultiResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                 use_deeplnafrica: bool = True,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-1,
                 gamma: float = 0.1,
                 atrous_rates = (12, 24, 36),
                 sched_step_size = 10,
                 buil_channels = 128,
                 buil_kernel = 5,
                 pos_weight = 1.0,
                 buil_out_chan = 4):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.sched_step_size = sched_step_size
        self.buil_channels = buil_channels
        self.buil_kernel1 = buil_kernel
        self.buil_out_chan = buil_out_chan

        # Modified buildings encoder
        self.buildings_encoder = nn.ModuleList([
            self._make_buildings_layer(1, 2, stride=2),    # 512x512 -> 256x256
            self._make_buildings_layer(2, 4, stride=4),   # 256x256 -> 64x64
            self._make_buildings_layer(4, 8, stride=2),  # 64x64 -> 32x32
            self._make_buildings_layer(8, 16, stride=1), # 32x32 -> 32x32
        ])

        # Modified main encoder
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)
        self.encoder.backbone.conv1 = nn.Conv2d(4 + 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        if use_deeplnafrica:
            allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
            checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
            original_state_dict = checkpoint["state_dict"]

            # Convert any float64 weights to float32
            for key, value in original_state_dict.items():
                if value.dtype == torch.float64:
                    original_state_dict[key] = value.to(torch.float32)
                
            # removing prefix
            state_dict = OrderedDict()
            for key, value in original_state_dict.items():
                if key.startswith('segm_model.'):
                    new_key = key[len('segm_model.'):]
                    state_dict[new_key] = value

            # Extract the original weights of the first convolutional layer
            original_conv1_weight = state_dict['backbone.conv1.weight']
            new_conv1_weight = torch.zeros((64, 4 + 2, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            nn.init.kaiming_normal_(new_conv1_weight[:, 4:, :, :], mode='fan_out', nonlinearity='relu')
            state_dict['backbone.conv1.weight'] = new_conv1_weight
                
            self.encoder.load_state_dict(state_dict, strict=False)
            
        # Modify ResNet layers to output intermediate features
        self.encoder.backbone.layer1.register_forward_hook(self._get_intermediate_feat)
        self.encoder.backbone.layer2.register_forward_hook(self._get_intermediate_feat)
        self.encoder.backbone.layer3.register_forward_hook(self._get_intermediate_feat)

        # Fusion layers for the encoder
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 + 4, 256, kernel_size=1),    # After layer1
            nn.Conv2d(512 + 8, 512, kernel_size=1),   # After layer2
            nn.Conv2d(1024 + 16, 1024, kernel_size=1), # After layer3
        ])
        
        # Additional fusion layer to reduce dimensions of fused features
        self.xtra_fusion = nn.Conv2d(256, 2, kernel_size=1, padding='same')

        # Fusion layer for the decoder
        self.decoder_fusion = nn.Conv2d(3, 1, kernel_size=1, padding='same')

    def _make_buildings_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _get_intermediate_feat(self, module, input, output):
        setattr(self, f'{module.__class__.__name__}_feat', output)

    def forward(self, batch):
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        buildings_data = buildings_data.to(self.device)
        sentinel_data, _ = sentinel_batch
        sentinel_data = sentinel_data.to(self.device)
        
        input_shape = sentinel_data.shape[-2:]

        # Process buildings data
        b_feats = []
        x = buildings_data
        for layer in self.buildings_encoder:
            x = layer(x)
            b_feats.append(x)
            # print(f"Shape of buildings features: {x.shape}")

        # Process sentinel data
        x = torch.cat([sentinel_data, b_feats[0]], dim=1)
        
        x = self.encoder.backbone.conv1(x)
        x = self.encoder.backbone.bn1(x)
        x = self.encoder.backbone.relu(x)
        x = self.encoder.backbone.maxpool(x)
        
        # Apply ResNet layers and fuse with buildings features
        x = self.encoder.backbone.layer1(x)
        # print(f"x layer 1: {x.shape}")
        x = self.fusion_layers[0](torch.cat([x, b_feats[1]], dim=1))
        x_fus64 = self.xtra_fusion(x)
        # print(f"x_fus64 shape: {x_fus64.shape}")

        x = self.encoder.backbone.layer2(x)
        x = self.fusion_layers[1](torch.cat([x, b_feats[2]], dim=1))
        
        x = self.encoder.backbone.layer3(x)
        x = self.fusion_layers[2](torch.cat([x, b_feats[3]], dim=1))

        x = self.encoder.backbone.layer4(x)

        x = self.encoder.classifier(x) # torch.Size([16, 1, 32, 32])
        
        # Upsample to 64x64
        x_64 = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False) #torch.Size([16, 1, 64, 64])
        # print(f"x_64 shape: {x_64.shape}")
        # Fuse with encoder features which are also 64x64
        fused_64 = self.decoder_fusion(torch.cat([x_64, x_fus64], dim=1))

        # Final upsampling to input size (256x256)
        x = F.interpolate(fused_64, size=input_shape, mode='bilinear', align_corners=False)
                
        return x.squeeze(1)

    def training_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)
        
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)
                
        return loss
    
    def validation_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)
        
        self.log('val_mean_iou', mean_iou)
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.to(self.device)
        sentinel_labels = sentinel_labels.squeeze(-1)

        segmentation = self.forward(batch)     
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, sentinel_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)
        
    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall
    
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,  # adjust step_size to your needs
            gamma=self.gamma      # adjust gamma to your needs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


# Feature Map Visualisation class
class FeatureMapVisualization:
    def __init__(self, model, device):
        self.model = model
        self.feature_maps = {}
        self.hooks = []
        self.device = device

    def add_hooks(self, layer_names):
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(self.save_feature_map(name)))

    def save_feature_map(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def visualize_feature_maps(self, layer_name, input_data, num_feature_maps='all', figsize=(20, 20)):
        self.model.eval()
        with torch.no_grad():
            # Force model to device again
            self.model = self.model.to(self.device)
            
            # Ensure input_data is on the correct device
            if isinstance(input_data, list):
                input_data = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in input_data]
            elif isinstance(input_data, dict):
                input_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
            elif isinstance(input_data, torch.Tensor):
                input_data = input_data.to(self.device)
            
            # Double-check that all model parameters are on the correct device
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            
            self.model(input_data)

        if layer_name not in self.feature_maps:
            print(f"Layer {layer_name} not found in feature maps.")
            return

        feature_maps = self.feature_maps[layer_name].cpu().detach().numpy()

        # Handle different dimensions
        if feature_maps.ndim == 4:  # (batch_size, channels, height, width)
            feature_maps = feature_maps[0]  # Take the first item in the batch
        elif feature_maps.ndim == 3:  # (channels, height, width)
            pass
        else:
            print(f"Unexpected feature map shape: {feature_maps.shape}")
            return

        total_maps = feature_maps.shape[0]
        
        if num_feature_maps == 'all':
            num_feature_maps = total_maps
        else:
            num_feature_maps = min(num_feature_maps, total_maps)

        # Calculate grid size
        grid_size = math.ceil(math.sqrt(num_feature_maps))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                if index < num_feature_maps:
                    feature_map_img = feature_maps[index]
                    im = axes[i, j].imshow(feature_map_img, cmap='viridis')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Channel {index+1}')
                else:
                    axes[i, j].axis('off')

        fig.suptitle(f'Feature Maps for Layer: {layer_name}\n({num_feature_maps} out of {total_maps} channels)')
        fig.tight_layout()
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.show()
        
class SimplifiedASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimplifiedASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False)
        
        self.project = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        
        x = torch.cat((x1, x2, x3), dim=1)
        return self.project(x)
    
class MultiResolutionFPNwASPP(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-2, weight_decay: float = 1e-1, gamma: float = 0.1, sched_step_size = 10, pos_weight = 1.0):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = torch.tensor(pos_weight, device='mps')
        self.gamma = gamma
        self.sched_step_size = sched_step_size
        
        # Sentinel encoder
        self.s1 = self._make_layer(4, 128, stride=2)  # 128x128
        self.s2 = self._make_layer(128, 256, stride=2)  # 64x64
        self.s3 = self._make_layer(256, 512, stride=2)  # 32x32
        
        self.s1_mid = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0)  # 256x256
        
        # Buildings encoder
        self.e0 = self._make_layer(1, 32, stride=1)  # 512x512
        self.e1 = self._make_layer(32, 64, stride=2)  # 256x256
        self.e2 = self._make_layer(64, 128, stride=2)  # 128x128
        self.e3 = self._make_layer(128, 256, stride=2)  # 64x64
        self.e4 = self._make_layer(256, 512, stride=2)  # 32x32
        
        # ASPP modules
        self.aspp1 = SimplifiedASPP(256, 128)
        self.aspp2 = SimplifiedASPP(512, 256)
        self.aspp3 = SimplifiedASPP(1024, 512)
        
        # Decoder
        self.d1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)


    def _make_layer(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch):
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        # Sentinel encoder
        s1_mid = self.s1_mid(sentinel_data)  # [4, 64, 256, 256]
        s1 = self.s1(sentinel_data)  # [4, 128, 128, 128]
        s2 = self.s2(s1)  # [4, 256, 64, 64]
        s3 = self.s3(s2)  # [4, 512, 32, 32]
        
        # Buildings encoder
        e0 = self.e0(buildings_data)  # [4, 32, 512, 512]
        e1 = self.e1(e0)  # [4, 64, 256, 256]
        e2 = self.e2(e1)  # [4, 128, 128, 128]
        e3 = self.e3(e2)  # [4, 256, 64, 64]
        e4 = self.e4(e3)  # [4, 512, 32, 32]
        
        # Lateral connections with ASPP
        l2 = self.aspp1(torch.cat([s1, e2], dim=1))  # [4, 128, 128, 128]
        l3 = self.aspp2(torch.cat([s2, e3], dim=1))  # [4, 256, 64, 64]
        l4 = self.aspp3(torch.cat([s3, e4], dim=1))  # [4, 512, 32, 32]
        
        # Decoder
        d1 = self.d1(l4)  # [4, 256, 64, 64]
        d2 = self.d2(torch.cat([d1, l3], dim=1))  # [4, 128, 128, 128]
        d3 = self.d3(torch.cat([d2, l2], dim=1))  # [4, 64, 256, 256]
        output = self.final(torch.cat([d3, s1_mid], dim=1))  # [4, 1, 256, 256]

        return output
    
    def training_step(self, batch):
        
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        
        sentinel_labels = sentinel_labels.unsqueeze(1)
        
        segmentation = self.forward(batch)
        
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_precision', mean_precision)
        self.log('train_recall', mean_recall)
                
        return loss
    
    def validation_step(self, batch):
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch
        sentinel_labels = sentinel_labels.unsqueeze(1)

        segmentation = self.forward(batch)      
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        val_loss = loss_fn(segmentation, sentinel_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('val_loss', val_loss)
        self.log('val_mean_iou', mean_iou)
        self.log('val_precision', mean_precision)
        self.log('val_recall', mean_recall)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        sentinel_batch, _ = batch
        _, sentinel_labels = sentinel_batch

        segmentation = self.forward(batch)     
        assert segmentation.shape == sentinel_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {sentinel_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, sentinel_labels)
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou, mean_precision, mean_recall = self.compute_metrics(preds, sentinel_labels)

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        self.log('test_precision', mean_precision)
        self.log('test_recall', mean_recall)
    
    def compute_metrics(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        
        # IoU computation
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        mean_iou = iou.mean()
        
        # Precision and Recall computation
        true_positives = (preds & target).sum((1, 2))
        predicted_positives = preds.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = true_positives.float() / (predicted_positives.float() + 1e-10)
        recall = true_positives.float() / (actual_positives.float() + 1e-10)
        
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        
        return mean_iou, mean_precision, mean_recall

        
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.sched_step_size,
            gamma=self.gamma
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def add_hooks(self):
        def hook_fn(module, input, output):
            self.feature_maps[module] = output

        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)