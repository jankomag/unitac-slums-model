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

from typing import Iterator, Optional
from torch.optim import AdamW
import torch
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    SemanticSegmentationLabelSource)

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from rastervision.core.data import (RasterioCRSTransformer,
                                    RasterioCRSTransformer)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)

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

def create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5):
    gdf = gpd.read_file(buildings_uri)
    gdf = gdf.to_crs('EPSG:3857')
    xmin, _, _, ymax = gdf.total_bounds
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings

    buildings_vector_source = GeoJSONVectorSource(
        buildings_uri,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source,
        background_class_id=0)

    print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")

    label_vector_source = GeoJSONVectorSource(label_uri,
        crs_transformer_buildings,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])

    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    buildings_label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)

    return rasterized_buildings_source, buildings_label_source, crs_transformer_buildings

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
    def __init__(self, model, sentinelGeoDataset, buildingsGeoDataset, device='cuda'):
        self.model = model
        self.sentinelGeoDataset = sentinelGeoDataset
        self.dataset = buildingsGeoDataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(buildingsGeoDataset)):
                buildings = buildingsGeoDataset[idx]
                sentinel = sentinelGeoDataset[idx]
                
                sentinel_data = sentinel[0].unsqueeze(0).to(device)
                sentlabels = sentinel[1].unsqueeze(0).to(device)

                buildings_data = buildings[0].unsqueeze(0).to(device)
                labels = buildings[1].unsqueeze(0).to(device)

                output = self.model(((sentinel_data,sentlabels), (buildings_data,labels)))
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = sentinelGeoDataset.windows[idx] # here has to be sentinelGeoDataset to work
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)

class MultiResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (12, 24, 36),
                sched_step_size = 10,
                buil_channels = 128,
                buil_kernel1 = 5,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.sched_step_size = sched_step_size
        self.buil_channels = buil_channels
        self.buil_kernel1 = buil_kernel1
        
        # Main encoder - 4 sentinel channels + 1 buildings channel
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        self.encoder.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.classifier = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)

        # Buildings footprint encoder
        self.buildings_encoder = nn.Sequential(
            nn.Conv2d(1, self.buil_channels, kernel_size=self.buil_kernel1, stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(self.buil_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.buil_channels, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
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
            new_conv1_weight = torch.zeros((64, 5, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            new_conv1_weight[:, 4, :, :] = original_conv1_weight.mean(dim=1) #[:, 3, :, :] if red to be copied
            state_dict['backbone.conv1.weight'] = new_conv1_weight
                
            self.encoder.load_state_dict(state_dict, strict=False)
        
    def forward(self, batch):
        
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        b_out = self.buildings_encoder(buildings_data)
        # print("Buildings encoder output stats:", b_out.min(), b_out.max(), b_out.mean())
        
        concatenated = torch.cat([sentinel_data, b_out], dim=1)    
        # print("Concatenated data stats:", concatenated.min(), concatenated.max(), concatenated.mean())

        segmentation = self.encoder(concatenated)['out']
        # print("Segmentation output stats:", segmentation.min(), segmentation.max(), segmentation.mean())
        
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
