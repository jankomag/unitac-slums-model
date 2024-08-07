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
import pandas as pd
from typing import Any, Optional, Tuple, Union, Sequence
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Subset
import math
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

from src.features.dataloaders import (ensure_tuple, MultiInputCrossValidator,
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

def merge_geojson_files(country_directory, output_file):
    merged_gdf = gpd.GeoDataFrame()
    for city in os.listdir(country_directory):
        city_path = os.path.join(country_directory, city)
        if os.path.isdir(city_path):
            for split in ['split0_predictions', 'split1_predictions']:
                split_path = os.path.join(city_path, split)
                vector_output_path = os.path.join(split_path, 'vector_output')
                if os.path.isdir(vector_output_path):
                    json_file = os.path.join(vector_output_path, 'class-1-slums.json')
                    if os.path.exists(json_file):
                        try:
                            gdf = gpd.read_file(json_file)
                            if not gdf.empty and 'geometry' in gdf.columns:
                                gdf['city'] = city
                                gdf['split'] = split
                                merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
                            else:
                                print(f"Skipping empty or invalid GeoJSON file: {json_file}")
                        except Exception as e:
                            print(f"Error reading file {json_file}: {str(e)}")
                    else:
                        print(f"JSON file not found: {json_file}")
                else:
                    print(f"Vector output directory not found: {vector_output_path}")
    
    if not merged_gdf.empty and 'geometry' in merged_gdf.columns:
        merged_gdf.to_file(output_file, driver='GeoJSON')
        print(f'Merged GeoJSON file saved to {output_file}')
    else:
        print("No valid geometries found. Merged GeoJSON file not created.")

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

def generate_and_display_predictions(model, eval_sent_scene, eval_buil_scene, device, epoch, class_config):
        # Set up the prediction dataset
        sent_strided_fullds = CustomSlidingWindowGeoDataset(eval_sent_scene, size=256, stride=256, padding=128, city='SD', transform=None, transform_type=TransformType.noop)
        buil_strided_fullds = CustomSlidingWindowGeoDataset(eval_buil_scene, size=512, stride=512, padding=256, city='SD', transform=None, transform_type=TransformType.noop)
        mergedds = MergeDataset(sent_strided_fullds, buil_strided_fullds)

        # Generate predictions
        predictions_iterator = MultiResPredictionsIterator(model, mergedds, device=device)
        windows, predictions = zip(*predictions_iterator)

        # Create SemanticSegmentationLabels from predictions
        pred_labels = SemanticSegmentationLabels.from_predictions(
            windows,
            predictions,
            extent=eval_sent_scene.extent,
            num_classes=len(class_config),
            smooth=True
        )

        gt_labels = eval_sent_scene.label_source.get_labels()

        # Show predictions
        # fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5)
        scores = pred_labels.get_score_arr(pred_labels.extent)

        # Create a figure with three subplots side by side
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot smooth predictions
        scores_class = scores[1]
        im1 = ax.imshow(scores_class, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Smooth Predictions (Class {1} Scores)')
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig, ax
        # Log the figure to WandB
        wandb.log({f"Predictions_Epoch_{epoch}": wandb.Image(fig)})
        
        plt.close(fig)  # Close the figure to free up memory
        
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

class PredictionsIterator:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                x, _ = self.get_item(dataset, idx)
                x = x.unsqueeze(0).to(device)
                
                output = model(x)
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                window = self.get_window(dataset, idx)
                self.predictions.append((window, probabilities))
    
    def get_item(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_item(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_item(dataset.datasets[dataset_idx], sample_idx)
        else:
            return dataset[idx]
    
    def get_window(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_window(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_window(dataset.datasets[dataset_idx], sample_idx)
        else:
            return dataset.windows[idx]
    
    def get_concat_dataset_indices(self, concat_dataset, idx):
        for dataset_idx, dataset in enumerate(concat_dataset.datasets):
            if idx < len(dataset):
                return dataset_idx, idx
            idx -= len(dataset)
        raise IndexError('Index out of range')
    
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
        x = self.encoder.backbone.layer1(x) # outputs torch.Size([16, 256, 64, 64])
        # print(f"x layer 1: {x.shape}")
        x = self.fusion_layers[0](torch.cat([x, b_feats[1]], dim=1)) # outputs torch.Size([16, 256, 64, 64])
        x_fus64 = self.xtra_fusion(x)
        # print(f"x_fus64 shape: {x_fus64.shape}")

        x = self.encoder.backbone.layer2(x)
        x = self.fusion_layers[1](torch.cat([x, b_feats[2]], dim=1)) # outputs torch.Size([16, 512, 32, 32])
        
        x = self.encoder.backbone.layer3(x)
        x = self.fusion_layers[2](torch.cat([x, b_feats[3]], dim=1)) # outputs torch.Size([16, 1024, 16, 16])

        x = self.encoder.backbone.layer4(x) # outputs torch.Size([16, 2048, 16, 16])

        x = self.encoder.classifier(x) # outputs torch.Size([16, 1, 32, 32])
        
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
                
    def visualize_maximized_activation(self, layer_name, num_iterations=100, learning_rate=0.1, num_feature_maps='all', figsize=(20, 20)):
        self.model.eval()

        # Create optimizable inputs
        sentinel_data = torch.randn(1, 4, 256, 256, requires_grad=True, device=self.device)
        buildings_data = torch.randn(1, 1, 256, 256, requires_grad=True, device=self.device)

        # Create synthetic labels (these won't be used in forward pass, but are needed for unpacking)
        sentinel_labels = torch.zeros(1, 1, 256, 256, device=self.device)
        buildings_labels = torch.zeros(1, 1, 256, 256, device=self.device)

        # Normalization parameters (adjust these based on your data)
        sentinel_mean = [0.485, 0.456, 0.406, 0.5]
        sentinel_std = [0.229, 0.224, 0.225, 0.25]
        buildings_mean = [0.5]
        buildings_std = [0.5]

        optimizer = optim.Adam([sentinel_data, buildings_data], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Normalize inputs
            norm_sentinel = Normalize(mean=sentinel_mean, std=sentinel_std)(sentinel_data)
            norm_buildings = Normalize(mean=buildings_mean, std=buildings_std)(buildings_data)

            # Create batch structure expected by the model
            sentinel_batch = (norm_sentinel, sentinel_labels)
            buildings_batch = (norm_buildings, buildings_labels)
            batch = (sentinel_batch, buildings_batch)

            # Forward pass
            self.model(batch)

            # Get activation of target layer
            if layer_name not in self.feature_maps:
                print(f"Layer {layer_name} not found in feature maps.")
                return

            activation = self.feature_maps[layer_name]

            # Define objective (e.g., mean activation)
            objective = activation.mean()

            # Backward pass
            objective.backward()
            optimizer.step()

        # Visualize the maximized activation
        self.visualize_feature_maps(layer_name, batch, num_feature_maps, figsize)

        return sentinel_data.detach(), buildings_data.detach()