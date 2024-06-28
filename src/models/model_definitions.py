import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from rastervision.core.box import Box
import stackstac
from pathlib import Path
import torch.nn as nn
import cv2
from os.path import join
from collections import OrderedDict

import json
from shapely.geometry import shape, mapping
from shapely.affinity import translate

from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader
from typing import List
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers.wandb import WandbLogger
import os
from datetime import datetime
import wandb
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Iterator, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torch.utils.data import ConcatDataset

from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    MinMaxTransformer, MultiRasterSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource, CRSTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    SemanticSegmentationLabelSource)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore

from rastervision.core.data.utils import pad_to_window_size
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
                                          SemanticSegmentationVisualizer, SlidingWindowGeoDataset)
from rastervision.pipeline.utils import repr_with_args
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
from xarray import DataArray

from affine import Affine
import numpy as np
import geopandas as gpd
from rastervision.core.box import Box
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from typing import List
from rasterio.features import rasterize
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from deeplnafrica.deepLNAfrica import init_segm_model
import folium

from src.data.dataloaders import (
    create_datasets,
    create_buildings_raster_source, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
# implements loading gdf - class CustomGeoJSONVectorSource
class CustomGeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON files or GeoDataFrames."""

    def __init__(self,
                 crs_transformer: 'CRSTransformer',
                 uris: Optional[Union[str, List[str]]] = None,
                 gdf: Optional[gpd.GeoDataFrame] = None,
                 vector_transformers: List['VectorTransformer'] = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            uris (Optional[Union[str, List[str]]]): URI(s) of the GeoJSON file(s).
            gdf (Optional[gpd.GeoDataFrame]): A GeoDataFrame with vector data.
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            bbox (Optional[Box]): User-specified crop of the extent. If None,
                the full extent available in the source file is used.
        """
        self.uris = listify_uris(uris) if uris is not None else None
        self.gdf = gdf
        super().__init__(
            crs_transformer,
            vector_transformers=vector_transformers,
            bbox=bbox)

    def _get_geojson(self) -> dict:
        if self.gdf is not None:
            # Convert GeoDataFrame to GeoJSON
            df = self.gdf.to_crs('epsg:4326')
            geojson = df.__geo_interface__
        elif self.uris is not None:
            geojsons = [self._get_geojson_single(uri) for uri in self.uris]
            geojson = merge_geojsons(geojsons)
        else:
            raise ValueError("Either 'uris' or 'gdf' must be provided.")
        return geojson

    def _get_geojson_single(self, uri: str) -> dict:
        # download first so that it gets cached
        path = download_if_needed(uri)
        df: gpd.GeoDataFrame = gpd.read_file(path)
        df = df.to_crs('epsg:4326')
        geojson = df.__geo_interface__
        return geojson

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
class SentinelDeeplabv3(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        self.deeplab = deeplabv3_resnet50(pretrained=False, progress=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.deeplab.classifier = DeepLabHead(2048, 1, atrous_rates = (12, 24, 36))
        
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
        
        # Move data to the device
        x = x.to(self.device)

        x = self.deeplab(x)['out']#.squeeze(dim=1)
        x = x.permute(0, 2, 3, 1)

        return x

    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        segmentation = self(img)
        groundtruth = groundtruth.float().to(self.device)
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('val_loss', loss)
        self.log('val_mean_iou', mean_iou.item())

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, informal_gt)

        self.log('test_loss', loss)
        self.log('test_mean_iou', mean_iou)

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

class SentinelSimpleSS(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        self.encoder1 = self.conv_block(4, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Move data to the device
        x = x.to(self.device)

        # Encoder
        e1 = self.encoder1(x)
        # print(f"e1 shape: {e1.shape}")
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        # print(f"e2 shape: {e2.shape}")
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        # print(f"e3 shape: {e3.shape}")
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        # print(f"e4 shape: {e4.shape}")

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        # print(f"b shape: {b.shape}")
        
        # Decoder
        d4 = self.upconv4(b)
        # print(f"d4 shape: {d4.shape}")
        d4 = torch.cat([d4, e4], dim=1)
        # print(f"d4 shape after cat: {d4.shape}")
        d4 = self.decoder4(d4)
        # print(f"d4 shape after decoder: {d4.shape}")
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final Convolution
        out = self.final_conv(d1)

        return out

    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
    
    def compute_dice_coefficient(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).permute(0,3,1,2)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)
        dice = self.compute_dice_coefficient(preds, groundtruth)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_dice', dice)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).permute(0,3,1,2)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)
        dice = self.compute_dice_coefficient(preds, groundtruth)

        self.log('val_loss', loss)
        self.log('val_mean_iou', mean_iou.item())
        self.log('val_dice', dice)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, informal_gt)

        self.log('test_loss', loss)
        self.log('test_mean_iou', mean_iou)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.parameters(),
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


# Buildings Only Models
class BuildingsDeeplabv3(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        self.deeplab = deeplabv3_resnet50(pretrained=False, progress=False)
        self.deeplab.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.deeplab.classifier = DeepLabHead(2048, 1, atrous_rates = (12, 24, 36))
        
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

    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        segmentation = self(img)
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('val_loss', loss)
        self.log('val_mean_iou', mean_iou.item())

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, informal_gt)

        self.log('test_loss', loss)
        self.log('test_mean_iou', mean_iou)

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

class BuildingsUNET(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Move data to the device
        x = x.to(self.device)

        # Encoder
        e1 = self.encoder1(x)
        # print(f"e1 shape: {e1.shape}")
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        # print(f"e2 shape: {e2.shape}")
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        # print(f"e3 shape: {e3.shape}")
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        # print(f"e4 shape: {e4.shape}")

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        # print(f"b shape: {b.shape}")
        
        # Decoder
        d4 = self.upconv4(b)
        # print(f"d4 shape: {d4.shape}")
        d4 = torch.cat([d4, e4], dim=1)
        # print(f"d4 shape after cat: {d4.shape}")
        d4 = self.decoder4(d4)
        # print(f"d4 shape after decoder: {d4.shape}")
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final Convolution
        out = self.final_conv(d1)

        return out

    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()
    
    def compute_dice_coefficient(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)
        dice = self.compute_dice_coefficient(preds, groundtruth)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
        self.log('train_dice', dice)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float().to(self.device).unsqueeze(1)
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)
        dice = self.compute_dice_coefficient(preds, groundtruth)

        self.log('val_loss', loss)
        self.log('val_mean_iou', mean_iou.item())
        self.log('val_dice', dice)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(self.device))

        informal_gt = groundtruth[:, 0, :, :].float().to(self.device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, informal_gt)

        self.log('test_loss', loss)
        self.log('test_mean_iou', mean_iou)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.parameters(),
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

class BuildingsOnlyPredictionsIterator:
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
    

# Mulitmodal Model definition
class FusionModule(nn.Module):
    def __init__(self, sentinel_channels, buildings_channels):
        super().__init__()
        self.buildings_conv = nn.Conv2d(buildings_channels, sentinel_channels, kernel_size=3, padding=1)
        self.fusion_conv = nn.Conv2d(sentinel_channels * 2, sentinel_channels, kernel_size=1)
        
    def forward(self, sentinel_features, buildings_features):
        buildings_processed = self.buildings_conv(buildings_features)
        buildings_upsampled = F.interpolate(buildings_processed, size=sentinel_features.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat([sentinel_features, buildings_upsampled], dim=1)
        fused = self.fusion_conv(combined)
        return fused
    
class MultiModalPredictionsIterator:
    def __init__(self, model, sentinelGeoDataset, buildingsGeoDataset, device='cuda'):
        self.model = model
        self.sentinelGeoDataset = sentinelGeoDataset
        self.dataset = buildingsGeoDataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(sentinelGeoDataset)):
                buildings = buildingsGeoDataset[idx]
                sentinel = sentinelGeoDataset[idx]
                
                sentinel_data = sentinel[0].unsqueeze(0).to(device)
                sentlabels = sentinel[1].unsqueeze(0).to(device)

                buildings_data = buildings[0].unsqueeze(0).to(device)
                labels = buildings[1].unsqueeze(0).to(device)

                output = self.model(((sentinel_data,sentlabels), (buildings_data,labels)))
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = buildingsGeoDataset.windows[idx]
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)


class MultiResolutionFPN(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                sched_step_size = 10,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')):
        super().__init__()
        super(MultiResolutionFPN, self).__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.sched_step_size = sched_step_size
        
        # Sentinel encoder
        self.s1 = self._make_layer(4, 128) # 128x72x72
        self.s2 = self._make_layer(128, 256) # 256x36x36
        self.s3 = self._make_layer(256, 512) # 512x18x18
        
        self.s1_mid = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0) # 64x144x144
        
        # Buildings encoder
        self.e1 = self._make_layer(1, 64) # 64x144x144
        self.e2 = self._make_layer(64, 128) # 128x72x72
        self.e3 = self._make_layer(128, 256) # 256x36x36
        self.e4 = self._make_layer(256, 512) # 512x18x18
        
        # Decoder
        self.d1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1) # 512x36x36
        self.d2 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1) # 256x72x72
        self.d3 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # 128x144x144
        self.d4 = nn.ConvTranspose2d(256, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 1x288x288
        
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
        l3 = torch.cat([s2, e3], dim=1)
        # print(f"l3 shape: {l3.shape}")
        l4 = torch.cat([s3, e4], dim=1)
        
        # Decoder
        d1 = self.d1(l4)
        # print(f"d1 shape: {d1.shape}")
        d2 = self.d2(torch.cat([d1, l3], dim=1))
        d3 = self.d3(torch.cat([d2, l2], dim=1))
        # print(f"d3 shape: {d3.shape}")
        out = self.d4(torch.cat([d3, l1], dim=1))
        # print(f"out shape: {out.shape}")
        return out#.squeeze(1)
    
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


class MultiModalSegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.learning_rate = 1e-4
        self.weight_decay = 0
        self.sentinel_encoder = deeplabv3_resnet50(pretrained=False, progress=False)
        self.buildings_encoder = deeplabv3_resnet50(pretrained=False, progress=True)
        
        self.sentinel_encoder.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.buildings_encoder.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # load pretrained deeplnafrica weights into sentinel channels
        allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
        checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
        state_dict = checkpoint["state_dict"]
        
        # Convert any float64 weights to float32
        for key, value in state_dict.items():
            if value.dtype == torch.float64:
                state_dict[key] = value.to(torch.float32)
                
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('segm_model.backbone.'):
                new_key = key[len('segm_model.backbone.'):]  # Remove the backbone prefix
                new_state_dict[new_key] = value
                
        self.sentinel_encoder.backbone.load_state_dict(new_state_dict, strict=True)

        # Intermediate Layer Getters
        self.sentinel_encoder_backbone = IntermediateLayerGetter(self.sentinel_encoder.backbone, {'layer4': 'out_sent', 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        self.buildings_encoder_backbone = IntermediateLayerGetter(self.buildings_encoder.backbone, {'layer4': 'out_buil', 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        self.buildings_downsampler = nn.Conv2d(2048, 2048, kernel_size=2, stride=2)
        
        self.fusion_layer = nn.Conv2d(4096, 2048, kernel_size=1)
        
        # fusion with an additional activation layer
        self.fusion = nn.Sequential(
            nn.Conv2d(4096, 2048, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 3, padding=1)
        )
        
        self.segmentation_head = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=(6, 12, 24))
             
    def forward(self, batch):

        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        # Move data to the device
        sentinel_data = sentinel_data.to(self.device)
        buildings_data = buildings_data.to(self.device)
        buildings_labels = buildings_labels.to(self.device)
        
        sentinel_features = self.sentinel_encoder_backbone(sentinel_data)
        buildings_features = self.buildings_encoder_backbone(buildings_data)
        
        sentinel_out = sentinel_features['out_sent']
        buildings_out = buildings_features['out_buil']
        buildings_out_downsampled = self.buildings_downsampler(buildings_out)
        
        # concatenated = sentinel_out+buildings_out_downsampled # addition works well on SD
        
        concatenated = torch.cat([sentinel_out, buildings_out_downsampled], dim=1)
        # print(f"concatenated features shape: {concatenated.shape}")
        
        # Decode the fused features
        # fused_features = self.fusion_layer(concatenated)
        fused_features = self.fusion(concatenated)
        # print(f"fused_features shape: {fused_features.shape}")

        segmentation = self.segmentation_head(fused_features)
        
        segmentation = F.interpolate(segmentation, size=288, mode="bilinear", align_corners=False)
        
        return segmentation.squeeze(1)
    
    def training_step(self, batch):
        
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)
        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(segmentation, buildings_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, buildings_labels)
        
        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)
                
        return loss
    
    def validation_step(self, batch):
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)      
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fn(segmentation, buildings_labels.float())
        
        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, buildings_labels)
        
        self.log('val_mean_iou', mean_iou)
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

        self.log('test_loss', test_loss)
        self.log('test_mean_iou', mean_iou)
        
    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,  # adjust step_size to your needs
            gamma=0.1      # adjust gamma to your needs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }



# code for loading 5 channel wieghts
# new_state_dict = OrderedDict()
        # for 5 channels backbone
        # for key, value in state_dict.items():
        #     if key.startswith('segm_model.backbone.conv1'):  # Assuming conv1 is directly under backbone
        #         # Modify the weights for the conv1 layer to adapt from 4 channels to 5 channels
        #         if 'weight' in key:
        #             # Assuming state_dict[key] has shape (64, 4, 7, 7)
        #             original_weight = state_dict[key]
        #             new_weight = torch.zeros((original_weight.shape[0], 5, original_weight.shape[2], original_weight.shape[3]), dtype=original_weight.dtype)
        #             # Copy the weights for the existing 4 channels and duplicate the first channel weights for the new 5th channel
        #             new_weight[:, :4, :, :] = original_weight
        #             new_weight[:, 4:5, :, :] = original_weight[:, 0:1, :, :]
        #             new_state_dict[key] = new_weight
        #         else:
        #             new_state_dict[key] = value
        #     else:
        #         new_state_dict[key] = value