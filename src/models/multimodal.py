import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
# multiprocessing.set_start_method('fork')
import cv2
import pytorch_lightning as pl
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import torch
from rastervision.core.box import Box
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.features import rasterize
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from typing import TYPE_CHECKING
import wandb
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.data import ConcatDataset

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from collections import OrderedDict
import duckdb


# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import MultiModalPredictionsIterator, CustomGeoJSONVectorSource, FusionModule
from deeplnafrica.deepLNAfrica import (Deeplabv3SegmentationModel, init_segm_model,
                                       CustomDeeplabv3SegmentationModel)
from src.data.dataloaders import (create_sentinel_raster_source, create_buildings_raster_source,
                                  create_datasets, show_windows, CustomStatsTransformer,
                                  CustomSemanticSegmentationSlidingWindowGeoDataset)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.pytorch_learner import SemanticSegmentationVisualizer
from rastervision.core.data import (Scene, ClassConfig, RasterioCRSTransformer,
                                    RasterioSource, GeoJSONVectorSource,
                                    ClassInferenceTransformer, RasterizedSource,
                                    SemanticSegmentationLabelSource, VectorSource)
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.utils import listify_uris, merge_geojsons
from rastervision.pipeline.file_system import (
    get_local_path, json_to_file, make_dir, sync_to_dir, file_exists,
    download_if_needed, NotReadableError, get_tmp_dir)
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (
    color_to_triple, plot_channel_groups, channel_groups_to_imgs)

from typing import (TYPE_CHECKING, Sequence, Optional, List, Dict, Union,
                    Tuple, Any)
from abc import ABC, abstractmethod

from torch import Tensor
import albumentations as A

from rastervision.pipeline.file_system import make_dir
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner.utils import (
    deserialize_albumentation_transform, validate_albumentation_transform,
    MinMaxNormalize)
from rastervision.pytorch_learner.learner_config import (
    RGBTuple,
    ChannelInds,
    validate_channel_display_groups,
    get_default_channel_display_groups,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from matplotlib.figure import Figure

from typing import TYPE_CHECKING, Iterator, List, Optional
from os.path import join

from rastervision.pipeline.config import register_config, Config, Field
from rastervision.core.data.label_store import (LabelStoreConfig,
                                                SemanticSegmentationLabelStore)
from rastervision.core.data.utils import (denoise, mask_to_building_polygons,
                                          mask_to_polygons)

if TYPE_CHECKING:
    import numpy as np
    from shapely.geometry.base import BaseGeometry

    from rastervision.core.box import Box
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig
    
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
    
class MergeDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'

label_uriGC = "../../data/SHP/Guatemala_PS.shp"
image_uriGC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
buildings_uriGC = '../../data/0/overture/GT_buildings3857.geojson'

label_uriTG = "../../data/SHP/Tegucigalpa_PS.shp"
image_uriTG = '../../data/0/sentinel_Gee/HND_Comayaguela_2023.tif'
buildings_uriTG = '../../data/0/overture/HND_buildings3857.geojson'

label_uriPN = "../../data/SHP/Panama_PS.shp"
image_uriPN = '../../data/0/sentinel_Gee/PAN_San_Miguelito_2023.tif'
buildings_uriPN = '../../data/0/overture/PN_buildings3857.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

# Santo Domingo
sentinel_source_normalizedSD, sentinel_label_raster_sourceSD = create_sentinel_raster_source(image_uri, label_uri, class_config, clip_to_label_source=True)
rasterized_buildings_sourceSD, buildings_label_sourceSD, crs_transformer_buildingsSD = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)    

SentinelScene_SD = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalizedSD,
        label_source = sentinel_label_raster_sourceSD)
        # aoi_polygons=[pixel_polygon])

BuildingsScene_SD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_sourceSD,
        label_source = buildings_label_sourceSD)

train_multiple_cities=False
if train_multiple_cities:
    # # Guatemala City
    # sentinel_source_normalizedGC, sentinel_label_raster_sourceGC = create_sentinel_raster_source(image_uriGC, label_uriGC, class_config, clip_to_label_source=True)
    # rasterized_buildings_sourceGC, buildings_label_sourceGC, crs_transformer_buildingsGC = create_buildings_raster_source(buildings_uriGC, image_uriGC, label_uriGC, class_config, resolution=5)    

    # BuildingsScene_GC = Scene(
    #         id='GC_buildings',
    #         raster_source = rasterized_buildings_sourceGC,
    #         label_source = buildings_label_sourceGC)
            
    # SentinelScene_GC = Scene(
    #         id='GC_sentinel',
    #         raster_source = sentinel_source_normalizedGC,
    #         label_source = sentinel_label_raster_sourceGC)

    # # Tegucigalpa
    # sentinel_source_normalizedTG, sentinel_label_raster_sourceTG = create_sentinel_raster_source(image_uriGC, label_uriTG, class_config, clip_to_label_source=True)
    # rasterized_buildings_sourceTG, buildings_label_sourceTG, crs_transformer_buildingsTG = create_buildings_raster_source(buildings_uriGC, image_uriGC, label_uriTG, class_config, resolution=5)

    # SentinelScene_TG = Scene(
    #     id='TG_sentinel',
    #     raster_source=sentinel_source_normalizedTG,
    #     label_source=sentinel_label_raster_sourceTG)

    # BuildingsScene_TG = Scene(
    #     id='TG_buildings',
    #     raster_source=rasterized_buildings_sourceTG,
    #     label_source=buildings_label_sourceTG)

    # # Panama City
    # sentinel_source_normalizedPN, sentinel_label_raster_sourcePN = create_sentinel_raster_source(image_uriPN, label_uriPN, class_config, clip_to_label_source=True)
    # rasterized_buildings_sourcePN, buildings_label_sourcePN, crs_transformer_buildingsPN = create_buildings_raster_source(buildings_uriPN, image_uriPN, label_uriPN, class_config, resolution=5)

    # SentinelScene_PN = Scene(
    #     id='PN_sentinel',
    #     raster_source=sentinel_source_normalizedPN,
    #     label_source=sentinel_label_raster_sourcePN)

    # BuildingsScene_PN = Scene(
    #     id='PN_buildings',
    #     raster_source=rasterized_buildings_sourcePN,
    #     label_source=buildings_label_sourcePN)
    
    # Guatemala City
    # buildingsGeoDataset_GC, train_buildings_dataset_GC, val_buildings_dataset_GC, test_buildings_dataset_GC = create_datasets(BuildingsScene_GC, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    # sentinelGeoDataset_GC, train_sentinel_dataset_GC, val_sentinel_dataset_GC, test_sentinel_dataset_GC = create_datasets(SentinelScene_GC, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    # Tegucigalpa
    # buildingsGeoDataset_TG, train_buildings_dataset_TG, val_buildings_dataset_TG, test_buildings_dataset_TG = create_datasets(BuildingsScene_TG, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    # sentinelGeoDataset_TG, train_sentinel_dataset_TG, val_sentinel_dataset_TG, test_sentinel_dataset_TG = create_datasets(SentinelScene_TG, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    # # Panama City
    # buildingsGeoDataset_PN, train_buildings_dataset_PN, val_buildings_dataset_PN, test_buildings_dataset_PN = create_datasets(BuildingsScene_PN, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    # sentinelGeoDataset_PN, train_sentinel_dataset_PN, val_sentinel_dataset_PN, test_sentinel_dataset_PN = create_datasets(SentinelScene_PN, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    pass

batch_size = 16

# Santo Domingo
buildingsGeoDataset_SD, train_buildings_dataset_SD, val_buildings_dataset_SD, test_buildings_dataset_SD = create_datasets(BuildingsScene_SD, imgsize=288, stride=288, padding=50, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset_SD, train_sentinel_dataset_SD, val_sentinel_dataset_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=144, stride=144, padding=25, val_ratio=0.2, test_ratio=0.1, seed=42)

all_cities_sentinel_train_ds = ConcatDataset([train_sentinel_dataset_SD]) #train_sentinel_dataset_GC, train_sentinel_dataset_TG, train_sentinel_dataset_PN
all_cities_sentinel_val_ds = ConcatDataset([val_sentinel_dataset_SD]) # val_sentinel_dataset_GC, val_sentinel_dataset_TG, val_sentinel_dataset_PN
all_cities_sentinel_test_ds = ConcatDataset([test_sentinel_dataset_SD]) # test_sentinel_dataset_GC, test_sentinel_dataset_TG, test_sentinel_dataset_PN

all_cities_build_train_ds = ConcatDataset([train_buildings_dataset_SD]) #train_buildings_dataset_GC train_buildings_dataset_TG, train_buildings_dataset_PN
all_cities_build_val_ds = ConcatDataset([val_buildings_dataset_SD]) #val_buildings_dataset_GC, val_buildings_dataset_TG, val_buildings_dataset_PN
all_cities_build_test_ds = ConcatDataset([test_buildings_dataset_SD]) #test_buildings_dataset_GC, test_buildings_dataset_TG, test_buildings_dataset_PN
    
train_dataset = MergeDataset(all_cities_sentinel_train_ds, all_cities_build_train_ds)
val_dataset = MergeDataset(all_cities_sentinel_val_ds, all_cities_build_val_ds)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# when one city for training:
train_dataset = MergeDataset(train_sentinel_dataset_SD, train_buildings_dataset_SD)
val_dataset = MergeDataset(val_sentinel_dataset_SD, val_buildings_dataset_SD)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

channel_display_groups_sent = {'RGB': (0,1,2), 'NIR': (3, )}
channel_display_groups_build = {'Buildings': (0,)}

vis_sent = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors,
    channel_display_groups=channel_display_groups_sent)

vis_build = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors,
    channel_display_groups=channel_display_groups_build)

x, y = vis_sent.get_batch(all_cities_sentinel_train_ds, 2)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(all_cities_build_train_ds, 2)
vis_build.plot_batch(x, y, show=True)

class MultiModalDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# buildings encoder testing 
buildings_encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)
buildings_encoder.backbone

# buildings_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# buildings_bn1 = buildings_encoder.backbone.bn1
# buildings_relu = buildings_encoder.backbone.relu
# buildings_maxpool = buildings_encoder.backbone.maxpool
# buildings_layer1 = buildings_encoder.backbone.layer1
# conv_extra = nn.Conv2d(256, 2048, kernel_size=3, padding=1)
# adaptive_pool = torch.nn.AdaptiveAvgPool2d((18, 18))

# for batch_idx, batch in enumerate(data_module.train_dataloader()):
#     sentinel_batch, buildings_batch = batch
#     buildings_data, buildings_labels = buildings_batch
#     sentinel_data, _ = sentinel_batch
    
#     # sentinel_data = sentinel_data.to(device)
#     # buildings_data = buildings_data.to(device)
#     x = buildings_conv1(buildings_data)
#     x = buildings_bn1(x)
#     x = buildings_relu(x)
#     x = buildings_maxpool(x)
#     x = buildings_layer1(x)
#     x = conv_extra(x)
#     x = adaptive_pool(x)
#     print(f"conv1 data shape: {x.shape}")
#     break

# Train the model
class MultiResolutionSegmentationModel(pl.LightningModule):
    def __init__(self,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (6, 12, 24),
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = pos_weight
        
        self.sentinel_encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        self.sentinel_encoder.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.buildings_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.buildings_bn1 = self.sentinel_encoder.backbone.bn1
        self.relu = self.sentinel_encoder.backbone.relu
        self.buildings_maxpool = self.sentinel_encoder.backbone.maxpool
        self.buildings_layer1 = self.sentinel_encoder.backbone.layer1
        self.buildings_conv2 = nn.Conv2d(256, 2048, kernel_size=3, padding=1)
        self.buildings_adapool = torch.nn.AdaptiveAvgPool2d((18, 18))

        # load pretrained deeplnafrica weights into sentinel channels
        allcts_path = '/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt'
        checkpoint = torch.load(allcts_path, map_location='cpu')  # Load to CPU first
        state_dict = checkpoint["state_dict"]
        
        # Convert any float64 weights to float32
        for key, value in state_dict.items():
            if value.dtype == torch.float64:
                state_dict[key] = value.to(torch.float32)
                
        # removing prefix
        final_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('segm_model.backbone.'):
                new_key = key[len('segm_model.backbone.'):]
                final_state_dict[new_key] = value

        self.sentinel_encoder.backbone.load_state_dict(final_state_dict, strict=True)
        
        # Intermediate Layer Getters
        self.sentinel_encoder_backbone = IntermediateLayerGetter(self.sentinel_encoder.backbone, {'layer4': 'out'})#, 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        
        self.fusion_layer = nn.Conv2d(4096, 2048, kernel_size=1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(4096, 2048, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, 3, padding=1)
        )
        
        self.segmentation_head = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)
           
    def forward(self, batch):

        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        # Move data to the device
        sentinel_data = sentinel_data.to(self.device)
        buildings_data = buildings_data.to(self.device)
        buildings_labels = buildings_labels.to(self.device)
        
        sentinel_features = self.sentinel_encoder_backbone(sentinel_data)
        sentinel_out = sentinel_features['out']
        # print(f"sentinel_features shape: {sentinel_features['out'].shape}")
        
        x = self.buildings_conv1(buildings_data)
        x = self.buildings_bn1(x)
        x = self.relu(x)
        x = self.buildings_maxpool(x)
        x = self.buildings_layer1(x)
        x = self.buildings_conv2(x)
        
        buildings_out = self.buildings_adapool(x)
        
        # concatenated = torch.cat([sentinel_out, buildings_out], dim=1)    
        concatenated = sentinel_out+buildings_out # addition works well on SD        
        # print(f"concatenated features shape: {concatenated.shape}")
        
        # Decode the fused features
        # fused_features = self.fusion_layer(concatenated)
        # fused_features = self.fusion(concatenated)
        # print(f"fused_features shape: {fused_features.shape}")

        segmentation = self.segmentation_head(concatenated)
        
        segmentation = F.interpolate(segmentation, size=288, mode="bilinear", align_corners=False)
        
        return segmentation.squeeze(1)
    
    def training_step(self, batch):
        
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)
        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
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

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
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

model = MultiResolutionSegmentationModel(weight_decay=0.001,
                            learning_rate=0.0001,
                            gamma=0.01,
                            atrous_rates = (6,12,24),
                            pos_weight=torch.tensor(1.0, device='mps'))
model.to(device)

# for batch_idx, batch in enumerate(data_module.train_dataloader()):
#     sentinel_batch, buildings_batch = batch
#     buildings_data, buildings_labels = buildings_batch
#     sentinel_data, _ = sentinel_batch
    
#     sentinel_data = sentinel_data.to(device)
#     buildings_data = buildings_data.to(device)
#     outsent = model(batch)
#     print(f"Sentinel data shape: {outsent.shape}")
#     break

output_dir = f'../UNITAC-trained-models/multi_modal/trained_SD_DLNAw/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal')
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}-{batch_size:02d}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=5,
    max_epochs=50,
    num_sanity_val_steps=3
)

# Train the model
trainer.fit(model, datamodule=data_module)

# # Use best model for evaluation
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/multi_modal/trained_SD_sentinel_DLNAw/multimodal_runidrun_id=0-batch_size=00-epoch=22-val_loss=0.2262.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = MultiResolutionSegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScene_SD, imgsize=288, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene_SD, imgsize=144, stride = 72, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
predictions_iterator = MultiModalPredictionsIterator(best_model, sentinelGeoDataset, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Ensure windows are Box instances
windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=BuildingsScene_SD.extent,
    num_classes=len(class_config),
    smooth=True
)

# Show predictions
scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('infs Scores')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/multi-modal/SD_pretrained_512/',
#     crs_transformer = crs_transformer_buildings,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)