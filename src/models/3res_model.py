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
import stackstac
import pystac_client

import folium
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.data import ConcatDataset
import math

from fvcore.nn import FlopCountAnalysis
# from torchinfo import summary  # Optional, for detailed summary

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from collections import OrderedDict


# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import MultiModalPredictionsIterator, CustomGeoJSONVectorSource, CustomVectorOutputConfig, MergeDataset, CustomGeoJSONVectorSource
from deeplnafrica.deepLNAfrica import (Deeplabv3SegmentationModel, init_segm_model, 
                                       CustomDeeplabv3SegmentationModel)
from src.data.dataloaders import (create_sentinel_raster_source, create_buildings_raster_source,
                                  create_datasets, show_windows, CustomStatsTransformer,
                                  CustomSemanticSegmentationSlidingWindowGeoDataset)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.pytorch_learner import SemanticSegmentationVisualizer
from rastervision.core.data import (Scene, ClassConfig, RasterioCRSTransformer, XarraySource,
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

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

label_uriSD = "../../data/0/SantoDomingo3857.geojson"
image_uriSD = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

# SD labels
gdf = gpd.read_file(label_uriSD)
gdf = gdf.to_crs('EPSG:4326')
gdf_filled = gdf.copy()
gdf_filled["geometry"] = gdf_filled.geometry.buffer(0.00008)
# map_center = gdf_filled.geometry.centroid.iloc[0].coords[0][::-1]
# mymap = folium.Map(location=map_center, zoom_start=13)
# for _, row in gdf.iterrows():
#     geojson = row['geometry'].__geo_interface__
#     folium.GeoJson(geojson, style_function=lambda x: {'color': 'blue'}).add_to(mymap)
# for _, row in gdf_filled.iterrows():
#     geojson = row['geometry'].__geo_interface__
#     folium.GeoJson(geojson, style_function=lambda x: {'color': 'red'}).add_to(mymap)
# mymap
labels4326 = gdf_filled.to_crs('EPSG:4326')
xmin4326, ymin4326, xmax4326, ymax4326 = labels4326.total_bounds

labels3857 = labels4326.to_crs('EPSG:3857')
xmin3857, _, _, ymax3857 = labels3857.total_bounds

# Santo Domingo
sentinel_source_normalizedSD, sentinel_label_raster_sourceSD = create_sentinel_raster_source(image_uriSD, label_uriSD, class_config, clip_to_label_source=True)
crs_transformer = RasterioCRSTransformer.from_uri(image_uriSD)
label_vector_source = CustomGeoJSONVectorSource(
    gdf = gdf_filled,
    crs_transformer = crs_transformer,
    vector_transformers=[
        ClassInferenceTransformer(
            default_class_id=class_config.get_class_id('slums'))])
sentinel_label_raster_sourceSD = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
# sentinel_label_sourceSD = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
chip = sentinel_source_normalizedSD[:, :, [0, 1, 2]]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(chip)
plt.show()

# Rasterised buildings
rasterized_buildings2m_sourceSD, buildings2m_label_sourceSD, crs_transformer_buildings2m_SD = create_buildings_raster_source(buildings_uri, image_uriSD, label_uriSD, class_config, resolution=2)    
rasterized_buildings5m_sourceSD, buildings5m_label_sourceSD, crs_transformer_buildings5m_SD = create_buildings_raster_source(buildings_uri, image_uriSD, label_uriSD, class_config, resolution=5)

SentinelScene_SD = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalizedSD,
        label_source = sentinel_label_raster_sourceSD)
        # aoi_polygons=[pixel_polygon])

Buildings2mScene_SD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings2m_sourceSD,
        label_source = buildings2m_label_sourceSD)

Buildings5mScene_SD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings5m_sourceSD,
        label_source = buildings5m_label_sourceSD)

buildings2mGeoDataset_SD, train_buildings2m_dataset_SD, val_buildings2m_dataset_SD, test_buildings2m_dataset_SD = create_datasets(Buildings2mScene_SD, imgsize=640, stride=640, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
buildings5mGeoDataset_SD, train_buildings5m_dataset_SD, val_buildings5m_dataset_SD, test_buildings5m_dataset_SD = create_datasets(Buildings5mScene_SD, imgsize=256, stride=256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset_SD, train_sentinel_dataset_SD, val_sentinel_dataset_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=128, stride=128, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildings2mGeoDataset_SD.scene.label_source)
train_windows = train_buildings2m_dataset_SD.windows
val_windows = val_buildings2m_dataset_SD.windows
test_windows = test_buildings2m_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

img_full = create_full_image(buildings5mGeoDataset_SD.scene.label_source)
train_windows = train_buildings5m_dataset_SD.windows
val_windows = val_buildings5m_dataset_SD.windows
test_windows = test_buildings5m_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source._get_chip(extent)    
    return chip

img_full = create_full_image(sentinelGeoDataset_SD.scene.label_source)
train_windows = train_sentinel_dataset_SD.windows
val_windows = val_sentinel_dataset_SD.windows
test_windows = test_sentinel_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')


batch_size = 16

train_multiple_cities=False

if train_multiple_cities:
    
    label_uriGC = "../../data/SHP/Guatemala_PS.shp"
    image_uriGC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
    buildings_uriGC = '../../data/0/overture/GT_buildings3857.geojson'

    label_uriTG = "../../data/SHP/Tegucigalpa_PS.shp"
    image_uriTG = '../../data/0/sentinel_Gee/HND_Comayaguela_2023.tif'
    buildings_uriTG = '../../data/0/overture/HND_buildings3857.geojson'

    label_uriPN = "../../data/SHP/Panama_PS.shp"
    image_uriPN = '../../data/0/sentinel_Gee/PAN_San_Miguelito_2023.tif'
    buildings_uriPN = '../../data/0/overture/PN_buildings3857.geojson'

    # Guatemala City
    sentinel_source_normalizedGC, sentinel_label_raster_sourceGC = create_sentinel_raster_source(image_uriGC, label_uriGC, class_config, clip_to_label_source=True)
    rasterized_buildings_sourceGC, buildings_label_sourceGC, crs_transformer_buildingsGC = create_buildings_raster_source(buildings_uriGC, image_uriGC, label_uriGC, class_config, resolution=5)    

    BuildingsScene_GC = Scene(
            id='GC_buildings',
            raster_source = rasterized_buildings_sourceGC,
            label_source = buildings_label_sourceGC)
            
    SentinelScene_GC = Scene(
            id='GC_sentinel',
            raster_source = sentinel_source_normalizedGC,
            label_source = sentinel_label_raster_sourceGC)

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
    buildingsGeoDataset_GC, _, _, _ = create_datasets(BuildingsScene_GC, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    sentinelGeoDataset_GC, _, _, _ = create_datasets(SentinelScene_GC, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    # Tegucigalpa
    # buildingsGeoDataset_TG, train_buildings_dataset_TG, val_buildings_dataset_TG, test_buildings_dataset_TG = create_datasets(BuildingsScene_TG, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    # sentinelGeoDataset_TG, train_sentinel_dataset_TG, val_sentinel_dataset_TG, test_sentinel_dataset_TG = create_datasets(SentinelScene_TG, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    # # Panama City
    # buildingsGeoDataset_PN, train_buildings_dataset_PN, val_buildings_dataset_PN, test_buildings_dataset_PN = create_datasets(BuildingsScene_PN, imgsize=288, stride=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    # sentinelGeoDataset_PN, train_sentinel_dataset_PN, val_sentinel_dataset_PN, test_sentinel_dataset_PN = create_datasets(SentinelScene_PN, imgsize=144, stride=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

    all_cities_sentinel_train_ds = ConcatDataset([train_sentinel_dataset_SD, sentinelGeoDataset_GC]) # train_sentinel_dataset_TG, train_sentinel_dataset_PN
    all_cities_sentinel_val_ds = ConcatDataset([val_sentinel_dataset_SD]) # val_sentinel_dataset_GC, val_sentinel_dataset_TG, val_sentinel_dataset_PN
    all_cities_sentinel_test_ds = ConcatDataset([test_sentinel_dataset_SD]) # test_sentinel_dataset_GC, test_sentinel_dataset_TG, test_sentinel_dataset_PN

    all_cities_build_train_ds = ConcatDataset([train_buildings_dataset_SD, buildingsGeoDataset_GC]) # train_buildings_dataset_TG, train_buildings_dataset_PN
    all_cities_build_val_ds = ConcatDataset([val_buildings_dataset_SD]) #val_buildings_dataset_GC, val_buildings_dataset_TG, val_buildings_dataset_PN
    all_cities_build_test_ds = ConcatDataset([test_buildings_dataset_SD]) #test_buildings_dataset_GC, test_buildings_dataset_TG, test_buildings_dataset_PN
        
    train_dataset = MergeDataset(all_cities_sentinel_train_ds, all_cities_build_train_ds)
    val_dataset = MergeDataset(all_cities_sentinel_val_ds, all_cities_build_val_ds)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
else:
    # when one city for training:
    train_dataset = MergeDataset(train_sentinel_dataset_SD, train_buildings5m_dataset_SD, train_buildings2m_dataset_SD)
    val_dataset = MergeDataset(val_sentinel_dataset_SD, val_buildings5m_dataset_SD, val_buildings2m_dataset_SD)

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

x, y = vis_sent.get_batch(sentinelGeoDataset_SD, 2)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildings5mGeoDataset_SD, 2)
vis_build.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(buildings2mGeoDataset_SD, 2)
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
    
    def setup(self, stage=None):
        pass

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# Train the model 
class Multi3ResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (6, 12, 24),
                sched_step_size = 10,
                build_depth = 128,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.sched_step_size = sched_step_size
        
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        
        self.buildings_encoder = nn.Sequential(
            nn.Conv2d(1, build_depth, kernel_size=(7, 7), stride=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(build_depth),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(build_depth, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        )
        self.encoder.backbone.conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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
                if key.startswith('segm_model.backbone.'):
                    new_key = key[len('segm_model.backbone.'):]
                    state_dict[new_key] = value

            # Extract the original weights of the first convolutional layer
            original_conv1_weight = state_dict['conv1.weight']
            new_conv1_weight = torch.zeros((64, 5, 7, 7))
            new_conv1_weight[:, :4, :, :] = original_conv1_weight
            new_conv1_weight[:, 4, :, :] = original_conv1_weight[:, 0, :, :]
            new_conv1 = nn.Conv2d(5, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            new_conv1.weight = nn.Parameter(new_conv1_weight)
            new_state_dict = state_dict.copy()
            new_state_dict['conv1.weight'] = new_conv1.weight

            self.encoder.backbone.load_state_dict(new_state_dict, strict=True)
        
        # Intermediate Layer Getter
        self.encoder = IntermediateLayerGetter(self.encoder.backbone, {'layer4': 'out'})
        self.segmentation_head = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)
        
    def forward(self, batch):
        sentinel_batch, buildings_batch = batch
        buildings_data, _ = buildings_batch
        sentinel_data, _ = sentinel_batch
        # Move data to the device
        sentinel_data = sentinel_data.to(self.device)
        buildings_data = buildings_data.to(self.device)
        
        b_out = self.buildings_encoder(buildings_data)
        concatenated = torch.cat([sentinel_data, b_out], dim=1)    

        encoder_outputs = self.encoder(concatenated)
        out = encoder_outputs['out']

        segmentation = self.segmentation_head(out)
        # print(segmentation.shape, "segmentation shape")
        segmentation = F.interpolate(segmentation, size=144, mode="bilinear", align_corners=False)
        # print(segmentation.shape, "segmentation shape after interpolation")
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



hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'SD',
    'use_deeplnafrica': True,
    'build_depth': 128,
    'atrous_rates': (3, 6, 12),
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'gamma': 0.5,
    'sched_step_size': 20,
    'pos_weight': 2.0,
}

model = Multi3ResolutionDeepLabV3(
    use_deeplnafrica=hyperparameters['use_deeplnafrica'],
    learning_rate=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    gamma=hyperparameters['gamma'],
    atrous_rates=hyperparameters['atrous_rates'],
    sched_step_size=hyperparameters['sched_step_size'],
    pos_weight=torch.tensor(hyperparameters['pos_weight'], device='mps')
)
model.to(device)

# for idx, batch in enumerate(data_module.train_dataloader()):
#     # sentinel_batch, buildings_batch = batch
#     # buildings_data, buildings_labels = buildings_batch
#     # sentinel_data, _ = sentinel_batch
#     # sentinel_data = sentinel_data.to(device)
#     # buildings_data = buildings_data.to(device)
#     out = model(batch)
#     print(f"out data shape: {out.shape}")
#     break

output_dir = f'../UNITAC-trained-models/multi_modal/SD_DLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=2,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=30)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=40,
    max_epochs=150,
    num_sanity_val_steps=3
)

# Train the model
trainer.fit(model, datamodule=data_module)

# # Use best model for evaluation # best dlv3 multimodal_runidrun_id=0-epoch=43-val_loss=0.4043.ckpt
best_model_path_dplv3 = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/multi_modal/SD_DLV3/multimodal_runidrun_id=0-epoch=43-val_loss=0.4043.ckpt"
# best_model_path_fpn = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/multi_modal/SD_FPN/multimodal_runidrun_id=0-epoch=60-val_loss=0.3382.ckpt"
# best_model_path_dplv3 = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3.load_from_checkpoint(best_model_path_dplv3) #MultiResolutionDeepLabV3 MultiResolutionFPN
best_model.eval()

class MultiRes144PredictionsIterator:
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

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScene_SD, imgsize=288, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene_SD, imgsize=144, stride = 72, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
predictions_iterator = MultiRes144PredictionsIterator(best_model, sentinelGeoDataset_SD, buildingsGeoDataset_SD, device=device)
windows, predictions = zip(*predictions_iterator)
assert len(windows) == len(predictions)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=SentinelScene_SD.extent,
    num_classes=len(class_config),
    smooth=True
)

# Show predictions
scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('probability map')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

crs_transformer = RasterioCRSTransformer.from_uri(image_uriSD)
affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
crs_transformer.transform = affine_transform_buildings

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/multi-modal/SD_DLV3/',
    crs_transformer = crs_transformer,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildingsGeoDataset.scene.label_source)
train_windows = train_buildings_dataset_SD.windows
val_windows = val_buildings_dataset_SD.windows
test_windows = test_buildings_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')


# Ensure windows are Box instances
# windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Visualise feature maps
class FeatureMapVisualization:
    def __init__(self, model):
        self.model = model
        self.feature_maps = {}
        self.hooks = []

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

visualizer = FeatureMapVisualization(best_model)
visualizer.add_hooks([
    'buildings_encoder.0', # conv2d
    'buildings_encoder.1', # batchnorm
    'buildings_encoder.2', # relu
    'buildings_encoder.3', # maxpool
    'buildings_encoder.4', # maxpool
    'encoder.conv1',
    'encoder.layer1.0.conv1',
    'encoder.layer4.2.conv3',
    'segmentation_head.0',
    'segmentation_head.1',
    'segmentation_head.4',
    'segmentation_head[-1]'
])

# Get the iterator for the DataLoader
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
data_iter = iter(val_loader)
first_batch = next(data_iter)
second_batch = next(data_iter)
third_batch = next(data_iter)
fourth_batch = next(data_iter)
fifth_batch = next(data_iter)

x, y = vis_build.get_batch(val_buildings_dataset_SD, 5)
vis_build.plot_batch(x, y, show=True)

visualizer.visualize_feature_maps('buildings_encoder.0', fifth_batch, num_feature_maps=16) # conv2d
visualizer.visualize_feature_maps('buildings_encoder.1', fifth_batch, num_feature_maps=16) # batchnorm
visualizer.visualize_feature_maps('buildings_encoder.2', fifth_batch, num_feature_maps=16) # relu
visualizer.visualize_feature_maps('buildings_encoder.3', fifth_batch, num_feature_maps=16) # maxpool
visualizer.visualize_feature_maps('buildings_encoder.4', third_batch, num_feature_maps='all') #conv2d
visualizer.visualize_feature_maps('encoder.layer1.0.conv1', third_batch, num_feature_maps=16)
visualizer.visualize_feature_maps('encoder.layer4.2.conv3', third_batch, num_feature_maps=16)
visualizer.visualize_feature_maps('segmentation_head.0', third_batch, num_feature_maps=16)
visualizer.visualize_feature_maps('segmentation_head.1', third_batch, num_feature_maps=16)
visualizer.visualize_feature_maps('segmentation_head.4', third_batch, num_feature_maps=16)
visualizer.remove_hooks()


# Visualise filters
def visualize_filters(model, layer_name, num_filters=8):
    # Get the layer by name
    layer = dict(model.named_modules())[layer_name]
    assert isinstance(layer, nn.Conv2d), "Layer should be of type nn.Conv2d"

    # Get the weights of the filters
    filters = layer.weight.data.clone().cpu().numpy()

    # Normalize the filters to [0, 1] range for visualization
    min_filter, max_filter = filters.min(), filters.max()
    filters = (filters - min_filter) / (max_filter - min_filter)
    
    # Plot the filters
    num_filters = min(num_filters, filters.shape[0])  # Limit to number of available filters
    fig, axes = plt.subplots(1, num_filters, figsize=(20, 10))
    
    for i, ax in enumerate(axes):
        filter_img = filters[i]
        
        # If the filter has more than one channel, average the channels for visualization
        if filter_img.shape[0] > 1:
            filter_img = np.mean(filter_img, axis=0)
        
        cax = ax.imshow(filter_img, cmap='viridis')
        ax.axis('off')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cax, cax=cbar_ax)

    plt.show()
    
visualize_filters(best_model, 'segmentation_head.0', num_filters=8)

# FLOPS
# flops = FlopCountAnalysis(model, batch)

# print(flops.total())
# print(flops.by_module())

# print(parameter_count_table(model))