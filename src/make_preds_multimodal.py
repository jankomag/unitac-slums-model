import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import torch.nn as nn
import cv2
from os.path import join
import pandas as pd
import json
import rasterio as rio
from tqdm import tqdm
from shapely.geometry import shape, mapping
from shapely.affinity import translate
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter
import pandas as pd
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader
from typing import List
from rasterio.features import rasterize
import rasterio
import matplotlib.pyplot as plt
from pytorch_lightning.loggers.wandb import WandbLogger
import os
from datetime import datetime
import wandb
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import stackstac
import pystac_client

from typing import Iterator, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torch.utils.data import ConcatDataset

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from deeplnafrica.deepLNAfrica import init_segm_model
from src.data.dataloaders import (
    query_buildings_data,
    create_datasets,
    create_buildings_raster_source, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
)
from src.models.model_definitions import (CustomGeoJSONVectorSource, MultiModalPredictionsIterator, MultiResolutionDeepLabV3, CustomVectorOutputConfig)

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
from rastervision.core.data.utils import listify_uris, merge_geojsons

from typing import Any, Optional, Tuple, Union, Sequence
from typing import List
from rasterio.features import rasterize
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)


# from src.models import BuildingsOnlyDeeplabv3SegmentationModel

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
from typing import TYPE_CHECKING, List, Optional, Union
import logging

from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.core.data.utils import listify_uris, merge_geojsons
from rastervision.pipeline.file_system import (
    get_local_path, json_to_file, make_dir, sync_to_dir, file_exists,
    download_if_needed, NotReadableError, get_tmp_dir)
from rastervision.core.box import Box
from rastervision.core.data import (CRSTransformer, ClassConfig)
from rastervision.core.data.label import (SemanticSegmentationLabels,
                                          SemanticSegmentationSmoothLabels)
from rastervision.core.data.label_store import LabelStore
from rastervision.core.data.label_source import SemanticSegmentationLabelSource
from rastervision.core.data.raster_transformer import RGBClassTransformer
from rastervision.core.data.raster_source import RasterioSource
from rastervision.core.data.utils import write_window
if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer, VectorTransformer

log = logging.getLogger(__name__)
import duckdb

class PredictionsIterator:
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
    
class CustomStatsTransformer(RasterTransformer):
    def __init__(self,
                 means: Sequence[float],
                 stds: Sequence[float],
                 max_stds: float = 3.):
        # shape = (1, 1, num_channels)
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)
        self.max_stds = max_stds

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[Sequence[int]] = None) -> np.ndarray:
        if chip.dtype == np.uint8:
            return chip

        means = self.means
        stds = self.stds
        max_stds = self.max_stds
        if channel_order is not None:
            means = means[channel_order]
            stds = stds[channel_order]

        # Don't transform NODATA zero values.
        nodata_mask = chip == 0

        # Convert chip to float (if not already)
        chip = chip.astype(float)

        # Subtract mean and divide by std to get z-scores.
        for i in range(chip.shape[-1]):  # Loop over channels
            chip[..., i] -= means[i]
            chip[..., i] /= stds[i]

        # Apply max_stds clipping
        chip = np.clip(chip, -max_stds, max_stds)
        
        # Normalise to [0, 1]
        # chip = (chip - chip.min()) / (chip.max() - chip.min())
    
        # Normalize to have standard deviation of 1
        for i in range(chip.shape[-1]):
            chip[..., i] /= np.std(chip[..., i])

        chip[nodata_mask] = 0
        
        return chip

    @classmethod
    def from_raster_sources(cls,
                            raster_sources: List['RasterSource'],
                            sample_prob: Optional[float] = 0.1,
                            max_stds: float = 3.,
                            chip_sz: int = 300) -> 'CustomStatsTransformer':
        stats = RasterStats()
        stats.compute(
            raster_sources=raster_sources,
            sample_prob=sample_prob,
            chip_sz=chip_sz)
        stats_transformer = cls.from_raster_stats(
            stats, max_stds=max_stds)
        return stats_transformer
    
    @classmethod
    def from_raster_stats(cls, stats: RasterStats,
                          max_stds: float = 3.) -> 'CustomStatsTransformer':
        stats_transformer = cls(stats.means, stats.stds, max_stds=max_stds)
        return stats_transformer
    
def rasterise_buildings(image_uri, buildings, xmin, ymax, resolution = 5):
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings

    crs_transformer = RasterioCRSTransformer.from_uri(image_uri)
    print("Got the crs transformer for buildings")
    buildings_vector_source_crsimage = CustomGeoJSONVectorSource(
            gdf = buildings,
            crs_transformer = crs_transformer,
            vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    buildings_vector_source_crsbuildings = CustomGeoJSONVectorSource(
            gdf = buildings,
            crs_transformer = crs_transformer_buildings,
            vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
       
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source_crsbuildings,
        background_class_id=0)
            
    # Define the bbox of buildings in the image crs 
    buildings_extent = buildings_vector_source_crsimage.extent
    
    print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")
    return rasterized_buildings_source, buildings_extent, crs_transformer_buildings

def get_sentinel_source(image_uri, buildings_extent):
    sentinel_source_unnormalized = RasterioSource(
        image_uri,
        allow_streaming=True)

    # Calculate statistics transformer from the unnormalized source
    calc_stats_transformer = CustomStatsTransformer.from_raster_sources(
        raster_sources=[sentinel_source_unnormalized],
        max_stds=3
    )

    # Define a normalized raster source using the calculated transformer
    sentinel_source = RasterioSource(
        image_uri,
        allow_streaming=True,
        raster_transformers=[calc_stats_transformer],
        channel_order=[2, 1, 0, 3],
        bbox=buildings_extent
    )
    print(f"Loaded Sentinel data of size {sentinel_source.shape}, and dtype: {sentinel_source.dtype}")
    
    return sentinel_source

def build_datasets(rasterized_buildings_source, sentinel_source):
    build_scene = Scene(
        id='city',
        raster_source = rasterized_buildings_source)

    sent_scene = Scene(
        id='portauprince_buildings',
        raster_source = sentinel_source)

    build_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
            scene=build_scene,
            size=288,
            stride=144,
            out_size=288,
            padding=0)

    sent_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
            scene=sent_scene,
            size=144,
            stride=72,
            out_size=144,
            padding=0)
    
    return build_ds, sent_ds, build_scene

def save_predictions(model, sent_ds, buil_ds, build_scene, crs_transformer, country_code, city_name):
    device = 'mps'
    predictions_iterator = MultiModalPredictionsIterator(model, sent_ds, buil_ds, device=device)
    windows, predictions = zip(*predictions_iterator)
    print("Got predictions for: ", city_name, country_code)

    # Ensure windows are Box instances
    windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

    # Create SemanticSegmentationLabels from predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=build_scene.extent,
        num_classes=len(class_config),
        smooth=True
    )
    
    vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

    pred_label_store = SemanticSegmentationLabelStore(
        uri=f'../../vectorised_model_predictions/multi-modal/DLV3/{country_code}/{city_name}_{country_code}',
        crs_transformer = crs_transformer,
        class_config = class_config,
        vector_outputs = [vector_output_config],
        discrete_output = True)

    pred_label_store.save(pred_labels)

def get_sentinel_item(bbox_geometry, bbox):
    items = catalog.search(
        intersects=bbox_geometry,
        collections=['sentinel-2-c1-l2a'],
        datetime='2023-01-01/2024-06-27',
        query={'eo:cloud_cover': {'lt': 3}},
        max_items=1,
    ).item_collection()
    
    if not items:
        print("No items found for this city.")
        return None
    
    sentinel_source_unnormalized = XarraySource.from_stac(
        items,
        bbox_map_coords=tuple(bbox),
        stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
        allow_streaming=True,
    )

    stats_tf = CustomStatsTransformer.from_raster_sources([sentinel_source_unnormalized],max_stds=3)

    sentinel_source = XarraySource.from_stac(
        items,
        bbox_map_coords=tuple(bbox),
        raster_transformers=[stats_tf],
        stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
        allow_streaming=True,
        channel_order=[2, 1, 0, 3],
    )
    
    print(f"Loaded Sentinel data of size {sentinel_source.shape}, and dtype: {sentinel_source.dtype}")
    
    chip = sentinel_source[:, :, [0, 1, 2]]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip)
    plt.show()
    
    return sentinel_source

class MultiResolutionDeepLabV3(pl.LightningModule):
    def __init__(self,
                use_deeplnafrica: bool = True,
                learning_rate: float = 1e-2,
                weight_decay: float = 1e-1,
                gamma: float = 0.1,
                atrous_rates = (6, 12, 24),
                sched_step_size = 10,
                pos_weight: torch.Tensor = torch.tensor(1.0, device='mps')):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.atrous_rates = atrous_rates
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.sched_step_size = sched_step_size
        
        self.encoder = deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)     
        
        self.bconv1 = nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # self.bconv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(2, 2), bias=False)
        # self.bconv2 = nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2), bias=False)
        
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
        self.encoder = IntermediateLayerGetter(self.encoder.backbone, {'layer4': 'out'})#, 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        self.segmentation_head = DeepLabHead(in_channels=2048, num_classes=1, atrous_rates=self.atrous_rates)
           
    def forward(self, batch):

        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        # Move data to the device
        sentinel_data = sentinel_data.to(self.device)
        buildings_data = buildings_data.to(self.device)
        buildings_labels = buildings_labels.to(self.device)
        
        b_out1 = self.bconv1(buildings_data)
        # b_out2 = self.bconv2(b_out1)
        
        concatenated = torch.cat([sentinel_data, b_out1], dim=1)    
        # print(concatenated.shape, "concatenated shape")
        out = self.encoder(concatenated)
        out = out['out']

        segmentation = self.segmentation_head(out)
        
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

# best_model_path_fpn = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/multi_modal/SD_DLV3/multimodal_runidrun_id=0-epoch=12-val_loss=0.2195.ckpt"
best_model_path_dplv3 = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3.load_from_checkpoint(best_model_path_dplv3) #MultiResolutionDeepLabV3 MultiResolutionFPN


# best_model = MultiResolutionDeepLabV3.load_from_checkpoint(best_model_path_fpn)
# best_model.eval()

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

con = duckdb.connect("../../data/0/data.db")

con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
gdf = gdf.to_crs('EPSG:3857')
# gdf = gdf.head(20)
# gdf = gdf[gdf['city_ascii'] == 'Tabarre']

for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_ascii']
    country_code = row['iso3']
    print("Doing predictions for: ", city_name, country_code)
    image_uri =  f"../../data/0/sentinel_Gee/{country_code}_{city_name}_2023.tif"
    
    # Check if the image_uri exists
    if not os.path.exists(image_uri):
        print(f"Warning: File {image_uri} does not exist. Skipping to next row.")
        continue
    
    gdf_xmin, gdf_ymin, gdf_xmax, gdf_ymax = row['geometry'].bounds
    
    try:
        with rasterio.open(image_uri) as src:
            bounds = src.bounds
            raster_xmin, raster_ymin, raster_xmax, raster_ymax = bounds.left, bounds.bottom, bounds.right, bounds.top
            
    except Exception as e:
        print(f"Error processing {image_uri}: {e}")
        continue
    
    # Define the common box in EPSG:3857
    common_xmin_3857 = max(gdf_xmin, raster_xmin)
    common_ymin_3857 = max(gdf_ymin, raster_ymin)
    common_xmax_3857 = min(gdf_xmax, raster_xmax)
    common_ymax_3857 = min(gdf_ymax, raster_ymax)

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    common_xmin_4326, common_ymin_4326 = transformer.transform(common_xmin_3857, common_ymin_3857)
    common_xmax_4326, common_ymax_4326 = transformer.transform(common_xmax_3857, common_ymax_3857)
    print("Got common extent for: ", city_name, country_code)
    
    buildings = query_buildings_data(con, common_xmin_4326, common_ymin_4326, common_xmax_4326, common_ymax_4326)
    print("Got buildings extent for: ", city_name, country_code)

    rasterized_buildings_source, buildings_extent, crs_transformer_buildings = rasterise_buildings(image_uri, buildings, common_xmin_3857, common_ymax_3857)

    sentinel_source = get_sentinel_source(image_uri, buildings_extent)
    
    buil_ds, sent_ds, build_scene = build_datasets(rasterized_buildings_source, sentinel_source)
    print("Got datasets extent for: ", city_name, country_code)

    save_predictions(best_model, sent_ds, buil_ds, build_scene, crs_transformer_buildings, country_code, city_name)
    print(f"Saved predictions data for {city_name}, {country_code}")
    
# # Merge geojson for cities
# def merge_geojson_files(country_directory, output_file):
#     # Create an empty GeoDataFrame with an appropriate schema
#     merged_gdf = gpd.GeoDataFrame()
    
#     # Traverse the directory structure
#     for city in os.listdir(country_directory):
#         city_path = os.path.join(country_directory, city)
#         vector_output_path = os.path.join(city_path, 'vector_output')
        
#         if os.path.isdir(vector_output_path):
#             # Find the .json file in the vector_output directory
#             for file in os.listdir(vector_output_path):
#                 if file.endswith('.json'):
#                     file_path = os.path.join(vector_output_path, file)
#                     # Load the GeoJSON file into a GeoDataFrame
#                     gdf = gpd.read_file(file_path)
#                     # Add the city name as an attribute to each feature
#                     gdf['city'] = city
#                     # Append to the merged GeoDataFrame
#                     merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
    
#     # Save the merged GeoDataFrame to a GeoJSON file
#     merged_gdf.to_file(output_file, driver='GeoJSON')
#     print(f'Merged GeoJSON file saved to {output_file}')

# # Specify the country directory and the output file path
# country_directory = '../vectorised_model_predictions/multi-modal/SD_GC/SLV/'
# output_file = os.path.join(country_directory, 'SLV_multimodal_SDGC.geojson')

# # Merge the GeoJSON files
# merge_geojson_files(country_directory, output_file)




# # From STAC
# BANDS = [
#     'blue', # B02
#     'green', # B03
#     'red', # B04
#     'nir', # B08
# ]

# URL = 'https://earth-search.aws.element84.com/v1'
# catalog = pystac_client.Client.open(URL)

# for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_ascii']
    country_code = row['iso3']
    print("Doing predictions for: ", city_name, country_code)
    
    gdf_xmin, gdf_ymin, gdf_xmax, gdf_ymax = row['geometry'].bounds
    
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xmin_4326, ymin_4326 = transformer.transform(gdf_xmin, gdf_ymin)
    xmax_4326, ymax_4326 = transformer.transform(gdf_xmax, gdf_ymax)

    bbox = Box(ymin=ymin_4326, xmin=xmin_4326, ymax=ymax_4326, xmax=xmax_4326)
    bbox_geometry = {
        'type': 'Polygon',
        'coordinates': [
            [
                (xmin_4326, ymin_4326),
                (xmin_4326, ymax_4326),
                (xmax_4326, ymax_4326),
                (xmax_4326, ymin_4326),
                (xmin_4326, ymin_4326)
            ]
        ]
    }
    
    # Getting Sentinel data
    try:
        sentinel_source = get_sentinel_item(bbox_geometry, bbox)
        if sentinel_source is None:
            continue
    except Exception as e:
        print(f"An error occurred for {city_name}, {country_code}: {e}")
        continue
    
    # Getting Buildings data    
    buildings = query_buildings_data(con, xmin_4326, ymin_4326, xmax_4326, ymax_4326)
    print("Got buildings for: ", city_name, country_code)

    rasterized_buildings_source, buildings_extent, crs_transformer_buildings = rasterise_buildings(image_uri, buildings, gdf_xmin, gdf_ymax)
    
    buil_ds, sent_ds, build_scene = build_datasets(rasterized_buildings_source, sentinel_source)
    print("Got datasets extent for: ", city_name, country_code)
    
    save_predictions(best_model, sent_ds, buil_ds, build_scene, crs_transformer_buildings, country_code, city_name)
    print(f"Saved predictions data for {city_name}, {country_code}")