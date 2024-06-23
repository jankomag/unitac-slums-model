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

from typing import Iterator, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torch.utils.data import ConcatDataset

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from deeplnafrica.deepLNAfrica import init_segm_model

from src.data.dataloaders import (
    create_datasets,
    create_buildings_raster_source, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
)

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

# Model definition
class MultiModalSegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.learning_rate = 1e-4
        self.weight_decay = 0
        self.sentinel_encoder = deeplabv3_resnet50(pretrained=False, progress=False)
        self.buildings_encoder = deeplabv3_resnet50(pretrained=False, progress=True)
        
        self.sentinel_encoder.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.buildings_encoder.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Intermediate Layer Getters
        self.sentinel_encoder_backbone = IntermediateLayerGetter(self.sentinel_encoder.backbone, {'layer4': 'out_sent', 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        self.buildings_encoder_backbone = IntermediateLayerGetter(self.buildings_encoder.backbone, {'layer4': 'out_buil', 'layer3': 'layer3','layer2': 'layer2','layer1': 'layer1'})
        self.buildings_downsampler = nn.Conv2d(2048, 2048, kernel_size=2, stride=2)

        self.segmentation_head = DeepLabHead(in_channels=2048*2, num_classes=1, atrous_rates=(6, 12, 24))
             
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
        
        concatenated_features = torch.cat([sentinel_out, buildings_out_downsampled], dim=1)
        
        # Decode the fused features
        segmentation = self.segmentation_head(concatenated_features)
        
        segmentation = F.interpolate(segmentation, size=288, mode="bilinear", align_corners=False)
        
        return segmentation.squeeze()
    
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
    
def query_buildings_data(con, xmin, ymin, xmax, ymax):
    query = f"""
        SELECT *
        FROM buildings
        WHERE bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax};
    """
    buildings_df = pd.read_sql(query, con=con)

    if not buildings_df.empty:
        buildings = gpd.GeoDataFrame(buildings_df, geometry=gpd.GeoSeries.from_wkb(buildings_df.geometry.apply(bytes)), crs='EPSG:4326')
        buildings = buildings[['id', 'geometry']]
        buildings = buildings.to_crs("EPSG:3857")
        buildings['class_id'] = 1

    return buildings

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
        id='portauprince_sent',
        raster_source = rasterized_buildings_source)

    HT_sent_scene = Scene(
            id='portauprince_buildings',
            raster_source = sentinel_source)

    build_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
            scene=build_scene,
            size=288,
            stride=144,
            out_size=288,
            padding=0)

    sent_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
            scene=HT_sent_scene,
            size=144,
            stride=72,
            out_size=144,
            padding=0)
    
    return build_ds, sent_ds, build_scene

def save_predictions(model, sent_ds, buil_ds, build_scene, crs_transformer, country_code, city_name):
    device = 'mps'
    predictions_iterator = PredictionsIterator(model, sent_ds, buil_ds, device=device)
    windows, predictions = zip(*predictions_iterator)

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
        uri=f'../../vectorised_model_predictions/multi-modal/SD_only_train/{country_code}/{city_name}_{country_code}',
        crs_transformer = crs_transformer,
        class_config = class_config,
        vector_outputs = [vector_output_config],
        discrete_output = True)

    pred_label_store.save(pred_labels)

best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/multi_modal/multimodal_runidrun_id=0-batch_size=00-epoch=15-val_loss=0.2595.ckpt"
best_model = MultiModalSegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

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

gdf = gdf.tail(2)
for index, row in gdf.iterrows():
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