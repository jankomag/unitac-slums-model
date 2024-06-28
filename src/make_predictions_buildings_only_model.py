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

from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader
from typing import List
from rasterio.features import rasterize
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
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from deeplnafrica.deepLNAfrica import init_segm_model
from src.models.model_definitions import (BuildingsDeeplabv3, BuildingsUNET, CustomVectorOutputConfig, BuildingsOnlyPredictionsIterator, CustomGeoJSONVectorSource)
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

#  Original code for making predictions that works 
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
    
# Model definition
best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
best_model_path_unet = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/unet/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=14-val_loss=0.4913.ckpt"

# BuildingsDeeplabv3 BuildingsUNET
best_model = BuildingsDeeplabv3.load_from_checkpoint(best_model_path_deeplab)
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

def query_buildings_data(row, con):
    city_name = row['city_ascii']
    xmin, ymin, xmax, ymax = row['geometry'].bounds

    query = f"""
        SELECT *
        FROM buildings
        WHERE bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax};
    """
    buildings = pd.read_sql(query, con=con)

    if not buildings.empty:
        buildings = gpd.GeoDataFrame(buildings, geometry=gpd.GeoSeries.from_wkb(buildings.geometry.apply(bytes)), crs='EPSG:4326')
        buildings = buildings[['id', 'geometry']]
        buildings = buildings.to_crs("EPSG:3857")

    return buildings

def create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5):
    xmin, _, _, ymax = buildings.total_bounds

    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
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

def predict_and_save(buildings, image_uri, label_uri, class_config, crs_transformer_common, iso3_country_code, city_name, best_model, device):
    rasterized_buildings_source, buildings_label_source, crs_transformer_common = create_buildings_raster_source(buildings, image_uri, label_uri, class_config, resolution=5)
    
    eval_scene = Scene(
        id='eval_scene',
        raster_source=rasterized_buildings_source,
        label_source=buildings_label_source)

    buildingsGeoDataset, _, _, _ = create_datasets(
    eval_scene, imgsize=288, stride = 144, padding=0, val_ratio=0.4, test_ratio=0.1, seed=42)

    predictions_iterator = PredictionsIterator(best_model, buildingsGeoDataset, device=device)
    windows, predictions = zip(*predictions_iterator)

    pred_labels_HT = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=eval_scene.extent,
        num_classes=len(class_config),
        smooth=True)

    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5)

    pred_label_store = SemanticSegmentationLabelStore(
        uri=f'../vectorised_model_predictions/buildings_model_only/{iso3_country_code}/{city_name}_{iso3_country_code}',
        crs_transformer=crs_transformer_common,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True,
        smooth_output = False)

    pred_label_store.save(pred_labels_HT)
    print(f"Saved predictions data for {city_name}")

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities).to_crs("EPSG:4326")
image_uri = '../../data/0/sentinel_Gee/HTI_Tabarre_2023.tif'
label_uri = "../../data/0/SantoDomingo3857.geojson"
crs_transformer_common = RasterioCRSTransformer.from_uri(image_uri)

device='mps'
for index, row in gdf.iterrows():
    buildings = query_buildings_data(row, con)
    buildings['class_id'] = 1
    
    if buildings is not None:
        predict_and_save(buildings, image_uri, label_uri, class_config, crs_transformer_common, row['iso3'], row['city_ascii'], best_model, device)
        
# Merge geojson for cities
def merge_geojson_files(country_directory, output_file):
    # Create an empty GeoDataFrame with an appropriate schema
    merged_gdf = gpd.GeoDataFrame()
    
    # Traverse the directory structure
    for city in os.listdir(country_directory):
        city_path = os.path.join(country_directory, city)
        vector_output_path = os.path.join(city_path, 'vector_output')
        
        if os.path.isdir(vector_output_path):
            # Find the .json file in the vector_output directory
            for file in os.listdir(vector_output_path):
                if file.endswith('.json'):
                    file_path = os.path.join(vector_output_path, file)
                    # Load the GeoJSON file into a GeoDataFrame
                    gdf = gpd.read_file(file_path)
                    # Add the city name as an attribute to each feature
                    gdf['city'] = city
                    # Append to the merged GeoDataFrame
                    merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
    
    # Save the merged GeoDataFrame to a GeoJSON file
    merged_gdf.to_file(output_file, driver='GeoJSON')
    print(f'Merged GeoJSON file saved to {output_file}')

# Specify the country directory and the output file path
country_directory = '../vectorised_model_predictions/deeplab_buildings_model_only/SLV/'
output_file = os.path.join(country_directory, 'SLV_spatial_data.geojson')

# Merge the GeoJSON files
merge_geojson_files(country_directory, output_file)