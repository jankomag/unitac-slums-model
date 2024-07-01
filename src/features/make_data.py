import duckdb
import geopandas as gpd
import json
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
import requests
import shutil
import sys
from pathlib import Path
from shapely import wkb, wkt
from shapely.geometry import Point, Polygon, box, shape
from shapely.wkb import loads as wkb_loads
from shapely.wkt import loads
from typing import Dict, Any, Union
from tqdm import tqdm

# donwloaded locally for easy access - code for that in notebooks/overture.ipynb
con = duckdb.connect("../../data/0/data.db")

con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')

con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

### Donwload Overture Maps Buildings from local copy ###
 
sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
# gdf.value_counts("iso3")
hti = gdf[gdf["iso3"] == "PAN"]
hti.tail()
port_au_prince = gdf[gdf["city_ascii"] == "San Miguelito"]
port_au_prince
# GTC = gpd.read_file("../../data/SHP/Guatemala_PS.shp").to_crs("EPSG:4326")
bounds = port_au_prince.geometry.bounds.iloc[0]
xmin, ymin, xmax, ymax = bounds.minx, bounds.miny, bounds.maxx, bounds.maxy
# xmin, ymin, xmax, ymax = GTC.total_bounds

# import rasterio
# image_uriHT = '../../data/0/sentinel_Gee/HTI_Tabarre_2023.tif'
# with rasterio.open(image_uriHT) as src:
#     bounds = src.bounds
#     xmin, ymin, xmax, ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

# coordinates = [[[-72.3545099443,18.5103936876],[-72.3012816531,18.5103936876],[-72.3012816531,18.5567336116],[-72.3545099443,18.5567336116],[-72.3545099443,18.5103936876]]]
# # coordinates = [[[-70.1560805913,18.372680107],[-69.5854513522,18.372680107],[-69.5854513522,18.6220991307],[-70.1560805913,18.6220991307],[-70.1560805913,18.372680107]]]
# flat_coordinates = [item for sublist in coordinates for item in sublist]

# xmin = min(coordinate[0] for coordinate in flat_coordinates)
# xmax = max(coordinate[0] for coordinate in flat_coordinates)
# ymin = min(coordinate[1] for coordinate in flat_coordinates)
# ymax = max(coordinate[1] for coordinate in flat_coordinates)

query = f"""
    SELECT
        *
        --ST_GeomFromWKB(geometry) as geometry
    FROM buildings
    WHERE bbox.xmin > {xmin}
      AND bbox.xmax < {xmax}
      AND bbox.ymin > {ymin}
      AND bbox.ymax < {ymax};
"""

# Convert to GeoDataFrame
buildings = con.sql(query).to_df()

buildings = gpd.GeoDataFrame(buildings, geometry=gpd.GeoSeries.from_wkb(buildings.geometry.apply(bytes)), crs='EPSG:4326')
# viz(buildings)
# # extract source and confidence of buildings
# buildings['source'] = buildings['sources'].apply(lambda x: [entry['dataset'].strip(',') for entry in x])
# buildings['confidence'] = buildings.apply(lambda row: [entry['confidence'] if 'OpenStreetMap' not in row['source'] else 1 for entry in row['sources']], axis=1)
# buildings['source'] = buildings['source'].apply(lambda x: str(x).strip('[]'))
# buildings['confidence'] = buildings['confidence'].apply(lambda x: str(x).strip('[]'))
# buildings['confidence'] = pd.to_numeric(buildings['confidence'], errors='coerce')

buildings = buildings[['id','geometry']]#, 'source', 'confidence']]

buildings = gpd.GeoDataFrame(buildings).to_crs("EPSG:3857")
buildings.to_file("../../data/0/overture/PN_buildings3857.geojson",driver="GeoJSON")
buildings.to_parquet("../../data/0/overture/PN_buildings3857.parquet")

        
### Download DEM using elevation ###
# command line call to download::
# ! eio clip -o ../../data/0/SD-DEM.tif --bounds -70.1560805913 18.372680107 -69.5854513522 18.6220991307

# orginal messy code from dataloaders.py
import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from rastervision.core.box import Box
import stackstac
import pystac_client

from typing import Any, Optional, Tuple, Union, Sequence, List
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader, Dataset
from typing import List
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import seaborn as sns
import random

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
from rastervision.core.data.utils import pad_to_window_size

from rastervision.pytorch_learner.dataset import SlidingWindowGeoDataset, TransformType
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
    
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)

# from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
#                                           SemanticSegmentationVisualizer, SlidingWindowGeoDataset)
# from rastervision.pipeline.utils import repr_with_args

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.utils import parse_array_slices_Nd, fill_overflow

if TYPE_CHECKING:
    from pystac import Item, ItemCollection
    from rastervision.core.data import RasterTransformer, CRSTransformer

log = logging.getLogger(__name__)

def customgeoms_to_raster(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool) -> np.ndarray:
    if len(df) == 0:
        return np.full(window.size, background_class_id, dtype=np.uint8)

    window_geom = window.to_shapely()

    # subset to shapes that intersect window
    df_int = df[df.intersects(window_geom)]
    # transform to window frame of reference
    shapes = df_int.translate(xoff=-window.xmin, yoff=-window.ymin)
    # class IDs of each shape
    class_ids = df_int['class_id']

    if len(shapes) > 0:
        raster = rasterize(
            shapes=list(zip(shapes, class_ids)),
            out_shape=window.size,
            fill=background_class_id,
            dtype='uint8',
            all_touched=all_touched)
    else:
        raster = np.full(window.size, background_class_id, dtype=np.uint8)

    return raster

def geoms_to_raster(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool) -> np.ndarray:
    if len(df) == 0:
        return np.full(window.size, background_class_id, dtype=np.uint8)

    window_geom = window.to_shapely()

    # subset to shapes that intersect window
    df_int = df[df.intersects(window_geom)]
    # transform to window frame of reference
    shapes = df_int.translate(xoff=-window.xmin, yoff=-window.ymin)
    # class IDs of each shape
    class_ids = df_int['class_id']

    if len(shapes) > 0:
        raster = rasterize(
            shapes=list(zip(shapes, class_ids)),
            out_shape=window.size,
            fill=background_class_id,
            dtype=np.uint8,
            all_touched=all_touched)
    else:
        raster = np.full(window.size, background_class_id, dtype=np.uint8)

    return raster

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

def show_windows(img, windows, window_labels, title=''):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img, cmap='gray_r')
    ax.axis('off')
    
    for w, label in zip(windows, window_labels):
        if label == 'train':
            color = 'blue'
        elif label == 'val':
            color = 'red'
        elif label == 'test':
            color = 'green'
        
        p = patches.Polygon(w.to_points(), edgecolor=color, linewidth=1, fill=False)
        ax.add_patch(p)
    
    ax.autoscale()
    ax.set_title(title)
    plt.show()

class CustomMinMaxTransformer(RasterTransformer):
    """Transforms chips by scaling values in each channel to span a specified range."""

    def __init__(self, min_val: Union[float, List[float]], max_val: Union[float, List[float]]):
        """
        Args:
            min_val: Minimum value(s) for scaling. If a single value is provided, it will be broadcasted
                across all channels. If a list of values is provided, it should match the number of channels.
            max_val: Maximum value(s) for scaling. Same broadcasting rules as min_val apply.
        """
        self.min_val = min_val
        self.max_val = max_val

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[List[int]] = None) -> np.ndarray:
        c = chip.shape[-1]
        pixels = chip.reshape(-1, c)
        
        # Broadcasting if single value provided
        if isinstance(self.min_val, (int, float)):
            channel_mins = np.array([self.min_val] * c)
        else:
            channel_mins = np.array(self.min_val)
        
        if isinstance(self.max_val, (int, float)):
            channel_maxs = np.array([self.max_val] * c)
        else:
            channel_maxs = np.array(self.max_val)
        
        chip_normalized = (chip - channel_mins) / (channel_maxs - channel_mins)
        chip_normalized = np.clip(chip_normalized, 0, 1)  # Clip values to [0, 1] range
        chip_normalized = (255 * chip_normalized).astype(np.uint8)
        return chip_normalized
          
class CustomRasterizedSource(RasterSource):
    def __init__(self,
                 vector_source: 'VectorSource',
                 background_class_id: int,
                 bbox: Optional['Box'] = None,
                 all_touched: bool = False,
                 raster_transformers: List['RasterTransformer'] = []):
        self.vector_source = vector_source
        self.background_class_id = background_class_id
        self.all_touched = all_touched

        self.df = self.vector_source.get_dataframe()
        self.validate_labels(self.df)

        if bbox is None:
            bbox = self.vector_source.extent

        super().__init__(
            channel_order=[0],
            num_channels_raw=1,
            bbox=bbox,
            raster_transformers=raster_transformers)

    @property
    def dtype(self) -> np.dtype:
        return np.uint8

    @property
    def crs_transformer(self):
        return self.vector_source.crs_transformer

    def _get_chip(self,
                  window: 'Box',
                  out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        window = window.to_global_coords(self.bbox)
        chip = geoms_to_raster(
            self.df,
            window,
            background_class_id=self.background_class_id,
            all_touched=self.all_touched)

        if out_shape is not None:
            chip = self.resize(chip, out_shape)

        return np.expand_dims(chip, 2)

    def validate_labels(self, df: gpd.GeoDataFrame) -> None:
        geom_types = set(df.geom_type)
        if 'Point' in geom_types or 'LineString' in geom_types:
            raise ValueError('LineStrings and Points are not supported '
                             'in RasterizedSource. Use BufferTransformer '
                             'to buffer them into Polygons. '
                             f'Geom types found in data: {geom_types}')

        if len(df) > 0 and 'class_id' not in df.columns:
            raise ValueError('All label polygons must have a class_id.')

# custom class to normalise pixel values as z-scores
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

class CustomSemanticSegmentationLabelSource(LabelSource):
    def __init__(self,
                 raster_source: RasterSource,
                 class_config: ClassConfig,
                 bbox: Optional[Box] = None):
        self.raster_source = raster_source
        self.class_config = class_config
        if bbox is not None:
            self.set_bbox(bbox)

    def get_labels(self,
                   window: Optional[Box] = None) -> SemanticSegmentationLabels:
        if window is None:
            window = self.extent

        label_arr = self.get_label_arr(window)
        labels = SemanticSegmentationLabels.make_empty(
            extent=self.extent,
            num_classes=len(self.class_config),
            smooth=False)
        labels[window] = label_arr
        
        labels = torch.tensor(labels, dtype=torch.float32)

        return labels

    def get_label_arr(self, window: Optional[Box] = None) -> np.ndarray:
        if window is None:
            window = self.extent

        label_arr = self.raster_source.get_chip(window)
        if label_arr.ndim == 3:
            label_arr = np.squeeze(label_arr, axis=2)
        h, w = label_arr.shape
        if h < window.height or w < window.width:
            label_arr = pad_to_window_size(label_arr, window, self.extent,
                                           self.class_config.null_class_id)
        return label_arr

    @property
    def bbox(self) -> Box:
        return self.raster_source.bbox

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.raster_source.crs_transformer

    def set_bbox(self, bbox: 'Box') -> None:
        self.raster_source.set_bbox(bbox)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Box):
            return self.get_label_arr(key)
        else:
            return super().__getitem__(key)

# Class to implement train, test, val split and show windows
class CustomSemanticSegmentationSlidingWindowGeoDataset(SlidingWindowGeoDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            transform_type=TransformType.semantic_segmentation)

    def split_train_val_test(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = None) -> Tuple['CustomSemanticSegmentationSlidingWindowGeoDataset', 'CustomSemanticSegmentationSlidingWindowGeoDataset', 'CustomSemanticSegmentationSlidingWindowGeoDataset']:
        """
        Split the dataset into training, validation, and test subsets.

        Args:
            val_ratio (float): Ratio of validation data to total data. Defaults to 0.2.
            test_ratio (float): Ratio of test data to total data. Defaults to 0.2.
            seed (int): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[CustomSemanticSegmentationSlidingWindowGeoDataset, CustomSemanticSegmentationSlidingWindowGeoDataset, CustomSemanticSegmentationSlidingWindowGeoDataset]: 
                Training, validation, and test subsets as CustomSemanticSegmentationSlidingWindowGeoDataset objects.
        """
        assert 0.0 < val_ratio < 1.0, "val_ratio should be between 0 and 1."
        assert 0.0 < test_ratio < 1.0, "test_ratio should be between 0 and 1."
        assert val_ratio + test_ratio < 1.0, "Sum of val_ratio and test_ratio should be less than 1."

        if seed is not None:
            random.seed(seed)

        # Calculate number of samples for validation, test, and training
        num_samples = len(self)
        num_val = int(val_ratio * num_samples)
        num_test = int(test_ratio * num_samples)
        num_train = num_samples - num_val - num_test

        # Create indices for train, validation, and test subsets
        indices = list(range(num_samples))
        random.shuffle(indices)  # Shuffle indices randomly

        val_indices = indices[:num_val]
        test_indices = indices[num_val:num_val + num_test]
        train_indices = indices[num_val + num_test:]

        # Create new datasets for training, validation, and test
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices)
        test_dataset = self._create_subset(test_indices)

        return train_dataset, val_dataset, test_dataset

    def _create_subset(self, indices):
        """
        Create a subset of the dataset using specified indices.

        Args:
            indices (list): List of indices to include in the subset.

        Returns:
            CustomSemanticSegmentationSlidingWindowGeoDataset: Subset of the dataset.
        """
        subset = CustomSemanticSegmentationSlidingWindowGeoDataset(
            scene=self.scene,
            size=self.size,
            stride=self.stride,
            out_size=self.out_size,
            padding=self.padding
            # Include other parameters as needed
        )

        # Initialize subset's windows based on provided indices
        subset.windows = [self.windows[i] for i in indices]

        return subset
    
### Label source ###
label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

gdf = gpd.read_file(label_uri)
gdf = gdf.to_crs('EPSG:3857')
xmin, ymin, xmax, ymax = gdf.total_bounds

class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

crs_transformer = RasterioCRSTransformer.from_uri(image_uri)
crs_transformer.transform

label_vector_source = GeoJSONVectorSource(label_uri,
    crs_transformer,
    vector_transformers=[
        ClassInferenceTransformer(
            default_class_id=class_config.get_class_id('slums'))])

label_raster_source = CustomRasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
print(f"Loaded UNITAC CustomRasterizedSource: {label_raster_source.shape}")

label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
print(f"Loaded UNITAC SemanticSegmentationLabelSource: {label_raster_source.shape}")

chip = label_source[:, :]
chip.shape
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(chip)
plt.show()

### SENTINEL source
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

# Define an unnormalized raster source
sentinel_source_unnormalized = RasterioSource(
    image_uri,
    allow_streaming=True)

# Calculate statistics transformer from the unnormalized source
calc_stats_transformer = CustomStatsTransformer.from_raster_sources(
    raster_sources=[sentinel_source_unnormalized],
    max_stds=3
)

# Define a normalized raster source using the calculated transformer
sentinel_source_normalized = RasterioSource(
    image_uri,
    allow_streaming=True,
    raster_transformers=[calc_stats_transformer],
    channel_order=[2, 1, 0, 3],
    bbox = label_raster_source.bbox
)

# Rescale the image data to be in the range [0, 1]
chip = sentinel_source_normalized[:, :, :3]
chip_scaled = (chip - np.min(chip)) / (np.max(chip) - np.min(chip))
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(chip_scaled)
plt.show()

print(f"Minimum pixel value after normalization: {chip.min()}")
print(f"Maximum pixel value after normalization: {chip.max()}")

# Get from STAC


# Density plot
# num_bands = chip.shape[-1]
# plt.figure(figsize=(8, 6))
# for band in range(num_bands):
#     band_data = chip[..., band].flatten()
#     sns.kdeplot(band_data, shade=True, label=f'Band {band}', color=plt.cm.viridis(band / num_bands))
# plt.legend()
# plt.grid(True)
# plt.show()

print(f"Loaded Sentinel data of size {sentinel_source_normalized.shape}, and dtype: {sentinel_source_normalized.dtype}")

### Evaluation Scene ###
pixel_polygon = Polygon([
    (xmin, ymin),
    (xmin, ymax),
    (xmax, ymax),
    (xmax, ymin),
    (xmin, ymin)
])

SentinelSDScene = Scene(
    id='santo_domingo_eval',
    raster_source=sentinel_source_normalized,
    label_source = label_raster_source, #label_source
    aoi_polygons=[pixel_polygon])

class_config = ClassConfig(
    names=['background', 'slums'],
    colors=['lightgray', 'darkred'],
    null_class='background')
channel_display_groups = {'RGB': (0, 1, 2), 'NIR': (3, )}

sentinel_train_ds = SemanticSegmentationSlidingWindowGeoDataset(
    scene=SentinelSDScene,
    size=144,
    stride=144,
    out_size=144,
    padding=0)

vis = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors,
    channel_display_groups=channel_display_groups)
x, y = vis.get_batch(sentinel_train_ds, 2)
vis.plot_batch(x, y, show=True)

### Mulitraster source ###
# raster_source_multi = MultiRasterSource(
#     raster_sources=[rasterized_buildings_source, sentinel_source_normalized],
#     primary_source_idx=0,
#     force_same_dtype = True)
# print(f"Loaded final multiraster: {raster_source_multi.shape, raster_source_multi.dtype}")

### Define functions ###
def get_senitnel_dl_ds(batch_size=8, size=256, stride=256, out_size=256, padding=256):
    
    sentinel_train_ds = SemanticSegmentationSlidingWindowGeoDataset(
        scene=SentinelSDScene,
        size=size,
        stride=stride,
        out_size=out_size,
        padding=padding,
        to_pytorch=True,
        normalize=False)
    
    sentinel_val_ds = sentinel_train_ds # for evaluation purposes the datasets are the same (no training is taking place here)
    
    sentinel_dl = DataLoader(sentinel_train_ds, batch_size=batch_size, shuffle=True)
    return sentinel_dl, sentinel_train_ds, sentinel_val_ds

def get_training_sentinelOnly(batch_size=8, imgsize=256, padding=50, seed=42):
    
    full_dataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
        scene=SentinelSDScene,
        size=imgsize,
        stride=imgsize,
        out_size=imgsize,
        padding=padding)

    # Splitting dataset into train, validation, and test
    train_dataset, val_dataset, test_dataset = full_dataset.split_train_val_test(val_ratio=0.2, test_ratio=0.1, seed=seed)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test (hold-out) dataset length: {len(test_dataset)}")

    # Define DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size)
    
    return full_dataset, train_dl, val_dl, train_dataset, val_dataset, test_dataset