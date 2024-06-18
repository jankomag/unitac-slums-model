import sys
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from rastervision.core.box import Box
import cv2
from os.path import join

from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import Dataset, DataLoader
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

from typing import Iterator, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch

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
from src.data.dataloaders import (create_datasets, create_sentinel_raster_source,
                                  create_buildings_raster_source, show_windows)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)

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

label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

sentinel_source_normalized, sentinel_label_raster_source = create_sentinel_raster_source(image_uri, label_uri, class_config)
rasterized_buildings_source, buildings_label_source = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)

gdf = gpd.read_file(label_uri)
gdf = gdf.to_crs('EPSG:3857')
xmin, ymin, xmax, ymax = gdf.total_bounds
pixel_polygon = Polygon([
    (xmin, ymin),
    (xmin, ymax),
    (xmax, ymax),
    (xmax, ymin),
    (xmin, ymin)
])

SentinelScene = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalized,
        label_source = sentinel_label_raster_source,
        aoi_polygons=[pixel_polygon])

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScence, imgsize=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene, imgsize=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildingsGeoDataset.scene.label_source)
train_windows = train_buildings_dataset.windows
val_windows = val_buildings_dataset.windows
test_windows = test_buildings_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')


def create_full_imagesent(source) -> np.ndarray:
    extent = source.extent
    chip = source._get_chip(extent)    
    return chip

img_full = create_full_imagesent(SentinelScene.label_source)
train_windows = train_sentinel_dataset.windows
val_windows = val_sentinel_dataset.windows
test_windows = test_sentinel_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')


class MultiModalDataset(Dataset):
    def __init__(self, sentinel_dataset, buildings_dataset):
        self.sentinel_dataset = sentinel_dataset
        self.buildings_dataset = buildings_dataset

    def __len__(self):
        return len(self.sentinel_dataset)

    def __getitem__(self, idx):
        sentinel_data = self.sentinel_dataset[idx]
        buildings_data = self.buildings_dataset[idx]

        return {
            'sentinel': sentinel_data[0],
            'buildings': buildings_data[0],
            'sentinel_label': sentinel_data[1],
            'buildings_label': buildings_data[1]
        }
        
train_dataset = MultiModalDataset(train_sentinel_dataset, train_buildings_dataset)
val_dataset = MultiModalDataset(val_sentinel_dataset, val_buildings_dataset)
test_dataset = MultiModalDataset(test_sentinel_dataset, test_buildings_dataset)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

sentinel_sample = train_sentinel_dataset[0]
buildings_sample = train_buildings_dataset[0]

print(sentinel_sample[0].shape, sentinel_sample[1].shape)  # Sentinel image and label shapes
print(buildings_sample[0].shape, buildings_sample[1].shape)  # Buildings image and label shapes

# for i in range(2):
#     for batch in train_loader:
#         sentinel_data = batch['sentinel']
        # buildings_data = batch['buildings']
        # labels = batch['sentinel_label']

        # optimizer.zero_grad()
        # outputs = model(sentinel_data, buildings_data)
        # loss = F.cross_entropy(outputs, labels)
        # loss.backward()
        # optimizer.step()