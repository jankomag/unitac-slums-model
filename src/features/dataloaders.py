import os
import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from rastervision.core.box import Box
from typing import Any, Optional, Tuple, Union, Sequence, List
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader, Dataset
from typing import List
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import random
from rastervision.pytorch_learner.dataset import GeoDataset
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union
import logging
import duckdb
import geopandas as gpd
from shapely import wkb
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset
from shapely.ops import unary_union
from sklearn.model_selection import train_test_split
# from pystac_client import Client

from rastervision.core.box import Box
from rastervision.core.data import Scene, BufferTransformer
from rastervision.core.data.utils import AoiSampler
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.transform import (TransformType,
                                                            TF_TYPE_TO_TF_FUNC)
from typing import TYPE_CHECKING, Optional, Sequence, Union
from torch.utils.data import SubsetRandomSampler
import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (
    color_to_triple, plot_channel_groups, channel_groups_to_imgs)

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes
    from matplotlib.colors import Colormap

from typing import (TYPE_CHECKING, Sequence, Optional, List, Dict, Union,
                    Tuple, Any)
from abc import ABC, abstractmethod

from torch import Tensor
import albumentations as A
import matplotlib.pyplot as plt

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
    from shapely.geometry import MultiPolygon, Polygon

log = logging.getLogger(__name__)
import albumentations as A
from typing import Literal, TypeVar
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union
from typing import List
T = TypeVar('T')
import pandas as pd
from rastervision.core.data.utils import listify_uris, merge_geojsons
from sklearn.model_selection import KFold
from typing import Literal, TypeVar
import math

from pydantic.types import NonNegativeInt as NonNegInt, PositiveInt as PosInt

T = TypeVar('T')

from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    MinMaxTransformer, MultiRasterSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource, CRSTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    SemanticSegmentationLabelSource, RasterizerConfig)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.utils import pad_to_window_size

from rastervision.pytorch_learner.dataset import SlidingWindowGeoDataset, TransformType
# from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
#                                           SemanticSegmentationVisualizer, SlidingWindowGeoDataset)
from rastervision.pipeline.utils import repr_with_args
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
from xarray import DataArray
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, Subset
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if TYPE_CHECKING:
    from pystac import Item, ItemCollection
    from rastervision.core.data import RasterTransformer, CRSTransformer

log = logging.getLogger(__name__)
from rastervision.core.data.label_store import SemanticSegmentationLabelStore
from os.path import join
import random
from typing import Tuple, List, Union, Optional
from rastervision.core.box import Box
from rastervision.core.data import Scene
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.dataset import GeoDataset
from rastervision.pytorch_learner import SemanticSegmentationVisualizer
import numpy as np
import albumentations as A
from sklearn.model_selection import KFold

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

cities = {
    'SanJose': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/CRI_SanJose_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanJose_PS.shp'),
        'use_augmentation': False
    },
    'Tegucigalpa': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/HND_Comayaguela_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Tegucigalpa_PS.shp'),
        'use_augmentation': False
    },
    'SantoDomingo': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/0/SantoDomingo3857_buffered.geojson'),
        'use_augmentation': False
    },
    'GuatemalaCity': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/GTM_Guatemala_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Guatemala_PS.shp'),
        'use_augmentation': False
    },
    'Managua': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/NIC_Tipitapa_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Managua_PS.shp'),
        'use_augmentation': False
    },
    'Panama': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/PAN_Panama_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Panama_PS.shp'),
        'use_augmentation': False
    },
    'SanSalvador': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/SLV_SanSalvador_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
        'use_augmentation': False
    },
    'BelizeCity': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/BelizeCity_PS.shp')
        },
    'Belmopan': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Belmopan_PS.shp')
        }
}

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def geoms_to_raster(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool) -> np.ndarray:
    """Rasterize geometries that intersect with the window.

    Args:
        df (gpd.GeoDataFrame): All label geometries in the scene.
        window (Box): The part of the scene to rasterize.
        background_class_id (int): Class ID to use for pixels that don't
            fall under any label geometry.
        all_touched (bool): If True, all pixels touched by geometries will be
            burned in. If false, only pixels whose center is within the
            polygon or that are selected by Bresenham's line algorithm will be
            burned in. (See :func:`.rasterize` for more details).
            Defaults to False.

    Returns:
        np.ndarray: A raster.
    """
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

def _to_tuple(x: T, n: int = 2) -> Tuple[T, ...]:
    """Convert to n-tuple if not already an n-tuple."""
    if isinstance(x, tuple):
        if len(x) != n:
            raise ValueError()
        return x
    return tuple([x] * n)

def ensure_tuple(x: T, n: int = 2) -> tuple[T, ...]:
    """Convert to n-tuple if not already an n-tuple."""
    if isinstance(x, tuple):
        if len(x) != n:
            raise ValueError()
        return x
    return tuple([x] * n)

class MergeDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
            
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

class CustomRasterizerConfig(RasterizerConfig):
    background_class_id: int = 0

    def build(self):
        def custom_rasterizer(features):
            return np.where(features == 1, 1.0, np.where(features == 2, 0.5, 0.0))
        return custom_rasterizer

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

class CustomStatsTransformer(RasterTransformer):
    # custom class to normalise pixel values as z-scores
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

class FixedStatsTransformer(RasterTransformer):
    def __init__(self, means, stds, max_stds: float = 3.):
        
        self.means = means
        self.stds = stds
        self.max_stds = max_stds

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[Sequence[int]] = None) -> np.ndarray:
        if chip.dtype == np.uint8:
            return chip

        means = np.array(self.means)
        stds = np.array(self.stds)
        max_stds = self.max_stds
        # if channel_order is not None:
        #     means = means[channel_order]
        #     stds = stds[channel_order]

        # Don't transform NODATA zero values.
        nodata_mask = chip == 0

        # Convert chip to float (if not already)
        chip = chip.astype(float)

        # Subtract mean and divide by std to get z-scores.
        for i in range(chip.shape[-1]):  # Loop over channels
            chip[..., i] -= means[i]
            chip[..., i] /= stds[i]

        # Apply max_stds clipping
        # chip = np.clip(chip, -max_stds, max_stds)
    
        # Normalize to have standard deviation of 1
        for i in range(chip.shape[-1]):
            chip[..., i] /= np.std(chip[..., i])

        chip[nodata_mask] = 0
        
        return chip

    @classmethod
    def create(cls, max_stds: float = 3.) -> 'FixedStatsTransformer':
        return cls(max_stds=max_stds)

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

class CustomSlidingWindowGeoDataset(GeoDataset):
    """Read the scene left-to-right, top-to-bottom, using a sliding window.
    """

    def __init__(
            self,
            scene: Scene,
            city: str,
            size: Union[PosInt, Tuple[PosInt, PosInt]],
            stride: Union[PosInt, Tuple[PosInt, PosInt]],
            out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
            padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                     NonNegInt]]] = None,
            pad_direction: Literal['both', 'start', 'end'] = 'end',
            within_aoi: bool = False,
            transform: Optional[A.BasicTransform] = None,
            transform_type: Optional[TransformType] = None,
            normalize: bool = False,
            to_pytorch: bool = True,
            return_window: bool = False):
        
        super().__init__(
            scene=scene,
            out_size=out_size,
            within_aoi=within_aoi,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch,
            return_window=return_window)
        self.city = city
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.padding = padding
        self.pad_direction = pad_direction
        self.init_windows()

    def init_windows(self) -> None:
        """Pre-compute windows."""
        extent = self.scene.extent
        
        windows = extent.get_windows(
            self.size,
            stride=self.stride,
            padding=self.padding,
            pad_direction=self.pad_direction)
                
        if len(self.scene.aoi_polygons_bbox_coords) > 0:
            windows = Box.filter_by_aoi(
                windows,
                self.scene.aoi_polygons_bbox_coords,
                within=self.within_aoi)
        
        self.windows = windows

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx >= len(self):
                raise IndexError("Index out of range")
            window = self.windows[idx]
            return super().__getitem__(window)
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.windows)

class PolygonWindowGeoDataset(GeoDataset):
    def __init__(
        self,
        scene: Scene,
        city: str,
        window_size: Union[PosInt, Tuple[PosInt, PosInt]],
        out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
        padding: Optional[Union[NonNegInt, Tuple[NonNegInt, NonNegInt]]] = None,
        transform: Optional[A.BasicTransform] = None,
        transform_type: Optional[TransformType] = None,
        normalize: bool = False,
        to_pytorch: bool = True,
        return_window: bool = False,
        within_aoi: bool = False,
    ):
        super().__init__(
            scene=scene,
            out_size=out_size,
            within_aoi=within_aoi,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch,
            return_window=return_window,
        )
        
        self.city = city
        self.window_size: tuple[PosInt, PosInt] = ensure_tuple(window_size)
        self.padding = padding
        if self.padding is None:
            self.padding = (self.window_size[0] // 2, self.window_size[1] // 2)
        self.padding: tuple[NonNegInt, NonNegInt] = ensure_tuple(self.padding)

        self.windows = self.get_polygon_windows()

    def get_polygon_windows(self) -> List[Box]:
        """
        Get a list of window coordinates around the labeled areas in the scene.
        """
        windows = []
        for y in range(0, self.scene.extent.ymax, self.window_size[0]):
            for x in range(0, self.scene.extent.xmax, self.window_size[1]):
                window = Box(y, x, y + self.window_size[0], x + self.window_size[1])
                if self.has_label_data(window):
                    windows.append(window)
        return windows

    def has_label_data(self, window: Box) -> bool:
        """
        Check if the given window contains any labeled data (i.e., non-zero values).
        """
        x, y, w, h = window
        label_arr = self.scene.label_source.get_label_arr(window)
        return np.any(label_arr != 0)

    def split_train_val_test(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = None) -> Tuple['PolygonWindowGeoDataset', 'PolygonWindowGeoDataset', 'PolygonWindowGeoDataset']:
        """
        Split the dataset into training, validation, and test subsets.

        Args:
            val_ratio (float): Ratio of validation data to total data. Defaults to 0.2.
            test_ratio (float): Ratio of test data to total data. Defaults to 0.2.
            seed (int): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[PolygonWindowGeoDataset, PolygonWindowGeoDataset, PolygonWindowGeoDataset]: 
                Training, validation, and test subsets as PolygonWindowGeoDataset objects.
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

    def __getitem__(self, index_or_indices):
        if isinstance(index_or_indices, int):
            if index_or_indices >= len(self):
                raise StopIteration()
            window = self.windows[index_or_indices]
            return super().__getitem__(window)
        elif isinstance(index_or_indices, slice) or isinstance(index_or_indices, list):
            windows = [self.windows[i] for i in index_or_indices]
            items = [super().__getitem__(w) for w in windows]
            if self.return_window:
                return [(item, w) for item, w in zip(items, windows)]
            return items
        else:
            raise TypeError(f"Invalid index type: {type(index_or_indices)}")

    def _create_subset(self, indices):
        """
        Create a subset of the dataset using specified indices.

        Args:
            indices (list): List of indices to include in the subset.

        Returns:
            PolygonWindowGeoDataset: Subset of the dataset.
        """
        subset = PolygonWindowGeoDataset(
            scene=self.scene,
            window_size=self.window_size,
            out_size=self.out_size,
            padding=self.padding,
            transform=self.transform,
            transform_type=self.transform_type,
            normalize=self.normalize,
            to_pytorch=self.to_pytorch,
            return_window=self.return_window,
            within_aoi=self.within_aoi
        )

        # Initialize subset's windows based on provided indices
        subset.windows = [self.windows[i] for i in indices]

        return subset

    def __len__(self):
        return len(self.windows)

class CustomSemanticSegmentationVisualizer(Visualizer):
    def plot_batch(self,
                   x: List[Tuple[torch.Tensor, torch.Tensor]],
                   y: List[Tuple[torch.Tensor, torch.Tensor]],
                   output_path: Optional[str] = None,
                   z: Optional[Sequence] = None,
                   batch_limit: Optional[int] = None,
                   show: bool = False):

        sentinel_data = [item[0] for item in x]
        buildings_data = [item[1] for item in x]
        sentinel_labels = [item[0] for item in y]
        buildings_labels = [item[1] for item in y]
        
        # Plot Sentinel data
        self._plot_single_source(sentinel_data, sentinel_labels, output_path, z, batch_limit, show, "Sentinel")
        
        # Plot Buildings data
        self._plot_single_source(buildings_data, buildings_labels, output_path, z, batch_limit, show, "Buildings")

    def _plot_single_source(self,
                            x: List[torch.Tensor],
                            y: List[torch.Tensor],
                            output_path: Optional[str] = None,
                            z: Optional[Sequence] = None,
                            batch_limit: Optional[int] = None,
                            show: bool = False,
                            title_prefix: str = ""):

        batch_size = len(x)
        if batch_limit is not None:
            batch_size = min(batch_size, batch_limit)

        fig, axs = plt.subplots(batch_size, 2, figsize=(10, 5*batch_size))
        if batch_size == 1:
            axs = axs.reshape(1, -1)

        for i in range(batch_size):
            # Plot input image
            if title_prefix == "Sentinel":
                img = x[i][:3].permute(1, 2, 0)  # Use first 3 channels for RGB
            else:  # Buildings
                img = x[i].squeeze()  # Remove singleton dimensions if any
            axs[i, 0].imshow(img.cpu().numpy())
            axs[i, 0].set_title(f"{title_prefix} Input")
            axs[i, 0].axis('off')

            # Plot ground truth
            gt = y[i].squeeze()  # Squeeze out singleton dimensions
            if gt.ndim > 2:
                gt = gt[0]  # If still more than 2D, take the first slice
            axs[i, 1].imshow(gt.cpu().numpy(), cmap='viridis')
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis('off')

        plt.tight_layout()

        if show:
            plt.show()
        if output_path is not None:
            make_dir(output_path, use_dirname=True)
            fig.savefig(f"{output_path}_{title_prefix.lower()}.png", bbox_inches='tight', pad_inches=0.2)

        plt.close(fig)
    
    def plot_xyz(self,
                 axs: Sequence,
                 x: torch.Tensor,
                 y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                 z: Optional[torch.Tensor] = None,
                 plot_title: bool = True) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])

        img_axes = axs[:len(channel_groups)]

        # plot image
        imgs = channel_groups_to_imgs(x, channel_groups)
        plot_channel_groups(
            img_axes, imgs, channel_groups, plot_title=plot_title)

        if y is None and z is None:
            return

        # plot labels
        class_colors = self.class_colors
        colors = [
            color_to_triple(c) if isinstance(c, str) else c
            for c in class_colors
        ]
        colors = np.array(colors) / 255.
        cmap = mcolors.ListedColormap(colors)

        if y is not None:
            label_ax: 'Axes' = axs[len(channel_groups)]
            self.plot_gt(label_ax, y, num_classes=len(colors), cmap=cmap)
            if plot_title:
                label_ax.set_title('Ground truth')

        if z is not None:
            pred_ax = axs[-1]
            self.plot_pred(pred_ax, z, num_classes=len(colors), cmap=cmap)
            if plot_title:
                pred_ax.set_title('Predicted labels')

        # add a legend to the rightmost subplot
        class_names = self.class_names
        if class_names:
            legend_items = [
                mpatches.Patch(facecolor=col, edgecolor='black', label=name)
                for col, name in zip(colors, class_names)
            ]
            axs[-1].legend(
                handles=legend_items,
                loc='center left',
                bbox_to_anchor=(1., 0.5))

    def plot_gt(self, ax: 'Axes', y: Union[torch.Tensor, np.ndarray],
                num_classes: int, cmap: 'Colormap', **kwargs):
        ax.imshow(
            y,
            vmin=0,
            vmax=num_classes,
            cmap=cmap,
            interpolation='none',
            **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_pred(self, ax: 'Axes', z: Union[torch.Tensor, np.ndarray],
                  num_classes: int, cmap: 'Colormap', **kwargs):
        if z.ndim == 3:
            z = z.argmax(dim=0)
        self.plot_gt(ax, y=z, num_classes=num_classes, cmap=cmap, **kwargs)

    def get_plot_ncols(self, **kwargs) -> int:
        x = kwargs['x']
        nb_img_channels = x.shape[1]
        ncols = len(self.get_channel_display_groups(nb_img_channels))
        if kwargs.get('y') is not None:
            ncols += 1
        if kwargs.get('z') is not None:
            ncols += 1
        return ncols

def collate_fn(batch):
    processed_batch = []
    for item in batch:
        image, label = item
        
        # Replace NaN values with 0 in the image
        image = torch.nan_to_num(image, nan=0.0)
        
        # Check if there are still any NaN values in the label
        if torch.isnan(label).any():
            print(f"NaN found in label")
            continue        
        processed_batch.append((image, label))
    
    return torch.utils.data.dataloader.default_collate(processed_batch)

def collate_multi_fn(batch):
    sentinel_data = []
    buildings_data = []
    labels = []

    for item in batch:
        sentinel_batch, buildings_batch = item
        
        sentinel_item, sentinel_label = sentinel_batch
        buildings_item, buildings_label = buildings_batch

        # Replace NaN values with 0 in the image data
        sentinel_item = torch.nan_to_num(sentinel_item, nan=0.0)
        buildings_item = torch.nan_to_num(buildings_item, nan=0.0)

        # Check if there are any NaN values in the labels
        if torch.isnan(sentinel_label).any() or torch.isnan(buildings_label).any():
            print(f"NaN found in label")
            continue
        
        sentinel_data.append(sentinel_item)
        buildings_data.append(buildings_item)
        labels.append(sentinel_label)  # Use sentinel_label as they should be the same

    # If all items were skipped due to NaN values, return None
    if len(sentinel_data) == 0:
        return None

    # Stack the data and labels
    sentinel_data = torch.stack(sentinel_data)
    buildings_data = torch.stack(buildings_data)
    labels = torch.stack(labels)

    return ((sentinel_data, labels), (buildings_data, labels))

class AugmentedSubset(Subset):
    def __init__(self, dataset, indices, augmentation):
        super().__init__(dataset, indices)
        self.augmentation = augmentation

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        if isinstance(item, tuple):
            # For multi-input datasets
            sentinel_data, buildings_data = item
            sentinel_image, sentinel_label = sentinel_data
            buildings_image, buildings_label = buildings_data
            
            # Augment sentinel data
            augmented = self.augmentation(image=sentinel_image.numpy(), mask=sentinel_label.numpy())
            aug_sentinel_image, aug_sentinel_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            
            # Augment buildings data
            augmented = self.augmentation(image=buildings_image.numpy(), mask=buildings_label.numpy())
            aug_buildings_image, aug_buildings_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            
            return ((aug_sentinel_image, aug_sentinel_label), (aug_buildings_image, aug_buildings_label))
        else:
            # For single-input datasets
            image, label = item
            augmented = self.augmentation(image=image.numpy(), mask=label.numpy())
            aug_image, aug_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            return (aug_image, aug_label)

class BaseCrossValidator:
    def __init__(self, datasets, n_splits=2, val_ratio=0.2, test_ratio=0.1):
        self.datasets = datasets
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.city_splits = self._create_splits()
        self.augmentation = A.Compose([
                    A.VerticalFlip(p=1.0),
                    A.HorizontalFlip(p=1.0),
                ])

    def _create_splits(self):
        city_splits = {}
        for city, dataset in self.datasets.items():
            n_samples = len(dataset)
            
            if self.test_ratio > 0:
                # Create test set
                train_val_idx, test_idx = train_test_split(
                    range(n_samples), 
                    test_size=self.test_ratio, 
                    random_state=42
                )
                
                # Convert to lists
                train_val_idx = list(train_val_idx)
                test_idx = list(test_idx)
            else:
                # Use all samples for train and validation
                train_val_idx = list(range(n_samples))
                test_idx = []
            
            # Create train-val splits
            n_val = int(len(train_val_idx) * self.val_ratio / (1 - self.test_ratio))
            
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(train_val_idx))
            
            # Adjust splits to maintain val_ratio
            adjusted_splits = []
            for train_idx, val_idx in splits:
                if len(val_idx) > n_val:
                    extra = len(val_idx) - n_val
                    train_idx = np.concatenate([train_idx, val_idx[:extra]])
                    val_idx = val_idx[extra:]
                adjusted_splits.append((
                    [train_val_idx[i] for i in train_idx],
                    [train_val_idx[i] for i in val_idx],
                    test_idx  # Now this is already a list
                ))
            
            city_splits[city] = adjusted_splits
        return city_splits

    def get_split(self, split_index):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        val_city_indices = {}
        current_val_index = 0

        for city, dataset in self.datasets.items():
            train_idx, val_idx, test_idx = self.city_splits[city][split_index]
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_datasets.append(train_subset)
            val_datasets.append(val_subset)
            
            # Store the indices for this city's validation set
            val_city_indices[city] = (current_val_index, len(val_idx))
            current_val_index += len(val_idx)

            if test_idx:
                test_subset = Subset(dataset, test_idx)
                test_datasets.append(test_subset)

        return (ConcatDataset(train_datasets), 
                ConcatDataset(val_datasets), 
                ConcatDataset(test_datasets) if test_datasets else None, 
                val_city_indices)

    def get_windows_and_labels_for_city(self, city, split_index):
        raise NotImplementedError("Subclasses must implement this method")

class SingleInputCrossValidator(BaseCrossValidator):
    def get_windows_and_labels_for_city(self, city, split_index):
        if city not in self.datasets:
            raise ValueError(f"City '{city}' not found in datasets.")

        dataset = self.datasets[city]
        train_idx, val_idx, test_idx = self.city_splits[city][split_index]

        windows = dataset.windows
        labels = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(windows))]

        return windows, labels

class AugmentedSubset(Subset):
    def __init__(self, dataset, indices, augmentation):
        super().__init__(dataset, indices)
        self.augmentation = augmentation

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        
        if isinstance(item, tuple):
            # For multi-input datasets
            sentinel_data, buildings_data = item
            sentinel_image, sentinel_label = sentinel_data
            buildings_image, buildings_label = buildings_data
            
            # Augment sentinel data
            augmented = self.augmentation(image=sentinel_image.numpy(), mask=sentinel_label.numpy())
            aug_sentinel_image, aug_sentinel_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            
            # Augment buildings data
            augmented = self.augmentation(image=buildings_image.numpy(), mask=buildings_label.numpy())
            aug_buildings_image, aug_buildings_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            
            return ((aug_sentinel_image, aug_sentinel_label), (aug_buildings_image, aug_buildings_label))
        else:
            # For single-input datasets
            image, label = item
            augmented = self.augmentation(image=image.numpy(), mask=label.numpy())
            aug_image, aug_label = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])
            return (aug_image, aug_label)

class AugMultiInputCrossValidator(BaseCrossValidator):
    def __init__(self, datasets, n_splits=2, val_ratio=0.2, test_ratio=0.1, use_augmentation=False):
        super().__init__(datasets, n_splits, val_ratio, test_ratio)
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation = A.Compose([
                A.VerticalFlip(p=1.0),
                A.HorizontalFlip(p=1.0),
            ])
        else:
            self.augmentation = None

    def get_split(self, split_index):
        train_datasets = []
        val_datasets = []
        test_datasets = []
        val_city_indices = {}
        current_val_index = 0

        for city, dataset in self.datasets.items():
            train_idx, val_idx, test_idx = self.city_splits[city][split_index]
            
            if self.use_augmentation:
                train_subset = self._create_augmented_subset(dataset, train_idx)
            else:
                train_subset = Subset(dataset, train_idx)
            
            val_subset = Subset(dataset, val_idx)
            
            train_datasets.append(train_subset)
            val_datasets.append(val_subset)
            
            val_city_indices[city] = (current_val_index, len(val_idx))
            current_val_index += len(val_idx)

            if test_idx:
                test_subset = Subset(dataset, test_idx)
                test_datasets.append(test_subset)

        return (ConcatDataset(train_datasets), 
                ConcatDataset(val_datasets), 
                ConcatDataset(test_datasets) if test_datasets else None, 
                val_city_indices)

    def _create_augmented_subset(self, dataset, indices):
        original_subset = Subset(dataset, indices)
        augmented_subset = AugmentedSubset(dataset, indices, self.augmentation)
        return ConcatDataset([original_subset, augmented_subset])

    def get_windows_and_labels_for_city(self, city, split_index):
        if city not in self.datasets:
            raise ValueError(f"City '{city}' not found in datasets.")

        dataset = self.datasets[city]
        train_idx, val_idx, test_idx = self.city_splits[city][split_index]

        windows = dataset.datasets[0].windows
        labels = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(windows))]

        return windows, labels

class MultiInputCrossValidator(BaseCrossValidator):
    def get_windows_and_labels_for_city(self, city, split_index):
        if city not in self.datasets:
            raise ValueError(f"City '{city}' not found in datasets.")

        dataset = self.datasets[city]
        train_idx, val_idx, test_idx = self.city_splits[city][split_index]

        windows = dataset.datasets[0].windows
        labels = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(windows))]

        return windows, labels
        
class CustomSemanticSegmentationLabelStore(SemanticSegmentationLabelStore):
    @property
    def vector_output_uri(self) -> str:
        return self.root_uri
    
class CustomAugmentation:
    def __init__(self):
        self.augmentations = A.Compose([
            A.VerticalFlip(p=1.0),
            A.HorizontalFlip(p=1.0),
        ])

    def __call__(self, image, mask):
        augmented = self.augmentations(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
# Visulaization helper functions
def show_single_tile_multi(datasets, city, window_index, show_sentinel=True, show_buildings=True):
    if city not in datasets:
        raise ValueError(f"City '{city}' not found in datasets.")
    
    dataset = datasets[city]
    
    # Get the data for the specified window index
    data = dataset[window_index]
    
    # Assuming data is a tuple (sentinel_data, buildings_data)
    sentinel_data, sentinel_labels = data[0]
    buildings_data, _ = data[1]  # We only use sentinel_label
    
    # Set up the plot
    n_cols = 3 if show_sentinel else 1
    if show_buildings:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    plot_index = 0
    
    if show_sentinel:
        # Plot Sentinel data as RGB (R, G, B are channels 1, 2, 3)
        sentinel_rgb = sentinel_data[1:4].permute(1, 2, 0)
        sentinel_rgb = sentinel_rgb.float()  # Ensure it's float
        
        # Normalize to [0, 1] for display
        sentinel_rgb = (sentinel_rgb - sentinel_rgb.min()) / (sentinel_rgb.max() - sentinel_rgb.min())
        
        axes[plot_index].imshow(sentinel_rgb.cpu().numpy())
        axes[plot_index].set_title(f"{city} - Sentinel RGB")
        axes[plot_index].axis('off')
        plot_index += 1
        
        # Plot NIR as a separate grayscale image
        nir = sentinel_data[0]
        nir = (nir - nir.min()) / (nir.max() - nir.min())  # Normalize NIR
        axes[plot_index].imshow(nir.cpu().numpy(), cmap='gray')
        axes[plot_index].set_title(f"{city} - Sentinel NIR")
        axes[plot_index].axis('off')
        plot_index += 1
        
        # Plot Sentinel label
        buildings_data_squeezed = buildings_data.squeeze()
        axes[plot_index].imshow(buildings_data_squeezed.cpu().numpy(), cmap='gray')
        axes[plot_index].set_title(f"{city} - Building Footprints")
        axes[plot_index].axis('off')
        plot_index += 1
    
    if show_buildings:
        # Plot Buildings data
        sentinel_labels_squeezed = sentinel_labels.squeeze()
        axes[plot_index].imshow(sentinel_labels_squeezed.cpu().numpy(), cmap='gray')
        axes[plot_index].set_title(f"{city} - GT Labels")
        axes[plot_index].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_single_tile_sentinel(datasets, city, window_index):
    dataset = datasets[city]
    
    # Get the data for the specified window index
    data = dataset[window_index]
    
    # Assuming data is a tuple (sentinel_data, sentinel_label)
    sentinel_data, sentinel_label = data
    
    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Sentinel data as RGB (R, G, B are channels 1, 2, 3)
    sentinel_rgb = sentinel_data[1:4].permute(1, 2, 0)
    sentinel_rgb = sentinel_rgb.float()  # Ensure it's float
    
    # Normalize to [0, 1] for display
    sentinel_rgb = (sentinel_rgb - sentinel_rgb.min()) / (sentinel_rgb.max() - sentinel_rgb.min())
    
    axes[0].imshow(sentinel_rgb.cpu().numpy())
    axes[0].set_title(f"{city} - Sentinel Data (RGB)")
    axes[0].axis('off')
    
    # Plot NIR as a separate grayscale image
    nir = sentinel_data[0]
    nir = (nir - nir.min()) / (nir.max() - nir.min())  # Normalize NIR
    axes[1].imshow(nir.cpu().numpy(), cmap='gray')
    axes[1].set_title(f"{city} - Sentinel NIR")
    axes[1].axis('off')
    
    # Plot Sentinel label
    sentinel_label_squeezed = sentinel_label.squeeze()
    axes[2].imshow(sentinel_label_squeezed.cpu().numpy(), cmap='viridis')
    axes[2].set_title(f"{city} - Sentinel Label")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_single_tile_buildings(datasets, city, window_index):
    dataset = datasets[city]
    data = dataset[window_index]
    buildings_data, buildings_label = data
    
    # Set up the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Buildings data
    buildings_data_squeezed = buildings_data.squeeze()
    axes[0].imshow(buildings_data_squeezed, cmap='gray')
    axes[0].set_title(f"{city} - Buildings Data")
    axes[0].axis('off')
    
    # Plot Buildings label
    axes[1].imshow(buildings_label, cmap='grey')
    axes[1].set_title(f"{city} - Buildings Label")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_windows(img, windows, window_labels, title=''):
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(12, 12))
    ax.imshow(img, cmap='gray_r')
    ax.axis('off')

    for i, w in enumerate(windows):
        if window_labels[i] == 'train':
            color = 'blue'
        elif window_labels[i] == 'val':
            color = 'red'
        elif window_labels[i] == 'test':
            color = 'green'
        else:
            color = 'black'

        rect = patches.Rectangle(
            (w.xmin, w.ymin), w.width, w.height,
            edgecolor=color, linewidth=2, fill=False
        )
        ax.add_patch(rect)
        
        # Add index number as small text in the top-left corner
        ax.text(w.xmin + 2, w.ymin + 2, str(i), 
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.7, pad=0.2))

    ax.set_title(title, fontsize=18)
    plt.show()

def senitnel_create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

def buil_create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source._get_chip(extent)    
    return chip

def get_label_source_from_merge_dataset(merge_dataset):
    sentinel_dataset = merge_dataset.datasets[0]
    return sentinel_dataset.scene.label_source

def singlesource_show_windows_for_city(city, split_index, cv, datasets):
    windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
    
    # Get the full image for the city
    img_full = senitnel_create_full_image(datasets[city].scene.label_source)
    
    # Show the windows
    show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')

def show_first_batch_item(batch, device='cpu'):
    # Unpack the batch
    sentinel_batch, buildings_batch = batch
    buildings_data, _ = buildings_batch
    sentinel_data, sentinel_labels = sentinel_batch

    # Move data to CPU for visualization
    buildings_data = buildings_data.to('cpu')
    sentinel_data = sentinel_data.to('cpu')
    sentinel_labels = sentinel_labels.to('cpu')

    # Get the first item in the batch
    sentinel_image = sentinel_data[0].squeeze()
    buildings_image = buildings_data[0].squeeze()
    label = sentinel_labels[0].squeeze()

    # Set up the plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot Sentinel data as RGB (R, G, B are channels 1, 2, 3)
    sentinel_rgb = sentinel_image[1:4].permute(1, 2, 0)
    sentinel_rgb = sentinel_rgb.float()  # Ensure it's float
    
    # Normalize to [0, 1] for display
    sentinel_rgb = (sentinel_rgb - sentinel_rgb.min()) / (sentinel_rgb.max() - sentinel_rgb.min())
    
    axes[0].imshow(sentinel_rgb.numpy())
    axes[0].set_title("Sentinel RGB")
    axes[0].axis('off')
    
    # Plot NIR as a separate grayscale image
    nir = sentinel_image[0]
    nir = (nir - nir.min()) / (nir.max() - nir.min())  # Normalize NIR
    axes[1].imshow(nir.numpy(), cmap='gray')
    axes[1].set_title("Sentinel NIR")
    axes[1].axis('off')
    
    # Plot Buildings data
    axes[2].imshow(buildings_image.numpy(), cmap='gray')
    axes[2].set_title("Building Footprints")
    axes[2].axis('off')
    
    # Plot Label
    axes[3].imshow(label.numpy(), cmap='gray')
    axes[3].set_title("GT Labels")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()



# Functions to get data from OMF
def query_buildings_data(xmin, ymin, xmax, ymax):
    import duckdb
    con = duckdb.connect(os.path.join(grandparent_dir, 'data/0/data.db'))
    con.install_extension('httpfs')
    con.install_extension('spatial')
    con.load_extension('httpfs')
    con.load_extension('spatial')
    con.execute("SET s3_region='us-west-2'")
    con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")
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
    else:
        print("No buildings found in the specified area.")
        return None
    return buildings

def query_roads_data(xmin, ymin, xmax, ymax):
    import duckdb

    con = duckdb.connect(os.path.join(grandparent_dir, 'data/0/data.db'))
    con.install_extension('httpfs')
    con.install_extension('spatial')
    con.load_extension('httpfs')
    con.load_extension('spatial')
    con.execute("SET s3_region='us-west-2'")
    con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

    query = f"""
        SELECT *
        FROM roads
        WHERE subtype = 'road'
          AND bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax};
    """
    roads_df = pd.read_sql(query, con=con)

    if not roads_df.empty:
        roads = gpd.GeoDataFrame(roads_df, geometry=gpd.GeoSeries.from_wkb(roads_df.geometry.apply(bytes)), crs='EPSG:4326')
        roads = roads[['id', 'geometry']]
        roads = roads.to_crs("EPSG:3857")
        roads['class_id'] = 2

    return roads

def query_poi_data(xmin, ymin, xmax, ymax):
    # Initialize DuckDB connection
    con = duckdb.connect(os.path.join(grandparent_dir, 'data/0/data.db'))
    con.install_extension('spatial')
    con.install_extension('httpfs')
    con.load_extension('spatial')
    con.load_extension('httpfs')

    # Query for POIs
    poi_query = f"""
    SELECT
        id,
        names.primary AS name,
        categories.main AS category,
        ROUND(confidence, 2) AS confidence,
        ST_GeomFromWKB(geometry) AS geometry
    FROM pois
    WHERE
        bbox.xmin > {xmin} AND bbox.xmax < {xmax}
        AND bbox.ymin > {ymin} AND bbox.ymax < {ymax}
    """
    poi_df = con.execute(poi_query).df()
    if not poi_df.empty:
        poi_gdf = gpd.GeoDataFrame(poi_df, geometry=gpd.GeoSeries.from_wkb(poi_df.geometry.apply(bytes)), crs='EPSG:4326')
        poi_gdf = poi_gdf[['id', 'geometry','name','category','confidence']]
        poi_gdf = poi_gdf.to_crs("EPSG:3857")
        poi_gdf['class_id'] = 3

    print(f"POI data loaded successfully with {len(poi_gdf)} total POIs.")

    return poi_gdf


### Functions to create raster sources ###
def make_buildings_raster(image_path, labels_path, resolution=5):
    gdf = gpd.read_file(labels_path)
    gdf = gdf.to_crs('EPSG:3857')
    xmin3857, ymin3857, xmax3857, ymax3857 = gdf.total_bounds
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(resolution, 0, xmin3857, 0, -resolution, ymax3857)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    gdf = gdf.to_crs('EPSG:4326')
    xmin4326, ymin4326, xmax4326, ymax4326 = gdf.total_bounds
    buildings = query_buildings_data(xmin4326, ymin4326, xmax4326, ymax4326)
    print(f"Buildings data loaded successfully with {len(buildings)} total buildings.")
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    label_vector_source = GeoJSONVectorSource(labels_path,
        crs_transformer_buildings,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])
    
    sentinel_label_raster_source = RasterizedSource(label_vector_source, background_class_id=class_config.null_class_id)
    label_source = SemanticSegmentationLabelSource(sentinel_label_raster_source, class_config=class_config)
    
    rasterized_buildings_source = CustomRasterizedSource(
        buildings_vector_source,
        background_class_id=0,
        bbox=label_source.bbox)
    
    return rasterized_buildings_source, crs_transformer_buildings

def make_sentinel_raster(image_uri, label_uri, class_config, clip_to_label_source=False):
    
    crs_transformer = RasterioCRSTransformer.from_uri(image_uri)

    label_vector_source = GeoJSONVectorSource(label_uri,
        crs_transformer,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])

    sentinel_label_raster_source = RasterizedSource(label_vector_source, background_class_id=class_config.null_class_id)
    label_source = SemanticSegmentationLabelSource(sentinel_label_raster_source, class_config=class_config)
    print(f"Loaded SemanticSegmentationLabelSource: {sentinel_label_raster_source.shape}")

    # Define an unnormalized raster source
    # sentinel_source_unnormalized = RasterioSource(
    #     image_uri,
    #     allow_streaming=True)

    # # Calculate statistics transformer from the unnormalized source
    # calc_stats_transformer = CustomStatsTransformer.from_raster_sources(
    #     raster_sources=[sentinel_source_unnormalized],
    #     max_stds=3
    # )
    
    # Define the means and stds in NIR-RGB order
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    # Define a normalized raster source using the calculated transformer
    sentinel_source_normalized = RasterioSource(
        image_uri,
        allow_streaming=True,
        raster_transformers=[fixed_stats_transformer],
        channel_order=[3, 2, 1, 0],
        bbox=label_source.bbox if clip_to_label_source else None
    )

    print(f"Loaded Sentinel data of size {sentinel_source_normalized.shape}, and dtype: {sentinel_source_normalized.dtype}")
    return sentinel_source_normalized, label_source


### Create scenes functions ###
def create_sentinel_scene(city_data, class_config):
    image_path = city_data['image_path']
    labels_path = city_data['labels_path']

    sentinel_source_normalized, sentinel_label_raster_source = make_sentinel_raster(
        image_path, labels_path, class_config, clip_to_label_source=True
    )
    
    sentinel_scene = Scene(
        id='scene_sentinel',
        raster_source=sentinel_source_normalized,
        label_source=sentinel_label_raster_source
    )
    return sentinel_scene

def create_building_scene(city_name, city_data):
    image_path = city_data['image_path']
    labels_path = city_data['labels_path']

    # Create Buildings scene
    rasterized_buildings_source, crs_transformer_buildings = make_buildings_raster(image_path, labels_path, resolution=5)
    
    label_vector_source = GeoJSONVectorSource(labels_path,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=class_config.get_class_id('slums'))])
    
    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    buildings_label_sourceSD = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
    
    buildings_scene = Scene(
        id=f'{city_name}_buildings',
        raster_source=rasterized_buildings_source,
        label_source = buildings_label_sourceSD)

    return buildings_scene
    
def create_scenes_for_city(city_name, city_data, class_config, resolution=5):
    image_path = city_data['image_path']
    labels_path = city_data['labels_path']

    # Create Sentinel scene
    sentinel_source_normalized, sentinel_label_raster_source = make_sentinel_raster(
        image_path, labels_path, class_config, clip_to_label_source=True
    )
    
    sentinel_scene = Scene(
        id=f'{city_name}_sentinel',
        raster_source=sentinel_source_normalized,
        label_source=sentinel_label_raster_source
    )

    # Create Buildings scene
    rasterized_buildings_source, crs_transformer_buildings = make_buildings_raster(image_path, labels_path, resolution=resolution)
    
    label_vector_source = GeoJSONVectorSource(labels_path,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=class_config.get_class_id('slums'))])
    
    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    buildings_label_sourceSD = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
    
    buildings_scene = Scene(
        id=f'{city_name}_buildings',
        raster_source=rasterized_buildings_source,
        label_source = buildings_label_sourceSD)

    return sentinel_scene, buildings_scene


### Others ###
def make_buildings_and_roads_raster(image_path, labels_path, resolution=5, road_buffer=2):
    gdf = gpd.read_file(labels_path)
    gdf = gdf.to_crs('EPSG:4326')
    xmin, ymin, xmax, ymax = gdf.total_bounds

    crs_transformer = RasterioCRSTransformer.from_uri(image_path)
    affine_transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    crs_transformer.transform = affine_transform

    # Query buildings
    buildings = query_buildings_data(xmin, ymin, xmax, ymax)
    buildings['class_id'] = 1  # Assign class_id 1 to buildings
    print(f"Buildings data loaded successfully with {len(buildings)} total buildings.")

    # Query roads
    roads = query_roads_data(xmin, ymin, xmax, ymax)
    roads['class_id'] = 2  # Assign class_id 2 to roads
    print(f"Roads data loaded successfully with {len(roads)} total road segments.")

    # Buffer the roads
    roads_buffered = roads.copy()
    roads_buffered['geometry'] = roads_buffered.geometry.buffer(road_buffer)

    combined_gdf = pd.concat([buildings, roads_buffered], ignore_index=True)

    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf=combined_gdf,
        crs_transformer=crs_transformer,
        vector_transformers=[]
    )

    rasterized_source = CustomRasterizedSource(
        buildings_vector_source,
        background_class_id=0
    )

    return rasterized_source, crs_transformer


# belize_data = cities['SantoDomingoDOM']
# image_path = belize_data['image_path']
# labels_path = belize_data['labels_path']

# gdf = gpd.read_file(labels_path)
# gdf = gdf.to_crs('EPSG:4326')
# xmin, ymin, xmax, ymax = gdf.total_bounds
# roads = query_roads_data(xmin, ymin, xmax, ymax)
# # buildings = query_buildings_data(xmin, ymin, xmax, ymax)
# pois = query_poi_data(xmin, ymin, xmax, ymax)
# from lonboard import viz
# viz(roads)

# class_config = ClassConfig(names=['background', 'slums'], 
#                                 colors=['lightgray', 'darkred'],
#                                 null_class='background')

# sentinel_source_normalized, sentinel_label_raster_source = make_sentinel_raster(
#     image_path, labels_path, class_config, clip_to_label_source=True
# )

# ovfmratser, _ = make_buildings_and_roads_raster(image_path, labels_path, resolution=5, road_buffer=2)

# chip = ovfmratser[:, :]
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.imshow(chip)
# plt.show()

# urbmscene = Scene(
#         id='_sentinel',
#         raster_source=ovfmratser,
#         label_source=sentinel_label_raster_source    )
# buildGeoDataset_PN = PolygonWindowGeoDataset(urbmscene,window_size=512,out_size=512,padding=200,transform_type=TransformType.noop,transform=None)

# # x, y = vis_sent.get_batch(buildingsGeoDataset_TG, 5)
# # vis_sent.plot_batch(x, y, show=True)

# x, y = vis_build.get_batch(buildingsGeoDataset_TG, 5)
# vis_build.plot_batch(x, y, show=True)


# From STAC
# BANDS = [
#     'blue', # B02
#     'green', # B03
#     'red', # B04
#     'nir', # B08
# ]
# URL = 'https://earth-search.aws.element84.com/v1'
# # catalog = pystac_client.Client.open(URL)

# bbox = Box(ymin=ymin4326, xmin=xmin4326, ymax=ymax4326, xmax=xmax4326)
# bbox_geometry = {
#         'type': 'Polygon',
#         'coordinates': [
#             [
#                 (xmin4326, ymin4326),
#                 (xmin4326, ymax4326),
#                 (xmax4326, ymax4326),
#                 (xmax4326, ymin4326),
#                 (xmin4326, ymin4326)
#             ]
#         ]
#     }

def save_sentinel_mosaic_as_geotiff(sentinel_source, bbox, crs):
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'sentinel_mosaic.tif')
    
    # Get the data and metadata
    data = sentinel_source.data_array.values
    height, width, num_bands = data.shape
    
    # Create the transform
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
    
    # Save as GeoTIFF
    with rasterio.open(temp_file, 'w', driver='GTiff', height=height, width=width, 
                       count=num_bands, dtype=data.dtype, crs=crs, transform=transform) as dst:
        for i in range(num_bands):
            dst.write(data[:,:,i], i+1)
    
    return temp_file

def get_sentinel_items(bbox_geometry, bbox):
    items = catalog.search(
        intersects=bbox_geometry,
        collections=['sentinel-2-c1-l2a'],
        datetime='2023-01-01/2024-06-27',
        query={'eo:cloud_cover': {'lt': 10}},
        # Remove max_items=1 to get multiple items
    ).item_collection()
    
    if not items:
        print("No items found for this area.")
        return None
    # items = get_sentinel_items(bbox_geometry, bbox)

    return items

def create_sentinel_mosaic(items, bbox):
    # Create the initial XarraySource with temporal=True
    sentinel_source = XarraySource.from_stac(
        items,
        bbox_map_coords=tuple(bbox),
        stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
        allow_streaming=True,
        temporal=True
    )
    
    print(f"Initial data shape: {sentinel_source.data_array.shape}")
    print(f"Initial bbox: {sentinel_source.bbox}")
    print(f"Data coordinates: {sentinel_source.data_array.coords}")
    
    crs_transformer = sentinel_source.crs_transformer
    
    # Get the CRS of the data
    data_crs = sentinel_source.crs_transformer.image_crs
    
    # Use the original bbox of the data
    data_bbox = sentinel_source.bbox
    
    print(f"Data bbox: {data_bbox}")
    
    # Apply mosaic function to combine the temporal dimension
    mosaic_data = sentinel_source.data_array.median(dim='time')
    
    print(f"Mosaic data shape: {mosaic_data.shape}")
    print(f"Mosaic data coordinates: {mosaic_data.coords}")
    
    # Create a temporary XarraySource from the mosaicked data
    temp_mosaic_source = XarraySource(
        mosaic_data,
        crs_transformer=sentinel_source.crs_transformer,
        bbox=data_bbox,
        channel_order=[2, 1, 0, 3],  # Assuming you want RGB-NIR order
        temporal=False
    )
    
    # Calculate stats transformer
    calc_stats_transformer = CustomStatsTransformer.from_raster_sources(
        raster_sources=[temp_mosaic_source],
        max_stds=3
    )
    
    # Create the final normalized XarraySource
    normalized_mosaic_source = XarraySource(
        mosaic_data,
        crs_transformer=sentinel_source.crs_transformer,
        bbox=data_bbox,
        channel_order=[2, 1, 0, 3],  # Assuming you want RGB-NIR order
        temporal=False,
        raster_transformers=[calc_stats_transformer]
    )
    
    print(f"Created normalized mosaic of size {normalized_mosaic_source.shape}, and dtype: {normalized_mosaic_source.dtype}")
    print(f"Normalized mosaic bbox: {normalized_mosaic_source.bbox}")
    print(f"Data CRS: {data_crs}")
    # sentinel_source_SD, crs_transformer = create_sentinel_mosaic(items, bbox)

    return normalized_mosaic_source, crs_transformer

def display_mosaic(mosaic_source):
    print(f"Mosaic source shape: {mosaic_source.shape}")
    
    if mosaic_source.shape[0] == 0 or mosaic_source.shape[1] == 0:
        print("The mosaic has zero width or height. There might be no data for the specified region.")
        return
    
    chip = mosaic_source[:, :, [0, 1, 2]]  # RGB channels
    
    print(f"Chip shape: {chip.shape}")
    print(f"Chip min: {np.min(chip)}, max: {np.max(chip)}")
    
    # For normalized data, we might want to clip to a reasonable range
    vmin, vmax = -3, 3
    normalized_chip = np.clip(chip, vmin, vmax)
    
    # Scale to [0, 1] for display
    normalized_chip = (normalized_chip - vmin) / (vmax - vmin)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(normalized_chip, interpolation='nearest', aspect='equal')
    ax.set_title("Normalized Mosaic RGB")
    plt.colorbar(im, ax=ax)
    plt.show()

# if __name__ == "__main__":
    label_uri = "../../data/0/SantoDomingo3857.geojson"
    image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
    class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

    ### SENTINEL source ###
    sentinel_source_normalized, sentinel_source_label = create_sentinel_raster_source(image_uri, label_uri, class_config)
    
    # show label source
    chip = sentinel_source_label[:, :]
    chip.shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip)
    plt.show()

    # show raster source
    chip = sentinel_source_normalized[:, :, :3]
    chip_scaled = (chip - np.min(chip)) / (np.max(chip) - np.min(chip))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip_scaled)
    plt.show()

    print(f"Minimum pixel value after normalization: {chip.min()}")
    print(f"Maximum pixel value after normalization: {chip.max()}")

    # Density plot of sentinel image
    num_bands = chip.shape[-1]
    plt.figure(figsize=(8, 6))
    for band in range(num_bands):
        band_data = chip[..., band].flatten()
        sns.kdeplot(band_data, shade=True, label=f'Band {band}', color=plt.cm.viridis(band / num_bands))
    plt.legend()
    plt.grid(True)
    plt.show()

    ### Scenes ###
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
        label_source = label_raster_source,
        aoi_polygons=[pixel_polygon])

    BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = label_source,
        aoi_polygons=[pixel_polygon])
    
    pass