import os
import torch
from affine import Affine
import numpy as np
import geopandas as gpd
from rastervision.core.box import Box
from typing import Any, Optional, Tuple, Union, Sequence, List

from torch.utils.data import Dataset
from typing import List
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import random
from rastervision.pytorch_learner.dataset import GeoDataset
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union
import logging
import duckdb
import geopandas as gpd
import albumentations as A
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from rastervision.core.box import Box
from rastervision.core.data import Scene
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.transform import TransformType
from typing import TYPE_CHECKING, Optional, Sequence, Union

from typing import (TYPE_CHECKING, Sequence, Optional, List, Dict, Union,
                    Tuple, Any)

from rastervision.core.data import ClassConfig

log = logging.getLogger(__name__)
import albumentations as A
from typing import Literal, TypeVar
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union
from typing import List
T = TypeVar('T')
import pandas as pd
from rastervision.core.data.utils import listify_uris, merge_geojsons
from sklearn.model_selection import KFold

from pydantic.types import NonNegativeInt as NonNegInt, PositiveInt as PosInt

T = TypeVar('T')

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSource, RasterizedSource, Scene, ClassInferenceTransformer,
                                    VectorSource, CRSTransformer, RasterioCRSTransformer, RasterioSource,
                                    SemanticSegmentationLabelSource, RasterizerConfig)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.utils import pad_to_window_size

from rastervision.pytorch_learner.dataset import TransformType
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, Subset
from typing import Any, Optional, Tuple, Union, Sequence
from typing import List
    
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import VectorSource, RasterioCRSTransformer, RasterioCRSTransformer
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer, CRSTransformer

log = logging.getLogger(__name__)
from rastervision.core.data.label_store import SemanticSegmentationLabelStore
import random
from typing import Tuple, List, Union, Optional
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pytorch_learner.dataset.dataset import GeoDataset
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

class CustomSemanticSegmentationLabelStore(SemanticSegmentationLabelStore):
    @property
    def vector_output_uri(self) -> str:
        return self.root_uri
        
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


# Classes implementing two tiling strategies
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


# Classes implementing cross validation for within-city tiling
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
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(train_val_idx))
            
            adjusted_splits = []
            for train_idx, val_idx in splits:
                adjusted_splits.append((
                    [train_val_idx[i] for i in train_idx],
                    [train_val_idx[i] for i in val_idx],
                    test_idx
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

class MultiInputCrossValidator(BaseCrossValidator):
    def get_windows_and_labels_for_city(self, city, split_index):
        if city not in self.datasets:
            raise ValueError(f"City '{city}' not found in datasets.")

        dataset = self.datasets[city]
        train_idx, val_idx, test_idx = self.city_splits[city][split_index]

        windows = dataset.datasets[0].windows
        labels = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(windows))]

        return windows, labels
        
class SingleInputCrossValidator(BaseCrossValidator):
    def get_windows_and_labels_for_city(self, city, split_index):
        if city not in self.datasets:
            raise ValueError(f"City '{city}' not found in datasets.")

        dataset = self.datasets[city]
        train_idx, val_idx, test_idx = self.city_splits[city][split_index]

        windows = dataset.windows
        labels = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(windows))]

        return windows, labels


# Visulaization helper functions
def show_single_tile_multi(datasets, city, window_index, show_sentinel=True, show_buildings=True, save_path=None):
    if city not in datasets:
        raise ValueError(f"City '{city}' not found in datasets.")
    
    dataset = datasets[city]
    data = dataset[window_index]
    sentinel_data, sentinel_labels = data[0]
    buildings_data, _ = data[1]
    
    fig = plt.figure(figsize=(20, 10))
    grid = plt.GridSpec(2, 4, figure=fig)
    
    if show_buildings:
        # Building Footprints (larger size)
        ax1 = fig.add_subplot(grid[:, :2])
        buildings_data_squeezed = buildings_data.squeeze()
        ax1.imshow(buildings_data_squeezed.cpu().numpy(), cmap='gray')
        ax1.set_title(f"{city} - Building Footprints", fontsize=16)
        ax1.axis('off')
    
    if show_sentinel:
        # Sentinel RGB
        ax2 = fig.add_subplot(grid[0, 2])
        sentinel_rgb = sentinel_data[1:4].permute(1, 2, 0).float()
        sentinel_rgb = (sentinel_rgb - sentinel_rgb.min()) / (sentinel_rgb.max() - sentinel_rgb.min())
        ax2.imshow(sentinel_rgb.cpu().numpy())
        ax2.set_title(f"{city} - Sentinel RGB", fontsize=16)
        ax2.axis('off')
        
        # Sentinel NIR
        ax3 = fig.add_subplot(grid[0, 3])
        nir = sentinel_data[0]
        nir = (nir - nir.min()) / (nir.max() - nir.min())
        ax3.imshow(nir.cpu().numpy(), cmap='gray')
        ax3.set_title(f"{city} - Sentinel NIR", fontsize=16)
        ax3.axis('off')
        
        # GT Labels
        ax4 = fig.add_subplot(grid[1, 2:])
        sentinel_labels_squeezed = sentinel_labels.squeeze()
        ax4.imshow(sentinel_labels_squeezed.cpu().numpy(), cmap='gray')
        ax4.set_title(f"{city} - Ground Truth Labels", fontsize=16)
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

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

def get_label_source_from_merge_dataset(merge_dataset):
    sentinel_dataset = merge_dataset.datasets[0]
    return sentinel_dataset.scene.label_source

def singlesource_show_windows_for_city(city, split_index, cv, datasets):
    windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
    
    # Get the full image for the city
    img_full = senitnel_create_full_image(datasets[city].scene.label_source)
    
    # Show the windows
    show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')

def show_first_batch_item(batch):
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
