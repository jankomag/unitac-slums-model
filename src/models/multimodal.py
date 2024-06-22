import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
multiprocessing.set_start_method('fork')

import pytorch_lightning as pl
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from rastervision.core.box import Box

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

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from src.data.dataloaders import create_sentinel_raster_source, create_buildings_raster_source, create_datasets
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore
from rastervision.core.data import Scene, ClassConfig
from rastervision.core.data.utils import all_equal, match_bboxes, geoms_to_bbox_coords
# from rastervision.core.raster_stats import RasterStats
# from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
# from rastervision.pipeline.utils import repr_with_args
# from rastervision.pytorch_learner.dataset.transform import (TransformType)
# from rastervision.pipeline.config import (Config,Field)
# from rastervision.pytorch_learner.dataset import SlidingWindowGeoDataset, TransformType

from typing import (TYPE_CHECKING, Callable, Dict, List, Literal, Optional,
                    Sequence, Tuple, Union)
from pydantic import PositiveInt as PosInt, conint
import math
import random

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from rasterio.windows import Window as RioWindow

from rastervision.pipeline.utils import repr_with_args

NonNegInt = conint(ge=0)

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon


class BoxSizeError(ValueError):
    pass


class CustomBox():
    """A multi-purpose box (ie. rectangle) representation."""

    def __init__(self, ymin, xmin, ymax, xmax):
        """Construct a bounding box.

        Unless otherwise stated, the convention is that these coordinates are
        in pixel coordinates and represent boxes that lie within a
        RasterSource.

        Args:
            ymin: minimum y value (y is row)
            xmin: minimum x value (x is column)
            ymax: maximum y value
            xmax: maximum x value
        """
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def __eq__(self, other: 'Box') -> bool:
        """Return true if other has same coordinates."""
        return self.tuple_format() == other.tuple_format()

    def __ne__(self, other: 'Box'):
        """Return true if other has different coordinates."""
        return self.tuple_format() != other.tuple_format()

    @property
    def height(self) -> int:
        """Return height of Box."""
        return self.ymax - self.ymin

    @property
    def width(self) -> int:
        """Return width of Box."""
        return self.xmax - self.xmin

    @property
    def extent(self) -> 'Box':
        """Return a (0, 0, h, w) Box representing the size of this Box."""
        return Box(0, 0, self.height, self.width)

    @property
    def size(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def area(self) -> int:
        """Return area of Box."""
        return self.height * self.width

    def normalize(self) -> 'Box':
        """Ensure ymin <= ymax and xmin <= xmax."""
        ymin, ymax = sorted((self.ymin, self.ymax))
        xmin, xmax = sorted((self.xmin, self.xmax))
        return Box(ymin, xmin, ymax, xmax)

    def rasterio_format(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return Box in Rasterio format: ((ymin, ymax), (xmin, xmax))."""
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def tuple_format(self) -> Tuple[int, int, int, int]:
        return (self.ymin, self.xmin, self.ymax, self.xmax)

    def shapely_format(self) -> Tuple[int, int, int, int]:
        return self.to_xyxy()

    def to_int(self):
        return Box(
            int(self.ymin), int(self.xmin), int(self.ymax), int(self.xmax))

    def npbox_format(self):
        """Return Box in npbox format used by TF Object Detection API.

        Returns:
            Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return np.array(
            [self.ymin, self.xmin, self.ymax, self.xmax], dtype=float)

    @staticmethod
    def to_npboxes(boxes):
        """Return nx4 numpy array from list of Box."""
        nb_boxes = len(boxes)
        npboxes = np.empty((nb_boxes, 4))
        for boxind, box in enumerate(boxes):
            npboxes[boxind, :] = box.npbox_format()
        return npboxes

    def __iter__(self):
        return iter(self.tuple_format())

    def __getitem__(self, i):
        return self.tuple_format()[i]

    def __repr__(self) -> str:
        return repr_with_args(self, **self.to_dict())

    def __hash__(self) -> int:
        return hash(self.tuple_format())

    def geojson_coordinates(self) -> List[Tuple[int, int]]:
        """Return Box as GeoJSON coordinates."""
        # Compass directions:
        nw = [self.xmin, self.ymin]
        ne = [self.xmin, self.ymax]
        se = [self.xmax, self.ymax]
        sw = [self.xmax, self.ymin]
        return [nw, ne, se, sw, nw]

    def make_random_square_container(self, size: int) -> 'Box':
        """Return a new square Box that contains this Box.

        Args:
            size: the width and height of the new Box
        """
        return self.make_random_box_container(size, size)

    def make_random_box_container(self, out_h: int, out_w: int) -> 'Box':
        """Return a new rectangular Box that contains this Box.

        Args:
            out_h (int): the height of the new Box
            out_w (int): the width of the new Box
        """
        self_h, self_w = self.size

        if out_h < self_h:
            raise BoxSizeError('size of random container cannot be < height')
        if out_w < self_w:
            raise BoxSizeError('size of random container cannot be < width')

        ymin, xmin, _, _ = self.normalize()

        lb = ymin - (out_h - self_h)
        ub = ymin
        out_ymin = random.randint(int(lb), int(ub))

        lb = xmin - (out_w - self_w)
        ub = xmin
        out_xmin = random.randint(int(lb), int(ub))

        return Box(out_ymin, out_xmin, out_ymin + out_h, out_xmin + out_w)

    def make_random_square(self, size: int) -> 'Box':
        """Return new randomly positioned square Box that lies inside this Box.

        Args:
            size: the height and width of the new Box
        """
        if size >= self.width:
            raise BoxSizeError('size of random square cannot be >= width')

        if size >= self.height:
            raise BoxSizeError('size of random square cannot be >= height')

        ymin, xmin, ymax, xmax = self.normalize()

        lb = ymin
        ub = ymax - size
        rand_y = random.randint(int(lb), int(ub))

        lb = xmin
        ub = xmax - size
        rand_x = random.randint(int(lb), int(ub))

        return Box.make_square(rand_y, rand_x, size)

    def intersection(self, other: 'Box') -> 'Box':
        """Return the intersection of this Box and the other.

        Args:
            other: The box to intersect with this one.

        Returns:
             The intersection of this box and the other one.
        """
        if not self.intersects(other):
            return Box(0, 0, 0, 0)

        box1 = self.normalize()
        box2 = other.normalize()

        xmin = max(box1.xmin, box2.xmin)
        ymin = max(box1.ymin, box2.ymin)
        xmax = min(box1.xmax, box2.xmax)
        ymax = min(box1.ymax, box2.ymax)
        return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def intersects(self, other: 'Box') -> bool:
        box1 = self.normalize()
        box2 = other.normalize()
        if box1.ymax <= box2.ymin or box1.ymin >= box2.ymax:
            return False
        if box1.xmax <= box2.xmin or box1.xmin >= box2.xmax:
            return False
        return True

    @staticmethod
    def from_npbox(npbox):
        """Return new Box based on npbox format.

        Args:
            npbox: Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return Box(*npbox)

    @staticmethod
    def from_shapely(shape):
        """Instantiate from the bounds of a shapely geometry."""
        xmin, ymin, xmax, ymax = shape.bounds
        return Box(ymin, xmin, ymax, xmax)

    @classmethod
    def from_rasterio(self, rio_window: RioWindow) -> 'Box':
        """Instantiate from a rasterio window."""
        yslice, xslice = rio_window.toslices()
        return Box(yslice.start, xslice.start, yslice.stop, xslice.stop)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Convert to (xmin, ymin, width, height) tuple"""
        return (self.xmin, self.ymin, self.width, self.height)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to (xmin, ymin, xmax, ymax) tuple"""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_points(self) -> np.ndarray:
        """Get (x, y) coords of each vertex as a 4x2 numpy array."""
        return np.array(self.geojson_coordinates()[:4])

    def to_shapely(self) -> Polygon:
        """Convert to shapely Polygon."""
        return Polygon.from_bounds(*self.shapely_format())

    def to_rasterio(self) -> RioWindow:
        """Convert to a Rasterio Window."""
        return RioWindow.from_slices(*self.normalize().to_slices())

    def to_slices(self,
                  h_step: Optional[int] = None,
                  w_step: Optional[int] = None) -> Tuple[slice, slice]:
        """Convert to slices: ymin:ymax[:h_step], xmin:xmax[:w_step]"""
        return slice(self.ymin, self.ymax, h_step), slice(
            self.xmin, self.xmax, w_step)

    def translate(self, dy: int, dx: int) -> 'Box':
        """Translate window along y and x axes by the given distances."""
        ymin, xmin, ymax, xmax = self
        return Box(ymin + dy, xmin + dx, ymax + dy, xmax + dx)

    def to_global_coords(self, bbox: 'Box') -> 'Box':
        """Go from bbox coords to global coords.

        E.g., Given a box Box(20, 20, 40, 40) and bbox Box(20, 20, 100, 100),
        the box becomes Box(40, 40, 60, 60).

        Inverse of Box.to_local_coords().
        """
        return self.translate(dy=bbox.ymin, dx=bbox.xmin)

    def to_local_coords(self, bbox: 'Box') -> 'Box':
        """Go from to global coords bbox coords.

        E.g., Given a box Box(40, 40, 60, 60) and bbox Box(20, 20, 100, 100),
        the box becomes Box(20, 20, 40, 40).

        Inverse of Box.to_global_coords().
        """
        return self.translate(dy=-bbox.ymin, dx=-bbox.xmin)

    def reproject(self, transform_fn: Callable) -> 'Box':
        """Reprojects this box based on a transform function.

        Args:
            transform_fn: A function that takes in a tuple (x, y) and
                reprojects that point to the target coordinate reference
                system.
        """
        (xmin, ymin) = transform_fn((self.xmin, self.ymin))
        (xmax, ymax) = transform_fn((self.xmax, self.ymax))

        return Box(ymin, xmin, ymax, xmax)

    @staticmethod
    def make_square(ymin, xmin, size) -> 'Box':
        """Return new square Box."""
        return Box(ymin, xmin, ymin + size, xmin + size)

    def center_crop(self, edge_offset_y: int, edge_offset_x: int) -> 'Box':
        """Return Box whose sides are eroded by the given offsets.

        Box(0, 0, 10, 10).center_crop(2, 4) ==  Box(2, 4, 8, 6)
        """
        return Box(self.ymin + edge_offset_y, self.xmin + edge_offset_x,
                   self.ymax - edge_offset_y, self.xmax - edge_offset_x)

    def erode(self, erosion_sz) -> 'Box':
        """Return new Box whose sides are eroded by erosion_sz."""
        return self.center_crop(erosion_sz, erosion_sz)

    def buffer(self, buffer_sz: float, max_extent: 'Box') -> 'Box':
        """Return new Box whose sides are buffered by buffer_sz.

        The resulting box is clipped so that the values of the corners are
        always greater than zero and less than the height and width of
        max_extent.
        """
        buffer_sz = max(0., buffer_sz)
        if buffer_sz < 1.:
            delta_width = int(round(buffer_sz * self.width))
            delta_height = int(round(buffer_sz * self.height))
        else:
            delta_height = delta_width = int(round(buffer_sz))

        return Box(
            max(0, math.floor(self.ymin - delta_height)),
            max(0, math.floor(self.xmin - delta_width)),
            min(max_extent.height,
                int(self.ymax) + delta_height),
            min(max_extent.width,
                int(self.xmax) + delta_width))

    def pad(self, ymin: int, xmin: int, ymax: int, xmax: int) -> 'Box':
        """Pad sides by the given amount."""
        return Box(
            ymin=self.ymin - ymin,
            xmin=self.xmin - xmin,
            ymax=self.ymax + ymax,
            xmax=self.xmax + xmax)

    def copy(self) -> 'Box':
        return Box(*self)

    def get_windows(self,
                    size: Union[PosInt, Tuple[PosInt, PosInt]],
                    stride: Union[PosInt, Tuple[PosInt, PosInt]],
                    padding: Optional[Union[NonNegInt, Tuple[
                        NonNegInt, NonNegInt]]] = None,
                    pad_direction: Literal['both', 'start', 'end'] = 'end'
                    ) -> List['Box']:
        """Returns a list of boxes representing windows generated using a
        sliding window traversal with the specified size, stride, and
        padding.

        Each of size, stride, and padding can be either a positive int or
        a tuple `(vertical-component, horizontal-component)` of positive ints.

        Padding currently only applies to the right and bottom edges.

        Args:
            size (Union[PosInt, Tuple[PosInt, PosInt]]): Size (h, w) of the
                windows.
            stride (Union[PosInt, Tuple[PosInt, PosInt]]): Step size between
                windows. Can be 2-tuple (h_step, w_step) or positive int.
            padding (Optional[Union[PosInt, Tuple[PosInt, PosInt]]], optional):
                Optional padding to accommodate windows that overflow the
                extent. Can be 2-tuple (h_pad, w_pad) or non-negative int.
                If None, will be set to (size[0]//2, size[1]//2).
                Defaults to None.
            pad_direction (Literal['both', 'start', 'end']): If 'end', only pad
                ymax and xmax (bottom and right). If 'start', only pad ymin and
                xmin (top and left). If 'both', pad all sides. Has no effect if
                padding is zero. Defaults to 'end'.

        Returns:
            List[Box]: List of Box objects.
        """
        if not isinstance(size, tuple):
            size = (size, size)

        if not isinstance(stride, tuple):
            stride = (stride, stride)

        if size[0] <= 0 or size[1] <= 0 or stride[0] <= 0 or stride[1] <= 0:
            raise ValueError('size and stride must be positive.')

        if padding is None:
            padding = (size[0] // 2, size[1] // 2)

        if not isinstance(padding, tuple):
            padding = (padding, padding)

        if padding[0] < 0 or padding[1] < 0:
            raise ValueError('padding must be non-negative.')

        if padding != (0, 0):
            h_pad, w_pad = padding
            if pad_direction == 'both':
                padded_box = self.pad(
                    ymin=h_pad, xmin=w_pad, ymax=h_pad, xmax=w_pad)
            elif pad_direction == 'end':
                padded_box = self.pad(ymin=0, xmin=0, ymax=h_pad, xmax=w_pad)
            elif pad_direction == 'start':
                padded_box = self.pad(ymin=h_pad, xmin=w_pad, ymax=0, xmax=0)
            else:
                raise ValueError('pad_directions must be one of: '
                                 '"both", "start", "end".')
            return padded_box.get_windows(
                size=size, stride=stride, padding=(0, 0))

        # padding is necessarily (0, 0) at this point, so we ignore it
        h, w = size
        h_step, w_step = stride
        # lb = lower bound, ub = upper bound
        ymin_lb = self.ymin
        xmin_lb = self.xmin
        ymin_ub = self.ymax - h
        xmin_ub = self.xmax - w

        windows = []
        for ymin in range(ymin_lb, ymin_ub + 1, h_step):
            for xmin in range(xmin_lb, xmin_ub + 1, w_step):
                windows.append(Box(ymin, xmin, ymin + h, xmin + w))
        return windows

    def to_dict(self) -> Dict[str, int]:
        """Convert to a dict with keys: ymin, xmin, ymax, xmax."""
        return {
            'ymin': self.ymin,
            'xmin': self.xmin,
            'ymax': self.ymax,
            'xmax': self.xmax,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Box':
        return cls(d['ymin'], d['xmin'], d['ymax'], d['xmax'])

    @staticmethod
    def filter_by_aoi(windows: List['Box'],
                      aoi_polygons: List[Polygon],
                      within: bool = True) -> List['Box']:
        """Filters windows by a list of AOI polygons

        Args:
            within: if True, windows are only kept if they lie fully within an
                AOI polygon. Otherwise, windows are kept if they intersect an
                AOI polygon.
        """
        # merge overlapping polygons, if any
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)

        if within:
            keep_window = aoi_polygons.contains
        else:
            keep_window = aoi_polygons.intersects

        out = [w for w in windows if keep_window(w.to_shapely())]
        return out

    @staticmethod
    def within_aoi(window: 'Box',
                   aoi_polygons: Polygon | List[Polygon]) -> bool:
        """Check if window is within the union of given AOI polygons."""
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)
        w = window.to_shapely()
        out = aoi_polygons.contains(w)
        return out

    @staticmethod
    def intersects_aoi(window: 'Box',
                       aoi_polygons: Polygon | List[Polygon]) -> bool:
        """Check if window intersects with the union of given AOI polygons."""
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)
        w = window.to_shapely()
        out = aoi_polygons.intersects(w)
        return out
    
    def from_tensor(tensor):
        """Create a Box object from a tensor."""
        ymin, xmin, ymax, xmax = tensor.tolist()
        return Box(ymin, xmin, ymax, xmax)

    def __contains__(self, query: Union['Box', Sequence]) -> bool:
        """Check if box or point is contained within this box.

        Args:
            query: Box or single point (x, y).

        Raises:
            NotImplementedError: if query is not a Box or tuple/list.
        """
        if isinstance(query, Box):
            ymin, xmin, ymax, xmax = query
            return (ymin >= self.ymin and xmin >= self.xmin
                    and ymax <= self.ymax and xmax <= self.xmax)
        elif isinstance(query, (tuple, list)):
            x, y = query
            return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
        else:
            raise NotImplementedError()


# class CustomVectorOutputConfig(Config):
#     """Config for vectorized semantic segmentation predictions."""
#     class_id: int = Field(
#         ...,
#         description='The prediction class that is to be turned into vectors.'
#     )
#     denoise: int = Field(
#         8,
#         description='Diameter of the circular structural element used to '
#         'remove high-frequency signals from the image. Smaller values will '
#         'reduce less noise and make vectorization slower and more memory '
#         'intensive (especially for large images). Larger values will remove '
#         'more noise and make vectorization faster but might also remove '
#         'legitimate detections.'
#     )
#     threshold: Optional[float] = Field(
#         None,
#         description='Probability threshold for creating the binary mask for '
#         'the pixels of this class. Pixels will be considered to belong to '
#         'this class if their probability for this class is >= ``threshold``. '
#         'Defaults to ``None``, which is equivalent to (1 / num_classes).'
#     )

#     def vectorize(self, mask: np.ndarray) -> Iterator['Polygon']:
#         """Vectorize binary mask representing the target class into polygons."""
#         # Apply denoising if necessary
#         if self.denoise > 0:
#             kernel = np.ones((self.denoise, self.denoise), np.uint8)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Convert contours to polygons
#         for contour in contours:
#             if contour.size >= 6:  # Minimum number of points for a valid polygon
#                 yield Polygon(contour.squeeze())

#     def get_uri(self, root: str, class_config: Optional['ClassConfig'] = None) -> str:
#         """Get the URI for saving the vector output."""
#         if class_config is not None:
#             class_name = class_config.get_name(self.class_id)
#             uri = join(root, f'class-{self.class_id}-{class_name}.json')
#         else:
#             uri = join(root, f'class-{self.class_id}.json')
#         return uri
    
label_uri = "../data/0/SantoDomingo3857.geojson"
image_uri = '../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../data/0/overture/santodomingo_buildings.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

sentinel_source_normalized, sentinel_label_raster_source = create_sentinel_raster_source(image_uri, label_uri, class_config)
rasterized_buildings_source, buildings_label_source = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)

# gdf = gpd.read_file(label_uri)
# gdf = gdf.to_crs('EPSG:3857')
# xmin, ymin, xmax, ymax = gdf.total_bounds
# pixel_polygon = Polygon([
#     (xmin, ymin),
#     (xmin, ymax),
#     (xmax, ymax),
#     (xmax, ymin),
#     (xmin, ymin)
# ])

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

SentinelScene = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalized,
        label_source = sentinel_label_raster_source)
        # aoi_polygons=[pixel_polygon])

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScence, imgsize=288, stride = 288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene, imgsize=144, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

# num_workers = 11
batch_size= 8
# train_sentinel_loader = DataLoader(train_sentinel_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# train_buildings_loader = DataLoader(train_buildings_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# val_sentinel_loader = DataLoader(val_sentinel_dataset, batch_size=batch_size, shuffle=False) #, num_workers=num_workers)
# val_buildings_loader = DataLoader(val_buildings_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# train_sentinel_loader.num_workers

# class MultiModalDataModule(LightningDataModule):
#     def __init__(self, train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader):
#         super().__init__()
#         self.train_sentinel_loader = train_sentinel_loader
#         self.train_buildings_loader = train_buildings_loader
#         self.val_sentinel_loader = val_sentinel_loader
#         self.val_buildings_loader = val_buildings_loader

#     def train_dataloader(self):
#         return zip(self.train_sentinel_loader, self.train_buildings_loader)

#     def val_dataloader(self):
#         return zip(self.val_sentinel_loader, self.val_buildings_loader)
    
# data_module = MultiModalDataModule(train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader)

# assert len(train_sentinel_loader) == len(train_buildings_loader), "DataLoaders must have the same length"
# assert len(val_sentinel_loader) == len(val_buildings_loader), "DataLoaders must have the same length"

# Other approach for dataloading # Concatenate datasets
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

train_dataset = ConcatDataset(train_sentinel_dataset, train_buildings_dataset)
val_dataset = ConcatDataset(val_sentinel_dataset, val_buildings_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) #, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)#, num_workers=num_workers)

train_loader.num_workers

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

# Train the model
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
        
        # print("Output of sent backbone out: ", sentinel_out.shape)
        # print("Output of sent backbone layer3: ", sentinel_features['layer3'].shape)
        # print("Output of sent backbone layer2: ", sentinel_features['layer2'].shape)
        # print("Output of sent backbone layer1: ", sentinel_features['layer1'].shape)
        # print("")
        # print("Output of build backbone out (before downsampling): ", buildings_out.shape)
        # print("Output of build backbone out (after downsampling): ", buildings_out_downsampled.shape)
        # print("Output of build backbone layer3: ", buildings_features['layer3'].shape)
        # print("Output of build backbone layer2: ", buildings_features['layer2'].shape)
        # print("Output of build backbone layer1: ", buildings_features['layer1'].shape)
        
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
        
        self.log('train_loss', loss)
                
        return loss
    
    def validation_step(self, batch):
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)      
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fn(segmentation, buildings_labels.float())
        
        wandb.log({'val_loss': val_loss.item(), 'epoch': self.current_epoch})
        self.log('val_loss', val_loss, prog_bar=True, logger=True)
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, buildings_batch = batch
        _, buildings_labels = buildings_batch

        segmentation = self.forward(batch)     
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, buildings_labels)
        self.log('test_loss', test_loss)

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
    
model = MultiModalSegmentationModel()
model.to(device)
# model.train()

for batch_idx, batch in enumerate(data_module.train_dataloader()):
    sentinel_batch, buildings_batch = batch
    buildings_data, buildings_labels = buildings_batch
    sentinel_data, _ = sentinel_batch
    
    sentinel_data = sentinel_data.to(device)
    buildings_data = buildings_data.to(device)

    print(f"Sentinel data shape: {sentinel_data.shape}")
    print(f"Buildings data shape: {buildings_data.shape}")
    print(f"Buildings labels shape: {buildings_labels.shape}")
    # Pass the data through the model
    model_out = model(batch)
    print(f"Segmentation output shape: {model_out.shape}")
    break  # Exit after the first batch for brevity

output_dir = f'../../UNITAC-trained-models/multi_modal/'
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
print("using num workers ", data_module.train_dataloader().num_workers)
# Train the model
trainer.fit(model, datamodule=data_module)

# # Use best model for evaluation
best_model_path = checkpoint_callback.best_model_path
best_model = MultiModalSegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

full_dataset = ConcatDataset(sentinelGeoDataset, buildingsGeoDataset)
full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)#, num_workers=num_workers)

class PredictionsIterator:
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
                window = buildingsGeoDataset.windows[idx]
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)
    
predictions_iterator = PredictionsIterator(best_model, sentinelGeoDataset, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# # Ensure windows are Box instances
windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=SentinelScene.extent,
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
#     uri='../../vectorised_model_predictions/buildings_model_only/',
#     crs_transformer = crs_transformer_buildings,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)