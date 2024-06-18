import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List

import numpy as np
import geopandas as gpd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import albumentations as A
from affine import Affine
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.features import rasterize
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from typing import TYPE_CHECKING
from pydantic import conint
from xarray import DataArray

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pystac import Item
from torch.utils.data import DataLoader, Dataset

from rastervision.core.box import Box
from rastervision.core.data import (
    RasterSource, RasterioSource, RasterTransformer, ClassConfig,
    GeoJSONVectorSourceConfig, GeoJSONVectorSource, MinMaxTransformer,
    MultiRasterSource, RasterizedSourceConfig, RasterizedSource, Scene,
    StatsTransformer, ClassInferenceTransformer, VectorSourceConfig,
    VectorSource, XarraySource, CRSTransformer, IdentityCRSTransformer,
    RasterioCRSTransformer, SemanticSegmentationLabelSource,
    LabelSource, LabelStore, SemanticSegmentationLabels,
    SemanticSegmentationLabelStore, SemanticSegmentationLabels,
    SemanticSegmentationLabelStore, pad_to_window_size
)

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer, CRSTransformer
    from rastervision.core.box import Box

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from src.data.dataloaders import (
    create_datasets, create_sentinel_raster_source,
    create_buildings_raster_source, show_windows
)

from rastervision.core.data.utils import all_equal, match_bboxes, geoms_to_bbox_coords
from rastervision.core.raster_stats import RasterStats
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pipeline.utils import repr_with_args
from rastervision.pytorch_learner.dataset.transform import (TransformType)
from rastervision.pipeline.config import (Config,Field)

class AlbumentationsDataset(Dataset):
    """An adapter to use arbitrary datasets with albumentations transforms."""

    def __init__(self,
                 orig_dataset: Any,
                 transform: Optional[A.BasicTransform] = None,
                 transform_type: TransformType = TransformType.noop,
                 normalize=True,
                 to_pytorch=True):
        """Constructor.

        Args:
            orig_dataset (Any): An object with a __getitem__ and __len__.
            transform (A.BasicTransform, optional): Albumentations
                transform to apply to the windows. Defaults to None.
                Each transform in Albumentations takes images of type uint8, and
                sometimes other data types. The data type requirements can be
                seen at https://albumentations.ai/docs/api_reference/augmentations/transforms/ # noqa
                If there is a mismatch between the data type of imagery and the
                transform requirements, a RasterTransformer should be set
                on the RasterSource that converts to uint8, such as
                MinMaxTransformer or StatsTransformer.
            transform_type (TransformType): The type of transform so that its
                inputs and outputs can be handled correctly. Defaults to
                TransformType.noop.
            normalize (bool, optional): If True, x is normalized to [0, 1]
                based on its data type. Defaults to True.
            to_pytorch (bool, optional): If True, x and y are converted to
                pytorch tensors. Defaults to True.
        """
        self.orig_dataset = orig_dataset
        self.normalize = normalize
        self.to_pytorch = to_pytorch
        self.transform_type = transform_type

        tf_func = TF_TYPE_TO_TF_FUNC[transform_type]
        self.transform = lambda inp: tf_func(inp, transform)

        if transform_type == TransformType.object_detection:
            self.normalize = False
            self.to_pytorch = False

    def __getitem__(self, key) -> Tuple[torch.Tensor, torch.Tensor]:
        val = self.orig_dataset[key]

        try:
            x, y = self.transform(val)
        except Exception as exc:
            log.warning(
                'Many albumentations transforms require uint8 input. Therefore, we '
                'recommend passing a MinMaxTransformer or StatsTransformer to the '
                'RasterSource so the input will be converted to uint8.')
            raise exc

        if self.normalize and np.issubdtype(x.dtype, np.unsignedinteger):
            max_val = np.iinfo(x.dtype).max
            x = x.astype(float) / max_val

        if self.to_pytorch:
            x = torch.from_numpy(x).float()
            # (..., H, W, C) --> (..., C, H, W)
            x = x.transpose_(-2, -1).transpose_(-3, -2)
            if y is not None:
                y = torch.from_numpy(y)

        if y is None:
            # Ideally, y should be None to semantically convey the absence of
            # any label, but PyTorch's default collate function doesn't handle
            # None values.
            y = torch.tensor(np.nan)

        return x, y

    def __len__(self):
        return len(self.orig_dataset)

class ImageDataset(AlbumentationsDataset):
    """ Dataset that reads from image files. """

class GeoDataset(AlbumentationsDataset):
    """ Dataset that reads directly from a Scene
        (i.e. a raster source and a label source).
    """

    def __init__(
            self,
            scene: Scene,
            out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
            within_aoi: bool = True,
            transform: Optional[A.BasicTransform] = None,
            transform_type: Optional[TransformType] = None,
            normalize: bool = True,
            to_pytorch: bool = True,
            return_window: bool = False):
        
        self.scene = scene
        self.within_aoi = within_aoi
        self.return_window = return_window
        self.out_size = None

        if out_size is not None:
            self.out_size = _to_tuple(out_size)
            transform = self.append_resize_transform(transform, self.out_size)

        super().__init__(
            orig_dataset=scene,
            transform=transform,
            transform_type=transform_type,
            normalize=normalize,
            to_pytorch=to_pytorch)

    def append_resize_transform(
            self, transform: A.BasicTransform | None,
            out_size: tuple[PosInt, PosInt]) -> A.Resize | A.Compose:
        """Get transform to use for resizing windows to out_size."""
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose([transform, resize_tf])
        return transform

    def __len__(self):
        raise NotImplementedError()

    @classmethod
    def from_uris(cls, *args, **kwargs) -> 'GeoDataset':
        raise NotImplementedError()

class MultiResolutionRasterSource(RasterSource):
    """Merge multiple ``RasterSources`` with different resolutions."""

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 resolutions: Sequence[int],
                 primary_source_idx: conint(ge=0) = 0,
                 force_same_dtype: bool = False,
                 channel_order: Optional[Sequence[conint(ge=0)]] = None,
                 raster_transformers: Sequence = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            raster_sources (Sequence[RasterSource]): Sequence of RasterSources.
            resolutions (Sequence[int]): Sequence of resolutions corresponding to each raster source.
            primary_source_idx (int): Index of the primary raster source.
            force_same_dtype (bool): If true, force all sub-chips to have the
                same dtype as the primary_source_idx-th sub-chip. Use with caution.
            channel_order (Sequence[conint(ge=0)], optional): Channel ordering
                that will be used by .get_chip(). Defaults to None.
            raster_transformers (Sequence, optional): Sequence of transformers.
                Defaults to [].
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If given, the primary raster source's bbox is set to this.
                If None, the full extent available in the source file of the
                primary raster source is used.
        """
        num_channels_raw = sum(rs.num_channels for rs in raster_sources)
        if not channel_order:
            num_channels = sum(rs.num_channels for rs in raster_sources)
            channel_order = list(range(num_channels))

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        if bbox is None:
            bbox = raster_sources[primary_source_idx].bbox
        else:
            raster_sources[primary_source_idx].set_bbox(bbox)

        super().__init__(
            channel_order,
            num_channels_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

        self.force_same_dtype = force_same_dtype
        self.raster_sources = raster_sources
        self.resolutions = resolutions
        self.primary_source_idx = primary_source_idx
        self.non_primary_sources = [
            rs for i, rs in enumerate(raster_sources)
            if i != primary_source_idx
        ]

        self.validate_raster_sources()

    def validate_raster_sources(self) -> None:
        """Validate sub-``RasterSources``.

        Checks if:

        - dtypes are same or ``force_same_dtype`` is True.

        """
        dtypes = [rs.dtype for rs in self.raster_sources]
        if not self.force_same_dtype and not all_equal(dtypes):
            raise ValueError(
                'dtypes of all sub raster sources must be the same. '
                f'Got: {dtypes} '
                '(Use force_same_dtype to cast all to the dtype of the '
                'primary source)')

    @property
    def primary_source(self) -> RasterSource:
        """Primary sub-``RasterSource``"""
        return self.raster_sources[self.primary_source_idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the raster as a (..., H, W, C) tuple."""
        *shape, _ = self.primary_source.shape
        return (*shape, self.num_channels)

    @property
    def dtype(self) -> np.dtype:
        return self.primary_source.dtype

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.primary_source.crs_transformer

    def _get_chip(self,
                  window: Box,
                  out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Get chip w/o applying channel_order and transformers.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (Optional[Tuple[int, int]]): (height, width) to resize
                the chip to.

        Returns:
            [height, width, channels] numpy array
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        return chip

    def get_chip(self,
                 window: Box,
                 out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Return the transformed chip in the window.

        Get processed chips from sub raster sources (with their respective
        channel orders and transformations applied), concatenate them along the
        channel dimension, apply channel_order, followed by transformations.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (Optional[Tuple[int, int]]): (height, width) to resize
                the chip to.

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip

    def _get_sub_chips(self,
                       window: Box,
                       out_shape: Optional[Tuple[int, int]] = None) -> list[np.ndarray]:
        """Return chips from sub raster sources as a list.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (Optional[Tuple[int, int]]): (height, width) to resize
                the chip to.

        Returns:
            List[np.ndarray]: List of chips from each sub raster source.
        """

        def get_chip(
                rs: RasterSource,
                window: Box,
                map: bool = False,
                out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
            if map:
                func = rs.get_chip_by_map_window
            else:
                func = rs.get_chip
            return func(window, out_shape=out_shape)

        primary_rs = self.primary_source
        other_rses = self.non_primary_sources

        primary_sub_chip = get_chip(primary_rs, window, out_shape=out_shape)
        if out_shape is None:
            out_shape = primary_sub_chip.shape[:2]
        window_map_coords = primary_rs.crs_transformer.pixel_to_map(
            window, bbox=primary_rs.bbox)
        sub_chips = [
            get_chip(rs, window_map_coords, map=True, out_shape=out_shape)
            for rs in other_rses
        ]
        sub_chips.insert(self.primary_source_idx, primary_sub_chip)

        if self.force_same_dtype:
            dtype = sub_chips[self.primary_source_idx].dtype
            sub_chips = [chip.astype(dtype) for chip in sub_chips]

        return sub_chips

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

raster_sources = [rasterized_buildings_source, sentinel_source_normalized]
raster_source_multi = MultiRasterSource(raster_sources=raster_sources, primary_source_idx=0, force_same_dtype=True)

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

MultiScene = Scene(
        id='santodomingo_multi',
        raster_source = raster_source_multi,
        label_source = rasterized_buildings_source,
        aoi_polygons=[pixel_polygon])

SentinelScene = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalized,
        label_source = sentinel_label_raster_source,
        aoi_polygons=[pixel_polygon])

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

multiGeoDataset, train_dataset, val_dataset, test_dataset = create_datasets(MultiScene, imgsize=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

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

img_full = create_full_imagesent(multiGeoDataset.scene.label_source)
train_windows = train_dataset.windows
val_windows = val_dataset.windows
test_windows = test_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

train_dl = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=6)

# Fine-tune the model
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

pretrained_checkpoint_path = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
model = CustomDeeplabv3SegmentationModel(num_bands=1, pretrained_checkpoint=pretrained_checkpoint_path)
model.to(device)
model.train()









class CustomScene:
    def __init__(self,
                 id: str,
                 raster_sources: Dict[str, RasterSource],
                 primary_source_key: str,
                 label_source: Optional[LabelSource] = None,
                 label_store: Optional[LabelStore] = None,
                 aoi_polygons: Optional[List[BaseGeometry]] = None):
        """Constructor.

        During initialization, ``Scene`` attempts to set the extents of the
        given ``label_source`` and the ``label_store`` to be identical to the
        extent of the given ``raster_source``.

        Args:
            id: ID for this scene.
            raster_source: Source of imagery for this scene.
            label_source: Source of labels for this scene.
            label_store: Store of predictions for this scene.
            aoi: Optional list of AOI polygons in pixel coordinates.
        """
        if label_source is not None:
            match_bboxes(raster_source, label_source)

        if label_store is not None:
            match_bboxes(raster_source, label_store)

        self.id = id
        self.raster_sources = raster_sources
        self.primary_source_key = primary_source_key
        self.multi_raster_source = MultiRasterSource(
            raster_sources=self.raster_sources,
            primary_source_key=self.primary_source_key,
            force_same_dtype = False,
            channel_order = None,
            raster_transformers = [],
            bbox = None)
        
        self.label_source = label_source
        self.label_store = label_store
            
        if aoi_polygons is None:
            self.aoi_polygons = []
            self.aoi_polygons_bbox_coords = []
        else:
            for p in aoi_polygons:
                if p.geom_type not in ['Polygon', 'MultiPolygon']:
                    raise ValueError(
                        'Expected all AOI geometries to be Polygons or '
                        f'MultiPolygons. Found: {p.geom_type}.')
            bbox = self.raster_source.bbox
            bbox_geom = bbox.to_shapely()
            self.aoi_polygons = [
                p for p in aoi_polygons if p.intersects(bbox_geom)
            ]
            self.aoi_polygons_bbox_coords = list(
                geoms_to_bbox_coords(self.aoi_polygons, bbox))

    @property
    def extent(self) -> 'Box':
        """Extent of the associated :class:`.RasterSource`."""
        return self.raster_source.extent

    @property
    def bbox(self) -> 'Box':
        """Bounding box applied to the source data."""
        return self.raster_source.bbox

    def __getitem__(self, key: Any) -> Tuple[Any, Any]:
        x = self.raster_source[key]
        if self.label_source is not None:
            y = self.label_source[key]
        else:
            y = None
        return x, y

class CustomMultiRasterSource(RasterSource):
    """Merge multiple ``RasterSources`` by concatenating along channel dim."""

    def __init__(self,
                 raster_sources: Dict[str, RasterSource],
                 primary_source_key: str,
                 force_same_dtype: bool = False,
                 channel_order: Optional[Sequence[conint(ge=0)]] = None,
                 raster_transformers: Sequence = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            raster_sources (Sequence[RasterSource]): Sequence of RasterSources.
            primary_source_idx (0 <= int < len(raster_sources)): Index of the
                raster source whose CRS, dtype, and other attributes will
                override those of the other raster sources.
            force_same_dtype (bool): If true, force all sub-chips to have the
                same dtype as the primary_source_idx-th sub-chip. No careful
                conversion is done, just a quick cast. Use with caution.
            channel_order (Sequence[conint(ge=0)], optional): Channel ordering
                that will be used by .get_chip(). Defaults to None.
            raster_transformers (Sequence, optional): Sequence of transformers.
                Defaults to [].
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If given, the primary raster source's bbox is set to this.
                If None, the full extent available in the source file of the
                primary raster source is used.
        """
        num_channels_raw = sum(rs.num_channels for rs in raster_sources)
        if not channel_order:
            num_channels = sum(rs.num_channels for rs in raster_sources)
            channel_order = list(range(num_channels))
            
        primary_source_idx = primary_source_key

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        if bbox is None:
            bbox = raster_sources[primary_source_idx].bbox
        else:
            raster_sources[primary_source_idx].set_bbox(bbox)

        super().__init__(
            channel_order,
            num_channels_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

        self.force_same_dtype = force_same_dtype
        
        self.raster_sources = raster_sources
        self.primary_source_key = primary_source_key
        self.primary_source = raster_sources[primary_source_key]
        self.non_primary_sources = {k: v for k, v in raster_sources.items() if k != primary_source_key}

        self.validate_raster_sources()

    def validate_raster_sources(self) -> None:
        """Validate sub-``RasterSources``.

        Checks if:

        - dtypes are same or ``force_same_dtype`` is True.

        """
        dtypes = [rs.dtype for rs in self.raster_sources]
        if not self.force_same_dtype and not all_equal(dtypes):
            raise ValueError(
                'dtypes of all sub raster sources must be the same. '
                f'Got: {dtypes} '
                '(Use force_same_dtype to cast all to the dtype of the '
                'primary source)')

    @property
    def primary_source(self) -> RasterSource:
        """Primary sub-``RasterSource``"""
        return self.raster_sources[self.primary_source_idx]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the raster as a (..., H, W, C) tuple."""
        *shape, _ = self.primary_source.shape
        return (*shape, self.num_channels)

    @property
    def dtype(self) -> np.dtype:
        return self.primary_source.dtype

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.primary_source.crs_transformer

    def _get_sub_chips(self,
                    window: Box,
                    out_shape: Optional[Tuple[int, int]] = None
                    ) -> Dict[str, np.ndarray]:
        sub_chips = {}
        primary_sub_chip = get_chip(self.primary_source, window, out_shape=out_shape)
        if out_shape is None:
            out_shape = primary_sub_chip.shape[:2]
        window_map_coords = self.primary_source.crs_transformer.pixel_to_map(
            window, bbox=self.primary_source.bbox)
        sub_chips[self.primary_source_key] = primary_sub_chip

        for key, rs in self.non_primary_sources.items():
            sub_chips[key] = get_chip(rs, window_map_coords, map=True, out_shape=out_shape)

        if self.force_same_dtype:
            dtype = sub_chips[self.primary_source_key].dtype
            sub_chips = {k: chip.astype(dtype) for k, chip in sub_chips.items()}

        return sub_chips

    def _get_chip(self,
                window: Box,
                out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        channels = []
        for key in sorted(sub_chips.keys()):
            channels.append(sub_chips[key])
        chip = np.concatenate(channels, axis=-1)
        return chip

    def get_chip(self,
                 window: Box,
                 out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip

class CustomSlidingWindowGeoDataset(GeoDataset):
    def __init__(
            self,
            scene: Scene,
            size: Union[PosInt, Tuple[PosInt, PosInt]],
            stride: Union[PosInt, Tuple[PosInt, PosInt]],
            out_size: Optional[Union[PosInt, Tuple[PosInt, PosInt]]] = None,
            padding: Optional[Union[NonNegInt, Tuple[NonNegInt,
                                                     NonNegInt]]] = None,
            pad_direction: Literal['both', 'start', 'end'] = 'end',
            within_aoi: bool = True,
            transform: Optional[A.BasicTransform] = None,
            transform_type: Optional[TransformType] = None,
            normalize: bool = True,
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
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        self.padding = padding
        self.pad_direction = pad_direction
        self.init_windows()

    def init_windows(self) -> None:
        windows = self.scene.multi_raster_source.extent.get_windows(
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

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration()
        window = self.windows[idx]
        out = super().__getitem__(window)
        if self.return_window:
            return (out, window)
        return out

    def __len__(self):
        return len(self.windows)
    
    def split_train_val_test(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = None) -> Tuple['CustomSemanticSegmentationSlidingWindowGeoDataset', 'CustomSemanticSegmentationSlidingWindowGeoDataset', 'CustomSemanticSegmentationSlidingWindowGeoDataset']:

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
        subset = CustomSlidingWindowGeoDataset(
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