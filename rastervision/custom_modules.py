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
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource, CRSTransformer,
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

class CustomCRSTransformer(CRSTransformer):
    def __init__(self,
                 transform: Affine,
                 image_crs: Any,
                 map_crs: Any = 'epsg:4326',
                 round_pixels: bool = True):

        if (image_crs is None) or (image_crs == map_crs):
            self.map2image = lambda *args, **kws: args[:2]
            self.image2map = lambda *args, **kws: args[:2]
        else:
            self.map2image = Transformer.from_crs(
                map_crs, image_crs, always_xy=True).transform
            self.image2map = Transformer.from_crs(
                image_crs, map_crs, always_xy=True).transform

        self.round_pixels = round_pixels

        super().__init__(transform, image_crs, map_crs)

    def __repr__(self) -> str:
        cls_name = type(self).__name__

        image_crs_str = str(self.image_crs)
        if len(image_crs_str) > 70:
            image_crs_str = image_crs_str[:70] + '...'

        map_crs_str = str(self.map_crs)
        if len(map_crs_str) > 70:
            map_crs_str = map_crs_str[:70] + '...'

        transform_str = (
            '\n\t\t' + (str(self.transform).replace('\n', '\n\t\t')))
        out = f"""{cls_name}(
            image_crs="{image_crs_str}",
            map_crs="{map_crs_str}",
            round_pixels="{self.round_pixels}",
            transform={transform_str})
        """
        return out

    def _map_to_pixel(self, map_point):
        image_point = self.map2image(*map_point)
        x, y = image_point
        if self.round_pixels:
            row, col = rowcol(self.transform, x, y)
        else:
            row, col = rowcol(self.transform, x, y, op=lambda x: x)
        pixel_point = (col, row)
        return pixel_point

    def _pixel_to_map(self, pixel_point):
        col, row = pixel_point
        if self.round_pixels:
            col = col.astype(int) if isinstance(col, np.ndarray) else int(col)
            row = row.astype(int) if isinstance(row, np.ndarray) else int(row)
        image_point = xy(self.transform, row, col, offset='center')
        map_point = self.image2map(*image_point)
        return map_point
    
    ### ADPATING THIS BIT ###
    @classmethod
    def from_affine_transfrom(
            cls, predefined_transform: Any, map_crs: Optional[str] = 'epsg:4326', image_crs='epsg:4326', **kwargs
    ) -> Union[IdentityCRSTransformer, 'CustomCRSTransformer']:
        transform = predefined_transform
        map_crs = image_crs if map_crs is None else map_crs
        # image_crs = map_crs

        no_crs_tf = (image_crs is None) or (image_crs == map_crs)
        no_affine_tf = (transform is None) or (transform == Affine.identity())
        if no_crs_tf and no_affine_tf:
            return IdentityCRSTransformer()

        if transform is None:
            transform = Affine.identity()

        return cls(transform, image_crs, map_crs, **kwargs)

class XarraySource(RasterSource):
    """A RasterSource for reading an Xarry DataArray.

    .. warning:: ``XarraySource`` API is in beta.
    """

    def __init__(self,
                 data_array: DataArray,
                 crs_transformer: 'CRSTransformer',
                 raster_transformers: List['RasterTransformer'] = [],
                 channel_order: Optional[Sequence[int]] = None,
                 bbox: Optional[Box] = None,
                 temporal: bool = False):

        self.temporal = temporal
        if self.temporal:
            if set(data_array.dims) != {'x', 'y', 'band', 'time'}:
                raise ValueError(
                    'If temporal=True, data_array must have 4 dimensions: '
                    '"x", "y", "band", and "time" (in any order).')
        else:
            if set(data_array.dims) != {'x', 'y', 'band'}:
                raise ValueError(
                    'If temporal=False, data_array must have 3 dimensions: '
                    '"x", "y", and "band" (in any order).')

        self.data_array = data_array.transpose(..., 'y', 'x', 'band')
        self.ndim = data_array.ndim
        self._crs_transformer = crs_transformer

        num_channels_raw = len(data_array.band)
        if channel_order is None:
            channel_order = np.arange(num_channels_raw, dtype=int)
        else:
            channel_order = np.array(channel_order, dtype=int)
        self._num_channels = None
        self._dtype = None
        if len(raster_transformers) == 0:
            self._num_channels = len(channel_order)
            self._dtype = data_array.dtype

        height, width = len(data_array.y), len(data_array.x)
        self.full_extent = Box(0, 0, height, width)
        if bbox is None:
            bbox = self.full_extent
        else:
            if bbox not in self.full_extent:
                new_bbox = bbox.intersection(self.full_extent)
                log.warning(f'Clipping ({bbox}) to the DataArray\'s '
                            f'full extent ({self.full_extent}). '
                            f'New bbox={new_bbox}')
                bbox = new_bbox

        super().__init__(
            channel_order,
            num_channels_raw,
            raster_transformers=raster_transformers,
            bbox=bbox)

    @classmethod
    def from_stac(
            cls,
            item_or_item_collection: Union['Item', 'ItemCollection'],
            raster_transformers: List['RasterTransformer'] = [],
            channel_order: Optional[Sequence[int]] = None,
            bbox: Optional[Box] = None,
            bbox_map_coords: Optional[Box] = None,
            temporal: bool = False,
            allow_streaming: bool = False,
            stackstac_args: dict = dict(rescale=False)) -> 'XarraySource':
        """Construct an ``XarraySource`` from a STAC Item or ItemCollection.

        Args:
            item_or_item_collection: STAC Item or ItemCollection.
            raster_transformers: RasterTransformers to use to transform chips
                after they are read.
            channel_order: List of indices of channels to extract from raw
                imagery. Can be a subset of the available channels. If None,
                all channels available in the image will be read.
                Defaults to None.
            bbox: User-specified crop of the extent. If None, the full extent
                available in the source file is used. Mutually exclusive with
                ``bbox_map_coords``. Defaults to ``None``.
            bbox_map_coords: User-specified bbox in EPSG:4326 coords of the
                form (ymin, xmin, ymax, xmax). Useful for cropping the raster
                source so that only part of the raster is read from. Mutually
                exclusive with ``bbox``. Defaults to ``None``.
            temporal: If True, data_array is expected to have a "time"
                dimension and the chips returned will be of shape (T, H, W, C).
            allow_streaming: If False, load the entire DataArray into memory.
                Defaults to True.
            stackstac_args: Optional arguments to pass to stackstac.stack().
        """
        import stackstac

        data_array = stackstac.stack(item_or_item_collection, **stackstac_args)

        if not temporal and 'time' in data_array.dims:
            if len(data_array.time) > 1:
                raise ValueError('temporal=False but len(data_array.time) > 1')
            data_array = data_array.isel(time=0)

        if not allow_streaming:
            from humanize import naturalsize
            log.info('Loading the full DataArray into memory '
                     f'({naturalsize(data_array.nbytes)}).')
            data_array.load()

        crs_transformer = RasterioCRSTransformer(
            transform=data_array.transform, image_crs=data_array.crs)

        if bbox is not None:
            if bbox_map_coords is not None:
                raise ValueError('Specify either bbox or bbox_map_coords, '
                                 'but not both.')
            bbox = Box(*bbox)
        elif bbox_map_coords is not None:
            bbox_map_coords = Box(*bbox_map_coords)
            bbox = crs_transformer.map_to_pixel(bbox_map_coords).normalize()
        else:
            bbox = None

        raster_source = XarraySource(
            data_array,
            crs_transformer=crs_transformer,
            raster_transformers=raster_transformers,
            channel_order=channel_order,
            bbox=bbox,
            temporal=temporal)
        return raster_source

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the raster as a (height, width, num_channels) tuple."""
        H, W = self.bbox.size
        if self.temporal:
            T = len(self.data_array.time)
            return T, H, W, self.num_channels
        return H, W, self.num_channels

    @property
    def num_channels(self) -> int:

        if self._num_channels is None:
            self._set_info_from_chip()
        return self._num_channels

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._set_info_from_chip()
        return self._dtype

    @property
    def crs_transformer(self) -> RasterioCRSTransformer:
        return self._crs_transformer

    def _set_info_from_chip(self):
        """Read 1x1 chip to get info not statically inferable."""
        test_chip = self.get_chip(Box(0, 0, 1, 1))
        self._dtype = test_chip.dtype
        self._num_channels = test_chip.shape[-1]

    def _get_chip(self,
                  window: Box,
                  bands: Union[int, Sequence[int], slice] = slice(None),
                  time: Union[int, Sequence[int], slice] = slice(None),
                  out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        window = window.to_global_coords(self.bbox)

        window_within_bbox = window.intersection(self.bbox)

        yslice, xslice = window_within_bbox.to_slices()
        if self.temporal:
            chip = self.data_array.isel(
                x=xslice, y=yslice, band=bands, time=time).to_numpy()
        else:
            chip = self.data_array.isel(
                x=xslice, y=yslice, band=bands).to_numpy()

        if window != window_within_bbox:
            *batch_dims, h, w, c = chip.shape
            # coords of window_within_bbox within window
            yslice, xslice = window_within_bbox.to_local_coords(
                window).to_slices()
            tmp = np.zeros((*batch_dims, *window.size, c))
            tmp[..., yslice, xslice, :] = chip
            chip = tmp

        chip = fill_overflow(self.bbox, window, chip)
        if out_shape is not None:
            chip = self.resize(chip, out_shape)
        return chip

    def get_chip(self,
                 window: Box,
                 bands: Optional[Union[int, Sequence[int], slice]] = None,
                 time: Union[int, Sequence[int], slice] = slice(None),
                 out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Read a chip specified by a window from the file.

        Args:
            window (Box): Bounding box of chip in pixel coordinates.
            bands (Optional[Union[Sequence[int], slice]], optional): Subset of
                bands to read. Note that this will be applied on top of the
                channel_order (if specified). So if this is an RGB image and
                channel_order=[2, 1, 0], then using bands=[0] will return the
                B-channel. Defaults to None.
            out_shape (Optional[Tuple[int, ...]], optional): (height, width) of
                the output chip. If None, no resizing is done.
                Defaults to None.

        Returns:
            np.ndarray: A chip of shape (height, width, channels).
        """
        if bands is None or bands == slice(None):
            bands = self.channel_order
        else:
            bands = self.channel_order[bands]
        chip = self._get_chip(
            window, bands=bands, time=time, out_shape=out_shape)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, bands)
        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, dim_slices = parse_array_slices_Nd(
            key, extent=self.extent, dims=self.ndim)
        if self.temporal:
            t, h, w, c = dim_slices
        else:
            h, w, c = dim_slices
            t = None

        out_shape = None
        if h.step is not None or w.step is not None:
            out_h, out_w = window.size
            if h.step is not None:
                out_h //= h.step
            if w.step is not None:
                out_w //= w.step
            out_shape = (int(out_h), int(out_w))

        chip = self.get_chip(window, bands=c, time=t, out_shape=out_shape)
        return chip
    
    def calculate_ndbi(chip, nir_band=3, swir_band=4):
        nir = chip[:, :, nir_band]  # NIR band
        swir = chip[:, :, swir_band]  # SWIR band
        ndbi = (swir - nir) / (swir + nir)

        return ndbi
    
    def patched_get_chip(self, window: Box, bands=None, time=slice(None), out_shape=None):
        chip = self._get_chip(window, bands=bands, time=time, out_shape=out_shape)

        # Compute NDBI if NIR and SWIR bands are present
        if 3 in bands and 4 in bands:  # Assuming NIR is at index 3 and SWIR is at index 4
            ndbi = calculate_ndbi(chip)
            # Add NDBI as the fifth channel
            ndbi = np.expand_dims(ndbi, axis=2)
            chip_with_ndbi = np.concatenate((chip, ndbi), axis=2)
            return chip_with_ndbi

        return chip
    
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
        chip = customgeoms_to_raster(
            self.df,
            window,
            background_class_id=self.background_class_id,
            all_touched=self.all_touched)

        if out_shape is not None:
            chip = self.resize(chip, out_shape)

        # Add third singleton dim since rasters must have >=1 channel.
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

        # Normalize to have standard deviation of 1 if desired
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
        stats_transformer = CustomStatsTransformer.from_raster_stats(
            stats, max_stds=max_stds)
        return stats_transformer
    
    @classmethod
    def from_raster_stats(cls, stats: RasterStats,
                          **kwargs) -> 'StatsTransformer':
        
        stats_transformer = CustomStatsTransformer(stats.means, stats.stds, **kwargs)
        return stats_transformer
    