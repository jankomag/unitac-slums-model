from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    MinMaxTransformer, MultiRasterSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource)

geom = Polygon([
    (-70.1560805913,18.372680107),
    (-69.5854513522,18.372680107),
    (-69.5854513522,18.6220991307),
    (-70.1560805913,18.6220991307),
    (-70.1560805913,18.372680107)
])

bbox = Box.from_shapely(geom.envelope)

BANDS = [
    'coastal', # B01
    'blue', # B02
    'green', # B03
    'red', # B04
    'rededge1', # B05
    'rededge2', # B06
    'rededge3', # B07
    'nir', # B08
    'nir08', # B8A
    'nir09', # B09
    'swir16', # B11
    'swir22', # B12
]

retry = Retry(
    total=5, backoff_factor=1, status_forcelist=[502, 503, 504], allowed_methods=None
)
stac_api_io = StacApiIO(max_retries=retry)

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1", stac_io=stac_api_io)

raster_source_multi = MultiRasterSource(raster_sources=[rasterized_source, sentinel_source], primary_source_idx=0)
