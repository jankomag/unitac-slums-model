import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from rasterio import features
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import gaussian_filter

# Load the building footprints GeoDataFrame
buildings_uri_SD = '../../data/0/overture/santodomingo_buildings.geojson'
label_uri_SD = "../../data/0/SantoDomingo3857.geojson"

buildings = gpd.read_file(buildings_uri_SD)
buildings = buildings.to_crs(epsg=3857)

# Calculate area and its reciprocal for each building
buildings['area'] = buildings.geometry.area
buildings['area_reciprocal'] = 100 / buildings.area

# Define raster properties
resolution = 5  # 5m resolution
xmin, ymin, xmax, ymax = buildings.total_bounds
width = int((xmax - xmin) / resolution)
height = int((ymax - ymin) / resolution)
transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
