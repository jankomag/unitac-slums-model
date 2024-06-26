import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import ee
import geemap
from tqdm import tqdm
from rasterio.transform import from_bounds
import pandas as pd
from rasterio.features import geometry_mask

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-yankomagn')

# Load your city GeoDataFrame
sica_cities = "../../data/0/SICA_cities.geojson"
gdf = gpd.read_file(sica_cities)
gdf = gdf.to_crs('EPSG:4326')
gdf = gdf[gdf['city_ascii'] == 'Los Minas']
gdf.head()

# Load informal settlements GeoJSON
informal_settlements = gpd.read_file('../../data/0/UNITAC_data/SantoDomingo_PS.geojson')
informal_settlements = informal_settlements.to_crs('EPSG:4326')

# Function to get population raster for a given bounding box
def get_population_raster(bbox, scale=30):
    ee_bbox = ee.Geometry.Rectangle(bbox)
    HRSL_general = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
    population = HRSL_general.median().clip(ee_bbox)
    
    # Get the population data as a numpy array
    # Note: We're removing the scale parameter here
    array = geemap.ee_to_numpy(population, region=ee_bbox)
    
    return array.squeeze()  # Remove single-dimensional entries

# Function to calculate population within a polygon
def calculate_population(raster, polygon, bbox):
    height, width = raster.shape
    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
    mask = geometry_mask([polygon], out_shape=raster.shape, transform=transform, invert=True)
    population = np.sum(raster[mask])
    print(f"Calculated population: {population}")
    return population


results = []

for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_ascii']
    country_code = row['iso3']
    city_geometry = row['geometry']
    
    print(f"\nProcessing city: {city_name}")
    
    # Get the bounding box of the city
    bbox = city_geometry.bounds
    bbox_list = [bbox[0], bbox[1], bbox[2], bbox[3]]
    
    # Get population raster for the city's bounding box
    population_raster = get_population_raster(bbox_list)
    print(f"Population raster shape: {population_raster.shape}")
    
    # Calculate total population in the city
    print("Calculating total population...")
    total_population = calculate_population(population_raster, city_geometry, bbox_list)
    
    # Find informal settlements within the city
    city_informal_settlements = informal_settlements[informal_settlements.intersects(city_geometry)]
    print(f"Number of intersecting informal settlements: {len(city_informal_settlements)}")
    
    # Calculate population in informal settlements
    print("Calculating informal settlement population...")
    informal_population = sum(calculate_population(population_raster, settlement, bbox_list) 
                              for settlement in city_informal_settlements.geometry)
    
    # Calculate proportion
    proportion_informal = informal_population / total_population if total_population > 0 else 0
    
    results.append({
        'city': city_name,
        'country_code': country_code,
        'total_population': total_population,
        'informal_population': informal_population,
        'proportion_informal': proportion_informal
    })

# Create a DataFrame with the results
results_df = pd.DataFrame(results)
print(results_df)