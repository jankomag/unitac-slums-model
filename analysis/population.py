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
import folium
from shapely.ops import unary_union
import re
from shapely.geometry import MultiPolygon, box, Polygon

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-yankomagn')

cities = {
    'Sansalvador': 'San Salvador, El Salvador',
    'SantoDomingo': 'Santo Domingo, Dominican Republic',
    'GuatemalaCity': 'Guatemala City, Guatemala',
    'Tegucigalpa': 'Tegucigalpa, Honduras',
    'Panama': 'Ciudad de Panamá, Pan',
    'Managua': 'Managua, Nicaragua',
}

sica_countries = [
    'Costa Rica', 'El Salvador', 'Guatemala',
    'Honduras', 'Nicaragua', 'Panama', 'Dominican Republic'
]

### DOWNLOADING URBAN BOUNDARY DATA FOR SICA COUNTRIES ###
ULU = ee.ImageCollection('projects/wri-datalab/cities/urban_land_use/V1')

def extract_urban_boundaries(country_name):
    # Get the country boundary
    country = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level0")
                         .filter(ee.Filter.eq('ADM0_NAME', country_name))
                         .first())
    
    # Generate image of 6-class land use from the highest probability class at each pixel
    ULUimage = ULU.select('lulc').reduce(ee.Reducer.firstNonNull()).rename('lulc')
    ULUimage = ULUimage.mask(ULUimage.mask().gt(0))
    
    # Generate image of road areas based on pixels with greater than 50% probability of being road
    roadsImage = ULU.select('road').reduce(ee.Reducer.firstNonNull()).rename('lulc')
    roadProb = 50
    roadsMask = roadsImage.updateMask(roadsImage.gt(roadProb)).where(roadsImage, 1)
    
    # Composite 6-class land use and roads into a single image
    ULUandRoads = ULUimage.where(roadsMask, 6).select('lulc')
    
    # Create a mask for all urban areas (excluding open space which has value 0)
    urbanMask = ULUandRoads.neq(0)
    
    # Convert urban areas to vectors
    urban_vectors = urbanMask.reduceToVectors(
        geometry=country.geometry(),
        scale=500,  # Increase scale for simplification
        eightConnected=False,
        maxPixels=1e13,
        geometryType='polygon'
    )
    
    # Merge overlapping polygons
    merged_vectors = urban_vectors.union(maxError=100)
    
    # Simplify the merged vectors
    simplified_vectors = merged_vectors.map(lambda f: f.simplify(500))
    
    # Add country name to each feature
    return simplified_vectors.map(lambda f: f.set('country', country_name))

# Extract urban boundaries for all SICA countries and save individually
for country in sica_countries:
    print(f"Processing {country}...")
    country_urban = extract_urban_boundaries(country)
    
    # Get the urban boundaries as a GeoJSON
    country_urban_geojson = country_urban.getInfo()

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(country_urban_geojson['features'])

    # Ensure the CRS is set (assuming WGS84)
    gdf = gdf.set_crs("EPSG:4326")

    # Save to file
    output_file = f"../data/1/urban_boundaries/{country.replace(' ', '_')}.gpkg"
    gdf.to_file(output_file, driver="GPKG")
    print(f"Urban boundaries for {country} saved to {output_file}")


### GET BBOX FOR URBAN BOUNDARIES ###
folder_path = "../data/1/urban_boundaries"
data = []

# Iterate through all .gpkg files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.gpkg'):
        country_name = filename.split('.')[0]
        file_path = os.path.join(folder_path, filename)
        
        # Read the GeoPackage file
        gdf = gpd.read_file(file_path)
        
        # Iterate through each row in the file
        for index, row in gdf.iterrows():
            geometry = row['geometry']
            
            # Check if the geometry is a MultiPolygon
            if isinstance(geometry, MultiPolygon):
                # If it's a MultiPolygon, iterate through its components
                for i, part in enumerate(geometry.geoms):
                    bbox = part.bounds
                    data.append({
                        'country': country_name,
                        'geometry_index': f"{index}_{i}",
                        'minx': bbox[0],
                        'miny': bbox[1],
                        'maxx': bbox[2],
                        'maxy': bbox[3]
                    })
            else:
                # If it's a single geometry, process it directly
                bbox = geometry.bounds
                data.append({
                    'country': country_name,
                    'geometry_index': str(index),
                    'minx': bbox[0],
                    'miny': bbox[1],
                    'maxx': bbox[2],
                    'maxy': bbox[3]
                })

# Create a dataframe from the collected data
result_df = pd.DataFrame(data)

# Display the first few rows of the dataframe
result_df.head()

# Create a GeoDataFrame with bbox geometries
gdf_bboxes = gpd.GeoDataFrame(
    result_df,
    geometry=[box(row.minx, row.miny, row.maxx, row.maxy) for _, row in result_df.iterrows()],
    crs="EPSG:4326"
)
gdf_bboxes.explore()

# Saved once
# output_filename = "../data/1/all_SICA_urban_boundaries.geojson"
# gdf_bboxes.to_file(output_filename, driver="GeoJSON")

### IMAGES DOWNLOADED FROM GEE using download_all_cities.ipynb ###

### CALCULATE POPULATION FOR PREDICTIONS WITHIN URBAN BOUNDARIES ###

urab_areas_folder_path = "../data/1/urban_boundaries"

def split_multipolygons(gdf):
    rows = []
    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, MultiPolygon):
            for geom in row.geometry.geoms:
                new_row = row.copy()
                new_row.geometry = geom
                rows.append(new_row)
        else:
            rows.append(row)
    return gpd.GeoDataFrame(rows, crs=gdf.crs)

gdfs = []
for filename in os.listdir(urab_areas_folder_path):
    if filename.endswith('.gpkg'):
        file_path = os.path.join(urab_areas_folder_path, filename)
        gdf = gpd.read_file(file_path)
        gdf = split_multipolygons(gdf)
        gdfs.append(gdf)
        print(f"Loaded and processed: {filename}")

# Combine all the GeoDataFrames into one
urban_areas = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
urban_areas.explore()

### Get Model Predictions ###
predictions_folder_path = "../data/1/SICA_final_predictions/sel_DLV3"
countries = ["Nicaragua", "Honduras"] #CRI, "GTM", , "SLV"

# get all predictions
predictions = {}
for country in countries:
    file_path = os.path.join(predictions_folder_path, country, "averaged_predictions.geojson")
    if os.path.exists(file_path):
        predictions[country] = gpd.read_file(file_path)
    else:
        print(f"No predictions file found for {country}")

# Calculating population proportion
def process_city(city_row, predictions):
    country = city_row['country']
    city_geometry = city_row['geometry']
    
    if not isinstance(city_geometry, Polygon):
        print(f"Warning: geometry for {country} is not a Polygon. Skipping.")
        return None
    
    print(f"\nProcessing city in: {country}")
    
    # Create an Earth Engine geometry
    ee_geometry = ee.Geometry.Polygon(list(city_geometry.exterior.coords))
    
    # Use HRSL dataset
    HRSL_general = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
    
    try:
        # Get the mosaic image from the collection
        hrsl_image = HRSL_general.mosaic()

        # Calculate total population in the city
        total_population = hrsl_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_geometry,
            scale=30,
            maxPixels=1e9
        ).get('b1').getInfo()
        
        print(f"Total population: {total_population}")
        
        # Find informal settlements within the city
        if country in predictions:
            country_predictions = predictions[country]
            city_informal_settlements = country_predictions[country_predictions.geometry.intersects(city_geometry)]
            print(f"Number of intersecting informal settlements: {len(city_informal_settlements)}")
            
            if not city_informal_settlements.empty:
                # Calculate population in informal settlements
                informal_population = 0
                for settlement in city_informal_settlements.geometry:
                    settlement_ee = ee.Geometry.Polygon(list(settlement.exterior.coords))
                    settlement_pop = hrsl_image.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=settlement_ee,
                        scale=30,
                        maxPixels=1e9
                    ).get('b1').getInfo()
                    informal_population += settlement_pop
                
                print(f"Informal settlement population: {informal_population}")
            else:
                print(f"No predictions intersect with this urban area in {country}")
                informal_population = 0
        else:
            print(f"No predictions available for {country}")
            informal_population = 0
        
        # Calculate proportion
        proportion_informal = informal_population / total_population if total_population > 0 else 0
        
        return {
            'country': country,
            'total_population': total_population,
            'informal_population': informal_population,
            'proportion_informal': proportion_informal
        }
    except Exception as e:
        print(f"Error processing city in {country}: {str(e)}")
        return None

results = []

for index, row in tqdm(urban_areas.iterrows(), total=urban_areas.shape[0]):
    result = process_city(row, predictions)
    if result is not None:
        results.append(result)

# Create a DataFrame with the results
results_df = pd.DataFrame(results)
# sort by proportion_informal
results_df = results_df.sort_values('proportion_informal', ascending=False)

results_df.head()

# # Create a map
# m = geemap.Map()

# # Add urban areas to the map
# for idx, row in urban_areas.iterrows():
#     geometry = row['geometry']
#     country = row['country']
#     ee_geometry = ee.Geometry.Polygon(list(geometry.exterior.coords))
#     m.addLayer(ee_geometry, {'color': 'blue'}, f'Urban Area - {country}')

# # Add predictions to the map
# colors = ['red', 'green', 'yellow', 'purple', 'orange']  # Add more colors if needed
# for i, (country, pred) in enumerate(predictions.items()):
#     for idx, row in pred.iterrows():
#         geometry = row['geometry']
#         ee_geometry = ee.Geometry.Polygon(list(geometry.exterior.coords))
#         m.addLayer(ee_geometry, {'color': colors[i % len(colors)]}, f'Prediction - {country}')

# # Set the map center to the centroid of the first urban area
# first_urban_area = urban_areas.iloc[0]['geometry']
# center = first_urban_area.centroid
# m.setCenter(center.x, center.y, 8)

# # Display the map
# m



### CODE TO CALCULATE RATIO OF POPULATION WITHIN PRECARIOUS AREAS WITHIN BBOXES ###
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


precariousAreas_SD = gpd.read_file('../data/0/UNITAC_data/SantoDomingo_PS.geojson')
precariousAreas_SD = precariousAreas_SD.to_crs('EPSG:4326')

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)

gdf = gdf.to_crs('EPSG:3857')

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


# works below
# Use HRSL dataset
HRSL_general = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
try:
    # Get the latest image from the collection
    hrsl_image = HRSL_general.mosaic()

    # Calculate total population
    total_population = hrsl_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=ee_geometry,
        scale=30,
        maxPixels=1e9
    ).get('b1').getInfo()  # HRSL uses 'b1' as the band name
    
    print(f"Total population: {total_population}")

    # Visualize the population data
    map_center = ee_geometry.centroid().getInfo()['coordinates']
    my_map = geemap.Map(center=map_center[::-1], zoom=10)
    
    # Add HRSL layer
    vis_params = {
        'min': 0,
        'max': 100,
        'palette': ['white', 'yellow', 'orange', 'red']
    }
    my_map.addLayer(hrsl_image.clip(ee_geometry), vis_params, 'HRSL Population')
    
    # Add city boundary
    my_map.addLayer(ee_geometry, {'color': '00FF00'}, 'City Boundary')
    
    display(my_map)
except Exception as e:
    print(f"Error processing city: {str(e)}")
