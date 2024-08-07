import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import seaborn as sns
import ee
import os
import sys
import geemap
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import make_valid, simplify, wkt
from tqdm import tqdm
from rasterio.transform import from_bounds
import pandas as pd
from rasterio.features import geometry_mask
import folium
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, box, Polygon
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import re
from matplotlib import rcParams
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import CRS
from shapely.geometry import box, Point

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

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize(project='ee-yankomagn')

sica_countries = [
    'Costa Rica', 'El Salvador', 'Guatemala',
    'Honduras', 'Nicaragua', 'Panama', 'Dominican Republic'
]

### DOWNLOADING URBAN BOUNDARY DATA FOR SICA COUNTRIES ###
# ULU = ee.ImageCollection('projects/wri-datalab/cities/urban_land_use/V1')

# def extract_urban_boundaries(country_name):
#     # Get the country boundary
#     country = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level0")
#                          .filter(ee.Filter.eq('ADM0_NAME', country_name))
#                          .first())
    
#     # Generate image of 6-class land use from the highest probability class at each pixel
#     ULUimage = ULU.select('lulc').reduce(ee.Reducer.firstNonNull()).rename('lulc')
#     ULUimage = ULUimage.mask(ULUimage.mask().gt(0))
    
#     # Generate image of road areas based on pixels with greater than 50% probability of being road
#     roadsImage = ULU.select('road').reduce(ee.Reducer.firstNonNull()).rename('lulc')
#     roadProb = 50
#     roadsMask = roadsImage.updateMask(roadsImage.gt(roadProb)).where(roadsImage, 1)
    
#     # Composite 6-class land use and roads into a single image
#     ULUandRoads = ULUimage.where(roadsMask, 6).select('lulc')
    
#     # Create a mask for all urban areas (excluding open space which has value 0)
#     urbanMask = ULUandRoads.neq(0)
    
#     # Convert urban areas to vectors
#     urban_vectors = urbanMask.reduceToVectors(
#         geometry=country.geometry(),
#         scale=500,  # Increase scale for simplification
#         eightConnected=False,
#         maxPixels=1e13,
#         geometryType='polygon'
#     )
    
#     # Merge overlapping polygons with a non-zero error margin
#     merged_vectors = urban_vectors.union(1)  # 1 meter error margin
    
#     # Simplify the merged vectors
#     simplified_vectors = merged_vectors.map(lambda f: ee.Feature(f.simplify(500)))
    
#     # Add country name to each feature
#     return simplified_vectors.map(lambda f: f.set('country', country_name))

# # Extract urban boundaries for all SICA countries and save individually
# for country in sica_countries:
#     print(f"Processing {country}...")
#     country_urban = extract_urban_boundaries(country)
    
#     # Get the urban boundaries as a GeoJSON
#     country_urban_geojson = country_urban.getInfo()

#     # Convert to GeoDataFrame
#     gdf = gpd.GeoDataFrame.from_features(country_urban_geojson['features'])

#     # Ensure the CRS is set (assuming WGS84)
#     gdf = gdf.set_crs("EPSG:4326")

#     # Split MultiPolygons into single Polygons
#     gdf = split_multipolygons(gdf)

#     # Save to file
#     output_file = f"../data/1/urban_boundaries/{country.replace(' ', '_')}.gpkg"
#     gdf.to_file(output_file, driver="GPKG")
#     print(f"Urban boundaries for {country} saved to {output_file}")

# ### GET BBOX FOR URBAN BOUNDARIES ###
# folder_path = "../data/1/urban_boundaries"
# data = []

# for filename in os.listdir(folder_path):
#     if filename.endswith('.gpkg'):
#         country_name = filename.split('.')[0]
#         file_path = os.path.join(folder_path, filename)
        
#         # Read the GeoPackage file
#         gdf = gpd.read_file(file_path)
        
#         # Iterate through each row in the file
#         for index, row in gdf.iterrows():
#             geometry = row['geometry']
            
#             # Check if the geometry is a MultiPolygon
#             if isinstance(geometry, MultiPolygon):
#                 # If it's a MultiPolygon, iterate through its components
#                 for i, part in enumerate(geometry.geoms):
#                     bbox = part.bounds
#                     data.append({
#                         'country': country_name,
#                         'geometry_index': f"{index}_{i}",
#                         'minx': bbox[0],
#                         'miny': bbox[1],
#                         'maxx': bbox[2],
#                         'maxy': bbox[3]
#                     })
#             else:
#                 # If it's a single geometry, process it directly
#                 bbox = geometry.bounds
#                 data.append({
#                     'country': country_name,
#                     'geometry_index': str(index),
#                     'minx': bbox[0],
#                     'miny': bbox[1],
#                     'maxx': bbox[2],
#                     'maxy': bbox[3]
#                 })

# result_df = pd.DataFrame(data)

# # Create a GeoDataFrame with bbox geometries
# gdf_bboxes = gpd.GeoDataFrame(
#     result_df,
#     geometry=[box(row.minx, row.miny, row.maxx, row.maxy) for _, row in result_df.iterrows()],
#     crs="EPSG:4326"
# )
# gdf_bboxes.explore()

# Saved once
# output_filename = "../data/1/all_SICA_urban_boundaries.geojson"
# gdf_bboxes.to_file(output_filename, driver="GeoJSON")

### IMAGES DOWNLOADED FROM GEE using download_all_cities.ipynb ###

#########################################
#### CALCULATE POPULATION PROPORTION ####
#########################################
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

urab_areas_folder_path = os.path.join(parent_dir, "data/1/urban_boundaries/final_boundaries")
predictions_folder_path = os.path.join(parent_dir, "data/1/SICA_final_predictions/selSJ_DLV3")

# Combine all the GeoDataFrames into one
gdfs = []
for filename in os.listdir(urab_areas_folder_path):
    if filename.endswith('.gpkg'):
        file_path = os.path.join(urab_areas_folder_path, filename)
        gdf = gpd.read_file(file_path)
        gdf = split_multipolygons(gdf)
        gdfs.append(gdf)
        print(f"Loaded and processed: {filename}")

urban_areas = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
urban_areas.explore()

# add a column with concatenated city and country Name
urban_areas['city_country'] = urban_areas['city'] + ", " + urban_areas['country']
urban_areas.value_counts('city_country')

### Get Model Predictions ###
countries = ["Honduras","Nicaragua","Dominican Republic", "Honduras", "Panama","El Salvador","Guatemala","Costa Rica"]

# get all predictions
predictions = {}
for country in countries:
    file_path = os.path.join(predictions_folder_path, country, "averaged_predictions.geojson")
    if os.path.exists(file_path):
        predictions[country] = gpd.read_file(file_path)
    else:
        print(f"No predictions file found for {country}")

# Calculating population proportion
def process_city(city_row, predictions, use_hrsl=True):
    country = city_row['country']
    city = city_row['city']
    city_geometry = city_row['geometry']
    
    if not isinstance(city_geometry, Polygon):
        print(f"Warning: geometry for {country} is not a Polygon. Skipping.")
        return None
    
    print(f"\nProcessing: {city, country}")
    
    # Create an Earth Engine geometry
    ee_geometry = ee.Geometry.Polygon(list(city_geometry.exterior.coords))
    
    # Choose dataset based on user input
    if use_hrsl:
        population_dataset = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
        dataset_name = "HRSL"
        population_image = population_dataset.mosaic()
        
    else:
        population_image = ee.Image("projects/sat-io/open-datasets/GHS/GHS_POP/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0")
        dataset_name = "GHS"
    
    try:
        # Calculate total population in the city
        total_population = population_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_geometry,
            scale=30,
            maxPixels=1e9
        ).get('b1').getInfo()
        
        print(f"Total population ({dataset_name}): {total_population}")
        
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
                    settlement_pop = population_image.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=settlement_ee,
                        scale=30,
                        maxPixels=1e9
                    ).get('b1').getInfo()
                    informal_population += settlement_pop
                
                print(f"Informal settlement population: {informal_population}")
            else:
                print(f"No predictions intersect with this urban area in {city}")
                informal_population = 0
        else:
            print(f"No predictions available for {country}")
            informal_population = 0
        
        # Calculate proportion
        proportion_informal = informal_population / total_population if total_population > 0 else 0
        print(f"Proportion informal: {proportion_informal} in {city}")
        return {
            'country': country,
            'city': city,
            'total_population': total_population,
            'informal_population': informal_population,
            'proportion_informal': proportion_informal,
            'dataset_used': dataset_name
        }
    except Exception as e:
        print(f"Error processing city in {country}: {str(e)}")
        return None

results = []

for index, row in tqdm(urban_areas.iterrows(), total=urban_areas.shape[0]):
    result = process_city(row, predictions, use_hrsl=True)
    if result is not None:
        results.append(result)

# Create a DataFrame with the results
df_hrls = pd.DataFrame(results)

results_hrls_df = df_hrls.sort_values('proportion_informal', ascending=False)
results_hrls_df.tail(8)

results_ghsl_df = df_ghsl.sort_values('proportion_informal', ascending=False)
results_ghsl_df.head()


#############################################################
#### CALCULATE POPULATION PROPORTION AND SAVE TO GEOJSON ####
#############################################################

def process_city(city_row, predictions, use_hrsl=True):
    country = city_row['country']
    city = city_row['city']
    city_geometry = city_row['geometry']
    
    if not isinstance(city_geometry, Polygon):
        print(f"Warning: geometry for {country} is not a Polygon. Skipping.")
        return None
    
    print(f"\nProcessing: {city, country}")
    
    ee_geometry = ee.Geometry.Polygon(list(city_geometry.exterior.coords))
    
    if use_hrsl:
        population_dataset = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
        dataset_name = "HRSL"
        population_image = population_dataset.mosaic()
    else:
        population_image = ee.Image("projects/sat-io/open-datasets/GHS/GHS_POP/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0")
        dataset_name = "GHS"
    
    try:
        total_population = population_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_geometry,
            scale=30,
            maxPixels=1e9
        ).get('b1').getInfo()
        
        print(f"Total population ({dataset_name}): {total_population}")
        
        informal_settlements = []
        
        if country in predictions:
            country_predictions = predictions[country]
            city_informal_settlements = country_predictions[country_predictions.geometry.intersects(city_geometry)]
            print(f"Number of intersecting informal settlements: {len(city_informal_settlements)}")
            
            if not city_informal_settlements.empty:
                for idx, settlement in city_informal_settlements.iterrows():
                    settlement_ee = ee.Geometry.Polygon(list(settlement.geometry.exterior.coords))
                    settlement_pop = population_image.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=settlement_ee,
                        scale=30,
                        maxPixels=1e9
                    ).get('b1').getInfo()
                    
                    informal_settlements.append({
                        'geometry': settlement.geometry,
                        'properties': {
                            'country': country,
                            'city': city,
                            'estimated_population': settlement_pop
                        }
                    })
            else:
                print(f"No predictions intersect with this urban area in {city}")
        else:
            print(f"No predictions available for {country}")
        
        return informal_settlements
    except Exception as e:
        print(f"Error processing city in {country}: {str(e)}")
        return None

all_informal_settlements = []

for index, row in tqdm(urban_areas.iterrows(), total=urban_areas.shape[0]):
    settlements = process_city(row, predictions, use_hrsl=True)
    if settlements:
        all_informal_settlements.extend(settlements)

# Create a GeoDataFrame with all informal settlements
gdf_informal_settlements = gpd.GeoDataFrame(all_informal_settlements)

# Save the GeoDataFrame as a GeoJSON file
output_path = os.path.join(grandparent_dir, "data/1/SICA_final_predictions/final_preds.geojson")
gdf_informal_settlements.to_file(output_path, driver="GeoJSON")

print(f"Saved informal settlements to: {output_path}")

###########################
#### Labels population ####
###########################

labels_folder_path = "../data/SHP"

def clean_geometry(geom, buffer_distance=0.00002):  # Approximately 1 meter at the equator
    if geom is None or geom.is_empty:
        return None
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
        geom = geom.buffer(buffer_distance).buffer(-buffer_distance)  # Remove small holes and smooth edges
        geom = simplify(geom, tolerance=0.001)
        if isinstance(geom, (Polygon, MultiPolygon)):
            geom = geom.buffer(0)  # Ensure valid polygon
        return geom
    except Exception as e:
        print(f"Error in clean_geometry: {str(e)}")
        return None

def coord_to_ee(coord):
    return [np.clip(coord[0], -180, 180), np.clip(coord[1], -90, 90)]

def process_city_shapefile(file_path, use_hrsl=True):
    city = os.path.basename(file_path).replace('.shp', '')
    
    try:
        gdf = gpd.read_file(file_path)
        
        # Clean geometries
        gdf['geometry'] = gdf['geometry'].apply(clean_geometry)
        gdf = gdf.dropna(subset=['geometry'])
        
        if gdf.empty:
            print(f"No valid geometries found in {city}")
            return None
        
        # Merge all geometries into a single MultiPolygon
        try:
            merged_geometry = unary_union(gdf['geometry'])
            merged_geometry = clean_geometry(merged_geometry)
        except Exception as e:
            print(f"Error merging geometries for {city}: {str(e)}")
            print(f"Problematic geometries WKT: {[wkt.dumps(geom) for geom in gdf['geometry']]}")
            return None
        
        if merged_geometry is None or merged_geometry.is_empty:
            print(f"Merged geometry is empty for {city}")
            return None
        
        city_bbox = merged_geometry.bounds
        bbox_geometry = box(*city_bbox)
        
        ee_bbox = ee.Geometry.Rectangle([
            bbox_geometry.bounds[0],
            bbox_geometry.bounds[1],
            bbox_geometry.bounds[2],
            bbox_geometry.bounds[3]
        ])
        
        if use_hrsl:
            population_dataset = ee.ImageCollection("projects/sat-io/open-datasets/hrsl/hrslpop")
            dataset_name = "HRSL"
            population_image = population_dataset.mosaic()
        else:
            population_image = ee.Image("projects/sat-io/open-datasets/GHS/GHS_POP/GHS_POP_E2020_GLOBE_R2023A_54009_100_V1_0")
            dataset_name = "GHS"
        
        total_population = population_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee_bbox,
            scale=30,
            maxPixels=1e9
        ).get('b1').getInfo()
        
        print(f"Total population in {city} bounding box ({dataset_name}): {total_population}")
        
        # Create EE geometry for merged geometry
        if isinstance(merged_geometry, Polygon):
            coords = [coord_to_ee(coord) for coord in merged_geometry.exterior.coords]
            prediction_ee = ee.Geometry.Polygon(coords)
        elif isinstance(merged_geometry, MultiPolygon):
            multi_coords = [[coord_to_ee(coord) for coord in poly.exterior.coords] for poly in merged_geometry.geoms]
            prediction_ee = ee.Geometry.MultiPolygon(multi_coords)
        else:
            print(f"Unexpected geometry type for {city}")
            return None
        
        try:
            predicted_population = population_image.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=prediction_ee,
                scale=30,
                maxPixels=1e9
            ).get('b1').getInfo()
            
            print(f"Population in predicted areas: {predicted_population}")
            
            proportion_predicted = predicted_population / total_population if total_population > 0 else 0
            
            return {
                'city': city,
                'country': city,
                'total_population': total_population,
                'formal_population': total_population,
                'informal_population': predicted_population,
                'proportion_informal': proportion_predicted,
                'dataset_used': dataset_name
            }
        except ee.EEException as e:
            print(f"Error processing geometry in {city}: {str(e)}")
            print(f"Problematic geometry WKT: {wkt.dumps(merged_geometry)}")
            return None
        
    except Exception as e:
        print(f"Error processing {city}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"File path: {file_path}")
        return None

# Process each shapefile in the folder
labels_population = []
for filename in tqdm(os.listdir(labels_folder_path)):
    if filename.endswith('.shp'):
        file_path = os.path.join(labels_folder_path, filename)
        result = process_city_shapefile(file_path, use_hrsl=True)
        if result is not None:
            labels_population.append(result)

# Create a DataFrame with the results
labels_df = pd.DataFrame(labels_population)
labels_df = labels_df.sort_values('proportion_informal', ascending=False)
labels_df.head()


##########################
#### PLOTTING RESULTS ####
##########################
# decide which population data source to use
df=results_hrls_df
# filter out cities with no informal population
df = df[df['informal_population'] > 0]

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond', 'Times New Roman', 'DejaVu Serif']

def clean_city_name(name):
    return re.sub(r'(\w)([A-Z])', r'\1 \2', name)

df_sorted = df.sort_values('proportion_informal', ascending=False)

# Get the top N cities overall (e.g., top 20)
df_top = df_sorted.head(29)

# Calculate the formal population
df_top['formal_population'] = df_top['total_population'] - df_top['informal_population']
df_top['clean_city'] = df_top['city'].apply(clean_city_name)

# Create a color palette for countries
n_colors = len(df_top['country'].unique())
color_palette = sns.color_palette("Set2", n_colors)
color_dict = dict(zip(df_top['country'].unique(), color_palette))

# Create the plot
fig, ax = plt.subplots(figsize=(18, 10))

# Create the stacked bar chart
formal_bars = ax.bar(range(len(df_top)), df_top['formal_population'], 
                     color=[color_dict[c] for c in df_top['country']])
informal_bars = ax.bar(range(len(df_top)), df_top['informal_population'], 
                       bottom=df_top['formal_population'],
                       color=[color_dict[c] for c in df_top['country']], 
                       alpha=0.5)

# Customize the plot
ax.set_title('SICA Cities by Highest Estimated Population (HRSL) Proportion in Precarious Areas', fontsize=22)
ax.set_xlabel('City', fontsize=17)
ax.set_ylabel('Urban Population', fontsize=17)
ax.set_xticks(range(len(df_top)))
ax.set_xticklabels(df_top['clean_city'], rotation=45, ha='right', fontsize=14)

def millions_formatter(x, pos):
    return f'{x/1e6:.1f}M'

ax.yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

# Create a custom legend
legend_elements = [plt.Rectangle((0,0),1,1, color=color_dict[c], label=c) for c in color_dict]
percentage_patch = mpatches.Patch(color='none', label='% = Proportion of Population in PAs')
legend_elements.append(percentage_patch)

# Create legend with larger font size
ax.legend(handles=legend_elements, title='Legend', bbox_to_anchor=(1, 1), 
          loc='upper left', fontsize=16, title_fontsize=18)

# Add percentage labels at the top of each bar
for bar, row in zip(formal_bars, df_top.iterrows()):
    total = row[1]['total_population']
    informal = row[1]['informal_population']
    percentage = (informal / total) * 100
    ax.text(bar.get_x() + bar.get_width()/2, total, f'{percentage:.1f}%', 
            ha='center', va='bottom', fontweight='bold')

# Adjust the top of the plot to make room for labels
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

plt.tight_layout()
plt.show()


# Map to preview the results over urban areas
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
# m

##########################
#### City-level Maps #####
##########################

def create_city_map(city_name, urban_areas, predictions_folder_path, padding=0.1, zoom=15):
    # Find the city in the urban_areas GeoDataFrame
    city = urban_areas[urban_areas['city'] == city_name]
    city = city.to_crs('EPSG:3857')

    if city.empty:
        print(f"City '{city_name}' not found in urban_areas dataset.")
        return
    
    country = city['country'].iloc[0]
    city_geometry = city['geometry'].iloc[0]
    
    # Load predictions for the country
    predictions_file = os.path.join(predictions_folder_path, country, "averaged_predictions.geojson")
    if not os.path.exists(predictions_file):
        print(f"No predictions file found for {city}")
        return
    
    country_predictions = gpd.read_file(predictions_file)
    country_predictions = country_predictions.to_crs('EPSG:3857')
    # Find informal settlements within the city
    city_informal_settlements = gpd.overlay(country_predictions, city, how='intersection')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(13, 13))
    
    # Plot the city boundary
    city.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=2, zorder=3)
    
    # Plot the informal settlements with lower alpha and boundary
    city_informal_settlements.plot(ax=ax, facecolor='red', alpha=0.2, zorder=4)
    city_informal_settlements.boundary.plot(ax=ax, edgecolor='red', linewidth=1, zorder=5)
    
    # Set the extent of the map to the city boundaries with padding
    minx, miny, maxx, maxy = city_geometry.bounds
    width = maxx - minx
    height = maxy - miny
    ax.set_xlim(minx - width * padding, maxx + width * padding)
    ax.set_ylim(miny - height * padding, maxy + height * padding)
    
    # Add high-resolution satellite imagery as background
    cx.add_basemap(ax, crs=city.crs.to_string(), source=cx.providers.Esri.WorldImagery, zoom=zoom, zorder=1)
    
    # Add title
    cl_city_name = clean_city_name(city_name)
    plt.title(f"{cl_city_name}, {country}", fontsize=26)
    
    # Add legend with bigger font and white background
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='red', alpha=0.5, label='Model Predictions for Precarious Areas '),
        Patch(facecolor='none', edgecolor='blue', label='Urban Boundary')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=20, 
              framealpha=1, facecolor='white', edgecolor='black')
    
    # Add scale bar
    scale_bar = ScaleBar(1, units='m', location='lower left', length_fraction=0.25)
    ax.add_artist(scale_bar)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

# Use the function to create a maps
predictions_folder_path = os.path.join(grandparent_dir, "data/1/SICA_final_predictions/selSJ_DLV3")
create_city_map("ElProgreso", urban_areas, predictions_folder_path, padding=0.01, zoom=15)
create_city_map("SanPedroDeMacoris", urban_areas, predictions_folder_path, padding=0.1, zoom=15)
create_city_map("SanFranciscoDeMacoris", urban_areas, predictions_folder_path, padding=0.1, zoom=15)
create_city_map("SanPedroSula", urban_areas, predictions_folder_path, padding=0.01, zoom=15)


### OLD CODE TO CALCULATE RATIO OF POPULATION WITHIN PRECARIOUS AREAS WITHIN BBOXES ###
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
