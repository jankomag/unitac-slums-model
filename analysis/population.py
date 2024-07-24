import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import seaborn as sns
import ee
import geemap
from tqdm import tqdm
from rasterio.transform import from_bounds
import pandas as pd
from rasterio.features import geometry_mask
import folium
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, box, Polygon
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import re
from matplotlib import rcParams
import contextily as cx
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import CRS

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

urab_areas_folder_path = "../data/1/urban_boundaries/final_boundaries"
predictions_folder_path = "../data/1/SICA_final_predictions/sel_DLV3"

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
def process_city(city_row, predictions):
    country = city_row['country']
    city = city_row['city']
    city_geometry = city_row['geometry']
    
    if not isinstance(city_geometry, Polygon):
        print(f"Warning: geometry for {country} is not a Polygon. Skipping.")
        return None
    
    print(f"\nProcessing: {city, country}")
    
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
                print(f"No predictions intersect with this urban area in {city}")
                informal_population = 0
        else:
            print(f"No predictions available for {country}")
            informal_population = 0
        
        # Calculate proportion
        proportion_informal = informal_population / total_population if total_population > 0 else 0
        
        return {
            'country': country,
            'city': city,
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
df = pd.DataFrame(results)
# sort by proportion_informal
results_df = df.sort_values('proportion_informal', ascending=False)
results_df.head()

##########################
#### PLOTTING RESULTS ####
##########################

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond', 'Times New Roman', 'DejaVu Serif']

def clean_city_name(name):
    return re.sub(r'(\w)([A-Z])', r'\1 \2', name)

df_sorted = df.sort_values('proportion_informal', ascending=False)

# Get the top N cities overall (e.g., top 20)
df_top = df_sorted.head(15)

# Calculate the formal population
df_top['formal_population'] = df_top['total_population'] - df_top['informal_population']
df_top['clean_city'] = df_top['city'].apply(clean_city_name)

# Create a color palette for countries
n_colors = len(df_top['country'].unique())
color_palette = sns.color_palette("Set2", n_colors)
color_dict = dict(zip(df_top['country'].unique(), color_palette))

# Create the plot
fig, ax = plt.subplots(figsize=(16, 10))

# Create the stacked bar chart
formal_bars = ax.bar(range(len(df_top)), df_top['formal_population'], 
                     color=[color_dict[c] for c in df_top['country']])
informal_bars = ax.bar(range(len(df_top)), df_top['informal_population'], 
                       bottom=df_top['formal_population'],
                       color=[color_dict[c] for c in df_top['country']], 
                       alpha=0.5)

# Customize the plot
ax.set_title('SICA Cities by Highest Estimated Population Proportion in Precarious Areas', fontsize=22)
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

######################
#### MAP BARCHART ####
######################

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Garamond', 'Times New Roman', 'DejaVu Serif']

# Load the urban areas
urban_areas_path = "../data/1/urban_boundaries/bboxes_SICA_urban_boundaries.geojson"
urban_areas = gpd.read_file(urban_areas_path)

# Ensure the CRS is EPSG:4326
urban_areas = urban_areas.to_crs(epsg=4326)

# Merge urban areas with population data
merged_df = urban_areas.merge(df, left_on='city_name', right_on='city')

# Calculate formal population and proportion of informal population
merged_df['formal_population'] = merged_df['total_population'] - merged_df['informal_population']
merged_df['proportion_informal'] = merged_df['informal_population'] / merged_df['total_population']

# Sort by proportion of informal population and keep top 5
merged_df = merged_df.sort_values('proportion_informal', ascending=False)

# Create the plot
fig, ax = plt.subplots(figsize=(20, 20))

# Get the bounds of all urban areas
minx, miny, maxx, maxy = urban_areas.total_bounds

# Set the plot limits with some padding
padding = 0.1  # 10% padding
ax.set_xlim(minx - padding * (maxx - minx), maxx + padding * (maxx - minx))
ax.set_ylim(miny - padding * (maxy - miny), maxy + padding * (maxy - miny))

# Plot all urban areas
urban_areas.plot(ax=ax, color='lightgrey', edgecolor='black')

# Add basemap
ctx.add_basemap(ax, crs=urban_areas.crs.to_string(), source=ctx.providers.CartoDB.Positron)

# Set aspect ratio
y_coord = np.mean([miny, maxy])
ax.set_aspect(1 / np.cos(y_coord * np.pi / 180))

# Function to create a stacked bar
def create_stacked_bar(x, y, formal, informal, city, percentage, ax):
    bar_width = (maxx - minx) * 0.01  # Adjust based on your data extent
    bar_height = (maxy - miny) * 0.05  # Adjust based on your data extent
    
    # Normalize the values
    total = formal + informal
    if total > 0:
        norm_formal = formal / total * bar_height
        norm_informal = informal / total * bar_height
    else:
        norm_formal = norm_informal = 0
    
    # Create bars
    ax.add_patch(plt.Rectangle((x, y), bar_width, norm_formal, color='blue', alpha=0.7))
    ax.add_patch(plt.Rectangle((x, y+norm_formal), bar_width, norm_informal, color='red', alpha=0.7))
    
    # Add city name and percentage
    ax.text(x, y+bar_height+0.05, f"{city}\n{percentage:.1f}%", ha='center', va='bottom', fontsize=8)

# Add stacked bars for top 5 cities
for idx, row in merged_df.iterrows():
    x, y = row.geometry.centroid.x, row.geometry.centroid.y
    create_stacked_bar(x, y, row['formal_population'], row['informal_population'], 
                       row['city_name'], row['proportion_informal']*100, ax)

# Remove axis
ax.axis('off')

# Add legend
ax.add_patch(plt.Rectangle((0.05, 0.05), 0.02, 0.02, color='blue', alpha=0.7, transform=ax.transAxes))
ax.add_patch(plt.Rectangle((0.05, 0.08), 0.02, 0.02, color='red', alpha=0.7, transform=ax.transAxes))
ax.text(0.08, 0.05, 'Formal Population', va='center', transform=ax.transAxes)
ax.text(0.08, 0.08, 'Informal Population', va='center', transform=ax.transAxes)

# Set title
plt.title('SICA Cities by Highest Estimated Population Proportion in Precarious Areas', fontsize=20)

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
        print(f"No predictions file found for {country}")
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
predictions_folder_path = "../data/1/SICA_final_predictions/sel_DLV3"
create_city_map("ElProgreso", urban_areas, predictions_folder_path, padding=0.01, zoom=15)
create_city_map("SanPedroSula", urban_areas, predictions_folder_path, padding=0.01, zoom=15)
create_city_map("SanPedroDeMacoris", urban_areas, predictions_folder_path, padding=0.1, zoom=15)
create_city_map("SanFranciscoDeMacoris", urban_areas, predictions_folder_path, padding=0.1, zoom=15)


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
