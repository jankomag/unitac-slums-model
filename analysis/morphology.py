import os
import sys
import geopandas as gpd
import pandas as pd
import momepy as mm
import osmnx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
import libpysal
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# sys.path.append(parent_dir)

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

    return buildings

cities = {
    'SanJoseCRI': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/CRI_San_Jose_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanJose_PS.shp'),
        'use_augmentation': False
    },
    'TegucigalpaHND': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/HND_Comayaguela_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Tegucigalpa_PS.shp'),
        'use_augmentation': False
    },
    'SantoDomingoDOM': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/0/SantoDomingo3857_buffered.geojson'),
        'use_augmentation': True
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
    'SanSalvador_PS': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/SLV_SanSalvador_2024.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
        'use_augmentation': False
    },
    'BelizeCity': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
                   'labels_path': os.path.join(grandparent_dir, 'data/SHP/BelizeCity_PS.shp')},
    'Belmopan': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
                 'labels_path': os.path.join(grandparent_dir, 'data/SHP/Belmopan_PS.shp')}
}

# Function to check if a file exists
def file_exists(file_path):
    return os.path.isfile(file_path)

# Check if all files exist
missing_files = []
for city_name, city_data in cities.items():
    if not file_exists(city_data['labels_path']):
        missing_files.append(f"{city_name}: {city_data['labels_path']}")

# If there are missing files, print them and exit
if missing_files:
    print("The following files are missing:")
    for file in missing_files:
        print(file)
    print("Please ensure all files exist before running the analysis.")

print("All files exist. Proceeding with the analysis.")

def calculate_city_morphometrics(slum_buildings, all_slums):
    # Function to check and fix geometry
    def check_and_fix_geometry(geom):
        if not geom.is_valid:
            try:
                return make_valid(geom)
            except:
                return None
        return geom
    
    # Apply geometry check and fix
    slum_buildings['geometry'] = slum_buildings['geometry'].apply(check_and_fix_geometry)
    
    # Remove any None geometries (invalid and unfixable)
    slum_buildings = slum_buildings.dropna(subset=['geometry'])
    
    # Ensure buildings are valid geometries
    slum_buildings['geometry'] = slum_buildings['geometry'].buffer(0)
    slum_buildings['geometry'] = slum_buildings['geometry'].apply(lambda geom: MultiPolygon([geom]) if geom.geom_type == 'Polygon' else geom)
    
    # Reset index and add unique ID
    slum_buildings = slum_buildings.reset_index(drop=True)
    slum_buildings['uID'] = range(len(slum_buildings))
    
    slum_buildings.to_crs(epsg=3857, inplace=True)
    
    try:
        # Calculate tessellation
        tessellation = mm.Tessellation(slum_buildings, "uID", limit=all_slums).tessellation
        tessellation['cell_area'] = tessellation.area
        
        # Calculate metrics
        car = mm.AreaRatio(tessellation, slum_buildings, "cell_area", slum_buildings.area, "uID").series
        
        # PerimeterWall calculation
        def calculate_perimeter(geom):
            if geom.geom_type == 'Polygon':
                return geom.exterior.length
            elif geom.geom_type == 'MultiPolygon':
                return sum(poly.exterior.length for poly in geom.geoms)
            else:
                return 0

        buildings_wall = slum_buildings.geometry.apply(calculate_perimeter)
        
        # Weight matrix based on contiguity for building adjacency
        W1 = libpysal.weights.Queen.from_dataframe(tessellation, ids="uID", use_index=False)
        W3 = mm.sw_high(k=3, weights=W1)
        
        buildings_adjacency = mm.BuildingAdjacency(slum_buildings, W3, "uID", verbose=False).series
        
        buildings_neighbour_distance = mm.NeighborDistance(slum_buildings, W1, 'uID', verbose=False).series
        
        # Create a DataFrame with all individual values
        metrics_df = pd.DataFrame({
            "tessellation_car": car,
            "buildings_wall": buildings_wall,
            "buildings_adjacency": buildings_adjacency,
            "buildings_neighbour_distance": buildings_neighbour_distance
        })
        
        return metrics_df
    
    except (TopologicalError, GEOSException) as e:
        print(f"Error in calculating morphometrics: {str(e)}")
        return None

# Dictionary to store DataFrames for each city
city_dataframes = {}

# Iterate through each city
for city_name, city_data in cities.items():
    print(f"Processing {city_name}...")
    
    # Load the slums data
    slums = gpd.read_file(city_data['labels_path'])
    slums = slums.to_crs(epsg=4326)  # Ensure it's in WGS84
    xmin, ymin, xmax, ymax = slums.total_bounds

    # Query buildings data
    buildings = query_buildings_data(xmin, ymin, xmax, ymax)

    # Reproject to Web Mercator
    slums = slums.to_crs(epsg=3857)
    buildings = buildings.to_crs(epsg=3857)
    print(f"Loaded {len(buildings)} buildings and {len(slums)} slum polygons for {city_name}.")

    # Merge all slum polygons into one
    all_slums = slums.unary_union

    print(f"Area of slums in {city_name}: {all_slums.area:.2f} square meters")
    
    # Clip buildings to the slums area
    slum_buildings = gpd.sjoin(buildings, slums, how="inner", predicate="intersects")
    print(f"Found {len(slum_buildings)} buildings within slum areas in city {city_name}.")
    
    # Calculate morphometrics
    city_metrics_df = calculate_city_morphometrics(slum_buildings, all_slums)

    if city_metrics_df is not None:
        # Add city name to the DataFrame
        city_metrics_df['city'] = city_name
        city_dataframes[city_name] = city_metrics_df
    else:
        print(f"Failed to calculate metrics for {city_name}")

# Save each city's DataFrame to a separate CSV file
for city_name, df in city_dataframes.items():
    df.to_csv(f"metrics/{city_name}_slum_morphometrics.csv", index=False)
    print(f"Results for {city_name} saved to {city_name}_slum_morphometrics.csv")

all_cities_df = pd.concat(city_dataframes.values(), ignore_index=True)
all_cities_df.to_csv("all_cities_slum_morphometrics.csv", index=False)
print("Combined results for all cities saved to all_cities_slum_morphometrics.csv")

# Display summary statistics for each city
for city_name, df in city_dataframes.items():
    print(f"\nSummary statistics for {city_name}:")
    print(df.describe())

# Plotting the distribution of morphometrics by city
file_path = os.path.join(parent_dir, 'analysis/metrics/all_cities_slum_morphometrics.csv')
all_cities_df = pd.read_csv(file_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# Set the style for a cleaner look
plt.style.use('ggplot')

# Load data
file_path = os.path.join(parent_dir, 'analysis/metrics/all_cities_slum_morphometrics.csv')
all_cities_df = pd.read_csv(file_path)

# Clean up city names
city_name_map = {
    'Sansalvador_Ps': 'San Salvador, El Salvador',
    'Santodomingodom': 'Santo Domingo, Dominican Republic',
    'Guatemalacity': 'Guatemala City, Guatemala',
    'Tegucigalpahnd': 'Tegucigalpa, Honduras',
    'Sanjosecri': 'San Jose, Costa Rica',
    'Panama': 'Panama City, Panama',
    'Belizecity': 'Belize City, Belize',
    'Managua': 'Managua, Nicaragua',
    'Belmopan': 'Belmopan, Belize'
}

all_cities_df['city'] = all_cities_df['city'].map(city_name_map)

# Define numeric columns
numeric_cols = ['tessellation_car', 'buildings_wall', 'buildings_adjacency', 'buildings_neighbour_distance']

# Standardize the numeric columns
scaler = MinMaxScaler()
all_cities_df[numeric_cols] = scaler.fit_transform(all_cities_df[numeric_cols])

# Melt the dataframe for seaborn
df_melted = all_cities_df.melt(id_vars=['city'], value_vars=numeric_cols, var_name='Variable', value_name='Value')

# Create the plot
plt.figure(figsize=(20, 12))

# Create the boxplot with limited outliers
ax = sns.boxplot(x='Variable', y='Value', hue='city', data=df_melted, 
                 palette="Set3", whis=(10, 90))

# Color the outliers the same as their respective bars
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    for j in range(i*6, i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# Customize the plot
plt.title('Standardized Distribution of Building Morphometrics by City', fontsize=20, pad=20)
plt.xlabel('Morphometric', fontsize=16, labelpad=15)
plt.ylabel('Standardized Value', fontsize=16, labelpad=15)

# Improve x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=14)
ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()])

# Improve y-axis labels
plt.yticks(fontsize=14)

# Customize legend
plt.legend(title='City', title_fontsize='16', fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()