import os
import sys
import geopandas as gpd
import pandas as pd
# import momepy as mm
# import osmnx
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
# import libpysal
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

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
    # 'SanJoseCRI': {
    #     'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/CRI_San_Jose_2023.tif'),
    #     'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanJose_PS.shp'),
    #     'use_augmentation': False
    # },
    'Tegucigalpa': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/HND_Comayaguela_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Tegucigalpa_PS.shp'),
        'use_augmentation': False
    },
    'SantoDomingo': {
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
    }
    # 'SanSalvador_PS': {
    #     'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/SLV_SanSalvador_2024.tif'),
    #     'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
    #     'use_augmentation': False
    # },
    # 'BelizeCity': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_BelizeCity_2024.tif'),
    #                'labels_path': os.path.join(grandparent_dir, 'data/SHP/BelizeCity_PS.shp')},
    # 'Belmopan': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/BLZ_Belmopan_2024.tif'),
    #              'labels_path': os.path.join(grandparent_dir, 'data/SHP/Belmopan_PS.shp')}
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



##################
#### PLOTTING ####
##################

# distribution of morphometrics by city
file_path = os.path.join(grandparent_dir, 'slums-model-unitac/analysis/metrics/all_cities_slum_morphometrics.csv')
all_cities_df = pd.read_csv(file_path)

# plt.style.use('ggplot')
city_name_map = {
    'Sansalvador_Ps_': 'San Salvador, El Salvador',
    'SantoDomingoDOM': 'Santo Domingo, Dominican Republic',
    'GuatemalaCity': 'Guatemala City, Guatemala',
    'TegucigalpaHND': 'Tegucigalpa, Honduras',
    'SanJoseCRI_': 'San Jose, Costa Rica',
    'Panama': 'Panama City, Panama',
    'Belizecity_': 'Belize City, Belize',
    'Managua': 'Managua, Nicaragua',
    'Belmopan_': 'Belmopan, Belize'
}
all_cities_df = all_cities_df[all_cities_df['city'].isin(['SantoDomingoDOM', 'GuatemalaCity', 'TegucigalpaHND','Panama', 'Managua'])]
all_cities_df['city'] = all_cities_df['city'].map(city_name_map)

# Define numeric columns
numeric_cols = ['tessellation_car', 'buildings_wall', 'buildings_adjacency', 'buildings_neighbour_distance']
numeric_cols_names = ['Tesselation CAR', 'Buildings Perimeter Length', 'Buildings Adjecency', 'Distance to Nearest Building']
col_name_map = dict(zip(numeric_cols, numeric_cols_names))

def remove_outliers(group):
    for col in numeric_cols:
        mean = group[col].mean()
        std = group[col].std()
        group = group[(group[col] >= mean - 3*std) & (group[col] <= mean + 3*std)]
    return group

# Remove outliers for each city separately
all_cities_df = all_cities_df.groupby('city').apply(remove_outliers).reset_index(drop=True)

# Standardize the numeric columns
scaler = MinMaxScaler()
all_cities_df[numeric_cols] = scaler.fit_transform(all_cities_df[numeric_cols])

df_melted = all_cities_df.melt(id_vars=['city'], value_vars=numeric_cols, var_name='Variable', value_name='Value')
df_melted['Variable'] = df_melted['Variable'].map(col_name_map)

# Set up the font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Garamond'] + plt.rcParams['font.serif']

# Set up the style manually
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#E6E6E6'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Create the plot
plt.figure(figsize=(20, 12))
ax = sns.boxplot(x='Variable', y='Value', hue='city', data=df_melted,
                 palette="Set2", whis=(10, 90))

# Color the outliers the same as their respective bars
for i, artist in enumerate(ax.artists):
    col = artist.get_facecolor()
    for j in range(i*6, i*6+6):
        line = ax.lines[j]
        line.set_color(col)
        line.set_mfc(col)
        line.set_mec(col)

# Customize the plot
plt.title('Standardized Distributions of Building Morphometrics by City', fontsize=28, pad=20)
plt.xlabel('Morphometric', fontsize=20, labelpad=15)
plt.ylabel('Standardized Value', fontsize=20, labelpad=15)
plt.xticks(rotation=45, ha='right', fontsize=18)
ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()])
plt.yticks(fontsize=14)
plt.legend(title='City', title_fontsize='24', fontsize='17', bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.show()



#### MAPPING ####
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box, Point
import os
import duckdb
from pyproj import Transformer

def create_square_bbox(lat, lon, size_meters):
    point = Point(lon, lat)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
    half_size = size_meters / 2
    bbox = box(x - half_size, y - half_size, x + half_size, y + half_size)
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    minx, miny = transformer.transform(bbox.bounds[0], bbox.bounds[1])
    maxx, maxy = transformer.transform(bbox.bounds[2], bbox.bounds[3])
    return minx, miny, maxx, maxy

def plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir):
    xmin, ymin, xmax, ymax = create_square_bbox(lat, lon, size_meters)
    
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
    buildings_df = gpd.read_postgis(query, con, geom_col='geometry', crs='EPSG:4326')
    
    city_name_map = {
        'Sansalvador_Ps_': 'San Salvador, El Salvador',
        'SantoDomingo': 'Santo Domingo, Dominican Republic',
        'GuatemalaCity': 'Guatemala City, Guatemala',
        'Tegucigalpa': 'Tegucigalpa, Honduras',
        'SanJoseCRI_': 'San Jose, Costa Rica',
        'Panama': 'Panama City, Panama',
        'BelizeCity': 'Belize City, Belize (excluded from the study)',
        'Managua': 'Managua, Nicaragua',
        'Belmopan_': 'Belmopan, Belize'
    }
    
    cities = {
        'Tegucigalpa': os.path.join(grandparent_dir, 'data/SHP/Tegucigalpa_PS.shp'),
        'SantoDomingo': os.path.join(grandparent_dir, 'data/0/SantoDomingo3857_buffered.geojson'),
        'GuatemalaCity': os.path.join(grandparent_dir, 'data/SHP/Guatemala_PS.shp'),
        'Managua': os.path.join(grandparent_dir, 'data/SHP/Managua_PS.shp'),
        'Panama': os.path.join(grandparent_dir, 'data/SHP/Panama_PS.shp'),
        'BelizeCity': os.path.join(grandparent_dir, 'data/SHP/BelizeCity_PS.shp')
    }
        
    precarious_areas = gpd.read_file(cities[city_name])
    
    buildings_df = buildings_df.to_crs('EPSG:3857')
    precarious_areas = precarious_areas.to_crs('EPSG:3857')
    
    bbox = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]}, crs='EPSG:4326').to_crs('EPSG:3857')
    
    precarious_areas = gpd.clip(precarious_areas, bbox_gdf)
    
    unified_precarious_area = precarious_areas.unary_union
    unified_precarious_gdf = gpd.GeoDataFrame({'geometry': [unified_precarious_area]}, crs='EPSG:3857')
    
    ax.set_facecolor('white')
    
    buildings_df.plot(ax=ax, edgecolor='none', facecolor='black', linewidth=0)
    
    # Plot precarious areas with faint fill and outline
    unified_precarious_gdf.plot(ax=ax, facecolor='red', edgecolor='red', alpha=0.05, linewidth=2)
    unified_precarious_gdf.boundary.plot(ax=ax, color='red', linewidth=2)
    
    ax.set_xlim(bbox_gdf.total_bounds[0], bbox_gdf.total_bounds[2])
    ax.set_ylim(bbox_gdf.total_bounds[1], bbox_gdf.total_bounds[3])
    
    ax.set_axis_off()
    
    cleaned_city_name = city_name_map.get(city_name, city_name)
    ax.set_title(f'{cleaned_city_name}', fontsize=14)

def plot_multiple_areas(coordinates_list, size_meters, grandparent_dir):
    n = len(coordinates_list)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
    fig.suptitle('Building Footprints and Precarious Areas in Different Cities', fontsize=16)
    
    for i, (lat, lon, city_name) in enumerate(coordinates_list):
        row = i // cols
        col = i % cols
        ax = axs[row, col]
        plot_buildings_and_precarious_areas(ax, lat, lon, size_meters, city_name, grandparent_dir)
    
    # Hide any unused subplots
    for i in range(n, rows*cols):
        row = i // cols
        col = i % cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
coordinates_list = [
    (14.643041208660227, -90.52696617369627, 'GuatemalaCity'),
    (14.09833557316007, -87.24716065494343, 'Tegucigalpa'),
    (18.506793321891678, -69.89322847545206, 'SantoDomingo'),
    (12.153548471297961, -86.25461143585959, 'Managua'),
    (8.925055642079421, -79.62752568376733, 'Panama'),
    (17.503577109876268, -88.21419733167714, 'BelizeCity'),
]

size_meters = 1500

plot_multiple_areas(coordinates_list, size_meters, grandparent_dir)