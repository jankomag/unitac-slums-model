import sys
import geopandas as gpd
import pandas as pd
import momepy as mm
import osmnx
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
import libpysal
from shapely.validation import make_valid
from shapely.errors import TopologicalError, GEOSException
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
# sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.data.dataloaders import query_buildings_data

cities = {
    'SanJoseCRI': {'labels_path': '../data/SHP/SanJose_PS.shp'},
    'TegucigalpaHND': {'labels_path': '../data/SHP/Tegucigalpa_PS.shp'},
    'SantoDomingoDOM': {'labels_path': '../data/SHP/SantoDomingo_PS.shp'},
    'GuatemalaCity': {'labels_path': '../data/SHP/Guatemala_PS.shp'},
    'Managua': {'labels_path': '../data/SHP/Managua_PS.shp'},
    'Panama': {'labels_path': '../data/SHP/Panama_PS.shp'},
    'SanSalvador_PS': {'labels_path': '../data/SHP/SanSalvador_PS.shp'},
    'BelizeCity': {'labels_path': '../data/SHP/BelizeCity_PS.shp'},
    'Belmopan': {'labels_path': '../data/SHP/Belmopan_PS.shp'},
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
        
        # Modified PerimeterWall calculation
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

    # Merge all slum polygons into one
    all_slums = slums.unary_union

    # Clip buildings to the slums area
    slum_buildings = gpd.sjoin(buildings, slums, how="inner", predicate="intersects")

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

# If you want to combine all cities into one large DataFrame
all_cities_df = pd.concat(city_dataframes.values(), ignore_index=True)
all_cities_df.to_csv("all_cities_slum_morphometrics.csv", index=False)
print("Combined results for all cities saved to all_cities_slum_morphometrics.csv")

# Display summary statistics for each city
for city_name, df in city_dataframes.items():
    print(f"\nSummary statistics for {city_name}:")
    print(df.describe())

# Examine boxplots of the standardized numeric columns
numeric_cols = ['tessellation_car', 'buildings_wall', 'buildings_adjacency','buildings_neighbour_distance']
df=all_cities_df
# Standardize the numeric columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Melt the dataframe for seaborn
df_melted = df.melt(id_vars=['city'], value_vars=numeric_cols, var_name='Variable', value_name='Value')

# Create the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Variable', y='Value', hue='city', data=df_melted)

# Customize the plot
plt.title('Standardized Boxplots of Numeric Columns by City')
plt.xlabel('Variable')
plt.ylabel('Standardized Value')
plt.legend(title='City')
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()