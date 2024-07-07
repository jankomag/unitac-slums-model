import geopandas as gpd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

def query_buildings_data(xmin, ymin, xmax, ymax):
    import duckdb
    import pandas as pd
    import geopandas as gpd
    import os

    con = duckdb.connect(os.path.join(grandparent_dir, 'data/0/data.db'))
    con.install_extension('httpfs')
    con.install_extension('spatial')
    con.load_extension('httpfs')
    con.load_extension('spatial')
    con.execute("SET s3_region='us-west-2'")
    con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

    query = f"""
        WITH source_data AS (
            SELECT 
                id,
                geometry,
                UNNEST(sources) AS source,
                COUNT(*) OVER () AS total_buildings
            FROM buildings
            WHERE bbox.xmin > {xmin}
              AND bbox.xmax < {xmax}
              AND bbox.ymin > {ymin}
              AND bbox.ymax < {ymax}
        )
        SELECT 
            source.dataset AS source_name,
            COUNT(*) AS source_count,
            COUNT(*) * 100.0 / MAX(total_buildings) AS percentage
        FROM source_data
        GROUP BY source.dataset
        ORDER BY percentage DESC
    """

    source_stats = pd.read_sql(query, con=con)
    
    buildings_query = f"""
        SELECT id, geometry
        FROM buildings
        WHERE bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax}
    """
    
    buildings_df = pd.read_sql(buildings_query, con=con)
    
    buildings = gpd.GeoDataFrame()
    if not buildings_df.empty:
        buildings = gpd.GeoDataFrame(buildings_df, geometry=gpd.GeoSeries.from_wkb(buildings_df.geometry.apply(bytes)), crs='EPSG:4326')
        buildings = buildings[['id', 'geometry']]
        buildings = buildings.to_crs("EPSG:3857")
        buildings['class_id'] = 1

    return buildings, source_stats

def get_shapefile_extent(shp_path):
    # Read the shapefile
    gdf = gpd.read_file(shp_path)
    
    # Check the current CRS
    if gdf.crs is None:
        print(f"Warning: Shapefile {shp_path} has no defined CRS. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs != "EPSG:4326":
        print(f"Reprojecting {shp_path} from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
    
    # Get the total bounds
    bounds = gdf.total_bounds
    return bounds[0], bounds[1], bounds[2], bounds[3]

# Process all cities
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
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Guatemala_PS.shp'),
        'use_augmentation': False
    },
    'Managua': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/NIC_Tipitapa_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Managua_PS.shp'),
        'use_augmentation': False
    },
    'Panama': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/PAN_San_Miguelito_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/Panama_PS.shp'),
        'use_augmentation': False
    },
    'SanSalvador_PS': {
        'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/SLV_Delgado_2023.tif'),
        'labels_path': os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_lotifi_ilegal.shp'),
        'use_augmentation': False
    },
    'BelizeCity': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/_2023.tif'), 'labels_path': os.path.join(grandparent_dir, 'data/SHP/BelizeCity_PS.shp')},
    'Belmopan': {'image_path': os.path.join(grandparent_dir, 'data/0/sentinel_Gee/_2023.tif'), 'labels_path': os.path.join(grandparent_dir, 'data/SHP/Belmopan_PS.shp')}
}

city_stats = {}

for city, city_data in cities.items():
    xmin, ymin, xmax, ymax = get_shapefile_extent(city_data['labels_path'])
    buildings, source_stats = query_buildings_data(xmin, ymin, xmax, ymax)
    city_stats[city] = source_stats

# Print results
for city, stats in city_stats.items():
    print(f"\nSource statistics for {city}:")
    print(stats)
    
import matplotlib.pyplot as plt
import pandas as pd

def create_stacked_chart(city_stats, output_file='building_sources_by_city.png', dpi=300):
    # City name mapping
    city_name_map = {
        'SanSalvador_PS': 'San Salvador, El Salvador',
        'SantoDomingoDOM': 'Santo Domingo, Dominican Republic',
        'GuatemalaCity': 'Guatemala City, Guatemala',
        'TegucigalpaHND': 'Tegucigalpa, Honduras',
        'SanJoseCRI': 'San Jose, Costa Rica',
        'Panama': 'Panama City, Panama',
        'BelizeCity': 'Belize City, Belize',
        'Managua': 'Managua, Nicaragua',
        'Belmopan': 'Belmopan, Belize'
    }

    # Prepare data
    data = []
    for city, stats in city_stats.items():
        city_data = {row['source_name']: row['percentage'] for _, row in stats.iterrows()}
        data.append({
            'City': city_name_map.get(city, city),  # Use mapped name if available
            'OpenStreetMap': city_data.get('OpenStreetMap', 0),
            'Google Open Buildings': city_data.get('Google Open Buildings', 0),
            'Microsoft ML Buildings': city_data.get('Microsoft ML Buildings', 0)
        })
    
    # Convert to DataFrame and sort by Google Open Buildings percentage
    df = pd.DataFrame(data)
    df = df.sort_values('Google Open Buildings', ascending=False)

    # Set up the plot
    plt.figure(figsize=(14, 18))
    # plt.style.use('seaborn-darkgrid')
    # Use a nice font
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Create the stacked bar chart with specified colors
    ax = df.plot(x='City', 
                 y=['OpenStreetMap', 'Google Open Buildings', 'Microsoft ML Buildings'],
                 kind='bar', 
                 stacked=True, 
                 width=0.8,
                 color=['green', 'orange', 'blue'])

    # Customize the chart
    plt.title('Building Sources by City', fontsize=16, pad=20, fontweight='semibold')
    plt.xlabel('Cities', fontsize=12, labelpad=10)
    plt.ylabel('Percentage', fontsize=12, labelpad=10)
    
    # Adjust legend
    plt.legend(title='Source', title_fontsize='10', fontsize='8', loc='upper left', 
               bbox_to_anchor=(1, 1), frameon=True, edgecolor='black')

    plt.xticks(rotation=55, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

# Assuming city_stats is your dictionary with the results
create_stacked_chart(city_stats, '../../../building_sources_chart.png', dpi=300)
 
# Function to plot a single city
def plot_city(ax, city_name, labels_path):
    # Read the shapefile
    gdf = gpd.read_file(labels_path)
    
    # Reproject to Web Mercator (EPSG:3857) for compatibility with contextily
    gdf = gdf.to_crs(epsg=3857)
    
    # Plot the labels
    gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.5)
    
    # Add satellite imagery basemap
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    
    # Set the extent to the bounding box of the data
    ax.set_xlim(gdf.total_bounds[[0, 2]])
    ax.set_ylim(gdf.total_bounds[[1, 3]])
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(city_name)

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()

# Plot each city
for i, (city_name, city_data) in enumerate(cities.items()):
    plot_city(axes[i], city_name, city_data['labels_path'])

# Adjust layout and display
plt.tight_layout()
plt.show()