import geopandas as gpd
import pandas as pd
import osmnx

import momepy as mm
from shapely.geometry import MultiPolygon
import libpysal

from shapely.validation import make_valid
from shapely.errors import TopologicalError

# Load the slums and buildings data
slums = gpd.read_file("../../data/0/SantoDomingo3857.geojson")
buildings = gpd.read_file('../../data/0/overture/santodomingo_buildings.geojson')

# Check CRS and reproject if necessary
if slums.crs.is_geographic:
    slums = slums.to_crs(epsg=3857)  # Web Mercator projection
if buildings.crs.is_geographic:
    buildings = buildings.to_crs(epsg=3857)  # Web Mercator projection

# Ensure both layers are in the same CRS
buildings = buildings.to_crs(slums.crs)

# Merge all slum polygons into one
all_slums = slums.unary_union

# Clip buildings to the slums area
slum_buildings = buildings[buildings.geometry.within(all_slums)]

def calculate_city_morphometrics(slum_buildings):
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
        car_md = car.median()
        car_std = car.std()

        buildings_wall = mm.PerimeterWall(slum_buildings, verbose=False).series
        buildings_wall_md = buildings_wall.median()
        buildings_wall_std = buildings_wall.std()

        # Weight matrix based on contiguity for building adjacency
        W1 = libpysal.weights.Queen.from_dataframe(tessellation, ids="uID", use_index=False)
        W3 = mm.sw_high(k=3, weights=W1)

        buildings_adjacency = mm.BuildingAdjacency(slum_buildings, W3, "uID", verbose=False).series
        buildings_adjacency_md = buildings_adjacency.median()
        buildings_adjacency_std = buildings_adjacency.std()

        buildings_neighbour_distance = mm.NeighborDistance(slum_buildings, W1, 'uID', verbose=False).series
        buildings_neighbour_distance_md = buildings_neighbour_distance.median()
        buildings_neighbour_distance_std = buildings_neighbour_distance.std()

        return {
            "tessellation_car_md": car_md,
            "tessellation_car_std": car_std,
            "buildings_wall_md": buildings_wall_md,
            "buildings_wall_std": buildings_wall_std,
            "buildings_adjacency_md": buildings_adjacency_md,
            "buildings_adjacency_std": buildings_adjacency_std,
            "buildings_neighbour_distance_md": buildings_neighbour_distance_md,
            "buildings_neighbour_distance_std": buildings_neighbour_distance_std
        }
    except (TopologicalError, GEOSException) as e:
        print(f"Error in calculating morphometrics: {str(e)}")
        return None
    
# Calculate morphometrics for the entire slum area
print("Calculating morphometrics for all slums in the city...")
city_metrics = calculate_city_morphometrics(slum_buildings)

if city_metrics is not None:
    # Create a DataFrame with the results
    results_df = pd.DataFrame([city_metrics])
    
    # Add a city identifier
    results_df['city'] = 'Santo Domingo'
    
    # Save the results
    # results_df.to_csv("santo_domingo_slum_morphometrics.csv", index=False)
    print("Results saved to santo_domingo_slum_morphometrics.csv")
else:
    print("Failed to calculate city metrics due to geometry errors.")
    
results_df.head()