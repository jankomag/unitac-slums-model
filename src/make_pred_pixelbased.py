import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from rasterio.features import shapes
from pyproj import Transformer
import os
import duckdb

# Load trained model and selector

import joblib
model = joblib.load('trained_rf_model.joblib')
selector = joblib.load('trained_selector.joblib')

# Your existing functions
def limit_extent(data, profile, labels_gdf):
    bounds = labels_gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
    
    # Calculate new dimensions
    width = int((xmax - xmin) / profile['transform'][0])
    height = int((ymax - ymin) / -profile['transform'][4])
    
    # Create new transform
    new_transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    # Update profile
    out_profile = profile.copy()
    out_profile.update({
        "height": height,
        "width": width,
        "transform": new_transform
    })
    
    # Check if input is 2D or 3D
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    # Reproject and crop data
    out_data = np.zeros((data.shape[0], height, width), dtype=data.dtype)
    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=out_data[i],
            src_transform=profile['transform'],
            src_crs=profile['crs'],
            dst_transform=new_transform,
            dst_crs=profile['crs'],
            resampling=Resampling.bilinear
        )
    
    # If input was 2D, return 2D output
    if data.shape[0] == 1:
        out_data = out_data[0]
    
    return out_data, out_profile

def normalize_sentinel(sentinel_data):
    scaler = StandardScaler()
    shape = sentinel_data.shape
    flat_data = sentinel_data.reshape((shape[0], -1))
    normalized_flat = scaler.fit_transform(flat_data.T).T
    return normalized_flat.reshape(shape)

def resample_rasters(data, src_profile, dst_profile):
    dst_shape = (dst_profile['height'], dst_profile['width'])
    
    # Check if input is 2D or 3D
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    resampled_data = np.zeros((data.shape[0],) + dst_shape, dtype=data.dtype)
    
    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=resampled_data[i],
            src_transform=src_profile['transform'],
            src_crs=src_profile['crs'],
            dst_transform=dst_profile['transform'],
            dst_crs=dst_profile['crs'],
            resampling=Resampling.bilinear
        )
    
    # If input was 2D, return 2D output
    if data.shape[0] == 1:
        resampled_data = resampled_data[0]
    
    return resampled_data

def create_features_and_labels(sentinel_data, density_data, labels_gdf, profile):
    # Rasterize labels
    labels = rasterize(
        [(geom, 1) for geom in labels_gdf.geometry],
        out_shape=(profile['height'], profile['width']),
        transform=profile['transform'],
        fill=0,  # Background value
        dtype='uint8'
    )
    
    # Ensure sentinel_data is 3D
    if sentinel_data.ndim == 2:
        sentinel_data = sentinel_data[np.newaxis, ...]
    
    # Ensure density_data is 3D
    if density_data.ndim == 2:
        density_data = density_data[np.newaxis, ...]
    
    # Stack features
    features = np.vstack((sentinel_data, density_data))
    
    return features.reshape((-1, features.shape[0])), labels.flatten()

def predict_and_visualize(model, selector, features, profile):
    # Reshape features to 2D array
    features_2d = features.reshape((-1, features.shape[0]))

    # Apply feature selection
    features_selected = selector.transform(features_2d)

    # Make predictions
    predictions_proba = model.predict_proba(features_selected)[:, 1]
    predictions_binary = (predictions_proba > 0.5).astype(np.uint8)

    # Reshape predictions to match the original image shape
    predictions_proba_2d = predictions_proba.reshape((profile['height'], profile['width']))
    predictions_binary_2d = predictions_binary.reshape((profile['height'], profile['width']))

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    im1 = ax1.imshow(predictions_proba_2d, cmap='viridis')
    ax1.set_title('Probability Map')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(predictions_binary_2d, cmap='binary')
    ax2.set_title('Binary Predictions')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    return predictions_proba_2d, predictions_binary_2d

def vectorize_predictions(binary_map, profile):
    # Generate shapes from the binary prediction map
    shapes_generator = shapes(binary_map, mask=binary_map.astype(bool), transform=profile['transform'])
    
    # Convert shapes to GeoDataFrame
    geometries = [shape(geom) for geom, value in shapes_generator]
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=profile['crs'])
    
    return gdf

def save_to_geojson(gdf, output_path):
    gdf.to_file(output_path, driver='GeoJSON')

# Modified query_buildings_data function to return a GeoDataFrame
def query_buildings_data(con, xmin, ymin, xmax, ymax):
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
    else:
        buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

    return buildings

# New function to create building density
def create_building_density(buildings, bounds, resolution=5):
    xmin, ymin, xmax, ymax = bounds
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    # Calculate area and its reciprocal for each building
    buildings['area'] = buildings.geometry.area
    buildings['area_reciprocal'] = 100 / buildings.area

    raster = rasterize(
        [(geom, value) for geom, value in zip(buildings.geometry, buildings.area_reciprocal)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32
    )

    # Apply Gaussian filter to get density
    from scipy.ndimage import gaussian_filter
    sigma = 5 / resolution
    density = gaussian_filter(raster, sigma=sigma)

    return density, transform

# Main prediction loop
sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
gdf = gdf.to_crs('EPSG:3857')

con = duckdb.connect("../../data/0/data.db")
con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

for index, row in gdf.iterrows():
    city_name = row['city_ascii']
    country_code = row['iso3']
    print(f"Doing predictions for: {city_name}, {country_code}")
    
    image_uri = f"../../data/0/sentinel_Gee/{country_code}_{city_name}_2023.tif"
    
    if not os.path.exists(image_uri):
        print(f"Warning: File {image_uri} does not exist. Skipping to next row.")
        continue
    
    gdf_xmin, gdf_ymin, gdf_xmax, gdf_ymax = row['geometry'].bounds
    
    try:
        with rasterio.open(image_uri) as src:
            bounds = src.bounds
            raster_xmin, raster_ymin, raster_xmax, raster_ymax = bounds
            sentinel_data = src.read()
            sentinel_profile = src.profile
    except Exception as e:
        print(f"Error processing {image_uri}: {e}")
        continue
    
    # Define the common box in EPSG:3857
    common_xmin = max(gdf_xmin, raster_xmin)
    common_ymin = max(gdf_ymin, raster_ymin)
    common_xmax = min(gdf_xmax, raster_xmax)
    common_ymax = min(gdf_ymax, raster_ymax)

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    common_xmin_4326, common_ymin_4326 = transformer.transform(common_xmin, common_ymin)
    common_xmax_4326, common_ymax_4326 = transformer.transform(common_xmax, common_ymax)
    
    buildings = query_buildings_data(con, common_xmin_4326, common_ymin_4326, common_xmax_4326, common_ymax_4326)
    
    # Create building density
    density_data, density_transform = create_building_density(buildings, (common_xmin, common_ymin, common_xmax, common_ymax))
    
    # Limit extent and normalize
    sentinel_data, sentinel_profile = limit_extent(sentinel_data, sentinel_profile, row['geometry'])
    sentinel_data = normalize_sentinel(sentinel_data)
    
    # Resample density data to match Sentinel resolution
    density_profile = sentinel_profile.copy()
    density_profile.update(transform=density_transform, width=density_data.shape[1], height=density_data.shape[0])
    density_data = resample_rasters(density_data, density_profile, sentinel_profile)
    
    # Create features
    features, _ = create_features_and_labels(sentinel_data, density_data, None, sentinel_profile)
    
    # Make predictions
    prob_map, binary_map = predict_and_visualize(model, selector, features, sentinel_profile)
    
    # Vectorize predictions
    predicted_polygons = vectorize_predictions(binary_map, sentinel_profile)
    
    # Save predictions
    output_path = f'../../vectorised_model_predictions/pixel_based/{country_code}/{city_name}_{country_code}.geojson'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_to_geojson(predicted_polygons, output_path)
    
    print(f"Saved predictions for {city_name}, {country_code}")

con.close()