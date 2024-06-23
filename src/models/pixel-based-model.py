import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio import features
from scipy.ndimage import gaussian_filter
from rasterio.mask import mask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.preprocessing import StandardScaler
from shapely.geometry import shape
from sklearn.feature_selection import SelectFromModel
from rasterio.features import shapes
from sklearn.model_selection import cross_val_score
import joblib

# Load the building footprints GeoDataFrame
buildings_uri_SD = '../../data/0/overture/santodomingo_buildings.geojson'
label_uri_SD = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

buildings = gpd.read_file(buildings_uri_SD)
buildings = buildings.to_crs(epsg=3857)

# Calculate area and its reciprocal for each building
buildings['area'] = buildings.geometry.area
buildings['area_reciprocal'] = 100 / buildings.area

# Define raster properties
resolution = 5
xmin, ymin, xmax, ymax = buildings.total_bounds
width = int((xmax - xmin) / resolution)
height = int((ymax - ymin) / resolution)
transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

# Rasterize the buildings using area_reciprocal as the burn value
raster = features.rasterize(
    [(geom, value) for geom, value in zip(buildings.geometry, buildings.area_reciprocal)],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    all_touched=True,
    dtype=np.float32
)

# Define Gaussian kernel
kernel_radius_pixels = int(5 / resolution)
sigma = 5/kernel_radius_pixels

# Apply Gaussian filter to get density
density = gaussian_filter(raster, sigma=sigma)

# Save the density raster
density_path = '../../data/1/pixelbasedmodel/SDbuilding_density.tif'
with rasterio.open(density_path, 'w', driver='GTiff',
                   height=height, width=width,
                   count=1, dtype=density.dtype,
                   crs=buildings.crs, transform=transform) as dst:
    dst.write(density, 1)

print(f"Density raster saved to {density_path}")

# Open the raster file
with rasterio.open(density_path) as src:
    raster_data = src.read(1)  # Read the first band
    bounds = src.bounds
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(raster_data, cmap='viridis', extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    
    plt.colorbar(im, ax=ax, label='Building Density')
    
    ax.set_title('Building Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()

# Print some basic information about the raster
print(f"Raster shape: {raster_data.shape}")


### Prepare data for the Random Forest model ###
with rasterio.open(image_uri) as src:
    sentinel_data = src.read()
    sentinel_profile = src.profile

# Load building density data
density_path = '../../data/1/pixelbasedmodel/SDbuilding_density.tif'
with rasterio.open(density_path) as src:
    density_data = src.read(1)
    density_profile = src.profile

# Load labels
labels_gdf = gpd.read_file(label_uri_SD)

# 2. Limit extent
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

sentinel_data, sentinel_profile = limit_extent(sentinel_data, sentinel_profile, labels_gdf)
density_data, density_profile = limit_extent(density_data, density_profile, labels_gdf)
    
# 3. Normalize Sentinel bands
def normalize_sentinel(sentinel_data):
    scaler = StandardScaler()
    shape = sentinel_data.shape
    flat_data = sentinel_data.reshape((shape[0], -1))
    normalized_flat = scaler.fit_transform(flat_data.T).T
    return normalized_flat.reshape(shape)

sentinel_data = normalize_sentinel(sentinel_data)

# 4. Resample all rasters to match resolution
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

density_data = resample_rasters(density_data, density_profile, sentinel_profile)

# 5. Create features and labels
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
    
features, labels = create_features_and_labels(sentinel_data, density_data, labels_gdf, sentinel_profile)
    
# Plot RGB bands
rgb = sentinel_data[:3,:,:].transpose(1, 2, 0)
plt.imshow(rgb)
plt.title("Sentinel RGB Bands")
plt.show()  
    
### RF MODEL ###
# Split data for training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the model
def create_and_evaluate_model(X, y):
    
    # Create model with adjusted hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        max_depth=20,
        random_state=42
    )

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")

    # Fit the model on all data
    rf_model.fit(X, y)

    return rf_model

def predict_and_visualize(model, features, profile):
    # Reshape features to 2D array
    features_2d = features.reshape((-1, features.shape[0]))

    # Make predictions
    predictions_proba = model.predict_proba(features_2d)[:, 1]
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

model = create_and_evaluate_model(features, labels)
prob_map, binary_map = predict_and_visualize(model, features, sentinel_profile)

def vectorize_predictions(binary_map, profile):
    # Generate shapes from the binary prediction map
    shapes_generator = shapes(binary_map, mask=binary_map.astype(bool), transform=profile['transform'])
    
    # Convert shapes to GeoDataFrame
    geometries = [shape(geom) for geom, value in shapes_generator]
    gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=profile['crs'])
    
    return gdf

def save_to_geojson(gdf, output_path):
    gdf.to_file(output_path, driver='GeoJSON')

predicted_polygons = vectorize_predictions(binary_map, sentinel_profile)
save_to_geojson(predicted_polygons, 'pixelbased-predicted_polygons.geojson')()

joblib.dump(model, '../../../UNITAC-trained-models/pixel-based_basemodel/trained_rf_model.joblib')