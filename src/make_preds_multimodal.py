import os
import torch
import geopandas as gpd
from tqdm import tqdm
from pyproj import Transformer
import rasterio
import pandas as pd
from rastervision.core.box import Box
from rastervision.core.data import ClassConfig, RasterioCRSTransformer
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore
from affine import Affine
from pystac_client import Client
import glob

import sys
import torch
from rastervision.core.data.label import SemanticSegmentationLabels

import numpy as np
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
import matplotlib.pyplot as plt
import os

from typing import Iterator, Optional

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from src.features.dataloaders import (
    query_buildings_data, CustomGeoJSONVectorSource,MergeDataset,
    FixedStatsTransformer,
    show_windows, CustomRasterizedSource
)
from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot,
                                          CustomVectorOutputConfig, FeatureMapVisualization)
from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple, MultiInputCrossValidator, create_sentinel_mosaic,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn, get_sentinel_items,
                                  MergeDataset, show_single_tile_multi, get_label_source_from_merge_dataset, create_scenes_for_city, PolygonWindowGeoDataset)

from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource, CRSTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from typing import Any, Optional, Tuple, Union, Sequence
from rastervision.core.data.raster_source import RasterSource

from typing import Tuple, Optional
import numpy as np
import torch
from rastervision.core.box import Box
from rastervision.pytorch_learner.dataset.transform import TransformType
from rastervision.core.data import Scene
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt


URL = 'https://earth-search.aws.element84.com/v1'
catalog = Client.open(URL)

def get_sentinel_items(bbox_geometry, bbox):
    items = catalog.search(
        intersects=bbox_geometry,
        collections=['sentinel-2-c1-l2a'],
        datetime='2024-01-01/2024-07-16',
        query={'eo:cloud_cover': {'lt': 15}},
        # max_items=1 to get multiple items
    ).item_collection()
    
    if not items:
        print("No items found for this area.")
        return None
    return items

def create_sentinel_mosaic(items, bbox):
    # Create the initial XarraySource with temporal=True
    sentinel_source = XarraySource.from_stac(
        items,
        bbox_map_coords=tuple(bbox),
        stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
        allow_streaming=True,
        temporal=True
    )
    
    print(f"Initial data shape: {sentinel_source.data_array.shape}")
    print(f"Initial bbox: {sentinel_source.bbox}")
    print(f"Data coordinates: {sentinel_source.data_array.coords}")
    
    crs_transformer = sentinel_source.crs_transformer
    
    # Get the CRS of the data
    data_crs = sentinel_source.crs_transformer.image_crs
    
    # Use the original bbox of the data
    data_bbox = sentinel_source.bbox
    
    print(f"Data bbox: {data_bbox}")
    
    # Apply mosaic function to combine the temporal dimension
    mosaic_data = sentinel_source.data_array.median(dim='time')
    
    print(f"Mosaic data shape: {mosaic_data.shape}")
    print(f"Mosaic data coordinates: {mosaic_data.coords}")
    
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    # Create the final normalized XarraySource
    normalized_mosaic_source = XarraySource(
        mosaic_data,
        crs_transformer=sentinel_source.crs_transformer,
        bbox=data_bbox,
        channel_order=[3,2,1,0],
        temporal=False,
        raster_transformers=[fixed_stats_transformer]
    )
    
    print(f"Created normalized mosaic of size {normalized_mosaic_source.shape}, and dtype: {normalized_mosaic_source.dtype}")
    print(f"Normalized mosaic bbox: {normalized_mosaic_source.bbox}")
    print(f"Data CRS: {data_crs}")
    # sentinel_source_SD, crs_transformer = create_sentinel_mosaic(items, bbox)

    return normalized_mosaic_source, crs_transformer

def display_mosaic(mosaic_source):
    print(f"Mosaic source shape: {mosaic_source.shape}")
    
    if mosaic_source.shape[0] == 0 or mosaic_source.shape[1] == 0:
        print("The mosaic has zero width or height. There might be no data for the specified region.")
        return
    
    chip = mosaic_source[:, :, [1, 2, 3]]  # RGB channels
    
    print(f"Chip shape: {chip.shape}")
    print(f"Chip min: {np.min(chip)}, max: {np.max(chip)}")
    
    # For normalized data, we might want to clip to a reasonable range
    vmin, vmax = -3, 3
    normalized_chip = np.clip(chip, vmin, vmax)
    
    # Scale to [0, 1] for display
    normalized_chip = (normalized_chip - vmin) / (vmax - vmin)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(normalized_chip, interpolation='nearest', aspect='equal')
    ax.set_title("Normalized Mosaic RGB")
    plt.colorbar(im, ax=ax)
    plt.show()
    
def make_buildings_raster(image_path, resolution=5):
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        xmin3857, ymin3857, xmax3857, ymax3857 = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(resolution, 0, xmin3857, 0, -resolution, ymax3857)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xmin4326, ymin4326 = transformer.transform(xmin3857, ymin3857)
    xmax4326, ymax4326 = transformer.transform(xmax3857, ymax3857)
    
    buildings = query_buildings_data(xmin4326, ymin4326, xmax4326, ymax4326)
    print(f"Buildings data loaded successfully with {len(buildings)} total buildings.")
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = CustomRasterizedSource(
        buildings_vector_source,
        background_class_id=0,
        bbox=Box(xmin3857, ymin3857, xmax3857, ymax3857))
    
    return rasterized_buildings_source, crs_transformer_buildings

def make_sentinel_raster(image_uri):
    # Define the means and stds in NIR-RGB order
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    # Define a normalized raster source using the calculated transformer
    sentinel_source_normalized = RasterioSource(
        image_uri,
        allow_streaming=True,
        raster_transformers=[fixed_stats_transformer],
        channel_order=[3, 2, 1, 0]
    )

    print(f"Loaded Sentinel data of size {sentinel_source_normalized.shape}, and dtype: {sentinel_source_normalized.dtype}")
    return sentinel_source_normalized

def create_sentinel_scene(city_data):
    image_path = city_data['image_path']

    sentinel_source_normalized = make_sentinel_raster(image_path)
    
    sentinel_scene = Scene(
        id='scene_sentinel',
        raster_source=sentinel_source_normalized
    )
    return sentinel_scene

def calculate_metrics(pred_labels, gt_labels):
    # Convert predicted labels to discrete format
    pred_labels_discrete = SemanticSegmentationDiscreteLabels.make_empty(
        extent=pred_labels.extent,
        num_classes=len(class_config))
    scores = pred_labels.get_score_arr(pred_labels.extent)
    pred_array_discrete = (scores > 0.5).astype(int)
    pred_labels_discrete[pred_labels.extent] = pred_array_discrete[1]

    # Evaluate predictions
    evaluator = SemanticSegmentationEvaluator(class_config)
    evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels_discrete)
    inf_eval = evaluation.class_to_eval_item[1]

    return {
        'f1': inf_eval.f1,
        'precision': inf_eval.precision,
        'recall': inf_eval.recall
    }

def make_predictions(model, dataset, device):
    predictions_iterator = MultiResPredictionsIterator(model, dataset, device=device)
    windows, predictions = zip(*predictions_iterator)
    return windows, predictions

def load_multimodal_model(model_path):
    model = MultiResolutionDeepLabV3()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    return model

def make_predictions(model, dataset, device):
    predictions_iterator = MultiResPredictionsIterator(model, dataset, device=device)
    windows, predictions = zip(*predictions_iterator)
    return windows, predictions

def average_predictions(pred1, pred2):
    return [(p1 + p2) / 2 for p1, p2 in zip(pred1, pred2)]

device = torch.device('mps')

class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')
    
model_paths = [
    os.path.join(parent_dir, 'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/multimodal_selSJ_cv0_epoch=19-val_loss=0.2787.ckpt'),
    os.path.join(parent_dir, 'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/multimodal_selSJ_cv1_epoch=27-val_loss=0.2128.ckpt')
]

models = [load_multimodal_model(path) for path in model_paths]

##########################################
#### PREDICTIONS FOR SICA URBAN AREAS ####
##########################################
sica_cities = os.path.join(parent_dir, 'data/1/urban_boundaries/bboxes_SICA_urban_boundaries.geojson')
gdf = gpd.read_file(sica_cities)
gdf = gdf.to_crs('EPSG:3857')
# gdf = gdf[gdf['city_name'].isin(['Leon'])] #'Managua', 'SantoDomingo', 'Tegucigalpa', 'GuatemalaCity', 'PanamaCity'])]
model_name_directory = 'selSJ_DLV3'
gdf.explore()


for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_name']
    country = row['country']
    print("Doing predictions for: ", city_name, country)
    
    xmin3857, ymin3857, xmax3857, ymax3857 = row['geometry'].bounds
    
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xmin_4326, ymin_4326 = transformer.transform(xmin3857, ymin3857)
    xmax_4326, ymax_4326 = transformer.transform(xmax3857, ymax3857)
    bbox_4326 = Box(ymin=ymin_4326, xmin=xmin_4326, ymax=ymax_4326, xmax=xmax_4326)

    # Find the image file with a pattern match
    image_pattern = os.path.join(grandparent_dir, f"data/1/urban_boundaries/sentinel/{country}_{city_name}*")
    matching_files = glob.glob(image_pattern)
    
    if not matching_files:
        print(f"No matching image file found for {country}_{city_name}. Skipping.")
        continue
    image_uri = matching_files[0]
    
    crs_transformer = RasterioCRSTransformer.from_uri(image_uri)

    # Define the means and stds in NIR-RGB order    
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    sentinel_source_normalized = RasterioSource(
        image_uri,
        allow_streaming=True,
        raster_transformers=[fixed_stats_transformer],
        channel_order=[3, 2, 1, 0]
    )
    
    sentinel_scene = Scene(
        id=f'{city_name}_sentinel',
        raster_source=sentinel_source_normalized,
        label_source=None
    )

    sentinel_extent = sentinel_scene.extent
    sentinel_bbox = sentinel_scene.bbox    
    print(f"Sentinel scene extent: {sentinel_extent}")
    
    chip = sentinel_source_normalized[:, :, :3]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip)
    plt.show()
        
    # Query buildings data
    buildings = query_buildings_data(xmin_4326, ymin_4326, xmax_4326, ymax_4326)
    print(f"Got buildings for {city_name}, {country}, in the amount of {len(buildings)}.")    
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(5, 0, xmin3857, 0, -5, ymax3857)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source,
        background_class_id=0)  
    
    chip = rasterized_buildings_source[:, :, :]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip, cmap='gray')
    plt.show()
        
    building_scene = Scene(
        id=f'{city_name}_sentinel',
        raster_source=rasterized_buildings_source,
        label_source=None
    )
    print(f"Buildings scene extent: {building_scene.extent}")

    # Create datasets
    sent_strided_fullds = CustomSlidingWindowGeoDataset(sentinel_scene, size=256, stride=128, padding=0, city=city_name, transform=None, transform_type=TransformType.noop)
    buil_strided_fullds = CustomSlidingWindowGeoDataset(building_scene, size=512, stride=256, padding=0, city=city_name, transform=None, transform_type=TransformType.noop)
    print(f"Number of samples in sen: {len(sent_strided_fullds)} and buil: {len(buil_strided_fullds)}")
    if len(sent_strided_fullds) != len(buil_strided_fullds):
        print(f"Dataset lengths don't match for {city_name}, {country}. Skipping to next row.")
        continue
    
    mergedds = MergeDataset(sent_strided_fullds, buil_strided_fullds)
    print(f"Number of samples in mergedds: {len(mergedds)}")

    # Create prediction iterator for both models
    all_predictions = []
    for model in models:
        windows, predictions = make_predictions(model, mergedds, device)
        all_predictions.append(predictions)

    # Aggregate predictions (e.g., by averaging)
    aggregated_predictions = average_predictions(*all_predictions)

    # Create SemanticSegmentationLabels from aggregated predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        aggregated_predictions,
        extent=sentinel_scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    # Save predictions
    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5
    )

    crs_transformer_buil = sentinel_scene.raster_source.crs_transformer
    buildings3857 = buildings.to_crs('EPSG:3857')
    xmin3857b, _, _, ymax3857b = buildings3857.total_bounds
    
    affine_transform_buildings = Affine(10, 0, xmin3857b, 0, -10, ymax3857b)
    crs_transformer_buil.transform = affine_transform_buildings
    
    output_dir = os.path.join(grandparent_dir, f"data/1/SICA_final_predictions/{model_name_directory}/{country}")
    os.makedirs(output_dir, exist_ok=True)

    pred_label_store = SemanticSegmentationLabelStore(
        uri=os.path.join(output_dir, f'{city_name}_{country}.geojson'),
        crs_transformer=crs_transformer_buil,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True
    )

    pred_label_store.save(pred_labels)
    print(f"Saved aggregated predictions data for {city_name}, {country}")

#### STAC Main loop
# gdf=gdf.head(2)
for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_name']
    country_code = row['country']
    xmin4326, ymin4326, xmax4326, ymax4326 = row.geometry.bounds

    # Create bbox and bbox_geometry for PySTAC query
    bbox = Box(ymin=ymin4326, xmin=xmin4326, ymax=ymax4326, xmax=xmax4326)
    bbox_geometry = {
        'type': 'Polygon',
        'coordinates': [
            [
                (xmin4326, ymin4326),
                (xmin4326, ymax4326),
                (xmax4326, ymax4326),
                (xmax4326, ymin4326),
                (xmin4326, ymin4326)
            ]
        ]
    }

    # Get Sentinel data
    items = get_sentinel_items(bbox_geometry, bbox)
    if items is None:
        print(f"No Sentinel data found for {city_name}. Skipping.")
        continue

    # Create Sentinel mosaic
    sentinel_source, crs_transformer = create_sentinel_mosaic(items, bbox)
    print(f"Created Sentinel mosaic for {city_name}, {country_code}.")
    display_mosaic(sentinel_source)

    # Query buildings data
    buildings = query_buildings_data(xmin4326, ymin4326, xmax4326, ymax4326)
    print(f"Got buildings for {city_name}, {country_code}, in the amoint of {len(buildings)}.")
    
    # Create scenes
    sentinel_scene = Scene(
        id=f'{city_name}_sentinel',
        raster_source=sentinel_source,
        label_source=None  # No labels for prediction
    )
    
    buildings_scene = create_building_scene(buildings, bbox, crs_transformer)

    # Create datasets
    sentinel_dataset = CustomSlidingWindowGeoDataset(sentinel_scene, city=city_name, size=256, stride=128, out_size=256, padding=256, transform_type=TransformType.noop, transform=None)
    buildings_dataset = CustomSlidingWindowGeoDataset(buildings_scene, city=city_name, size=512, stride=256, out_size=512, padding=512, transform_type=TransformType.noop, transform=None)

    # Create merged dataset
    merged_dataset = MergeDataset(sentinel_dataset, buildings_dataset)

    # Make predictions with both models and average
    windows = None
    avg_predictions = None

    for model in models:
        windows, predictions = make_predictions(model, merged_dataset, device)
        if avg_predictions is None:
            avg_predictions = predictions
        else:
            avg_predictions = average_predictions(avg_predictions, predictions)

    # Create SemanticSegmentationLabels from averaged predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        avg_predictions,
        extent=sentinel_scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    # Save predictions
    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5
    )

    output_dir = f'../../vectorised_model_predictions/other_cities/{country_code}'
    os.makedirs(output_dir, exist_ok=True)

    pred_label_store = SemanticSegmentationLabelStore(
        uri=os.path.join(output_dir, f'{city_name}_{country_code}.geojson'),
        crs_transformer=crs_transformer,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True
    )

    pred_label_store.save(pred_labels)
    print(f"Saved predictions data for {city_name}, {country_code}")

print("Finished processing all cities.")

def merge_geojson_files(country_directory, output_file):
    # Create an empty GeoDataFrame with an appropriate schema
    merged_gdf = gpd.GeoDataFrame()
    
    # Traverse the directory structure
    for city in os.listdir(country_directory):
        city_path = os.path.join(country_directory, city)
        vector_output_path = os.path.join(city_path, 'vector_output')
        
        if os.path.isdir(vector_output_path):
            # Find the .json file in the vector_output directory
            for file in os.listdir(vector_output_path):
                if file.endswith('.json'):
                    file_path = os.path.join(vector_output_path, file)
                    # Load the GeoJSON file into a GeoDataFrame
                    gdf = gpd.read_file(file_path)
                    # Add the city name as an attribute to each feature
                    gdf['city'] = city
                    # Append to the merged GeoDataFrame
                    merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
    
    # Save the merged GeoDataFrame to a GeoJSON file
    merged_gdf.to_file(output_file, driver='GeoJSON')
    print(f'Merged GeoJSON file saved to {output_file}')

countries = [
    "Nicaragua",
    "Costa Rica",
    "El Salvador",
    "Dominican Republic",
    "Guatemala",
    "Honduras",
    "Panama"
]

# Base directory
base_directory = os.path.join(grandparent_dir, f'data/1/SICA_final_predictions/{model_name_directory}')

# Aggregate predictions for each country
for country in countries:
    country_directory = os.path.join(base_directory, country)
    output_file = os.path.join(country_directory, 'averaged_predictions.geojson')
    
    print(f"Processing {country}...")
    merge_geojson_files(country_directory, output_file)
    print(f"Finished processing {country}. Output saved to {output_file}")

print("All countries processed.")

# Aggregate predictions for all countries
gdfs = []
for country in countries:
    file_path = os.path.join(base_directory, country, "averaged_predictions.geojson")
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the GeoJSON file
        gdf = gpd.read_file(file_path)
        
        # Add a column to identify the country
        gdf['country'] = country
        
        # Append to the list of GeoDataFrames
        gdfs.append(gdf)
    else:
        print(f"Warning: {file_path} not found")

merged_gdf = pd.concat(gdfs, ignore_index=True)
output_file = os.path.join(base_directory, "merged_predictions.geojson")
merged_gdf.to_file(output_file, driver="GeoJSON")

print(f"Merged GeoJSON saved as {output_file}")