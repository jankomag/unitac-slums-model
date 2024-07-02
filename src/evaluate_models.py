import os
import sys
import torch
from affine import Affine
import geopandas as gpd
import matplotlib.pyplot as plt

from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.core.data import (ClassConfig, Scene, StatsTransformer, ClassInferenceTransformer,
                                    RasterizedSource, RasterioCRSTransformer, SemanticSegmentationLabelSource,
                                    GeoJSONVectorSource)
from rastervision.core.data.label import SemanticSegmentationLabels

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
# sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (BuildingsDeeplabv3, BuildingsOnlyPredictionsIterator, CustomGeoJSONVectorSource, MultiResSentLabelPredictionsIterator, MultiResolutionDeepLabV3)
from src.data.dataloaders import query_buildings_data, create_datasets, create_sentinel_raster_source

# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
models = {
    'deeplabv3_buildings': {
        'name': 'deeplabv3_buildings',
        'framework': 'pytorch',
        'weights_path': "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
    },
    'deeplabv3_multimodal': {
        'name': 'deeplabv3_multimodal',
        'framework': 'pytorch',
        'weights_path': ''
    }#, 'pixel_based_RF': {
    #     'framework': 'scikit_learn',
    #     'weights_path': ''
    # },
}

cities = {
    'SanJoseCRI': {
        'image_path': '../data/0/sentinel_Gee/CRI_San_Jose_2023.tif',
        'labels_path': '../data/SHP/SanJose_PS.shp'
    },
    'TegucigalpaHND': {
        'image_path': '../data/0/sentinel_Gee/HND_Comayaguela_2023.tif',
        'labels_path': '../data/SHP/Tegucigalpa_PS.shp'
    },
    'SantoDomingoDOM': {
        'image_path': '../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif',
        'labels_path': '../data/SHP/Tegucigalpa_PS.shp'
    },
    'GuatemalaCity': {
        'image_path': '../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif',
        'labels_path': '../data/SHP/Guatemala_PS.shp'
    },
    'Managua': {
        'image_path': '../data/0/sentinel_Gee/NIC_Tipitapa_2023.tif',
        'labels_path': '../data/SHP/Managua_PS.shp'
    },
    'Panama': {
        'image_path': '../data/0/sentinel_Gee/PAN_San_Miguelito_2023.tif',
        'labels_path': '../data/SHP/Panama_PS.shp'
    },
    'SanSalvador_PS': {
        'image_path': '../data/0/sentinel_Gee/SLV_Delgado_2023.tif',
        'labels_path': '../data/SHP/SanSalvador_PS.shp'
    }#,
    # 'BelizeCity': {'image_path': '../../data/0/sentinel_Gee/HND__2023.tif','labels_path': '../data/SHP/BelizeCity_PS.shp'},
    # 'Belmopan': {'image_path': '../../data/0/sentinel_Gee/HND__2023.tif','labels_path': '../data/SHP/Belmopan_PS.shp'}
}

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

# function to rasterise buildings data
def make_buildings_raster(image_path, labels_path):
    gdf = gpd.read_file(labels_path)
    gdf = gdf.to_crs('EPSG:4326')
    xmin, ymin, xmax, ymax = gdf.total_bounds

    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(5, 0, xmin, 0, -5, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    buildings = query_buildings_data(xmin, ymin, xmax, ymax)
    print(f"Buildings data loaded successfully with {len(buildings)} total buildings.")
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source,
        background_class_id=0)
    return rasterized_buildings_source

# functions to prepare data for different models
def prepare_data_for_deeplabv3_buildings(image_path, labels_path, buildings_raster):
    
    gdf = gpd.read_file(labels_path)
    gdf = gdf.to_crs('EPSG:4326')
    xmin, _, _, ymax = gdf.total_bounds

    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(5, 0, xmin, 0, -5, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    label_vector_source = GeoJSONVectorSource(labels_path,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=class_config.get_class_id('slums'))])
    
    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
    
    BuilScene = Scene(
        id='buildings',
        raster_source = buildings_raster,
        label_source = label_source)
    
    location_labels = BuilScene.label_source.get_labels()
    
    return (BuilScene, location_labels)

def prepare_data_for_deeplabv3_multimodal(image_path, labels_path, buildings_raster):
    
    ### Load Senitnel data ###
    sentinel_source_normalized, sentinel_label_raster_source = create_sentinel_raster_source(image_path, labels_path, class_config, clip_to_label_source=True)
    
    chip = sentinel_source_normalized[:, :, [0, 1, 2]]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(chip)
    plt.show()
    
    SentinelScene = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalized,
        label_source = sentinel_label_raster_source)
    
    ### Load building data ###
    gdf = gpd.read_file(labels_path)
    gdf = gdf.to_crs('EPSG:4326')
    xmin, _, _, ymax = gdf.total_bounds

    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(5, 0, xmin, 0, -5, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    # label_vector_source = GeoJSONVectorSource(labels_path,
    #     crs_transformer_buildings,
    #     vector_transformers=[ClassInferenceTransformer(default_class_id=class_config.get_class_id('slums'))])
    
    # label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    # label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
    
    BuilScene = Scene(
        id='buildings',
        raster_source = buildings_raster)
        # label_source = label_source)
    
    location_labels = SentinelScene.label_source.get_labels()
    scenes = (SentinelScene, BuilScene)
    return (scenes, location_labels)

def prepare_data_for_pixel_based_RF(image_path, labels_path):
    # Add code to prepare data for model_1
    return (data, location_labels)

# functions to make predictions
def make_predictions_buildings(model, inputs):
    model.eval()
    ds, _, _, _ = create_datasets(inputs, imgsize=288, stride = 144, padding=0, val_ratio=0.1, test_ratio=0.1, augment=False, seed=42)
    
    predictions_iterator = BuildingsOnlyPredictionsIterator(model, ds, device=device)
    windows, predictions = zip(*predictions_iterator)
    
    pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=inputs.extent,
    num_classes=len(class_config),
    smooth=True)
    
    scores = pred_labels.get_score_arr(pred_labels.extent)
    scores_building = scores[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    image = ax.imshow(scores_building)
    plt.show()

    return pred_labels

def make_predictions_multimodal(model, inputs):
    model.eval()
    sent_scene, buildings_scene = inputs
    
    buildingsGeoDataset, _, _, _ = create_datasets(buildings_scene, imgsize=512, stride = 256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    sentinelGeoDataset, _, _, _ = create_datasets(sent_scene, imgsize=256, stride = 128, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
    
    predictions_iterator = MultiResSentLabelPredictionsIterator(model, sentinelGeoDataset, buildingsGeoDataset, device=device)
    windows, predictions = zip(*predictions_iterator)
    
    pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=sent_scene.extent,
    num_classes=len(class_config),
    smooth=True)
    
    scores = pred_labels.get_score_arr(pred_labels.extent)
    scores_building = scores[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    image = ax.imshow(scores_building)
    plt.show()

    return pred_labels

# function to compute mean IOU, confusion matrix
def evaluate_model(pred_labels, labels):
    evaluator = SemanticSegmentationEvaluator(class_config)
    
    evaluation = evaluator.evaluate_predictions(
        ground_truth=labels,
        predictions=pred_labels)
    
    eval_metrics_dict = evaluation.class_to_eval_item[0]
    f1_score = eval_metrics_dict.f1
    print("F1 score: ", f1_score)
    
    return eval_metrics_dict

# function to load best model
def load_model(model_info):
    if model_info['framework'] == 'pytorch':
        
        if model_info['name'] == 'deeplabv3_multimodal':
            
            model = MultiResolutionDeepLabV3(buil_channels=64)
            checkpoint = torch.load(model_info['weights_path'])
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            
        elif model_info['name'] == 'deeplabv3_buildings':
            model = BuildingsDeeplabv3.load_from_checkpoint(model_info['weights_path'])
        
    elif model_info['framework'] == 'scikit_learn':
        model = None
        
    else:
        raise ValueError("Unsupported framework")
    
    return model

def run_evaluations(locations, models):
    results = {}
    
    # takes longest to rasterise only do it once here
    buildings_raster = make_buildings_raster(image_path, labels_path)
    
    for model_name, model_info in models.items():
        model = load_model(model_info)
        model_results = []
        
        for location_name, paths in locations.items():
            print(f"Evaluating {model_name} on {location_name}")
            image_path = paths['image_path']
            labels_path = paths['labels_path']
            
            # Prepare data for the specific model
            if model_name == 'pixel_based_RF':
                data = prepare_data_for_pixel_based_RF(image_path, labels_path)
            elif model_name == 'deeplabv3_buildings':
                data = prepare_data_for_deeplabv3_buildings(image_path, labels_path, buildings_raster)
            elif model_name == 'deeplabv3_multimodal':
                data = prepare_data_for_deeplabv3_multimodal(image_path, labels_path, buildings_raster)
            else:
                raise ValueError("Model preparation function not found")
            
            inputs, location_labels = data
            
            # Make predictions
            if model_info['framework'] == 'pytorch':
                pred_labels = make_predictions_buildings(model, inputs)
            elif model_info['framework'] == 'scikit_learn':
                predictions = model.predict(inputs)
            else:
                raise ValueError("Unsupported framework")
            
            # Evaluate predictions
            evaluation_metrics = evaluate_model(pred_labels, location_labels)
            print(f"Got f1 score of {evaluation_metrics.f1} for {location_name} model {model_name}")
            model_results.append({
                'location': location_name,
                'metrics': evaluation_metrics
            })
        
        results[model_name] = model_results
    
    return results

def save_evaluations(results, output_path):
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f)

# Run evaluations
results = run_evaluations(cities, models)
    
# Save the evaluation results
save_evaluations(results, 'evaluation_results.json')