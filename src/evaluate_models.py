import os
import sys
import torch
import duckdb
from affine import Affine

from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.core.data import (ClassConfig, Scene, StatsTransformer, ClassInferenceTransformer,
                                    RasterizedSource, RasterioCRSTransformer, SemanticSegmentationLabelSource,
                                    GeoJSONVectorSource)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from src.models.model_definitions import (BuildingsOnlyDeeplabv3SegmentationModel, BuildingsOnlyPredictionsIterator,
                                          CustomGeoJSONVectorSource)
                                        #   MultiModalSegmentationModel,
                                        #   BuildingsOnlyPredictionsIterator)
from src.data.dataloaders import query_buildings_data

con = duckdb.connect("../../data/0/data.db")
con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

models = {
    'pixel_based_RF': {
        'framework': 'scikit_learn',
        'weights_path': ''
    },
    'deeplabv3_buildings': {
        'framework': 'pytorch',
        'weights_path': '/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=09-val_loss=0.1590.ckpt'
    },
    'deeplabv3_multimodal': {
        'framework': 'pytorch',
        'weights_path': ''
    }
}

cities = {
    'SanJoseCRI': {
        'image_path': '../../data/0/sentinel_Gee/CRI_San_Jose_2023.tif',
        'labels_path': '../../data/SHP/SanJose_PS.shp'
    },
    'TegucigalpaHND': {
        'image_path': '../../data/0/sentinel_Gee/HND_Comayaguela_2023.tif',
        'labels_path': '../../data/SHP/Tegucigalpa_PS.shp'
    },
    'SantoDomingoDOM': {
        'image_path': '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif',
        'labels_path': '../../data/SHP/Tegucigalpa_PS.shp'
    },
    'GuatemalaCity': {
        'image_path': '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif',
        'labels_path': '../data/0/Guatemala_PS.shp'
    },
    'Managua': {
        'image_path': '../../data/0/sentinel_Gee/NIC_Tipitapa_2023.tif',
        'labels_path': '../data/0/Managua_PS.shp'
    },
    'Panama': {
        'image_path': '../../data/0/sentinel_Gee/PAN_San_Miguelito_2023.tif',
        'labels_path': '../data/0/Panama_PS.shp'
    },
    'SanSalvador_PS': {
        'image_path': '../../data/0/sentinel_Gee/SLV_Delgado_2023.tif',
        'labels_path': '../data/0/SanSalvador_PS.shp'
    },
    'BelizeCity': {
        'image_path': '../../data/0/sentinel_Gee/HND__2023.tif',
        'labels_path': '../data/0/BelizeCity_PS.shp'
    }
}

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

def prepare_data_for_pixel_based_RF(image_path, labels_path):
    # Add code to prepare data for model_1
    return (data, location_labels)

def prepare_data_for_deeplabv3_buildings(image_path, labels_path):
    sentinel_uri = image_path

    label_vector_source = GeoJSONVectorSource(labels_path,
        crs_transformer_buildings,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])
    
    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
    
    label_extent = label_source.get_extent()
    
    buildings = query_buildings_data(con, xmin, ymin, xmax, ymax)
    
    xmin, _, _, ymax = buildings.total_bounds

    crs_transformer_buildings = RasterioCRSTransformer.from_uri(sentinel_uri)
    affine_transform_buildings = Affine(5, 0, xmin, 0, -5, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source,
        background_class_id=0)
    
    BuilScene = Scene(
        id='buildings',
        raster_source = rasterized_buildings_source,
        label_source = label_source)
    
    data = BuilScene.raster_source
    location_labels = BuilScene.label_source.get_labels()
    
    return (data, location_labels)

def prepare_data_for_deeplabv3_multimodal(image_path, labels_path):
    # Add code to prepare data for model_3
    return (data, location_labels)


# function to compute mean IOU, confusion matrix
def evaluate_model(predictions, labels):
    pass

# function to load best model
def load_model(model_info):
    if model_info['framework'] == 'pytorch':
        
        if model_info['name'] == 'deeplabv3_multimodal':
            model = MultiModalSegmentationModel.load_from_checkpoint(model_info['weights_path'])
            
        elif model_info['name'] == 'deeplabv3_buildings':
            model = BuildingsOnlyDeeplabv3SegmentationModel.load_from_checkpoint(model_info['weights_path'])
        
    elif model_info['framework'] == 'scikit_learn':
        model = None
        
    else:
        raise ValueError("Unsupported framework")
    
    return model

def run_evaluations(locations, models):
    results = {}
    
    for model_name, model_info in models.items():
        model = load_model(model_info)
        model_results = []
        
        for location_name, paths in locations.items():
            image_path = paths['image_path']
            labels_path = paths['labels_path']
            
            # Prepare data for the specific model
            if model_name == 'pixel_based_RF':
                data = prepare_data_for_pixel_based_RF(image_path, labels_path)
            elif model_name == 'deeplabv3_buildings':
                data = prepare_data_for_deeplabv3_buildings(image_path, labels_path)
            elif model_name == 'deeplabv3_multimodal':
                data = prepare_data_for_deeplabv3_multimodal(image_path, labels_path)
            else:
                raise ValueError("Model preparation function not found")
            
            # Assuming data is a tuple (inputs, labels)
            inputs, location_labels = data
            
            # Make predictions
            if model_info['framework'] == 'pytorch':
                model.eval()
                with torch.no_grad():
                    predictions = model(inputs)
            elif model_info['framework'] == 'scikit_learn':
                predictions = model.predict(inputs)
            else:
                raise ValueError("Unsupported framework")
            
            # Evaluate predictions
            evaluation_metrics = evaluate_model(predictions, location_labels)
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
save_evaluations(results, 'path/to/evaluation_results.json')