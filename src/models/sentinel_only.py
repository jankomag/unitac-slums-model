from os.path import join
from subprocess import check_output
import os

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()
from torch.utils.data import ConcatDataset, Subset
import math
import sys
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime
from torchvision.models.segmentation import deeplabv3_resnet50
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
from affine import Affine
import geopandas as gpd
import torch
import torch.nn as nn
from torch.optim import AdamW
import tempfile
import wandb
import numpy as np
import cv2
# import lightning.pytorch as pl
from torch.utils.data import ConcatDataset
from rastervision.pytorch_learner.dataset.transform import (TransformType,
                                                            TF_TYPE_TO_TF_FUNC)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import Polygon
from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, make_dir, zipdir, download_if_needed,
    download_or_copy, sync_from_dir, get_local_path, unzip, is_local,
    get_tmp_dir)
from torch.utils.data import DataLoader
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import SentinelDeepLabV3, PredictionsIterator, create_predictions_and_ground_truth_plot, check_nan_params
from src.features.dataloaders import (create_sentinel_scene, cities, CustomSlidingWindowGeoDataset, collate_fn,
                                      senitnel_create_full_image, show_windows, PolygonWindowGeoDataset,
                                      SingleInputCrossValidator, singlesource_show_windows_for_city, show_single_tile_sentinel)

from rastervision.core.data.label_store import (SemanticSegmentationLabelStore)
from rastervision.core.data import (Scene, ClassInferenceTransformer, RasterizedSource,
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
from rastervision.pytorch_learner import (
    SolverConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner, SemanticSegmentationGeoDataConfig, SemanticSegmentationVisualizer,
)
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from rastervision.core.data.utils import get_polygons_from_uris

# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
# Preproceess SD labels for simplification (buffering)
# label_uriSD = "../../data/0/SantoDomingo3857.geojson"
# gdf = gpd.read_file(label_uriSD)
# gdf = gdf.to_crs('EPSG:4326')
# gdf_filled = gdf.copy()
# gdf_filled["geometry"] = gdf_filled.geometry.buffer(0.00008)
# gdf_filled = gdf_filled.to_crs('EPSG:3857')
# gdf_filled.to_file("../../data/0/SantoDomingo3857_buffered.geojson", driver="GeoJSON")

# Load training data
class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

# SantoDomingo
SentinelScene_SD = create_sentinel_scene(cities['SantoDomingo'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(SentinelScene_SD, city= 'SantoDomingo', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# GuatemalaCity
sentinel_sceneGC = create_sentinel_scene(cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, city='GuatemalaCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneTG = create_sentinel_scene(cities['Tegucigalpa'], class_config)
sentinelGeoDataset_TG = CustomSlidingWindowGeoDataset(sentinel_sceneTG, city='Tegucigalpa', size=256, stride = 256, out_size=256, padding=256, transform_type=TransformType.noop, transform=None)

# Managua
sentinel_sceneMN = create_sentinel_scene(cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
sentinel_scenePN = create_sentinel_scene(cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets
train_cities = 'sel'
split_index = 1 # 0 or 1

def get_sentinel_datasets(train_cities):
    all_datasets = {
        'SD': sentinelGeoDataset_SD,
        'GC': sentinelGeoDataset_GC,
        'TG': sentinelGeoDataset_TG,
        'MN': sentinelGeoDataset_MN,
        'PN': sentinelGeoDataset_PN,
    }
    if train_cities == 'sel':
        return all_datasets
    elif isinstance(train_cities, str):
        return {train_cities: all_datasets[train_cities]}
    elif isinstance(train_cities, list):
        return {city: all_datasets[city] for city in train_cities if city in all_datasets}
    else:
        raise ValueError("train_cities must be 'sel', a string, or a list of strings")

sentinel_datasets = get_sentinel_datasets(train_cities)

cv = SingleInputCrossValidator(sentinel_datasets, n_splits=2, val_ratio=0.5, test_ratio=0)

# Preview a city with sliding windows
city = 'SD'
singlesource_show_windows_for_city(city, split_index, cv, sentinel_datasets)
show_single_tile_sentinel(sentinel_datasets, city, 4)

train_dataset, val_dataset, _, val_city_indices = cv.get_split(split_index)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create DataLoaders
batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

# Create model
hyperparameters = {
    'model': 'DLV3',
    'split_index': split_index,
    'train_cities': train_cities,
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'gamma': 0.9,
    'sched_step_size': 10,
    'pos_weight': 2.0
}

model = SentinelDeepLabV3(use_deeplnafrica = hyperparameters['use_deeplnafrica'],
                    learning_rate = hyperparameters['learning_rate'],
                    weight_decay = hyperparameters['weight_decay'],
                    gamma = hyperparameters['gamma'],
                    atrous_rates = hyperparameters['atrous_rates'],
                    sched_step_size = hyperparameters['sched_step_size'],
                    pos_weight = hyperparameters['pos_weight'])
model.to(device)

output_dir = f'../../UNITAC-trained-models/sentinel_only/{train_cities}_DLV3'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-finetune-sentinel-only', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename=f'sentinel_{train_cities}_cv{split_index}-{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=5,
    save_last=True,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=30)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=55,
    max_epochs=250,
    num_sanity_val_steps=3,
    # overfit_batches=0.5
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Make predictions
model_id = 'last-v1.ckpt'
best_model_path = os.path.join(grandparent_dir, f'UNITAC-trained-models/sentinel_only/{train_cities}_DLV3/', model_id)
# best_model_path = checkpoint_callback.best_model_path

scene_eval = SentinelScene_SD  # Assuming this is your sentinel scene for evaluation
best_model = SentinelDeepLabV3.load_from_checkpoint(best_model_path)
best_model.eval()
check_nan_params(best_model)

class PredictionsIterator:
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.predictions = []
        
        for idx in range(len(dataset)):
            x, _ = dataset[idx]
            x = x.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(x)
            
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
            
            window = self.get_window(dataset, idx)
            
            self.predictions.append((window, probabilities))
    
    def get_window(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_window(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_window(dataset.datasets[dataset_idx], sample_idx)
        else:
            return dataset.windows[idx]
    
    def get_concat_dataset_indices(self, concat_dataset, idx):
        for dataset_idx, dataset in enumerate(concat_dataset.datasets):
            if idx < len(dataset):
                return dataset_idx, idx
            idx -= len(dataset)
        raise IndexError('Index out of range')
    
    def __iter__(self):
        return iter(self.predictions)

strided_fullds = CustomSlidingWindowGeoDataset(scene_eval, size=256, stride=128, padding=0, city='SD', transform=None, transform_type=TransformType.noop)
predictions_iterator = PredictionsIterator(best_model, strided_fullds, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=scene_eval.extent,
    num_classes=len(class_config),
    smooth=True
)
gt_labels = scene_eval.label_source.get_labels()

# Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels)
plt.show()

def calculate_sentinel_metrics(model, dataset, device, scene):
    predictions_iterator = PredictionsIterator(model, dataset, device=device)
    windows, predictions = zip(*predictions_iterator)

    # Create SemanticSegmentationLabels from predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    gt_labels = scene.label_source.get_labels()

    # Evaluate against labels:
    pred_labels_discrete = SemanticSegmentationDiscreteLabels.make_empty(
        extent=pred_labels.extent,
        num_classes=len(class_config))
    scores = pred_labels.get_score_arr(pred_labels.extent)
    pred_array_discrete = (scores > 0.5).astype(int)
    pred_labels_discrete[pred_labels.extent] = pred_array_discrete[1]
    evaluator = SemanticSegmentationEvaluator(class_config)
    evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels_discrete)
    inf_eval = evaluation.class_to_eval_item[1]
    return {
        'f1': inf_eval.f1,
        'precision': inf_eval.precision,
        'recall': inf_eval.recall
    }

city_metrics = {}
excluded_cities = []

for city, (dataset_index, num_samples) in val_city_indices.items():
    # Skip cities with no validation samples
    if num_samples == 0:
        print(f"Skipping {city} as it has no validation samples.")
        continue

    # Get the subset of the validation dataset for this city
    city_val_dataset = Subset(val_dataset, range(dataset_index, dataset_index + num_samples))
    
    # Get the scene for this city
    city_scene = sentinel_datasets[city].scene
    
    try:
        # Calculate metrics for this city
        metrics = calculate_sentinel_metrics(best_model, city_val_dataset, device, city_scene)
        
        city_metrics[city] = metrics
    except Exception as e:
        print(f"Error calculating metrics for {city}: {str(e)}")

# Print metrics for each city
print(f"\nMetrics for split {split_index}:")
for city, metrics in city_metrics.items():
    print(f"\nMetrics for {city}:")
    if any(math.isnan(value) for value in metrics.values()):
        print("No correct predictions made. Metrics are NaN.")
        excluded_cities.append(city)
    else:
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

# Calculate and print average metrics across all cities
valid_metrics = {k: v for k, v in city_metrics.items() if not any(math.isnan(value) for value in v.values())}

if valid_metrics:
    avg_metrics = {
        'f1': sum(m['f1'] for m in valid_metrics.values()) / len(valid_metrics),
        'precision': sum(m['precision'] for m in valid_metrics.values()) / len(valid_metrics),
        'recall': sum(m['recall'] for m in valid_metrics.values()) / len(valid_metrics)
    }

    print("\nAverage metrics across cities:")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    
    if excluded_cities:
        print(f"\n(Note: {', '.join(excluded_cities)} {'was' if len(excluded_cities) == 1 else 'were'} excluded due to NaN metrics)")
else:
    print("\nUnable to calculate average metrics. All cities have NaN values.")



# # # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# gdf = gpd.read_file('../data/0/SantoDomingo3857_buffered.geojson')
# gdf = gdf.to_crs('EPSG:3857')
# xmin, ymin, xmax, ymax = gdf.total_bounds

# crs_transformer_buildings = RasterioCRSTransformer.from_uri('../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif')
# affine_transform_buildings = Affine(10, 0, xmin, 0, -10, ymax)
# crs_transformer_buildings.transform = affine_transform_buildings

# crs_transformer = RasterioCRSTransformer.from_uri('../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif')

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/sentinel_only/DLV3_!/',
#     crs_transformer = crs_transformer_buildings,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)