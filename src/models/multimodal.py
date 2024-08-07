import os
import sys
import argparse
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import geopandas as gpd
import matplotlib.pyplot as plt
import albumentations as A
from affine import Affine
from torch.utils.data import ConcatDataset, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
from pytorch_lightning import LightningDataModule
import math
from ptflops import get_model_complexity_info
import torch
from typing import Dict
import shutil
import torch
import torch.optim as optim
from torchvision.transforms import Normalize

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params, MultiResolutionDeepLabV3,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot, merge_geojson_files,
                                          CustomVectorOutputConfig, FeatureMapVisualization, generate_and_display_predictions)
from src.features.dataloaders import (cities, show_windows,ensure_tuple, MultiInputCrossValidator, clear_directory, BaseCrossValidator,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn,show_first_batch_item,
                                  MergeDataset, show_single_tile_multi, get_label_source_from_merge_dataset, create_scenes_for_city, PolygonWindowGeoDataset)
from rastervision.core.box import Box
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data import (ClassConfig, XarraySource, Scene,RasterioCRSTransformer,
                                    ClassInferenceTransformer, SemanticSegmentationDiscreteLabels)
from rastervision.pytorch_learner.dataset.transform import (TransformType, TF_TYPE_TO_TF_FUNC)
from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
                                          SlidingWindowGeoDataset)
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.core.data.label_store import (SemanticSegmentationLabelStore)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiResolutionFPN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

from pytorch_lightning.callbacks import Callback
                        
# Check if MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

# SantoDomingo
sentinel_sceneSD, buildings_sceneSD = create_scenes_for_city('SantoDomingo', cities['SantoDomingo'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(sentinel_sceneSD, city= 'SantoDomingo', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, city= 'SantoDomingo',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# GuatemalaCity
sentinel_sceneGC, buildings_sceneGC = create_scenes_for_city('GuatemalaCity', cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, city='GuatemalaCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, city='GuatemalaCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneTG, buildings_sceneTG = create_scenes_for_city('Tegucigalpa', cities['Tegucigalpa'], class_config)
sentinelGeoDataset_TG = CustomSlidingWindowGeoDataset(sentinel_sceneTG, city='Tegucigalpa', size=256, stride = 256, out_size=256, padding=256, transform_type=TransformType.noop, transform=None)
buildGeoDataset_TG = CustomSlidingWindowGeoDataset(buildings_sceneTG, city='Tegucigalpa', size=512, stride = 512, out_size=512, padding=512, transform_type=TransformType.noop, transform=None)

# Managua
sentinel_sceneMN, buildings_sceneMN = create_scenes_for_city('Managua', cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, city='Managua', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
sentinel_scenePN, buildings_scenePN = create_scenes_for_city('Panama', cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN, city='Panama', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# SanJose
sentinel_sceneSJ, buildings_sceneSJ = create_scenes_for_city('SanJose', cities['SanJose'], class_config)
sentinelGeoDataset_SJ = PolygonWindowGeoDataset(sentinel_sceneSJ, city='SanJose', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SJ = PolygonWindowGeoDataset(buildings_sceneSJ, city='SanJose', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# BelizeCity
sentinel_sceneBL, buildings_sceneBL = create_scenes_for_city('BelizeCity', cities['BelizeCity'], class_config)
sentinelGeoDataset_BL = PolygonWindowGeoDataset(sentinel_sceneBL, city='BelizeCity', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_BL = PolygonWindowGeoDataset(buildings_sceneBL, city='BelizeCity', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets
train_cities = 'SD' # 'sel'
split_index = 1 # 0 or 1

multimodal_datasets = {
    'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD),
    'GuatemalaCity': MergeDataset(sentinelGeoDataset_GC, buildGeoDataset_GC),
    'Tegucigalpa': MergeDataset(sentinelGeoDataset_TG, buildGeoDataset_TG),
    'Managua': MergeDataset(sentinelGeoDataset_MN, buildGeoDataset_MN),
    'Panama': MergeDataset(sentinelGeoDataset_PN, buildGeoDataset_PN),
    'SanJose': MergeDataset(sentinelGeoDataset_SJ, buildGeoDataset_SJ),
    'BelizeCity': MergeDataset(sentinelGeoDataset_BL, buildGeoDataset_BL)
    } if train_cities == 'sel_wSSBM' else {'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD)}

cv = MultiInputCrossValidator(multimodal_datasets, n_splits=2, val_ratio=0.5, test_ratio=0)

# Preview a city with sliding windows
city = 'SantoDomingo'
windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
img_full = senitnel_create_full_image(get_label_source_from_merge_dataset(multimodal_datasets[city]))
show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')

# save_path = os.path.join(grandparent_dir, f"multimodal_visualization_{city}.png")
show_single_tile_multi(multimodal_datasets, city, 7, show_sentinel=True, show_buildings=True)#, save_path=save_path)

train_dataset, val_dataset, _, val_city_indices = cv.get_split(split_index)
print(f"Train dataset size: {len(train_dataset)}") 
print(f"Validation dataset size: {len(val_dataset)}")

batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_multi_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_multi_fn)

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# Train the model 
hyperparameters = {
    'model': 'DLV3',
    'train_cities': train_cities,
    'split_index': split_index,
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'gamma': 0.7,
    'sched_step_size': 6,
    'pos_weight': 2.0,
    'buil_out_chan': 4
}

output_dir = f'../../UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_mean_iou',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_{train_cities}_cv{split_index}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=5,
    mode='max')

class PredictionVisualizationCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            device = pl_module.device  # Get the device from the model
            generate_and_display_predictions(pl_module, sentinel_sceneSD, buildings_sceneSD, device, trainer.current_epoch, class_config)

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=15)
visualization_callback = PredictionVisualizationCallback(every_n_epochs=5)  # Adjust the frequency as needed

model = MultiResolutionDeepLabV3(
    use_deeplnafrica=hyperparameters['use_deeplnafrica'],
    learning_rate=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    gamma=hyperparameters['gamma'],
    atrous_rates=hyperparameters['atrous_rates'],
    sched_step_size = hyperparameters['sched_step_size'],
    pos_weight=hyperparameters['pos_weight'],
    buil_out_chan = hyperparameters['buil_out_chan']
)

model.to(device)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback, visualization_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=30,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
tiles_per_city = {
    'San Jose': 41,
    'Guatemala City': 20,
    'Santo Domingo': 13,
    'Tegucigalpa': 12,
    'Panama City': 10,
    'Managua': 3
}

# model_id = 'last-v1.ckpt'
# best_model_path = os.path.join(grandparent_dir, f'UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/', model_id)

best_model_path = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3()
checkpoint = torch.load(best_model_path)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()
check_nan_params(best_model)

eval_sent_scene = sentinel_sceneSD
eval_buil_scene = buildings_sceneSD
sent_strided_fullds = CustomSlidingWindowGeoDataset(eval_sent_scene, size=256, stride=256, padding=0, city='SD', transform=None, transform_type=TransformType.noop)
buil_strided_fullds = CustomSlidingWindowGeoDataset(eval_buil_scene, size=512, stride=512, padding=0, city='SD', transform=None, transform_type=TransformType.noop)
mergedds = MergeDataset(sent_strided_fullds, buil_strided_fullds)

predictions_iterator = MultiResPredictionsIterator(best_model, mergedds, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=eval_sent_scene.extent,
    num_classes=len(class_config),
    smooth=True
)

gt_labels = eval_sent_scene.label_source.get_labels()

# Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5)
plt.show()

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
print(f"F1 on city full DS:{inf_eval.f1}")

def calculate_multimodal_metrics(model, merged_dataset, device, scene):
    predictions_iterator = MultiResPredictionsIterator(model, merged_dataset, device=device)
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
        'recall': inf_eval.recall,
        'num_tiles': len(windows)
    }

city_metrics = {}
excluded_cities = []
total_tiles = 0

for city, (dataset_index, num_samples) in val_city_indices.items():
    if num_samples == 0:
        print(f"Skipping {city} as it has no validation samples.")
        continue

    city_val_dataset = Subset(val_dataset, range(dataset_index, dataset_index + num_samples))
    city_scene = multimodal_datasets[city].datasets[0].scene

    try:
        metrics = calculate_multimodal_metrics(best_model, city_val_dataset, device, city_scene)
        city_metrics[city] = metrics
        total_tiles += metrics['num_tiles']
    except Exception as e:
        print(f"Error calculating metrics for {city}: {str(e)}")

# Print metrics for each city
print(f"\nMetrics for split {split_index}:")
for city, metrics in city_metrics.items():
    print(f"\nMetrics for {city} ({metrics['num_tiles']} tiles):")
    if any(math.isnan(value) for value in metrics.values()):
        print("No correct predictions made. Metrics are NaN.")
        excluded_cities.append(city)
    else:
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")

# Calculate and print weighted average metrics
valid_metrics = {k: v for k, v in city_metrics.items() if not any(math.isnan(value) for value in v.values())}

if valid_metrics:
    weighted_metrics = {
        'f1': sum(metrics['f1'] * metrics['num_tiles'] for metrics in valid_metrics.values()) / total_tiles,
        'precision': sum(metrics['precision'] * metrics['num_tiles'] for metrics in valid_metrics.values()) / total_tiles,
        'recall': sum(metrics['recall'] * metrics['num_tiles'] for metrics in valid_metrics.values()) / total_tiles
    }

    print("\nWeighted average metrics across cities:")
    print(f"F1 Score: {weighted_metrics['f1']:.4f}")
    print(f"Precision: {weighted_metrics['precision']:.4f}")
    print(f"Recall: {weighted_metrics['recall']:.4f}")
    
    if excluded_cities:
        print(f"\n(Note: {', '.join(excluded_cities)} {'was' if len(excluded_cities) == 1 else 'were'} excluded due to NaN metrics)")
else:
    print("\nUnable to calculate weighted average metrics. All cities have NaN values.")

# ################################
# ###### Unseen Predictions ######
# ################################

# model_paths_original_noSJ = [
#     os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv0_epoch=18-val_loss=0.4542.ckpt'),
#     os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv1_epoch=35-val_loss=0.3268.ckpt')
# ]
# model_paths = [
#     os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/multimodal_selSJ_cv0_epoch=19-val_loss=0.2787.ckpt'),
#     os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/multimodal_selSJ_cv1_epoch=27-val_loss=0.2128.ckpt')
# ]

# def save_predictions_as_geojson(pred_labels, city, split_index, output_dir, cities, class_config):
#     vector_output_config = CustomVectorOutputConfig(
#         class_id=1,
#         denoise=8,
#         threshold=0.5
#     )

#     crs_transformer = RasterioCRSTransformer.from_uri(cities[city]['image_path'])
#     gdf = gpd.read_file(cities[city]['labels_path']).to_crs('epsg:3857')
#     xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
#     affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
#     crs_transformer.transform = affine_transform_buildings

#     city_output_dir = os.path.join(output_dir, city)
#     os.makedirs(city_output_dir, exist_ok=True)  # Create directory if it doesn't exist, but don't clear it
#     label_store = SemanticSegmentationLabelStore(
#         uri=os.path.join(city_output_dir, f'split{split_index}_predictions'),
#         crs_transformer=crs_transformer,
#         class_config=class_config,
#         vector_outputs=[vector_output_config],
#         discrete_output=True
#     )
#     label_store.save(pred_labels)

# def load_multimodal_model(model_path, device):
#     model = MultiResolutionDeepLabV3()
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'], strict=True)
#     model = model.to(device)
#     model.eval()
#     return model

# def save_unseen_predictions_multimodal(models, cv, multimodal_datasets, cities, class_config, device, output_dir):
#     # clear_directory(output_dir)

#     for split_index, model in enumerate(models):
#         model.eval()
        
#         _, val_dataset, _, val_city_indices = cv.get_split(split_index)
        
#         for city, (val_start, val_count) in val_city_indices.items():
#             if val_count == 0:
#                 continue
            
#             print(f"Processing unseen data for {city} using model from split {split_index}")
            
#             # Get the full dataset for this city
#             city_full_dataset = multimodal_datasets[city]
            
#             # Get the validation indices for this city and split
#             city_val_indices = cv.city_splits[city][split_index][1]  # [1] corresponds to val_idx
            
#             # Create a subset using only the validation indices
#             city_val_dataset = Subset(city_full_dataset, city_val_indices)
            
#             # Get the sentinel scene for extent information
#             city_sent_scene = get_sentinel_scene_from_merge_dataset(city_full_dataset)
            
#             predictions_iterator = MultiResPredictionsIterator(model, city_val_dataset, device=device)
#             windows, predictions = zip(*predictions_iterator)
            
#             pred_labels = SemanticSegmentationLabels.from_predictions(
#                 windows,
#                 predictions,
#                 extent=city_sent_scene.extent,
#                 num_classes=len(class_config),
#                 smooth=True
#             )
            
#             save_predictions_as_geojson(pred_labels, city, split_index, output_dir, cities, class_config)

# def get_sentinel_scene_from_merge_dataset(merged_dataset):
#     # Assuming the first dataset in the MergeDataset is the sentinel dataset
#     return merged_dataset.datasets[0].scene

# models = [load_multimodal_model(path, device) for path in model_paths]

# output_dir = '../../vectorised_model_predictions/multimodal/unseen_predictions_selSJ/'

# try:
#     save_unseen_predictions_multimodal(models, cv, multimodal_datasets, cities, class_config, device, output_dir)

#     merged_output_file = os.path.join(output_dir, 'combined_predictions.geojson')
#     merge_geojson_files(output_dir, merged_output_file)

#     print("Predictions saved and combined successfully.")
# except Exception as e:
#     print(f"An error occurred during prediction or merging: {str(e)}")

##############################
###### Model Complexity ######
##############################

# def input_constructor(input_res):
#     # Create sample inputs matching your data structure
#     sentinel = torch.randn(1, 4, 256, 256)  # 4 channels, 256x256 image size
#     buildings = torch.randn(1, 1, 512, 512)  # 1 channel, 512x512 image size
#     return ((sentinel, None), (buildings, None))

# model = MultiResolutionDeepLabV3()
# macs, params = get_model_complexity_info(
#     model, 
#     (4, 256, 256),  # This is ignored, but needed for the function
#     input_constructor=input_constructor,
#     as_strings=True, 
#     print_per_layer_stat=True, 
#     verbose=True
# )

# print(f'MultiResolutionDeepLabV3 - Computational complexity: {macs}')
# print(f'MultiResolutionDeepLabV3 - Number of parameters: {params}')


# ##############################
# ### Visualise feature maps ###
# ##############################

model_id = 'multimodal_selSJ_cv1_epoch=27-val_loss=0.2128.ckpt'
best_model_path = os.path.join(parent_dir, f'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/', model_id)
best_model = MultiResolutionDeepLabV3()#buil_channels=buil_channels, buil_kernel=buil_kernel, buil_out_chan=4)
checkpoint = torch.load(best_model_path)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()
check_nan_params(best_model)

# Feature Map Visualisation class
class FeatureMapVisualization:
    def __init__(self, model, device):
        self.model = model
        self.feature_maps = {}
        self.hooks = []
        self.device = device

    def add_hooks(self, layer_names):
        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(self.save_feature_map(name)))

    def save_feature_map(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def visualize_feature_maps(self, layer_name, input_data, num_feature_maps='all', figsize=(20, 20)):
        self.model.eval()
        with torch.no_grad():
            # Force model to device again
            self.model = self.model.to(self.device)
            
            # Ensure input_data is on the correct device
            if isinstance(input_data, list):
                input_data = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in input_data]
            elif isinstance(input_data, dict):
                input_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
            elif isinstance(input_data, torch.Tensor):
                input_data = input_data.to(self.device)
            
            # Double-check that all model parameters are on the correct device
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            
            self.model(input_data)

        if layer_name not in self.feature_maps:
            print(f"Layer {layer_name} not found in feature maps.")
            return

        feature_maps = self.feature_maps[layer_name].cpu().detach().numpy()

        # Handle different dimensions
        if feature_maps.ndim == 4:  # (batch_size, channels, height, width)
            feature_maps = feature_maps[0]  # Take the first item in the batch
        elif feature_maps.ndim == 3:  # (channels, height, width)
            pass
        else:
            print(f"Unexpected feature map shape: {feature_maps.shape}")
            return

        total_maps = feature_maps.shape[0]
        
        if num_feature_maps == 'all':
            num_feature_maps = total_maps
        else:
            num_feature_maps = min(num_feature_maps, total_maps)

        # Calculate grid size
        grid_size = math.ceil(math.sqrt(num_feature_maps))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and axes.ndim == 1:
            axes = axes.reshape(1, -1)

        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                if index < num_feature_maps:
                    feature_map_img = feature_maps[index]
                    im = axes[i, j].imshow(feature_map_img, cmap='viridis')
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'Channel {index+1}')
                else:
                    axes[i, j].axis('off')

        fig.suptitle(f'Feature Maps for Layer: {layer_name}\n({num_feature_maps} out of {total_maps} channels)')
        fig.tight_layout()
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.show()
                
    def visualize_maximized_activation(self, layer_name, num_iterations=100, learning_rate=0.1, num_feature_maps='all', figsize=(20, 20)):
        self.model.eval()

        # Create optimizable inputs
        sentinel_data = torch.randn(1, 4, 256, 256, requires_grad=True, device=self.device)
        buildings_data = torch.randn(1, 1, 512, 512, requires_grad=True, device=self.device)

        # Create synthetic labels (these won't be used in forward pass, but are needed for unpacking)
        sentinel_labels = torch.zeros(1, 1, 256, 256, device=self.device)
        buildings_labels = torch.zeros(1, 1, 512, 512, device=self.device)

        # Normalization parameters (adjust these based on your data)
        sentinel_mean = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
        sentinel_std = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B
        buildings_mean = [0.1]
        buildings_std = [0.8]

        optimizer = optim.Adam([sentinel_data, buildings_data], lr=learning_rate)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Normalize inputs
            norm_sentinel = Normalize(mean=sentinel_mean, std=sentinel_std)(sentinel_data)
            norm_buildings = Normalize(mean=buildings_mean, std=buildings_std)(buildings_data)

            # Create batch structure expected by the model
            sentinel_batch = (norm_sentinel, sentinel_labels)
            buildings_batch = (norm_buildings, buildings_labels)
            batch = (sentinel_batch, buildings_batch)

            # Forward pass
            self.model(batch)

            # Get activation of target layer
            if layer_name not in self.feature_maps:
                print(f"Layer {layer_name} not found in feature maps.")
                return

            activation = self.feature_maps[layer_name]

            # Define objective (e.g., mean activation)
            objective = activation.mean()

            # Backward pass
            objective.backward()
            optimizer.step()

        # Visualize the maximized activation
        self.visualize_feature_maps(layer_name, batch, num_feature_maps, figsize)

        return sentinel_data.detach(), buildings_data.detach()
    
# # Use it like this:
visualizer = FeatureMapVisualization(best_model, device=device)
visualizer.add_hooks([
    'buildings_encoder.0.0', # conv2d
    'buildings_encoder.1.0', # conv2d
    'xtra_fusion',
    'decoder_fusion'
])

# Get the iterator for the DataLoader
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_multi_fn)
data_iter = iter(val_loader)
first_batch = next(data_iter)

show_first_batch_item(first_batch)

visualizer.visualize_feature_maps('buildings_encoder.0.0', first_batch, num_feature_maps='all')
visualizer.visualize_feature_maps('buildings_encoder.1.0', first_batch, num_feature_maps='all')
visualizer.visualize_feature_maps('xtra_fusion', first_batch, num_feature_maps='all')
visualizer.visualize_feature_maps('decoder_fusion', first_batch, num_feature_maps='all')

# Feature Visaualization #
for layer_name in ['buildings_encoder.1.0', 'xtra_fusion', 'decoder_fusion']:
    print(f"Visualizing maximized activation for layer: {layer_name}")
    visualizer.visualize_maximized_activation(layer_name, num_iterations=500, learning_rate=0.1)

# visualizer.remove_hooks()