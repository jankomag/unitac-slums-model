import os
import sys
import argparse
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import wandb
import geopandas as gpd
import matplotlib.pyplot as plt
# from fvcore.nn import FlopCountAnalysis
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

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params, MultiResolutionDeepLabV3,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot, MultiResolutionFPN,
                                          CustomVectorOutputConfig, FeatureMapVisualization, MultiResolution128DeepLabV3)
from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple, MultiInputCrossValidator,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn,
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

# # SantoDomingo
sentinel_sceneSD, buildings_sceneSD = create_scenes_for_city('SantoDomingo', cities['SantoDomingo'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(sentinel_sceneSD, city= 'SantoDomingo', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, city= 'SantoDomingo',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# # GuatemalaCity
sentinel_sceneGC, buildings_sceneGC = create_scenes_for_city('GuatemalaCity', cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, city='GuatemalaCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, city='GuatemalaCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneTG, buildings_sceneTG = create_scenes_for_city('Tegucigalpa', cities['Tegucigalpa'], class_config)
sentinelGeoDataset_TG = CustomSlidingWindowGeoDataset(sentinel_sceneTG, city='Tegucigalpa', size=256, stride = 256, out_size=256, padding=256, transform_type=TransformType.noop, transform=None)
buildGeoDataset_TG = CustomSlidingWindowGeoDataset(buildings_sceneTG, city='Tegucigalpa', size=512, stride = 512, out_size=512, padding=512, transform_type=TransformType.noop, transform=None)

# # Managua
sentinel_sceneMN, buildings_sceneMN = create_scenes_for_city('Managua', cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, city='Managua', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# # Panama
sentinel_scenePN, buildings_scenePN = create_scenes_for_city('Panama', cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN, city='Panama', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets
train_cities = 'sel' # 'sel'
split_index = 0 # 0 or 1

multimodal_datasets = {
    'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD),
    'GuatemalaCity': MergeDataset(sentinelGeoDataset_GC, buildGeoDataset_GC),
    'Tegucigalpa': MergeDataset(sentinelGeoDataset_TG, buildGeoDataset_TG),
    'Managua': MergeDataset(sentinelGeoDataset_MN, buildGeoDataset_MN),
    'Panama': MergeDataset(sentinelGeoDataset_PN, buildGeoDataset_PN),
} if train_cities == 'sel' else {'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD)}

cv = MultiInputCrossValidator(multimodal_datasets, n_splits=2, val_ratio=0.5, test_ratio=0)

# Preview a city with sliding windows
city = 'Panama'
windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
img_full = senitnel_create_full_image(get_label_source_from_merge_dataset(multimodal_datasets[city]))
show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')
show_single_tile_multi(multimodal_datasets, city, 1, show_sentinel=True, show_buildings=True)

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
    # 'labels_size': labels_size,
    # 'buil_channels': buil_channels,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0.7,
    'gamma': 1,
    'sched_step_size': 15,
    'pos_weight': 2.0,
    # 'buil_kernel': buil_kernel,
    'buil_out_chan': 4
}

output_dir = f'../../UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_{train_cities}_cv{split_index}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=5,
    mode='min')

def generate_and_display_predictions(model, eval_sent_scene, eval_buil_scene, device, epoch):
        # Set up the prediction dataset
        sent_strided_fullds = CustomSlidingWindowGeoDataset(eval_sent_scene, size=256, stride=256, padding=128, city='SD', transform=None, transform_type=TransformType.noop)
        buil_strided_fullds = CustomSlidingWindowGeoDataset(eval_buil_scene, size=512, stride=512, padding=256, city='SD', transform=None, transform_type=TransformType.noop)
        mergedds = MergeDataset(sent_strided_fullds, buil_strided_fullds)

        # Generate predictions
        predictions_iterator = MultiResPredictionsIterator(model, mergedds, device=device)
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
        # fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5)
        scores = pred_labels.get_score_arr(pred_labels.extent)

        # Create a figure with three subplots side by side
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

        # Plot smooth predictions
        scores_class = scores[1]
        im1 = ax.imshow(scores_class, cmap='viridis', vmin=0, vmax=1)
        ax.axis('off')
        ax.set_title(f'Smooth Predictions (Class {1} Scores)')
        cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig, ax
        # Log the figure to WandB
        wandb.log({f"Predictions_Epoch_{epoch}": wandb.Image(fig)})
        
        plt.close(fig)  # Close the figure to free up memory
        
class PredictionVisualizationCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            device = pl_module.device  # Get the device from the model
            generate_and_display_predictions(pl_module, sentinel_sceneSD, buildings_sceneSD, device, trainer.current_epoch)

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=25)
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

# for batch in train_loader:
#     sentinel, buildings = batch
#     sentinel_data = sentinel[0].to(device)
#     sentlabels = sentinel[1].to(device)

#     buildings_data = buildings[0].to(device)
#     labels = buildings[1].to(device)

#     output = model(batch)# ((sentinel_data,sentlabels), (buildings_data,labels)))
#     print("Output of the model",output.shape)
#     break

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback, visualization_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=55,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
# model_id = 'last-v3.ckpt'
# best_model_path = os.path.join(grandparent_dir, f'UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/', model_id)

best_model_path = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3()#buil_channels=buil_channels, buil_kernel=buil_kernel, buil_out_chan=4)
checkpoint = torch.load(best_model_path)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()
check_nan_params(best_model)

eval_sent_scene = sentinel_sceneSD
eval_buil_scene = buildings_sceneSD
sent_strided_fullds = CustomSlidingWindowGeoDataset(eval_sent_scene, size=256, stride=128, padding=128, city='SD', transform=None, transform_type=TransformType.noop)
buil_strided_fullds = CustomSlidingWindowGeoDataset(eval_buil_scene, size=512, stride=256, padding=256, city='SD', transform=None, transform_type=TransformType.noop)
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
    city_scene = multimodal_datasets[city].datasets[0].scene  # Assuming the first dataset is sentinel
    
    try:
        # Calculate metrics for this city
        metrics = calculate_multimodal_metrics(best_model, city_val_dataset, device, city_scene)
        
        city_metrics[city] = metrics
    except Exception as e:
        print(f"Error calculating metrics for {city}: {str(e)}")

# Print metrics for each city
print(f"\nMetrics for split {split_index}:")
# print(f"Model id: {model_id}")
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


# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# crs_transformer = RasterioCRSTransformer.from_uri(cities['SantoDomingo']['image_path'])
# gdf = gpd.read_file(cities['SantoDomingo']['labels_path'])
# gdf = gdf.to_crs('epsg:3857')
# xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
# affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
# crs_transformer.transform = affine_transform_buildings

# pred_label_store = SemanticSegmentationLabelStore(
#     uri=f'../../vectorised_model_predictions/multi-modal/{train_cities}_DLV3/',
#     crs_transformer = crs_transformer,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)


### SAVING ALL PREDICTIONS FOR UNSEEN DATA ###
def load_model(model_path):
    model = MultiResolutionDeepLabV3()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    return model

def make_predictions(model, dataset, device):
    predictions_iterator = MultiResPredictionsIterator(model, dataset, device=device)
    windows, predictions = zip(*predictions_iterator)
    return windows, predictions

def save_predictions_as_geojson(pred_labels, city, output_dir, cities):
    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5
    )

    crs_transformer = RasterioCRSTransformer.from_uri(cities[city]['image_path'])
    gdf = gpd.read_file(cities[city]['labels_path'])
    gdf = gdf.to_crs('epsg:3857')
    xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
    affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
    crs_transformer.transform = affine_transform_buildings

    pred_label_store = SemanticSegmentationLabelStore(
        uri=os.path.join(output_dir, f'{city}_predictions.geojson'),
        crs_transformer=crs_transformer,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True
    )

    pred_label_store.save(pred_labels)

# Load both models
model_paths = [
    os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv0_res256_BCH128_BKR5_epoch=08-val_loss=0.5143.ckpt'),
    os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv1_res256_BCH128_BKR5_epoch=23-val_loss=0.3972.ckpt')
]
models = [load_model(path) for path in model_paths]

# Create datasets for all cities
all_cities = ['SantoDomingo', 'GuatemalaCity', 'Tegucigalpa', 'Managua', 'Panama']
all_datasets = {city: multimodal_datasets[city] for city in all_cities}

# Predict and evaluate for each split
output_dir = '../../vectorised_model_predictions/multi-modal/unseen_predictions/'
os.makedirs(output_dir, exist_ok=True)

all_metrics = {}

for split_index in [0, 1]:
    model = models[split_index]
    _, val_dataset, _, val_city_indices = cv.get_split(split_index)
    
    for city, (dataset_index, num_samples) in val_city_indices.items():
        if num_samples == 0:
            continue
        
        city_val_dataset = Subset(val_dataset, range(dataset_index, dataset_index + num_samples))
        city_scene = all_datasets[city].datasets[0].scene
        
        windows, predictions = make_predictions(model, city_val_dataset, device)
        
        pred_labels = SemanticSegmentationLabels.from_predictions(
            windows,
            predictions,
            extent=city_scene.extent,
            num_classes=len(class_config),
            smooth=True
        )
        
        save_predictions_as_geojson(pred_labels, city, output_dir, cities)
        
        metrics = calculate_multimodal_metrics(model, city_val_dataset, device, city_scene)
        all_metrics[city] = metrics

# Print and save metrics
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    for city, metrics in all_metrics.items():
        print(f"\nMetrics for {city}:")
        f.write(f"\nMetrics for {city}:\n")
        if any(math.isnan(value) for value in metrics.values()):
            print("No correct predictions made. Metrics are NaN.")
            f.write("No correct predictions made. Metrics are NaN.\n")
        else:
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
                f.write(f"{metric.capitalize()}: {value:.4f}\n")

    # Calculate and print average metrics
    valid_metrics = {k: v for k, v in all_metrics.items() if not any(math.isnan(value) for value in v.values())}
    if valid_metrics:
        avg_metrics = {
            metric: sum(city_metrics[metric] for city_metrics in valid_metrics.values()) / len(valid_metrics)
            for metric in ['f1', 'precision', 'recall']
        }
        print("\nAverage metrics across cities:")
        f.write("\nAverage metrics across cities:\n")
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        excluded_cities = set(all_metrics.keys()) - set(valid_metrics.keys())
        if excluded_cities:
            note = f"\n(Note: {', '.join(excluded_cities)} {'was' if len(excluded_cities) == 1 else 'were'} excluded due to NaN metrics)"
            print(note)
            f.write(note + "\n")
    else:
        print("\nUnable to calculate average metrics. All cities have NaN values.")
        f.write("\nUnable to calculate average metrics. All cities have NaN values.\n")

print(f"Predictions and metrics have been saved to {output_dir}")



### SAVING ALL PREDICTIONS FOR Strided and avergaed tiles ###
from typing import Dict

def create_strided_dataset(scene, city, size, stride):
    return CustomSlidingWindowGeoDataset(
        scene,
        city=city,
        size=size,
        stride=stride,
        out_size=size,
        padding=size,
        transform_type=TransformType.noop,
        transform=None
    )

def load_model(model_path, device):
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

def save_predictions_as_geojson(pred_labels, city, output_dir, cities):
    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5
    )

    crs_transformer = RasterioCRSTransformer.from_uri(cities[city]['image_path'])
    gdf = gpd.read_file(cities[city]['labels_path'])
    gdf = gdf.to_crs('epsg:3857')
    xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
    affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
    crs_transformer.transform = affine_transform_buildings

    pred_label_store = SemanticSegmentationLabelStore(
        uri=os.path.join(output_dir, f'{city}_predictions.geojson'),
        crs_transformer=crs_transformer,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True
    )

    pred_label_store.save(pred_labels)

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

def process_city(city: str, sentinel_scene: Scene, buildings_scene: Scene, models: list, device: torch.device, output_dir: str, cities: dict):
    sentinel_dataset = create_strided_dataset(sentinel_scene, city, size=256, stride=128)
    buildings_dataset = create_strided_dataset(buildings_scene, city, size=512, stride=256)
    merged_dataset = MergeDataset(sentinel_dataset, buildings_dataset)

    windows = None
    avg_predictions = None

    for model in models:
        windows, predictions = make_predictions(model, merged_dataset, device)
        if avg_predictions is None:
            avg_predictions = predictions
        else:
            avg_predictions = average_predictions(avg_predictions, predictions)

    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        avg_predictions,
        extent=sentinel_scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    save_predictions_as_geojson(pred_labels, city, output_dir, cities)

    # Calculate metrics using the averaged predictions
    gt_labels = sentinel_scene.label_source.get_labels()
    metrics = calculate_metrics(pred_labels, gt_labels)
    
    return metrics

# Main execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = [load_model(path, device) for path in model_paths]

output_dir = '../../vectorised_model_predictions/multi-modal/final_predictions_averaged/'
os.makedirs(output_dir, exist_ok=True)

# Use existing scenes and datasets
scenes = {
    'SantoDomingo': (sentinel_sceneSD, buildings_sceneSD),
    'GuatemalaCity': (sentinel_sceneGC, buildings_sceneGC),
    'Tegucigalpa': (sentinel_sceneTG, buildings_sceneTG),
    'Managua': (sentinel_sceneMN, buildings_sceneMN),
    'Panama': (sentinel_scenePN, buildings_scenePN)
}

all_metrics = {}

for city, (sentinel_scene, buildings_scene) in scenes.items():
    print(f"Processing {city}...")
    all_metrics[city] = process_city(city, sentinel_scene, buildings_scene, models, device, output_dir, cities)

# Print and save metrics
with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
    for city, metrics in all_metrics.items():
        print(f"\nMetrics for {city}:")
        f.write(f"\nMetrics for {city}:\n")
        if any(math.isnan(value) for value in metrics.values()):
            print("No correct predictions made. Metrics are NaN.")
            f.write("No correct predictions made. Metrics are NaN.\n")
        else:
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
                f.write(f"{metric.capitalize()}: {value:.4f}\n")

    # Calculate and print average metrics
    valid_metrics = {k: v for k, v in all_metrics.items() if not any(math.isnan(value) for value in v.values())}
    if valid_metrics:
        avg_metrics = {
            metric: sum(city_metrics[metric] for city_metrics in valid_metrics.values()) / len(valid_metrics)
            for metric in ['f1', 'precision', 'recall']
        }
        print("\nAverage metrics across cities:")
        f.write("\nAverage metrics across cities:\n")
        for metric, value in avg_metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
            f.write(f"{metric.capitalize()}: {value:.4f}\n")
        
        excluded_cities = set(all_metrics.keys()) - set(valid_metrics.keys())
        if excluded_cities:
            note = f"\n(Note: {', '.join(excluded_cities)} {'was' if len(excluded_cities) == 1 else 'were'} excluded due to NaN metrics)"
            print(note)
            f.write(note + "\n")
    else:
        print("\nUnable to calculate average metrics. All cities have NaN values.")
        f.write("\nUnable to calculate average metrics. All cities have NaN values.\n")

print(f"Predictions and metrics have been saved to {output_dir}")







# # Visualise feature maps

# def recursive_to_device(module, device):
#     for child in module.children():
#         recursive_to_device(child, device)
#     module.to(device)

# # Use it like this:
# recursive_to_device(best_model, device)

# visualizer = FeatureMapVisualization(best_model, device=device)
# visualizer.add_hooks([
#     'buildings_encoder.0', # conv2d
#     'buildings_encoder.1', # batchnorm
#     'buildings_encoder.2', # relu
#     'buildings_encoder.3', # maxpool
#     'buildings_encoder.4', # maxpool
#     'encoder.conv1',
#     'encoder.layer1.0.conv1',
#     'encoder.layer4.2.conv3',
#     'segmentation_head.0',
#     'segmentation_head.1',
#     'segmentation_head.4',
#     'segmentation_head[-1]'
# ])

# # Get the iterator for the DataLoader
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# best_model = best_model.to(device)

# visualizer = FeatureMapVisualization(best_model, device)

# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

# data_iter = iter(val_loader)
# first_batch = next(data_iter)

# visualizer.visualize_feature_maps('buildings_encoder.0', first_batch, num_feature_maps=16)
# # Move the batches to the device

# visualizer.visualize_feature_maps('buildings_encoder.0', first_batch, num_feature_maps=16) # conv2d
# visualizer.visualize_feature_maps('buildings_encoder.1', second_batch, num_feature_maps=16) # batchnorm
# visualizer.visualize_feature_maps('buildings_encoder.2', second_batch, num_feature_maps=16) # relu
# visualizer.visualize_feature_maps('buildings_encoder.3', second_batch, num_feature_maps=16) # maxpool
# visualizer.visualize_feature_maps('buildings_encoder.4', second_batch, num_feature_maps='all') #conv2d
# visualizer.visualize_feature_maps('encoder.layer1.0.conv1', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('encoder.layer4.2.conv3', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.0', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.1', second_batch, num_feature_maps=16)
# visualizer.visualize_feature_maps('segmentation_head.4', second_batch, num_feature_maps=16)
# visualizer.remove_hooks()


# # # Visualise filters
# # def visualize_filters(model, layer_name, num_filters=8):
#     # Get the layer by name
#     layer = dict(model.named_modules())[layer_name]
#     assert isinstance(layer, nn.Conv2d), "Layer should be of type nn.Conv2d"

#     # Get the weights of the filters
#     filters = layer.weight.data.clone().cpu().numpy()

#     # Normalize the filters to [0, 1] range for visualization
#     min_filter, max_filter = filters.min(), filters.max()
#     filters = (filters - min_filter) / (max_filter - min_filter)
    
#     # Plot the filters
#     num_filters = min(num_filters, filters.shape[0])  # Limit to number of available filters
#     fig, axes = plt.subplots(1, num_filters, figsize=(20, 10))
    
#     for i, ax in enumerate(axes):
#         filter_img = filters[i]
        
#         # If the filter has more than one channel, average the channels for visualization
#         if filter_img.shape[0] > 1:
#             filter_img = np.mean(filter_img, axis=0)
        
#         cax = ax.imshow(filter_img, cmap='viridis')
#         ax.axis('off')
    
#     fig.subplots_adjust(right=0.8)
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(cax, cax=cbar_ax)

#     plt.show()
    
# visualize_filters(best_model, 'segmentation_head.0', num_filters=8)

# # FLOPS
# # flops = FlopCountAnalysis(model, batch)

# # print(flops.total())
# # print(flops.by_module())

# # print(parameter_count_table(model))