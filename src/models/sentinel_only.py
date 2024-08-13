import os
from rastervision.core.box import Box
import math
from subprocess import check_output

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()
from torch.utils.data import Subset
import math
import sys
from datetime import datetime
import torch
import wandb
from ptflops import get_model_complexity_info

from rastervision.pytorch_learner.dataset.transform import TransformType
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (SentinelDeepLabV3, PredictionsIterator, create_predictions_and_ground_truth_plot, ModelLoader,
                                          check_nan_params, merge_geojson_files, save_unseen_predictions_sentinel)
from src.features.dataloaders import (create_sentinel_scene, cities, CustomSlidingWindowGeoDataset, collate_fn, PolygonWindowGeoDataset,
                                      SingleInputCrossValidator, singlesource_show_windows_for_city, show_single_tile_sentinel)

from rastervision.core.data import ClassConfig, SemanticSegmentationLabels, SemanticSegmentationDiscreteLabels
from rastervision.core.evaluation import SemanticSegmentationEvaluator

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
SentinelScene_MN = create_sentinel_scene(cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(SentinelScene_MN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
sentinel_scenePN = create_sentinel_scene(cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# San Salvador
sentinel_sceneSJ = create_sentinel_scene(cities['SanJose'], class_config)
sentinelGeoDataset_SJ = PolygonWindowGeoDataset(sentinel_sceneSJ, city='SanSalvador', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets
train_cities = 'selSJ'
split_index = 1 # 0 or 1

def get_sentinel_datasets(train_cities):
    all_datasets = {
        'SantoDomingo': sentinelGeoDataset_SD,
        'GuatemalaCity': sentinelGeoDataset_GC,
        'Tegucigalpa': sentinelGeoDataset_TG,
        'Managua': sentinelGeoDataset_MN,
        'Panama': sentinelGeoDataset_PN,
        'SanJose': sentinelGeoDataset_SJ
    }
    if train_cities == 'selSJ':
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
city = 'SanJose'
singlesource_show_windows_for_city(city, split_index, cv, sentinel_datasets)
show_single_tile_sentinel(sentinel_datasets, city, 39)

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
    'weight_decay': 0.00001,
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

output_dir = f'../UNITAC-trained-models/sentinel_only/{train_cities}_DLV3'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-finetune-sentinel-only', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_mean_iou',
    dirpath=output_dir,
    filename=f'sentinel_{train_cities}_cv{split_index}-{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=4,
    save_last=True,
    mode='max')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=20)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=30,
    max_epochs=250,
    num_sanity_val_steps=3,
)

# Train the model
trainer.fit(model, train_loader, val_loader)

##########################################
###### Individual Model Predictions ######
##########################################

model_id = 'sentinel_selSJ_cv1-epoch=36-val_loss=0.2694.ckpt'
best_model_path = os.path.join(parent_dir, f'../UNITAC-trained-models/sentinel_only/{train_cities}_DLV3/', model_id)
# best_model_path = checkpoint_callback.best_model_path
scene_eval = SentinelScene_SD

model = SentinelDeepLabV3.load_from_checkpoint(best_model_path)
model.eval()
check_nan_params(model)

strided_fullds = CustomSlidingWindowGeoDataset(scene_eval, size=256, stride=128, padding=0, city='SD', transform=None, transform_type=TransformType.noop)
predictions_iterator = PredictionsIterator(model, strided_fullds, device=device)
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
    city_scene = sentinel_datasets[city].scene

    try:
        metrics = calculate_sentinel_metrics(model, city_val_dataset, device, city_scene)
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

    unweighted_metrics = {
        'f1': sum(metrics['f1'] for metrics in valid_metrics.values()) / len(valid_metrics),
        'precision': sum(metrics['precision'] for metrics in valid_metrics.values()) / len(valid_metrics),
        'recall': sum(metrics['recall'] for metrics in valid_metrics.values()) / len(valid_metrics)
    }

    print("\nWeighted average metrics across cities:")
    print(f"F1 Score: {weighted_metrics['f1']:.4f}")
    print(f"Precision: {weighted_metrics['precision']:.4f}")
    print(f"Recall: {weighted_metrics['recall']:.4f}")

    print("\nUnweighted average metrics across cities:")
    print(f"F1 Score: {unweighted_metrics['f1']:.4f}")
    print(f"Precision: {unweighted_metrics['precision']:.4f}")
    print(f"Recall: {unweighted_metrics['recall']:.4f}")
    
    if excluded_cities:
        print(f"\n(Note: {', '.join(excluded_cities)} {'was' if len(excluded_cities) == 1 else 'were'} excluded due to NaN metrics)")
else:
    print("\nUnable to calculate average metrics. All cities have NaN values.")

################################
###### Unseen Predictions ######
################################

# model_paths = [
#     os.path.join(parent_dir, 'UNITAC-trained-models/sentinel_only/selSJ_DLV3/sentinel_selSJ_cv0-epoch=23-val_loss=0.3669.ckpt'),
#     os.path.join(parent_dir, 'UNITAC-trained-models/sentinel_only/selSJ_DLV3/sentinel_selSJ_cv1-epoch=36-val_loss=0.2694.ckpt')
# ]

# models = [ModelLoader.load_model(path) for path in model_paths]

# selected_cities = ['SantoDomingo', 'GuatemalaCity', 'Tegucigalpa', 'Managua', 'Panama', 'SanJose']
# all_datasets = {city: sentinel_datasets[city] for city in selected_cities}

# output_dir = '../vectorised_model_predictions/sentinel_only/unseen_predictions_selSJ_final/'
# save_unseen_predictions_sentinel(models, cv, all_datasets, cities, class_config, device, output_dir)

# merged_output_file = os.path.join(output_dir, 'combined_predictions_final.geojson')
# merge_geojson_files(output_dir, merged_output_file)

# print("Predictions saved and combined successfully.")

# ##############################
# ###### Model Complexity ######
# ##############################

# def input_constructor(input_res):
#     return torch.randn(1, 4, 256, 256)

# model = SentinelDeepLabV3()
# macs, params = get_model_complexity_info(
#     model, 
#     (4, 256, 256),
#     input_constructor=input_constructor,
#     as_strings=True, 
#     print_per_layer_stat=True, 
#     verbose=True
# )

# print(f'SentinelDeepLabV3 - Computational complexity: {macs}')
# print(f'SentinelDeepLabV3 - Number of parameters: {params}')