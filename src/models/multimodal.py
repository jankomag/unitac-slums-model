import os
import sys
import matplotlib.pyplot as plt
import torch
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import pytorch_lightning as pl

from torchvision.models.segmentation import deeplabv3_resnet50
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import math
from ptflops import get_model_complexity_info
import torch
import torch
from pytorch_lightning.callbacks import Callback

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params, MultiResolutionDeepLabV3, save_unseen_predictions_multimodal,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot, merge_geojson_files, generate_and_display_predictions,
                                          FeatureMapVisualization, PredictionVisualizationCallback, ModelLoader)

from src.features.dataloaders import (cities, show_windows, MultiInputCrossValidator,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn,show_first_batch_item,
                                  MergeDataset, show_single_tile_multi, get_label_source_from_merge_dataset, create_scenes_for_city, PolygonWindowGeoDataset)

from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data import (ClassConfig, SemanticSegmentationDiscreteLabels)
from rastervision.pytorch_learner.dataset.transform import TransformType
from rastervision.core.evaluation import SemanticSegmentationEvaluator
             
class PredictionVisualizationCallback(Callback):
    def __init__(self, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            device = pl_module.device  # Get the device from the model
            generate_and_display_predictions(pl_module, sentinel_sceneSD, buildings_sceneSD, device, trainer.current_epoch, class_config)

           
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

# Create datasets
train_cities = 'sel' # 'sel'
split_index = 1 # 0 or 1

multimodal_datasets = {
    'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD),
    'GuatemalaCity': MergeDataset(sentinelGeoDataset_GC, buildGeoDataset_GC),
    'Tegucigalpa': MergeDataset(sentinelGeoDataset_TG, buildGeoDataset_TG),
    'Managua': MergeDataset(sentinelGeoDataset_MN, buildGeoDataset_MN),
    'Panama': MergeDataset(sentinelGeoDataset_PN, buildGeoDataset_PN),
    'SanJose': MergeDataset(sentinelGeoDataset_SJ, buildGeoDataset_SJ),
    } if train_cities == 'sel' else {'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD)}

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

batch_size = 32
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
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'gamma': 0.8,
    'sched_step_size': 5,
    'pos_weight': 2.0,
    'buil_out_chan': 4
}

output_dir = f'../UNITAC-trained-models/multi_modal/{train_cities}_MultiResDLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_f1',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_{train_cities}_cv{split_index}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=6,
    mode='max')

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=20)
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
    min_epochs=50,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)


# Train the model
trainer.fit(model, datamodule=data_module)

##########################################
###### Individual Model Predictions ######
##########################################
# model_id = 'multimodal_selSJ2_cv1_epoch=36-val_loss=0.1872.ckpt'
# best_model_path = os.path.join(parent_dir, f'UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/', model_id)
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
#     os.path.join(parent_dir, 'UNITAC-trained-models/multi_modal/selSJ2_CustomDLV3/multimodal_selSJ2_cv0_epoch=17-val_loss=0.2936.ckpt'),
#     os.path.join(parent_dir, 'UNITAC-trained-models/multi_modal/selSJ2_CustomDLV3/multimodal_selSJ2_cv1_epoch=44-val_loss=0.2413.ckpt')
# ]

# models = [ModelLoader.load_model(path, device) for path in model_paths]

# output_dir = '../../vectorised_model_predictions/multimodal/unseen_predictions_selSJ2/'

# try:
#     save_unseen_predictions_multimodal(models, cv, multimodal_datasets, cities, class_config, device, output_dir)

#     merged_output_file = os.path.join(output_dir, 'combined_predictions.geojson')
#     merge_geojson_files(output_dir, merged_output_file)

#     print("Predictions saved and combined successfully.")
# except Exception as e:
#     print(f"An error occurred during prediction or merging: {str(e)}")

# #############################
# ##### Model Complexity ######
# #############################

# def multimodal_input_constructor(input_res):
#     sentinel = torch.randn(1, 4, 256, 256)  # 4 channels, 256x256 image size
#     buildings = torch.randn(1, 1, 512, 512)  # 1 channel, 512x512 image size
#     return ((sentinel, None), (buildings, None))

# model = MultiResolutionDeepLabV3()
# macs, params = get_model_complexity_info(
#     model, 
#     (4, 256, 256),  # This is ignored, but needed for the function
#     input_constructor=multimodal_input_constructor,
#     as_strings=True, 
#     print_per_layer_stat=True, 
#     verbose=True
# )

# print(f'MultiResolutionDeepLabV3 - Computational complexity: {macs}')
# print(f'MultiResolutionDeepLabV3 - Number of parameters: {params}')


# ##############################
# ### Visualise feature maps ###
# ##############################

# model_id = 'multimodal_selSJ_cv1_epoch=27-val_loss=0.2128.ckpt'
# best_model_path = os.path.join(parent_dir, f'UNITAC-trained-models/multi_modal/selSJ_CustomDLV3/', model_id)
# best_model = MultiResolutionDeepLabV3()#buil_channels=buil_channels, buil_kernel=buil_kernel, buil_out_chan=4)
# checkpoint = torch.load(best_model_path)
# state_dict = checkpoint['state_dict']
# best_model.load_state_dict(state_dict, strict=True)
# best_model = best_model.to(device)
# best_model.eval()
# check_nan_params(best_model)

# # Feature Map Visualisation class
# visualizer = FeatureMapVisualization(best_model, device=device)
# visualizer.add_hooks([
#     'buildings_encoder.0.0', # conv2d
#     'buildings_encoder.1.0', # conv2d
#     'xtra_fusion',
#     'decoder_fusion'
# ])

# # Get the iterator for the DataLoader
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, collate_fn=collate_multi_fn)
# data_iter = iter(val_loader)
# first_batch = next(data_iter)

# show_first_batch_item(first_batch)

# visualizer.visualize_feature_maps('buildings_encoder.0.0', first_batch, num_feature_maps='all')
# visualizer.visualize_feature_maps('buildings_encoder.1.0', first_batch, num_feature_maps='all')
# visualizer.visualize_feature_maps('xtra_fusion', first_batch, num_feature_maps='all')
# visualizer.visualize_feature_maps('decoder_fusion', first_batch, num_feature_maps='all')

# # Feature Visaualization #
# for layer_name in ['buildings_encoder.1.0', 'xtra_fusion', 'decoder_fusion']:
#     print(f"Visualizing maximized activation for layer: {layer_name}")
#     visualizer.visualize_maximized_activation(layer_name, num_iterations=500, learning_rate=0.1)

# visualizer.remove_hooks()