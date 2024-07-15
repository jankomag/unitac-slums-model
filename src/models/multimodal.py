import os
import sys
import argparse
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import wandb
import geopandas as gpd
from torch.utils.data import ConcatDataset
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

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params, CustomInterpolateMultiResolutionDeepLabV3,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot, MultiResolutionFPN, CustomMultiResolutionDeepLabV3,
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
train_cities = 'SD' # 'sel'
split_index = 0 # 0 or 1
buil_channels = 128
buil_kernel = 5
labels_size = 256

multimodal_datasets = {
    'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD),
    'GuatemalaCity': MergeDataset(sentinelGeoDataset_GC, buildGeoDataset_GC),
    'Tegucigalpa': MergeDataset(sentinelGeoDataset_TG, buildGeoDataset_TG),
    'Managua': MergeDataset(sentinelGeoDataset_MN, buildGeoDataset_MN),
    'Panama': MergeDataset(sentinelGeoDataset_PN, buildGeoDataset_PN),
} if train_cities == 'sel' else {'SD': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD)}

cv = MultiInputCrossValidator(multimodal_datasets, n_splits=2, val_ratio=0.5, test_ratio=0)

# Preview a city with sliding windows
city = 'SD'
windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
img_full = senitnel_create_full_image(get_label_source_from_merge_dataset(multimodal_datasets[city]))
show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')
show_single_tile_multi(multimodal_datasets, city, 11, show_sentinel=True, show_buildings=True)

train_dataset, val_dataset, _, val_city_indices = cv.get_split(split_index)
print(f"Train dataset size: {len(train_dataset)}") 
print(f"Validation dataset size: {len(val_dataset)}")

batch_size = 16
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
    'labels_size': labels_size,
    'buil_channels': buil_channels,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0.7,
    'gamma': 0.5,
    'sched_step_size': 14,
    'pos_weight': 2.0,
    'buil_kernel': buil_kernel,
    'buil_out_chan': 4
}

output_dir = f'../../UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_{train_cities}_cv{split_index}_res{labels_size}_BCH{buil_channels}_BKR{buil_kernel}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=3,
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

early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=8)
visualization_callback = PredictionVisualizationCallback(every_n_epochs=5)  # Adjust the frequency as needed

model = CustomInterpolateMultiResolutionDeepLabV3(
    use_deeplnafrica=hyperparameters['use_deeplnafrica'],
    learning_rate=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    gamma=hyperparameters['gamma'],
    atrous_rates=hyperparameters['atrous_rates'],
    sched_step_size = hyperparameters['sched_step_size'],
    buil_channels = buil_channels,
    buil_kernel = buil_kernel,
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
    min_epochs=25,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
# model_id = 'last-v1.ckpt'
# best_model_path = os.path.join(grandparent_dir, f'UNITAC-trained-models/multi_modal/{train_cities}_CustomDLV3/', model_id)

best_model_path = checkpoint_callback.best_model_path
best_model = CustomInterpolateMultiResolutionDeepLabV3()#buil_channels=buil_channels, buil_kernel=buil_kernel, buil_out_chan=4)
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
print(f"F1:{inf_eval.f1}")

# Calculate F1 scores
def calculate_multimodal_f1_score(model, merged_dataset, device, scene):
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
    return inf_eval.f1

city_f1_scores = {}

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
        # Calculate F1 score for this city
        f1_score = calculate_multimodal_f1_score(best_model, city_val_dataset, device, city_scene)
        
        city_f1_scores[city] = f1_score
    except Exception as e:
        print(f"Error calculating F1 score for {city}: {str(e)}")

# Print summary of cities with F1 scores
print(f"\nSummary of multimodal F1 scores for split {split_index}:")
for city, score in city_f1_scores.items():
    print(f"{city}: {score}")

vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

crs_transformer = RasterioCRSTransformer.from_uri(cities['SantoDomingo']['image_path'])
gdf = gpd.read_file(cities['SantoDomingo']['labels_path'])
gdf = gdf.to_crs('epsg:3857')
xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
crs_transformer.transform = affine_transform_buildings

pred_label_store = SemanticSegmentationLabelStore(
    uri=f'../../vectorised_model_predictions/multi-modal/{train_cities}_DLV3/',
    crs_transformer = crs_transformer,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)







# best_model = best_model.to(device)

# def recursive_to_device(module, device):
#     for child in module.children():
#         recursive_to_device(child, device)
#     module.to(device)

# # Use it like this:
# recursive_to_device(best_model, device)

# # Visualise feature maps
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