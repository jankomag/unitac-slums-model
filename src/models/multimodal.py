import os
import sys
from datetime import datetime

# import multiprocessing
# multiprocessing.set_start_method('fork')
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from affine import Affine
import wandb
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import stackstac
import pystac_client
import albumentations as A
from rastervision.core.box import Box
import rasterio
from rasterio.transform import from_bounds
import tempfile
import seaborn as sns
# import folium
from torch.utils.data import ConcatDataset
import math
# from sklearn.model_selection import KFold

# from fvcore.nn import FlopCountAnalysis
# from torchinfo import summary  # Optional, for detailed summary

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResSentLabelPredictionsIterator, MultiModalDataModule)
from deeplnafrica.deepLNAfrica import (Deeplabv3SegmentationModel, init_segm_model, 
                                       CustomDeeplabv3SegmentationModel)
from src.data.dataloaders import (buil_create_full_image, show_windows, cities,
                                  create_datasets, CustomStatsTransformer, senitnel_create_full_image,
                                  CustomSemanticSegmentationSlidingWindowGeoDataset, MergeDataset, create_scenes_for_city)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.pytorch_learner import SemanticSegmentationVisualizer
from rastervision.core.data import (Scene, ClassConfig, RasterioCRSTransformer, XarraySource,
                                    RasterioSource, GeoJSONVectorSource, StatsTransformer,
                                    ClassInferenceTransformer, RasterizedSource,
                                    SemanticSegmentationLabelSource, VectorSource)

from rastervision.core.data.label_store import (LabelStoreConfig, SemanticSegmentationLabelStore)

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

# Santo Domingo with augmentation
sentinel_sceneSD, buildings_sceneSD = create_scenes_for_city('SantoDomingoDOM', cities['SantoDomingoDOM'], class_config)

sentinelGeoDataset_SD, train_sentinel_ds_SD, val_sent_ds_SD, test_sentinel_ds_SD = create_datasets(sentinel_sceneSD, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
sentinelGeoDataset_SD_aug, train_sentinel_ds_SD_aug, val_sent_ds_SD_aug, test_sentinel_ds_SD_aug = create_datasets(sentinel_sceneSD, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=True, seed=12)

sent_train_ds_SD = ConcatDataset([train_sentinel_ds_SD, train_sentinel_ds_SD_aug])
sent_val_ds_SD = ConcatDataset([val_sent_ds_SD, val_sent_ds_SD_aug])

buildingsGeoDataset_SD, train_buil_ds_SD, val_buil_ds_SD, test_buil_ds_SD = create_datasets(buildings_sceneSD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
buildingsGeoDataset_SD_aug, train_buil_ds_SD_aug, val_buil_ds_SD_aug, test_buil_ds_SD_aug = create_datasets(buildings_sceneSD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = True, seed=12)

build_train_ds_SD = ConcatDataset([train_buil_ds_SD, train_buil_ds_SD_aug])
build_val_ds_SD = ConcatDataset([val_buil_ds_SD, val_buil_ds_SD_aug])

train_dataset = MergeDataset(sent_train_ds_SD, build_train_ds_SD)
val_dataset = MergeDataset(sent_val_ds_SD, build_val_ds_SD)

batch_size = 6
train_multiple_cities=False

if train_multiple_cities:
    # Guatemala City
    sentinel_sceneGC, buildings_sceneGC = create_scenes_for_city('GuatemalaCity', cities['GuatemalaCity'], class_config)
    sentinelGeoDataset_GC, train_sentinel_ds_GC, val_sent_ds_GC, test_sentinel_ds_GC = create_datasets(sentinel_sceneGC, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    buildingsGeoDataset_GC, train_buil_ds_GC, val_buil_ds_GC, test_buil_ds_GC = create_datasets(buildings_sceneGC, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    train_datasetGC = MergeDataset(train_sentinel_ds_GC, train_buil_ds_GC)
    val_datasetGC = MergeDataset(val_sent_ds_GC, val_buil_ds_GC)

    # TegucigalpaHND
    sentinel_sceneTG, buildings_sceneTG = create_scenes_for_city('TegucigalpaHND', cities['TegucigalpaHND'], class_config)
    sentinelGeoDataset_TG, train_sentinel_ds_TG, val_sent_ds_TG, test_sentinel_ds_TG = create_datasets(sentinel_sceneTG, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    buildingsGeoDataset_TG, train_buil_ds_TG, val_buil_ds_TG, test_buil_ds_TG = create_datasets(buildings_sceneTG, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    train_datasetTG = MergeDataset(train_sentinel_ds_TG, train_buil_ds_TG)
    val_datasetTG = MergeDataset(val_sent_ds_TG, val_buil_ds_TG)
    
    # Managua
    sentinel_sceneMN, buildings_sceneMN = create_scenes_for_city('Managua', cities['Managua'], class_config)
    sentinelGeoDataset_MN, train_sentinel_ds_MN, val_sent_ds_MN, test_sentinel_ds_MN = create_datasets(sentinel_sceneMN, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    buildingsGeoDataset_MN, train_buil_ds_MN, val_buil_ds_MN, test_buil_ds_MN = create_datasets(buildings_sceneMN, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    train_datasetMN = MergeDataset(train_sentinel_ds_MN, train_buil_ds_MN)
    val_datasetMN = MergeDataset(val_sent_ds_MN, val_buil_ds_MN)
    
    # Panama
    sentinel_scenePN, buildings_scenePN = create_scenes_for_city('Panama', cities['Panama'], class_config)
    sentinelGeoDataset_PN, train_sentinel_ds_PN, val_sent_ds_PN, test_sentinel_ds_PN = create_datasets(sentinel_scenePN, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    buildingsGeoDataset_PN, train_buil_ds_PN, val_buil_ds_PN, test_buil_ds_PN = create_datasets(buildings_scenePN, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    train_datasetPN = MergeDataset(train_sentinel_ds_PN, train_buil_ds_PN)
    val_datasetPN = MergeDataset(val_sent_ds_PN, val_buil_ds_PN)
    
    # San Salvador
    # sentinel_sceneSS, buildings_sceneSS = create_scenes_for_city('SanSalvador_PS', cities['SanSalvador_PS'], class_config)
    # sentinelGeoDataset_SS, train_sentinel_ds_SS, val_sent_ds_SS, test_sentinel_ds_SS = create_datasets(sentinel_sceneSS, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    # buildingsGeoDataset_SS, train_buil_ds_SS, val_buil_ds_SS, test_buil_ds_SS = create_datasets(buildings_sceneSS, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    # train_datasetSS = MergeDataset(train_sentinel_ds_SS, train_buil_ds_SS)
    # val_datasetSS = MergeDataset(val_sent_ds_SS, val_buil_ds_SS)
    
    # SanJoseCRI
    # sentinel_sceneSJ, buildings_sceneSJ = create_scenes_for_city('SanJoseCRI', cities['SanJoseCRI'], class_config)
    # sentinelGeoDataset_SJ, train_sentinel_ds_SJ, val_sent_ds_SJ, test_sentinel_ds_SJ = create_datasets(sentinel_sceneSJ, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
    # buildingsGeoDataset_SJ, train_buil_ds_SJ, val_buil_ds_SJ, test_buil_ds_SJ = create_datasets(buildings_sceneSJ, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
    # train_datasetSJ = MergeDataset(train_sentinel_ds_SJ, train_buil_ds_SJ)
    # val_datasetSJ = MergeDataset(val_sent_ds_SJ, val_buil_ds_SJ)

    train_dataset = ConcatDataset([train_dataset, train_datasetGC, train_datasetTG, train_datasetMN, train_datasetPN]) # train_datasetSJ, train_datasetSS
    val_dataset = ConcatDataset([val_dataset, val_datasetGC, val_datasetTG, val_datasetMN, val_datasetPN]) 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# # Preview sliding windows:
# img_full = senitnel_create_full_image(sentinelGeoDataset_SD.scene.label_source)
# train_windows = train_sentinel_ds_SD.windows
# val_windows = val_sent_ds_SD.windows
# test_windows = test_sentinel_ds_SD.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# img_full = buil_create_full_image(buildingsGeoDataset_SD.scene.raster_source)
# train_windows = train_buil_ds_SD.windows
# val_windows = val_buil_ds_SD.windows
# test_windows = test_buil_ds_SD.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# # Preview Sentinel and Buildings batches
# vis_sent = SemanticSegmentationVisualizer(
#     class_names=class_config.names, class_colors=class_config.colors,
#     channel_display_groups={'RGB': (0,1,2), 'NIR': (3, )})

# vis_build = SemanticSegmentationVisualizer(
#     class_names=class_config.names, class_colors=class_config.colors,
#     channel_display_groups={'Buildings': (0,)})

# x, y = vis_sent.get_batch(sentinelGeoDataset_SD, 2)
# vis_sent.plot_batch(x, y, show=True)

# x, y = vis_build.get_batch(buildingsGeoDataset_SD, 2)
# vis_build.plot_batch(x, y, show=True)

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# Train the model 
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'SD',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'labels_size': 256,
    'buil_channels': 16,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-2,
    'weight_decay': 0,
    'gamma': 1,
    'sched_step_size': 40,
    'pos_weight': 2.0,
    'buil_kernel1': 3
}

output_dir = f'../UNITAC-trained-models/multi_modal/SD_DLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
buil_channels = hyperparameters['buil_channels']
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_last=False,
    dirpath=output_dir,
    filename=f'multimodal_builchan{buil_channels}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=3,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=30)

model = MultiResolutionDeepLabV3(
    use_deeplnafrica=hyperparameters['use_deeplnafrica'],
    learning_rate=hyperparameters['learning_rate'],
    weight_decay=hyperparameters['weight_decay'],
    gamma=hyperparameters['gamma'],
    atrous_rates=hyperparameters['atrous_rates'],
    sched_step_size = hyperparameters['sched_step_size'],
    buil_channels = hyperparameters['buil_channels'],
    buil_kernel1 = hyperparameters['buil_kernel1'],
    pos_weight=torch.tensor(hyperparameters['pos_weight'], device='mps')
)
model.to(device)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=20,
    max_epochs=150,
    num_sanity_val_steps=3,
    # overfit_batches=0.35
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation # best multimodal_epoch=09-val_loss=0.2434.ckpt
# best_model_path_dplv3 = "/Users/janmagnuszewski/dev/UNITAC-trained-models/multi_modal/SD_DLV3/multimodal_builchan16_epoch=03-val_loss=0.7799.ckpt"
best_model_path_dplv3 = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3(buil_channels=16, buil_kernel1=3) #MultiResolutionDeepLabV3 MultiResolutionFPN
checkpoint = torch.load(best_model_path_dplv3)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict)
best_model.eval()

class MultiResSentLabelPredictionsIterator:
    def __init__(self, model, sentinelGeoDataset, buildingsGeoDataset, device='cuda'):
        self.model = model
        self.sentinelGeoDataset = sentinelGeoDataset
        self.dataset = buildingsGeoDataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(sentinelGeoDataset)):
                # print(f"Predicting for window {idx+1}/{len(buildingsGeoDataset)}")
                buildings = buildingsGeoDataset[idx]
                sentinel = sentinelGeoDataset[idx]
                
                sentinel_data = sentinel[0].unsqueeze(0).to(device)
                sentlabels = sentinel[1].unsqueeze(0).to(device)

                buildings_data = buildings[0].unsqueeze(0).to(device)
                labels = buildings[1].unsqueeze(0).to(device)

                output = self.model(((sentinel_data,sentlabels), (buildings_data,labels)))
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = sentinelGeoDataset.windows[idx] # here has to be sentinelGeoDataset to work
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)

buildingsGeoDataset, _, _, _ = create_datasets(buildings_sceneSD, imgsize=512, stride = 256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, _, _, _ = create_datasets(sentinel_sceneSD, imgsize=256, stride = 128, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

predictions_iterator = MultiResSentLabelPredictionsIterator(best_model, sentinelGeoDataset, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)
assert len(windows) == len(predictions)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=sentinel_sceneSD.extent,
    num_classes=len(class_config),
    smooth=True
)

# Show predictions
scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('probability map')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# crs_transformer = RasterioCRSTransformer.from_uri(image_uriSD)
# affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
# crs_transformer.transform = affine_transform_buildings

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/multi-modal/SD_DLV3/',
#     crs_transformer = crs_transformer,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)

# # Evaluate predictions
from rastervision.core.evaluation import SemanticSegmentationEvaluator

evaluator = SemanticSegmentationEvaluator(class_config)
gt_labels = sentinel_sceneSD.label_source.get_labels()
evaluation = evaluator.evaluate_predictions(
    ground_truth=gt_labels, predictions=pred_labels)
eval_metrics_dict = evaluation.class_to_eval_item[1]
f1_score = eval_metrics_dict.f1
precision = eval_metrics_dict.precision
print(eval_metrics_dict)
# # Visualise feature maps
# class FeatureMapVisualization:
#     def __init__(self, model):
#         self.model = model
#         self.feature_maps = {}
#         self.hooks = []

#     def add_hooks(self, layer_names):
#         for name, module in self.model.named_modules():
#             if name in layer_names:
#                 self.hooks.append(module.register_forward_hook(self.save_feature_map(name)))

#     def save_feature_map(self, name):
#         def hook(module, input, output):
#             self.feature_maps[name] = output
#         return hook

#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()

#     def visualize_feature_maps(self, layer_name, input_data, num_feature_maps='all', figsize=(20, 20)):
#         self.model.eval()
#         with torch.no_grad():
#             self.model(input_data)

#         if layer_name not in self.feature_maps:
#             print(f"Layer {layer_name} not found in feature maps.")
#             return

#         feature_maps = self.feature_maps[layer_name].cpu().detach().numpy()

#         # Handle different dimensions
#         if feature_maps.ndim == 4:  # (batch_size, channels, height, width)
#             feature_maps = feature_maps[0]  # Take the first item in the batch
#         elif feature_maps.ndim == 3:  # (channels, height, width)
#             pass
#         else:
#             print(f"Unexpected feature map shape: {feature_maps.shape}")
#             return

#         total_maps = feature_maps.shape[0]
        
#         if num_feature_maps == 'all':
#             num_feature_maps = total_maps
#         else:
#             num_feature_maps = min(num_feature_maps, total_maps)

#         # Calculate grid size
#         grid_size = math.ceil(math.sqrt(num_feature_maps))
        
#         fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        
#         if grid_size == 1:
#             axes = np.array([[axes]])
#         elif grid_size > 1 and axes.ndim == 1:
#             axes = axes.reshape(1, -1)

#         for i in range(grid_size):
#             for j in range(grid_size):
#                 index = i * grid_size + j
#                 if index < num_feature_maps:
#                     feature_map_img = feature_maps[index]
#                     im = axes[i, j].imshow(feature_map_img, cmap='viridis')
#                     axes[i, j].axis('off')
#                     axes[i, j].set_title(f'Channel {index+1}')
#                 else:
#                     axes[i, j].axis('off')

#         fig.suptitle(f'Feature Maps for Layer: {layer_name}\n({num_feature_maps} out of {total_maps} channels)')
#         fig.tight_layout()
        
#         # Add colorbar
#         fig.subplots_adjust(right=0.9)
#         cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#         fig.colorbar(im, cax=cbar_ax)
        
#         plt.show()

# visualizer = FeatureMapVisualization(best_model)
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
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
# data_iter = iter(val_loader)
# first_batch = next(data_iter)
# second_batch = next(data_iter)
# # third_batch = next(data_iter)

# x, y = vis_build.get_batch(val_buildings_dataset_SD, 2)
# vis_build.plot_batch(x, y, show=True)

# visualizer.visualize_feature_maps('buildings_encoder.0', second_batch, num_feature_maps=16) # conv2d
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


# # Visualise filters
# def visualize_filters(model, layer_name, num_filters=8):
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