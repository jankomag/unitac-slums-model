import os
import sys
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

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot,
                                          CustomVectorOutputConfig, FeatureMapVisualization)
from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple, MultiInputCrossValidator,
                                  create_datasets, senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_fn,
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
sentinel_sceneSD, buildings_sceneSD = create_scenes_for_city('SantoDomingo', cities['SantoDomingoDOM'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(sentinel_sceneSD, city= 'SantoDomingo', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, city= 'SantoDomingo',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# GuatemalaCity
sentinel_sceneGC, buildings_sceneGC = create_scenes_for_city('GuatemalaCity', cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, city='GuatemalaCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, city='GuatemalaCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneTG, buildings_sceneTG = create_scenes_for_city('Tegucigalpa', cities['TegucigalpaHND'], class_config)
sentinelGeoDataset_TG = CustomSlidingWindowGeoDataset(sentinel_sceneTG, city='Tegucigalpa', size=256, stride = 256, out_size=256, padding=0, transform_type=TransformType.noop, transform=None)
buildGeoDataset_TG = CustomSlidingWindowGeoDataset(buildings_sceneTG, city='Tegucigalpa', size=512, stride = 512, out_size=512, padding=0, transform_type=TransformType.noop, transform=None)

# Managua
sentinel_sceneMN, buildings_sceneMN = create_scenes_for_city('Managua', cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, city='Managua', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
sentinel_scenePN, buildings_scenePN = create_scenes_for_city('Panama', cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN, city='Panama', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# San Salvador - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneSS, buildings_sceneSS = create_scenes_for_city('SanSalvador_PS', cities['SanSalvador_PS'], class_config)
sentinelGeoDataset_SS = CustomSlidingWindowGeoDataset(sentinel_sceneSS, city='SanSalvador', size=256,stride=256,out_size=256,padding=0, transform_type=TransformType.noop,transform=None)
buildGeoDataset_SS = CustomSlidingWindowGeoDataset(buildings_sceneSS, city='SanSalvador', size=512,stride=256,out_size=512,padding=0, transform_type=TransformType.noop, transform=None)

# SanJose - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneSJ, buildings_sceneSJ = create_scenes_for_city('SanJoseCRI', cities['SanJoseCRI'], class_config)
sentinelGeoDataset_SJ = CustomSlidingWindowGeoDataset(sentinel_sceneSJ, city='SanJose', size=256, stride = 256, out_size=256,padding=0, transform_type=TransformType.noop, transform=None)
buildGeoDataset_SJ = CustomSlidingWindowGeoDataset(buildings_sceneSJ, city='SanJose', size=512, stride = 256, out_size=512,padding=0, transform_type=TransformType.noop, transform=None)

# BelizeCity
sentinel_sceneBL, buildings_sceneBL = create_scenes_for_city('BelizeCity', cities['BelizeCity'], class_config)
sentinelGeoDataset_BL = PolygonWindowGeoDataset(sentinel_sceneBL,city='BelizeCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_BL = PolygonWindowGeoDataset(buildings_sceneBL,city='BelizeCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Belmopan - data mostly rural exluding from training
sentinel_sceneBM, buildings_sceneBM = create_scenes_for_city('Belmopan', cities['Belmopan'], class_config, resolution=5)
sentinelGeoDataset_BM = PolygonWindowGeoDataset(sentinel_sceneBM, city='Belmopan', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)
buildGeoDataset_BM = PolygonWindowGeoDataset(buildings_sceneBM, city='Belmopan', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets for each city
multimodal_datasets = {
    'SantoDomingo': MergeDataset(sentinelGeoDataset_SD, buildGeoDataset_SD),
    'GuatemalaCity': MergeDataset(sentinelGeoDataset_GC, buildGeoDataset_GC),
    'Tegucigalpa': MergeDataset(sentinelGeoDataset_TG, buildGeoDataset_TG),
    'Managua': MergeDataset(sentinelGeoDataset_MN, buildGeoDataset_MN),
    'Panama': MergeDataset(sentinelGeoDataset_PN, buildGeoDataset_PN),
    'SanSalvador': MergeDataset(sentinelGeoDataset_SS, buildGeoDataset_SS),
    'SanJose': MergeDataset(sentinelGeoDataset_SJ, buildGeoDataset_SJ),
    'BelizeCity': MergeDataset(sentinelGeoDataset_BL, buildGeoDataset_BL),
    'Belmopan': MergeDataset(sentinelGeoDataset_BM, buildGeoDataset_BM)
}

cv = MultiInputCrossValidator(multimodal_datasets, n_splits=2, val_ratio=0.2, test_ratio=0.1)

# Preview a city with sliding windows
city = 'SantoDomingo'
split_index = 1

windows, labels = cv.get_windows_and_labels_for_city(city, split_index)
img_full = senitnel_create_full_image(get_label_source_from_merge_dataset(multimodal_datasets[city]))
show_windows(img_full, windows, labels, title=f'{city} Sliding windows (Split {split_index + 1})')

show_single_tile_multi(multimodal_datasets, city, 5, show_sentinel=True, show_buildings=True)

train_dataset, val_dataset, test_dataset, val_city_indices = cv.get_split(split_index)
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

batch_size = 24

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# Train the model 
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'all',
    'split_index': split_index,
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'labels_size': 256,
    'buil_channels': 128,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'gamma': 0.8,
    'sched_step_size': 5,
    'pos_weight': 2.0,
    'buil_kernel1': 7
}

traincities=hyperparameters['train_cities']
output_dir = f'../../UNITAC-trained-models/multi_modal/{traincities}_DLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
buil_channels = hyperparameters['buil_channels']
buil_kernel = hyperparameters['buil_kernel1']
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_last=True,
    dirpath=output_dir,
    filename=f'multimodal_{split_index}_BCH{buil_channels}_BKR{buil_kernel}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=6,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=40)

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
    min_epochs=50,
    max_epochs=250,
    num_sanity_val_steps=3,
    precision='16-mixed',
    benchmark=True,
)

# Train the model
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
# # best SD_DLV3/last.ckpt
best_model_path_dplv3 = os.path.join(grandparent_dir, "UNITAC-trained-models/multi_modal/all_DLV3/last-v1.ckpt")
# best_model_path_dplv3 = checkpoint_callback.best_model_path
best_model = MultiResolutionDeepLabV3(buil_channels=64, buil_kernel1=3)
checkpoint = torch.load(best_model_path_dplv3)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()
check_nan_params(best_model)

class MultiResPredictionsIterator:
    def __init__(self, model, merged_dataset, device='cuda'):
        self.model = model
        self.dataset = merged_dataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(merged_dataset)):
                sentinel, buildings = self.get_item(merged_dataset, idx)
                
                sentinel_data = sentinel[0].unsqueeze(0).to(device)
                sentlabels = sentinel[1].unsqueeze(0).to(device)

                buildings_data = buildings[0].unsqueeze(0).to(device)
                labels = buildings[1].unsqueeze(0).to(device)

                output = self.model(((sentinel_data,sentlabels), (buildings_data,labels)))
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = self.get_window(merged_dataset, idx)
                self.predictions.append((window, probabilities))

    def get_item(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_item(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_item(dataset.datasets[dataset_idx], sample_idx)
        elif isinstance(dataset, MergeDataset):
            return dataset[idx]
        else:
            raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    def get_window(self, dataset, idx):
        if isinstance(dataset, Subset):
            return self.get_window(dataset.dataset, dataset.indices[idx])
        elif isinstance(dataset, ConcatDataset):
            dataset_idx, sample_idx = self.get_concat_dataset_indices(dataset, idx)
            return self.get_window(dataset.datasets[dataset_idx], sample_idx)
        elif isinstance(dataset, MergeDataset):
            # Assume the first dataset in MergeDataset has the windows
            return dataset.datasets[0].windows[idx]
        else:
            raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    def get_concat_dataset_indices(self, concat_dataset, idx):
        for dataset_idx, dataset in enumerate(concat_dataset.datasets):
            if idx < len(dataset):
                return dataset_idx, idx
            idx -= len(dataset)
        raise IndexError('Index out of range')

    def __iter__(self):
        return iter(self.predictions)
    
sent_strided_fullds_SD = CustomSlidingWindowGeoDataset(sentinel_sceneSD, size=256, stride=128, padding=0, city='SantoDomingo', transform=None, transform_type=TransformType.noop)
buil_strided_fullds_SD = CustomSlidingWindowGeoDataset(buildings_sceneSD, size=512, stride=256, padding=0, city='SantoDomingo', transform=None, transform_type=TransformType.noop)
mergedds = MergeDataset(sent_strided_fullds_SD, buil_strided_fullds_SD)

predictions_iterator = MultiResPredictionsIterator(best_model, mergedds, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=sentinel_sceneSD.extent,
    num_classes=len(class_config),
    smooth=True
)

gt_labels = sentinel_sceneSD.label_source.get_labels()

# Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels, threshold=0.5)
plt.show()

# save if needed
# fig.savefig('predictions_and_ground_truth.png', dpi=300, bbox_inches='tight')

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
        print(f"F1 score for {city}: {f1_score}")
    except Exception as e:
        print(f"Error calculating F1 score for {city}: {str(e)}")

# Calculate overall F1 score
try:
    overall_f1 = calculate_multimodal_f1_score(best_model, val_dataset, device, val_dataset.datasets[0].datasets[0].scene)
    print(f"Overall F1 score: {overall_f1}")
except Exception as e:
    print(f"Error calculating overall F1 score: {str(e)}")

# Print summary of cities with F1 scores
print("\nSummary of F1 scores:")
for city, score in city_f1_scores.items():
    print(f"{city}: {score}")
print(f"Number of cities with F1 scores: {len(city_f1_scores)}")

# # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# crs_transformer = RasterioCRSTransformer.from_uri(cities['TegucigalpaHND']['image_path'])
# gdf = gpd.read_file(cities['TegucigalpaHND']['labels_path'])
# gdf = gdf.to_crs('epsg:3857')
# xmin3857, ymin, xmax, ymax3857 = gdf.total_bounds
# affine_transform_buildings = Affine(10, 0, xmin3857, 0, -10, ymax3857)
# crs_transformer.transform = affine_transform_buildings

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/multi-modal/all_DLV3/',
#     crs_transformer = crs_transformer,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)

# Visualise feature maps
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