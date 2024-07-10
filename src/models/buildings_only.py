import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from rastervision.core.box import Box
import stackstac
from pathlib import Path
import torch.nn as nn
import cv2
from os.path import join
import torch.nn.functional as F
import json
from shapely.geometry import shape, mapping
from shapely.affinity import translate

from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader
from typing import List
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers.wandb import WandbLogger
import os
from datetime import datetime
import wandb
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torchvision.models.segmentation import deeplabv3_resnet50
from collections import OrderedDict

from typing import Iterator, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torch.utils.data import ConcatDataset

from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    MinMaxTransformer, MultiRasterSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource, CRSTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    SemanticSegmentationLabelSource)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore

from rastervision.core.data.utils import pad_to_window_size
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from rastervision.pytorch_learner import (SemanticSegmentationSlidingWindowGeoDataset,
                                          SemanticSegmentationVisualizer, SlidingWindowGeoDataset)
from rastervision.pipeline.utils import repr_with_args
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
from xarray import DataArray

from affine import Affine
import numpy as np
import geopandas as gpd
from rastervision.core.box import Box
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from typing import List

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.pytorch_learner.dataset.transform import (TransformType, TF_TYPE_TO_TF_FUNC)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from src.models.model_definitions import (BuildingsDeepLabV3, CustomVectorOutputConfig,
                                          PredictionsIterator, create_predictions_and_ground_truth_plot)

from src.features.dataloaders import (buil_create_full_image, senitnel_create_full_image, create_scenes_for_city,
    create_datasets, create_building_scene,cities, show_windows, SingleInputCrossValidator, singlesource_show_windows_for_city, show_single_tile_buildings,
    PolygonWindowGeoDataset, MergeDataset, CustomSlidingWindowGeoDataset)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)

# Define device
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

# Santo Domingo
buildings_sceneSD = create_building_scene('SantoDomingoDOM', cities['SantoDomingoDOM'])
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, city= 'SantoDomingo',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# GuatemalaCity
buildings_sceneGC = create_building_scene('GuatemalaCity', cities['GuatemalaCity'])
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, city='GuatemalaCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
buildings_sceneTG = create_building_scene('Tegucigalpa', cities['TegucigalpaHND'])
buildGeoDataset_TG = CustomSlidingWindowGeoDataset(buildings_sceneTG, city='Tegucigalpa', size=512, stride = 512, out_size=512, padding=0, transform_type=TransformType.noop, transform=None)

# Managua
buildings_sceneMN = create_building_scene('Managua', cities['Managua'])
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, city='Managua', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
buildings_scenePN = create_building_scene('Panama', cities['Panama'])
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN, city='Panama', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# San Salvador - UNITAC report mentions data is complete, so using all tiles
buildings_sceneSS = create_building_scene('SanSalvador_PS', cities['SanSalvador_PS'])
buildGeoDataset_SS = CustomSlidingWindowGeoDataset(buildings_sceneSS, city='SanSalvador', size=512,stride=256,out_size=512,padding=0, transform_type=TransformType.noop, transform=None)

# SanJose - UNITAC report mentions data is complete, so using all tiles
buildings_sceneSJ = create_building_scene('SanJoseCRI', cities['SanJoseCRI'])
buildGeoDataset_SJ = CustomSlidingWindowGeoDataset(buildings_sceneSJ, city='SanJose', size=512, stride = 256, out_size=512,padding=0, transform_type=TransformType.noop, transform=None)

# BelizeCity
buildings_sceneBL = create_building_scene('BelizeCity', cities['BelizeCity'])
buildGeoDataset_BL = PolygonWindowGeoDataset(buildings_sceneBL,city='BelizeCity',window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)

# Belmopan
buildings_sceneBM = create_building_scene('Belmopan', cities['Belmopan'])
buildGeoDataset_BM = PolygonWindowGeoDataset(buildings_sceneBM, city='Belmopan', window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buildings_sceneBM.raster_source.get_chip(Box(0, 0, 512, 512)).max()

# Create datasets for each city
buildings_datasets = {
    'SantoDomingo': buildGeoDataset_SD,
    'GuatemalaCity': buildGeoDataset_GC,
    'Tegucigalpa': buildGeoDataset_TG,
    'Managua': buildGeoDataset_MN,
    'Panama': buildGeoDataset_PN,
    'SanSalvador': buildGeoDataset_SS,
    'SanJose': buildGeoDataset_SJ,
    'BelizeCity': buildGeoDataset_BL,
    'Belmopan': buildGeoDataset_BM
}

cv = SingleInputCrossValidator(buildings_datasets, n_splits=2, val_ratio=0.2, test_ratio=0.1)
split_index = 1

# Preview a city with sliding windows
city = 'SantoDomingo'

singlesource_show_windows_for_city(city, split_index, cv, buildings_datasets)
show_single_tile_buildings(buildings_datasets, city, 3)

train_dataset, val_dataset, test_dataset, val_city_indices = cv.get_split(split_index)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

# Train the model
hyperparameters = {
    'model': 'DLV3',
    'split_index': split_index,
    'train_cities': 'all',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'gamma': 0.5,
    'sched_step_size': 20,
    'pos_weight': 2.0,
}

model = BuildingsDeepLabV3(
    use_deeplnafrica = hyperparameters['use_deeplnafrica'],
    learning_rate = hyperparameters['learning_rate'],
    weight_decay = hyperparameters['weight_decay'],
    gamma = hyperparameters['gamma'],
    atrous_rates = hyperparameters['atrous_rates'],
    sched_step_size = hyperparameters['sched_step_size'],
    pos_weight = hyperparameters['pos_weight'])

model.to(device)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'../../UNITAC-trained-models/buildings_only/DLV3/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-buildings-only', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-buildings-only', log_model=True)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename=f'buildingsDLV3_{split_index}_{{epoch:02d}}-{{val_loss:.4f}}',
    save_top_k=4,
    mode='min',
    save_last=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=20)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=25,
    max_epochs=150,
    num_sanity_val_steps=1,
    # overfit_batches=0.2,
)

trainer.fit(model, train_loader, val_loader)

# Best deeplab model path val=0.3083
best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/DLV3/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
# best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsDeepLabV3.load_from_checkpoint(best_model_path_deeplab)
best_model.eval()

strided_fullds_SD = CustomSlidingWindowGeoDataset(buildings_sceneSD, size=256, stride=128, padding=0, city='SantoDomingo', transform=None, transform_type=TransformType.noop)


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

predictions_iterator = PredictionsIterator(best_model, strided_fullds_SD, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=buildings_sceneSD.extent,
    num_classes=len(class_config),
    smooth=True
)
gt_labels = buildings_sceneSD.label_source.get_labels()

# Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels)
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
inf_eval.f1


# Calculate F1 scores
def calculate_f1_score(model, dataset, device, scene):
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
    city_scene = buildings_datasets[city].scene
    
    try:
        # Calculate F1 score for this city
        f1_score = calculate_f1_score(best_model, city_val_dataset, device, city_scene)
        
        city_f1_scores[city] = f1_score
        print(f"F1 score for {city}: {f1_score}")
    except Exception as e:
        print(f"Error calculating F1 score for {city}: {str(e)}")

# Calculate overall F1 score
try:
    overall_f1 = calculate_f1_score(best_model, val_dataset, device, val_dataset.datasets[0].scene)
    print(f"Overall F1 score: {overall_f1}")
except Exception as e:
    print(f"Error calculating overall F1 score: {str(e)}")

# Print summary of cities with F1 scores
print("\nSummary of F1 scores:")
for city, score in city_f1_scores.items():
    print(f"{city}: {score}")
print(f"Number of cities with F1 scores: {len(city_f1_scores)}")


# Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/buildings_model_only/2/',
#     crs_transformer = crs_transformer_SD,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)

# # Map interactive visualization
# predspath = '/Users/janmagnuszewski/dev/slums-model-unitac/vectorised_model_predictions/buildings_model_only/2/vector_output/class-1-slums.json'
# label_uri = "../../data/0/SantoDomingo3857.geojson"
# extent_gdf = gpd.read_file(label_uri)
# gdf = gpd.read_file(predspath)
# m = folium.Map(location=[gdf.geometry[0].centroid.y, gdf.geometry[0].centroid.x], zoom_start=12)
# folium.GeoJson(gdf).add_to(m) 
# folium.GeoJson(extent_gdf, style_function=lambda x: {'color':'red'}).add_to(m)
# m