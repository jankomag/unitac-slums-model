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

train_dataset, val_dataset, test_dataset = cv.get_split(split_index)

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
    filename='buildingsDLV3_{epoch:02d}-{val_loss:.4f}',
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

trainer.fit(model, train_dl, val_dl)

# Best deeplab model path val=0.3083
# best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/DLV3/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsDeepLabV3.load_from_checkpoint(best_model_path)
best_model.eval()

# fulldataset_SD, train_sentinel_datasetSD, val_sent_ds_SD, test_sentinel_dataset_SD = create_datasets(buildings_sceneSD, imgsize=256, stride=256, padding=128, val_ratio=0.15, test_ratio=0.08, augment=False, seed=22)
strided_fullds_SD, _, _, _ = create_datasets(buildings_sceneSD, imgsize=512, stride=256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

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

# scores = pred_labels.get_score_arr(pred_labels.extent)
# scores_building = scores[0]
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# image = ax.imshow(scores_building)
# ax.axis('off')
# # ax.set_title('Only buildings footprints model predictions')
# # cbar = fig.colorbar(image, ax=ax)
# plt.show()

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