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
from rasterio.features import rasterize
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)
from rastervision.core.evaluation import SemanticSegmentationEvaluator


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from src.models.model_definitions import (BuildingsDeepLabV3, CustomVectorOutputConfig,
                                          PredictionsIterator, create_predictions_and_ground_truth_plot)

from src.data.dataloaders import (buil_create_full_image, senitnel_create_full_image,
    create_datasets, create_buildings_scene,cities, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
)

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

# Santo Domingo with augmentation
buildings_sceneSD = create_buildings_scene(cities['SantoDomingoDOM'], 'SantoDomingoDOM')

buildingsGeoDataset_SD, train_buil_ds_SD, val_buil_ds_SD, test_buil_ds_SD = create_datasets(buildings_sceneSD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
buildingsGeoDataset_SD_aug, train_buil_ds_SD_aug, val_buil_ds_SD_aug, test_buil_ds_SD_aug = create_datasets(buildings_sceneSD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = True, seed=12)

build_train_ds_SD = ConcatDataset([train_buil_ds_SD, train_buil_ds_SD_aug])
build_val_ds_SD = ConcatDataset([val_buil_ds_SD, val_buil_ds_SD_aug])

batch_size = 16
train_multiple_cities=False

if train_multiple_cities:
    # Guatemala City
    buildings_sceneGC = create_buildings_scene(cities['GuatemalaCity'], 'GuatemalaCity')
    buildingsGeoDataset_GC, train_buil_ds_GC, val_buil_ds_GC, test_buil_ds_GC = create_datasets(buildings_sceneGC, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)

    # TegucigalpaHND
    buildings_sceneTG = create_buildings_scene(cities['TegucigalpaHND'], 'TegucigalpaHND')
    buildingsGeoDataset_TG, train_buil_ds_TG, val_buil_ds_TG, test_buil_ds_TG = create_datasets(buildings_sceneTG, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Managua
    buildings_sceneMN = create_buildings_scene(cities['Managua'], 'Managua')
    buildingsGeoDataset_MN, train_buil_ds_MN, val_buil_ds_MN, test_buil_ds_MN = create_datasets(buildings_sceneMN, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Panama
    buildings_scenePN = create_buildings_scene(cities['Panama'], 'Panama')
    buildingsGeoDataset_PN, train_buil_ds_PN, val_buil_ds_PN, test_buil_ds_PN = create_datasets(buildings_scenePN, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Combine datasets
    train_dataset = ConcatDataset([build_train_ds_SD, train_buil_ds_GC, train_buil_ds_TG, train_buil_ds_MN, train_buil_ds_PN])
    val_dataset = ConcatDataset([build_val_ds_SD, val_buil_ds_GC, val_buil_ds_TG, val_buil_ds_MN, val_buil_ds_PN])
else:
    train_dataset = build_train_ds_SD
    val_dataset = build_val_ds_SD

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

img_full = senitnel_create_full_image(buildingsGeoDataset_SD.scene.label_source)
train_windows = train_buil_ds_SD.windows
val_windows = val_buil_ds_SD.windows
test_windows = test_buil_ds_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# Train the model
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'all',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'labels_size': 256,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-4,
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
    save_top_k=2,
    mode='min',
    save_last=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=35)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=30,
    max_epochs=150,
    num_sanity_val_steps=1,
    # overfit_batches=0.2,
)

trainer.fit(model, train_dl, val_dl)

# Best deeplab model path val=0.3083
best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/DLV3/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
# best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsDeepLabV3.load_from_checkpoint(best_model_path_deeplab)
best_model.eval()

# fulldataset_SD, train_sentinel_datasetSD, val_sent_ds_SD, test_sentinel_dataset_SD = create_datasets(buildings_sceneSD, imgsize=256, stride=256, padding=128, val_ratio=0.15, test_ratio=0.08, augment=False, seed=22)
strided_fullds_SD, _, _, _ = create_datasets(buildings_sceneSD, imgsize=512, stride=256, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

predictions_iterator = PredictionsIterator(best_model, val_buil_ds_SD, device=device)
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
# fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels)
# plt.show()

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

scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('Only buildings footprints model predictions')
cbar = fig.colorbar(image, ax=ax)
plt.show()

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