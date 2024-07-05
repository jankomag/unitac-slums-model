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


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from src.models.model_definitions import (BuildingsDeepLabV3, CustomVectorOutputConfig, BuildingsOnlyPredictionsIterator)
from deeplnafrica.deepLNAfrica import init_segm_model

from src.data.dataloaders import (buil_create_full_image,
    create_datasets, create_buildings_scene,cities,
    create_buildings_raster_source, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
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

# img_full = buil_create_full_image(buildingsGeoDataset_SD.scene.label_source)
# train_windows = build_train_ds_SD.windows
# val_windows = val_buil_ds_SD.windows
# test_windows = test_buil_ds_SD.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

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
# best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsDeepLabV3.load_from_checkpoint(best_model_path)
best_model.eval()

# Make predictions
buildingsGeoDataset, _, _, _ = create_datasets(buildings_sceneSD, imgsize=512, stride = 256, padding=50, val_ratio=0.2, test_ratio=0.1, augment = False, seed=42)
predictions_iterator = BuildingsOnlyPredictionsIterator(best_model, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=buildingsGeoDataset.scene.extent,
    num_classes=len(class_config),
    smooth=True)

# Show predictions
scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
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

# # Evaluate predictions
from rastervision.core.evaluation import SemanticSegmentationEvaluator

evaluator = SemanticSegmentationEvaluator(class_config)
gt_labels = BuildingsScenceSD.label_source.get_labels()
evaluation = evaluator.evaluate_predictions(
    ground_truth=gt_labels, predictions=pred_labels)
eval_metrics_dict = evaluation.class_to_eval_item[0]
f1_score = eval_metrics_dict.f1
f1_score

# # Map interactive visualization
# predspath = '/Users/janmagnuszewski/dev/slums-model-unitac/vectorised_model_predictions/buildings_model_only/2/vector_output/class-1-slums.json'
# label_uri = "../../data/0/SantoDomingo3857.geojson"
# extent_gdf = gpd.read_file(label_uri)
# gdf = gpd.read_file(predspath)
# m = folium.Map(location=[gdf.geometry[0].centroid.y, gdf.geometry[0].centroid.x], zoom_start=12)
# folium.GeoJson(gdf).add_to(m) 
# folium.GeoJson(extent_gdf, style_function=lambda x: {'color':'red'}).add_to(m)
# m



### Make predictions on another city ###
image_uri = '../../data/0/sentinel_Gee/HTI_Tabarre_2023.tif'
label_uri = "../../data/0/SantoDomingo3857.geojson"
buildings_uriHT = '../../data/0/overture/portauprince.geojson'

rasterized_buildings_sourceHT, buildings_label_sourceHT, crs_transformer_HT = create_buildings_raster_source(buildings_uriHT, image_uri, label_uri, class_config, resolution=5)

HT_eval_scene = Scene(
        id='portauprince_buildings',
        raster_source = rasterized_buildings_sourceHT,
        label_source = buildings_label_sourceHT)

HTGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(HT_eval_scene, imgsize=288, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

predictions_iterator = BuildingsOnlyPredictionsIterator(best_model, HTGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels_HT = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=HTGeoDataset.scene.extent,
    num_classes=len(class_config),
    smooth=True)

# Show predictions
scores = pred_labels_HT.get_score_arr(pred_labels_HT.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('infs Scores')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_storeHT = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/buildings_model_only/Haiti_unet/',
    crs_transformer = crs_transformer_HT,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_storeHT.save(pred_labels_HT)