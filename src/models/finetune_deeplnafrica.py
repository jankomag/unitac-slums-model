import os
from os.path import join
from subprocess import check_output

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()

import sys
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime
from torchvision.models.segmentation import deeplabv3_resnet50
from collections import OrderedDict
from torch.optim.lr_scheduler import MultiStepLR
from affine import Affine
import geopandas as gpd
import torch
import torch.nn as nn
from torch.optim import AdamW
import tempfile
import wandb
import numpy as np
import cv2
# import lightning.pytorch as pl
from torch.utils.data import ConcatDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import Polygon
from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, make_dir, zipdir, download_if_needed,
    download_or_copy, sync_from_dir, get_local_path, unzip, is_local,
    get_tmp_dir)
from torch.utils.data import DataLoader
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from src.models.model_definitions import CustomGeoJSONVectorSource, CustomVectorOutputConfig, SentinelSimpleSS, SentinelDeeplabv3
from src.data.dataloaders import create_datasets, create_sentinel_scene, cities, senitnel_create_full_image
from rastervision.core.data.label_store import (SemanticSegmentationLabelStore)
from rastervision.core.data import (Scene, ClassInferenceTransformer, RasterizedSource,
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
from rastervision.pytorch_learner import (
    SolverConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner, SemanticSegmentationGeoDataConfig, SemanticSegmentationVisualizer,
)
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from src.data.dataloaders import create_full_image, show_windows
from rastervision.core.data.utils import get_polygons_from_uris

# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
class PredictionsIterator:
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, label = dataset[idx]
                image = image.unsqueeze(0).to(device)

                output = self.model(image)
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = dataset.windows[idx]
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)
    
# Preproceess SD labels for simplification (buffering)
label_uriSD = "../../data/0/SantoDomingo3857.geojson"
gdf = gpd.read_file(label_uriSD)
gdf = gdf.to_crs('EPSG:4326')
gdf_filled = gdf.copy()
gdf_filled["geometry"] = gdf_filled.geometry.buffer(0.00008)
gdf_filled = gdf_filled.to_crs('EPSG:3857')
gdf_filled.to_file("../../data/0/SantoDomingo3857_buffered.geojson", driver="GeoJSON")

# Load training data
class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

SentinelScene_SD = create_sentinel_scene(cities['SantoDomingoDOM'], class_config)

sentinelGeoDataset_SD, train_sentinel_datasetSD, val_sent_ds_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)
sentinelGeoDataset_SD_aug, train_sentinel_datasetSD_aug, val_sent_ds_SD_aug, test_sentinel_dataset_SD_aug = create_datasets(SentinelScene_SD, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=True, seed=12)

# Preview sliding window
# img_full = senitnel_create_full_image(sentinelGeoDataset_SD.scene.label_source)
# train_windows = train_sentinel_datasetSD.windows
# val_windows = val_sent_ds_SD.windows
# test_windows = test_sentinel_dataset_SD.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

sent_train_ds_SD = ConcatDataset([train_sentinel_datasetSD, train_sentinel_datasetSD_aug])
sent_val_ds_SD = ConcatDataset([val_sent_ds_SD, val_sent_ds_SD_aug])

batch_size = 16

train_dl = DataLoader(sent_train_ds_SD, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(sent_val_ds_SD, batch_size=batch_size, shuffle=False, pin_memory=True)    

# Create model
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'SD',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-4,
    'weight_decay': 0.00001,
    'gamma': 0.5,
    'sched_step_size': 40,
    'pos_weight': torch.tensor(1.0, device='mps')
}

model = SentinelDeeplabv3(use_deeplnafrica = hyperparameters['use_deeplnafrica'],
                    learning_rate = hyperparameters['learning_rate'],
                    weight_decay = hyperparameters['weight_decay'],
                    gamma = hyperparameters['gamma'],
                    atrous_rates = hyperparameters['atrous_rates'],
                    sched_step_size = hyperparameters['sched_step_size'],
                    pos_weight = hyperparameters['pos_weight'])
model.to(device)

for batch in train_dl: 
    img, groundtruth = batch
    print(img.shape)
    print(groundtruth.shape)
    out = model(img)
    print(f"Out shape:{out.shape}")
    break

output_dir = f'../UNITAC-trained-models/sentinel_only/DLV3'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-finetune-sentinel-only', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=20)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=30,
    max_epochs=200,
    num_sanity_val_steps=3
)

# Train the model
trainer.fit(model, train_dl, val_dl)

# Make predictions
best_model_path = checkpoint_callback.best_model_path
# best_unet_path = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/sentinel_only/UNET/multimodal_runidrun_id=0-epoch=04-val_loss=0.6201.ckpt"
best_deeplab_path = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/sentinel_only/multimodal_runidrun_id=0-epoch=02-val_loss=0.3667.ckpt"
best_model = SentinelDeeplabv3.load_from_checkpoint(best_model_path) # SentinelSimpleSS SentinelDeeplabv3
best_model.eval()

# label_uriGC = "../../data/SHP/Guatemala_PS.shp"
# image_uriGC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
# sentinel_source_normalizedGC, sentinel_label_raster_sourceGC = create_sentinel_raster_source(image_uriGC, label_uriGC, class_config, clip_to_label_source=True)
# SentinelScene_GC = Scene(id='GC_sentinel', raster_source = sentinel_source_normalizedGC)

# GC_ds, _, _, _ = create_datasets(SentinelScene_GC, imgsize=144, stride=72, padding=8, val_ratio=0.2, test_ratio=0.1, seed=42)
SD_ds, _, _, _ = create_datasets(SentinelScene_SD, imgsize=256, stride=128, padding=8, val_ratio=0.2, test_ratio=0.1, seed=42)

predictions_iterator = PredictionsIterator(best_model, SD_ds, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=SentinelScene_SD.extent,
    num_classes=len(class_config),
    smooth=True
)

# Show predictions
scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('infs Scores')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/sentinel_only/UNET/',
#     crs_transformer = crs_transformer,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)

# # Evaluate against labels:
# gt_labels = SentinelScene_SD.label_source.get_labels()
# gt_extent = gt_labels.extent
# pred_extent = pred_labels.extent
# print(f"Ground truth extent: {gt_extent}")
# print(f"Prediction extent: {pred_extent}")

# evaluator = SemanticSegmentationEvaluator(class_config)
# evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels)

# evaluation.class_to_eval_item[0]
# # evaluation.class_to_eval_item[1]

# # # Discrete labels
# # pred_labels_dis = SemanticSegmentationLabels.from_predictions(
# #     sentinel_train_ds.windows,
# #     predictions,
# #     smooth=False,
# #     extent=sentinel_train_ds.scene.extent,
# #     num_classes=len(class_config))

# # scores_dis = pred_labels.get_class_mask(window=sentinel_train_ds.windows[6],class_id=1,threshold=0.005)

# # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# # fig.tight_layout()
# # image = ax.imshow(scores_dis, cmap='plasma')
# # ax.axis('off')
# # ax.set_title('infs')
# # cbar = fig.colorbar(image, ax=ax)
# # cbar.set_label('Score')
# # plt.show()