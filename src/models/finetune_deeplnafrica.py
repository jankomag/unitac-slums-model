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
from src.data.dataloaders import create_datasets, create_sentinel_raster_source
from rastervision.core.data.label_store import (SemanticSegmentationLabelStore)
from rastervision.core.data import (Scene,
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
    
# Load data
class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

sentinel_source_normalizedSD, sentinel_label_raster_sourceSD = create_sentinel_raster_source(image_uri, label_uri, class_config, clip_to_label_source=True)

gdf = gpd.read_file(label_uri).to_crs('EPSG:3857')
xmin, _, _, ymax = gdf.total_bounds

crs_transformer = RasterioCRSTransformer.from_uri(image_uri)
affine_transform_buildings = Affine(10, 0, xmin, 0, -10, ymax)
crs_transformer.transform = affine_transform_buildings
    
SentinelScene_SD = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalizedSD,
        label_source = sentinel_label_raster_sourceSD)

sentinelGeoDataset_SD, train_sentinel_dataset_SD, val_sentinel_dataset_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=144, stride=144, padding=25, val_ratio=0.2, test_ratio=0.1, seed=42)

batch_size=16
train_dl = DataLoader(train_sentinel_dataset_SD, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_sentinel_dataset_SD, batch_size=batch_size, shuffle=False, pin_memory=True)    

# Create model
model = SentinelSimpleSS(weight_decay=0.01,
        learning_rate=0.001,
        gamma=0.1,
        pos_weight=torch.tensor(2, device='mps'))
model.to(device)

# for batch in train_dl: 
#     img, groundtruth = batch
#     print(img.shape)
#     print(groundtruth.shape)
#     out = model(img)
#     print(f"Out shape:{out.shape}")
#     break

output_dir = f'../UNITAC-trained-models/sentinel_only/UNET'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only')
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
# best_model_path = checkpoint_callback.best_model_path
best_unet_path = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/sentinel_only/UNET/multimodal_runidrun_id=0-epoch=04-val_loss=0.6201.ckpt"
best_deeplab_path = "/Users/janmagnuszewski/dev/slums-model-unitac/src/UNITAC-trained-models/sentinel_only/multimodal_runidrun_id=0-epoch=02-val_loss=0.3667.ckpt"
best_model = SentinelDeeplabv3.load_from_checkpoint(best_deeplab_path) # SentinelSimpleSS SentinelDeeplabv3
best_model.eval()


# label_uriGC = "../../data/SHP/Guatemala_PS.shp"
# image_uriGC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
# sentinel_source_normalizedGC, sentinel_label_raster_sourceGC = create_sentinel_raster_source(image_uriGC, label_uriGC, class_config, clip_to_label_source=True)
# SentinelScene_GC = Scene(id='GC_sentinel', raster_source = sentinel_source_normalizedGC)

# GC_ds, _, _, _ = create_datasets(SentinelScene_GC, imgsize=144, stride=72, padding=8, val_ratio=0.2, test_ratio=0.1, seed=42)
SD_ds, _, _, _ = create_datasets(SentinelScene_SD, imgsize=144, stride=72, padding=8, val_ratio=0.2, test_ratio=0.1, seed=42)

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

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/sentinel_only/UNET/',
    crs_transformer = crs_transformer,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)

# Evaluate against labels:
gt_labels = SentinelScene_SD.label_source.get_labels()
gt_extent = gt_labels.extent
pred_extent = pred_labels.extent
print(f"Ground truth extent: {gt_extent}")
print(f"Prediction extent: {pred_extent}")

evaluator = SemanticSegmentationEvaluator(class_config)
evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels)

evaluation.class_to_eval_item[0]
# evaluation.class_to_eval_item[1]

# # Discrete labels
# pred_labels_dis = SemanticSegmentationLabels.from_predictions(
#     sentinel_train_ds.windows,
#     predictions,
#     smooth=False,
#     extent=sentinel_train_ds.scene.extent,
#     num_classes=len(class_config))

# scores_dis = pred_labels.get_class_mask(window=sentinel_train_ds.windows[6],class_id=1,threshold=0.005)

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# fig.tight_layout()
# image = ax.imshow(scores_dis, cmap='plasma')
# ax.axis('off')
# ax.set_title('infs')
# cbar = fig.colorbar(image, ax=ax)
# cbar.set_label('Score')
# plt.show()