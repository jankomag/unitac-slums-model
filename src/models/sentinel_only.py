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
sys.path.append(parent_dir)

from src.models.model_definitions import CustomVectorOutputConfig, SentinelDeepLabV3, PredictionsIterator, create_predictions_and_ground_truth_plot
from src.data.dataloaders import create_datasets, create_sentinel_scene, cities, senitnel_create_full_image, show_windows

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
    
# Preproceess SD labels for simplification (buffering)
# label_uriSD = "../../data/0/SantoDomingo3857.geojson"
# gdf = gpd.read_file(label_uriSD)
# gdf = gdf.to_crs('EPSG:4326')
# gdf_filled = gdf.copy()
# gdf_filled["geometry"] = gdf_filled.geometry.buffer(0.00008)
# gdf_filled = gdf_filled.to_crs('EPSG:3857')
# gdf_filled.to_file("../../data/0/SantoDomingo3857_buffered.geojson", driver="GeoJSON")

# Preproceess SS labels to select Lotificacion Illegal
# label_uriSS = os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_particular.shp')
# gdf = gpd.read_file(label_uriSS)
# gdf = gdf[gdf['TIPO_AUP'] == 'Lotificación Ilegal']
# gdf = gdf.to_crs('EPSG:3857')
# gdf.to_file(os.path.join(grandparent_dir, 'data/SHP/SanSalvador_PS_lotifi_ilegal.shp'), driver="GeoJSON")

# Load training data
class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

SentinelScene_SD = create_sentinel_scene(cities['SantoDomingoDOM'], class_config)

sentinelGeoDataset_SD, train_sentinel_datasetSD, val_sent_ds_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=256, stride=256, padding=128, val_ratio=0.15, test_ratio=0.08, augment=False, seed=22)
sentinelGeoDataset_SD_aug, train_sentinel_datasetSD_aug, val_sent_ds_SD_aug, test_sentinel_dataset_SD_aug = create_datasets(SentinelScene_SD, imgsize=256, stride=256, padding=128, val_ratio=0.15, test_ratio=0.08, augment=True, seed=22)

sent_train_ds_SD = ConcatDataset([train_sentinel_datasetSD, train_sentinel_datasetSD_aug])
sent_val_ds_SD = ConcatDataset([val_sent_ds_SD, val_sent_ds_SD_aug])

batch_size = 32
train_multiple_cities = False

if train_multiple_cities:
    # Guatemala City
    SentinelScene_GC = create_sentinel_scene(cities['GuatemalaCity'], class_config)
    sentinelGeoDataset_GC, train_sentinel_ds_GC, val_sent_ds_GC, test_sentinel_ds_GC = create_datasets(SentinelScene_GC, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # TegucigalpaHND
    SentinelScene_TG = create_sentinel_scene(cities['TegucigalpaHND'], class_config)
    sentinelGeoDataset_TG, train_sentinel_ds_TG, val_sent_ds_TG, test_sentinel_ds_TG = create_datasets(SentinelScene_TG, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Managua
    SentinelScene_MN = create_sentinel_scene(cities['Managua'], class_config)
    sentinelGeoDataset_MN, train_sentinel_ds_MN, val_sent_ds_MN, test_sentinel_ds_MN = create_datasets(SentinelScene_MN, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Panama
    SentinelScene_PN = create_sentinel_scene(cities['Panama'], class_config)
    sentinelGeoDataset_PN, train_sentinel_ds_PN, val_sent_ds_PN, test_sentinel_ds_PN = create_datasets(SentinelScene_PN, imgsize=256, stride=256, padding=50, val_ratio=0.2, test_ratio=0.1, augment=False, seed=12)

    # Combine datasets
    train_dataset = ConcatDataset([sent_train_ds_SD, train_sentinel_ds_GC, train_sentinel_ds_TG, train_sentinel_ds_MN, train_sentinel_ds_PN])
    val_dataset = ConcatDataset([sent_val_ds_SD, val_sent_ds_GC, val_sent_ds_TG, val_sent_ds_MN, val_sent_ds_PN])
else:
    train_dataset = sent_train_ds_SD
    val_dataset = sent_val_ds_SD

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)    

# Preview sliding window
img_full = senitnel_create_full_image(sentinelGeoDataset_SD.scene.label_source)
train_windows = train_sentinel_datasetSD.windows
val_windows = val_sent_ds_SD.windows
test_windows = test_sentinel_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# Create model
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'SD',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'gamma': 1,
    'sched_step_size': 15,
    'pos_weight': torch.tensor(2.0, device='mps')
}

model = SentinelDeepLabV3(use_deeplnafrica = hyperparameters['use_deeplnafrica'],
                    learning_rate = hyperparameters['learning_rate'],
                    weight_decay = hyperparameters['weight_decay'],
                    gamma = hyperparameters['gamma'],
                    atrous_rates = hyperparameters['atrous_rates'],
                    sched_step_size = hyperparameters['sched_step_size'],
                    pos_weight = hyperparameters['pos_weight'])
model.to(device)

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
    save_top_k=2,
    save_last=True,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=25)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=15,
    max_epochs=250,
    num_sanity_val_steps=3,
    # overfit_batches=0.5
)

# Train the model
trainer.fit(model, train_dl, val_dl)

# Make predictions
# best_model_path = checkpoint_callback.best_model_path
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/sentinel_only/DLV3/last.ckpt"
# best_model = SentinelDeepLabV3.load_from_checkpoint(best_model_path) # SentinelSimpleSS SentinelDeeplabv3
best_model = SentinelDeepLabV3()
best_model.eval()

fulldataset_SD, train_sentinel_datasetSD, val_sent_ds_SD, test_sentinel_dataset_SD = create_datasets(SentinelScene_SD, imgsize=256, stride=256, padding=128, val_ratio=0.15, test_ratio=0.08, augment=False, seed=22)
strided_fullds_SD, _, _, _ = create_datasets(SentinelScene_SD, imgsize=256, stride=128, padding=8, val_ratio=0.2, test_ratio=0.1, seed=42)

predictions_iterator = PredictionsIterator(best_model, strided_fullds_SD, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=SentinelScene_SD.extent,
    num_classes=len(class_config),
    smooth=True
)
gt_labels = SentinelScene_SD.label_source.get_labels()

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

# # # Saving predictions as GEOJSON
# vector_output_config = CustomVectorOutputConfig(
#     class_id=1,
#     denoise=8,
#     threshold=0.5)

# gdf = gpd.read_file('../data/0/SantoDomingo3857_buffered.geojson')
# gdf = gdf.to_crs('EPSG:3857')
# xmin, ymin, xmax, ymax = gdf.total_bounds

# crs_transformer_buildings = RasterioCRSTransformer.from_uri('../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif')
# affine_transform_buildings = Affine(10, 0, xmin, 0, -10, ymax)
# crs_transformer_buildings.transform = affine_transform_buildings

# crs_transformer = RasterioCRSTransformer.from_uri('../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif')

# pred_label_store = SemanticSegmentationLabelStore(
#     uri='../../vectorised_model_predictions/sentinel_only/DLV3_!/',
#     crs_transformer = crs_transformer_buildings,
#     class_config = class_config,
#     vector_outputs = [vector_output_config],
#     discrete_output = True)

# pred_label_store.save(pred_labels)