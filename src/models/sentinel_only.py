from os.path import join
from subprocess import check_output
import os

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
from rastervision.pytorch_learner.dataset.transform import (TransformType,
                                                            TF_TYPE_TO_TF_FUNC)

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

from src.models.model_definitions import SentinelDeepLabV3, PredictionsIterator, create_predictions_and_ground_truth_plot, check_nan_params
from src.features.dataloaders import (create_datasets, create_sentinel_scene, cities, CustomSlidingWindowGeoDataset,
                                      senitnel_create_full_image, show_windows, PolygonWindowGeoDataset)
                                      #SingleInputCrossValidator, singlesource_show_windows_for_city)

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

# SantoDomingo
SentinelScene_SD = create_sentinel_scene(cities['SantoDomingoDOM'], class_config)
sentinelGeoDataset_SD = PolygonWindowGeoDataset(SentinelScene_SD, city= 'SantoDomingo', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# GuatemalaCity
sentinel_sceneGC = create_sentinel_scene(cities['GuatemalaCity'], class_config)
sentinelGeoDataset_GC = PolygonWindowGeoDataset(sentinel_sceneGC, city='GuatemalaCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Tegucigalpa - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneTG = create_sentinel_scene(cities['TegucigalpaHND'], class_config)
sentinelGeoDataset_TG = CustomSlidingWindowGeoDataset(sentinel_sceneTG, city='Tegucigalpa', size=256, stride = 256, out_size=256, padding=0, transform_type=TransformType.noop, transform=None)

# Managua
sentinel_sceneMN = create_sentinel_scene(cities['Managua'], class_config)
sentinelGeoDataset_MN = PolygonWindowGeoDataset(sentinel_sceneMN, city='Managua', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Panama
sentinel_scenePN = create_sentinel_scene(cities['Panama'], class_config)
sentinelGeoDataset_PN = PolygonWindowGeoDataset(sentinel_scenePN, city='Panama', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# San Salvador - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneSS = create_sentinel_scene(cities['SanSalvador_PS'], class_config)
sentinelGeoDataset_SS = CustomSlidingWindowGeoDataset(sentinel_sceneSS, city='SanSalvador', size=256,stride=256,out_size=256,padding=0, transform_type=TransformType.noop,transform=None)

# SanJose - UNITAC report mentions data is complete, so using all tiles
sentinel_sceneSJ = create_sentinel_scene(cities['SanJoseCRI'], class_config)
sentinelGeoDataset_SJ = CustomSlidingWindowGeoDataset(sentinel_sceneSJ, city='SanJose', size=256, stride = 256, out_size=256,padding=0, transform_type=TransformType.noop, transform=None)

# BelizeCity
sentinel_sceneBL = create_sentinel_scene(cities['BelizeCity'], class_config)
sentinelGeoDataset_BL = PolygonWindowGeoDataset(sentinel_sceneBL,city='BelizeCity',window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Belmopan - data mostly rural exluding from training
sentinel_sceneBM = create_sentinel_scene(cities['Belmopan'], class_config)
sentinelGeoDataset_BM = PolygonWindowGeoDataset(sentinel_sceneBM, city='Belmopan', window_size=256,out_size=256,padding=0,transform_type=TransformType.noop,transform=None)

# Create datasets for each city
sentinel_datasets = {
    'SantoDomingo': sentinelGeoDataset_SD,
    'GuatemalaCity': sentinelGeoDataset_GC,
    'Tegucigalpa': sentinelGeoDataset_TG,
    'Managua': sentinelGeoDataset_MN,
    'Panama': sentinelGeoDataset_PN,
    'SanSalvador': sentinelGeoDataset_SS,
    'SanJose': sentinelGeoDataset_SJ,
    'BelizeCity': sentinelGeoDataset_BL,
    'Belmopan': sentinelGeoDataset_BM
}

cv = SingleInputCrossValidator(sentinel_datasets, n_splits=2, val_ratio=0.2, test_ratio=0.1)
split_index = 0

# Preview a city with sliding windows
city = 'Belmopan'
singlesource_show_windows_for_city(city, 1, cv, sentinel_datasets)
show_single_tile_sentinel(sentinel_datasets, city, 3)

train_dataset, val_dataset, test_dataset = cv.get_split(split_index)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 24
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)

# Create model
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'all',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
    'atrous_rates': (12, 24, 36),
    'learning_rate': 1e-3,
    'weight_decay': 0,
    'gamma': 0.5,
    'sched_step_size': 5,
    'pos_weight': 2.0
}

model = SentinelDeepLabV3(use_deeplnafrica = hyperparameters['use_deeplnafrica'],
                    learning_rate = hyperparameters['learning_rate'],
                    weight_decay = hyperparameters['weight_decay'],
                    gamma = hyperparameters['gamma'],
                    atrous_rates = hyperparameters['atrous_rates'],
                    sched_step_size = hyperparameters['sched_step_size'],
                    pos_weight = hyperparameters['pos_weight'])
model.to(device)

output_dir = f'../../UNITAC-trained-models/sentinel_only/DLV3'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only', config=hyperparameters)
wandb_logger = WandbLogger(project='UNITAC-finetune-sentinel-only', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=3,
    save_last=True,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=25)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=25,
    max_epochs=250,
    num_sanity_val_steps=3,
    # overfit_batches=0.5
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Make predictions
best_model_path = checkpoint_callback.best_model_path
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/sentinel_only/DLV3/last-1.ckpt"
best_model = SentinelDeepLabV3.load_from_checkpoint(best_model_path) # SentinelSimpleSS SentinelDeeplabv3
best_model.eval()
check_nan_params(best_model)

strided_fullds_SD, _, _, _ = create_datasets(SentinelScene_SD, imgsize=256, stride=128, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

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

# # Show predictions
fig, axes = create_predictions_and_ground_truth_plot(pred_labels, gt_labels)
plt.show()
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

def calculate_f1_score(model, sent_val_ds, device):
    predictions_iterator = PredictionsIterator(model, sent_val_ds, device=device)
    windows, predictions = zip(*predictions_iterator)

    # Create SemanticSegmentationLabels from predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=sent_val_ds.scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    gt_labels = sent_val_ds.scene.label_source.get_labels()

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

# List of cities and their corresponding val datasets
citie_valds = [
    ('Santo Domingo', sent_val_ds_SD),
    ('Guatemala City', sent_val_ds_GC),
    ('Tegucigalpa', sent_val_ds_TG),
    ('Managua', sent_val_ds_MN),
    ('Panama', sent_val_ds_PN),
    ('San Salvador', sent_val_ds_SS),
    ('San Jose', sent_val_ds_SJ),
    ('Belize City', sent_val_ds_BL),
    ('Belmopan', sent_val_ds_BM)
]

# Iterate through each city and calculate F1 score
for city_name, sent_val_ds in citie_valds:
    f1_score = calculate_f1_score(best_model, sent_val_ds, device)
    print(f"{city_name} - F1 Score: {f1_score:.4f}")



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