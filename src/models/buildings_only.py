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
sys.path.append(grandparent_dir)
from src.models.model_definitions import (BuildingsDeeplabv3, BuildingsUNET, CustomVectorOutputConfig, BuildingsOnlyPredictionsIterator, CustomGeoJSONVectorSource)
from deeplnafrica.deepLNAfrica import init_segm_model

import folium

from src.data.dataloaders import (
    create_datasets,
    create_buildings_raster_source, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset
)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)

### raster source ###
label_uriSD = "../../data/0/SantoDomingo3857.geojson"
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'
image_uriSD = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

gdf = gpd.read_file(label_uriSD)
gdf = gdf.to_crs('EPSG:4326')
gdf_filled = gdf.copy()
gdf_filled["geometry"] = gdf_filled.geometry.buffer(0.00008)

rasterized_buildings_sourceSD, _, _ = create_buildings_raster_source(buildings_uri, image_uriSD, label_uriSD, class_config, resolution=5)    
gdf = gpd.read_file(buildings_uri)
gdf = gdf.to_crs('EPSG:3857')
xmin, _, _, ymax = gdf.total_bounds

crs_transformer = RasterioCRSTransformer.from_uri(image_uriSD)

crs_transformer_buildings = crs_transformer
affine_transform_buildings = Affine(5, 0, xmin, 0, -5, ymax)
crs_transformer_buildings.transform = affine_transform_buildings

label_vector_source = CustomGeoJSONVectorSource(
    gdf = gdf_filled,
    crs_transformer = crs_transformer_buildings,
    vector_transformers=[
        ClassInferenceTransformer(
            default_class_id=class_config.get_class_id('slums'))])

label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
buildings_label_sourceSD = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)

BuildingsScene_SD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_sourceSD,
        label_source = buildings_label_sourceSD)

# create datasets
buildingsGeoDataset_SD, train_buil_datasetSD, val_buil_ds_SD, test_buil_dataset_SD = create_datasets(BuildingsScene_SD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = False, seed=12)
buildingsGeoDataset_SD_aug, train_buil_datasetSD_aug, val_buil_ds_SD_aug, test_buil_dataset_SD_aug = create_datasets(BuildingsScene_SD, imgsize=512, stride=512, padding=100, val_ratio=0.2, test_ratio=0.1, augment = True, seed=12)

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildingsGeoDataset_SD.scene.label_source)
train_windows = train_buil_datasetSD.windows
val_windows = val_buil_ds_SD.windows
test_windows = test_buil_dataset_SD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

build_train_ds_SD = ConcatDataset([train_buil_datasetSD, train_buil_datasetSD_aug])
build_val_ds_SD = ConcatDataset([val_buil_ds_SD, val_buil_ds_SD_aug])
len(build_train_ds_SD)

batch_size=8
train_dl = DataLoader(build_train_ds_SD, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(build_val_ds_SD, batch_size=batch_size, shuffle=False)

# Fine-tune the model
# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

# Train the model
model = BuildingsDeeplabv3(
    use_deeplnafrica = True,
    learning_rate = 1e-2,
    weight_decay = 1e-1,
    gamma = 0.1,
    atrous_rates = (12, 24, 36),
    sched_step_size = 10,
    pos_weight = torch.tensor(1.0, device='mps'))
model.to(device)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'../../UNITAC-trained-models/buildings_only/unet/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-buildings-only')
wandb_logger = WandbLogger(project='UNITAC-buildings-only', log_model=True)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='buildings_runid{run_id}_{image_size:02d}-{batch_size:02d}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min')
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=10,
    max_epochs=120,
    num_sanity_val_steps=1
)

trainer.fit(model, train_dl, val_dl)

# Best deeplab model path val=0.3083
best_model_path_deeplab = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
best_model_path_unet = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/unet/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=14-val_loss=0.4913.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsDeeplabv3.load_from_checkpoint(best_model_path) # BuildingsDeeplabv3 BuildingsUNET
best_model.eval()

# Make predictions
buildingsGeoDataset, _, _, _ = create_datasets(BuildingsScene_SD, imgsize=512, stride = 256, padding=50, val_ratio=0.2, test_ratio=0.1, augment = False, seed=42)
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