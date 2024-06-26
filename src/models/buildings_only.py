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
from src.models.model_definitions import (BuildingsDeeplabv3, BuildingsSimpleSS, CustomVectorOutputConfig, BuildingsOnlyPredictionsIterator)
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

def create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5):
    gdf = gpd.read_file(buildings_uri)
    gdf = gdf.to_crs('EPSG:3857')
    xmin, _, _, ymax = gdf.total_bounds
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(resolution, 0, xmin, 0, -resolution, ymax)
    crs_transformer_buildings.transform = affine_transform_buildings

    buildings_vector_source = GeoJSONVectorSource(
        buildings_uri,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = RasterizedSource(
        buildings_vector_source,
        background_class_id=0)

    print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")

    label_vector_source = GeoJSONVectorSource(label_uri,
        crs_transformer_buildings,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])

    label_raster_source = RasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    buildings_label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)

    return rasterized_buildings_source, buildings_label_source, crs_transformer_buildings

### Label source ###
label_uri_SD = "../../data/0/SantoDomingo3857.geojson"
image_uri_SD = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri_SD = '../../data/0/overture/santodomingo_buildings.geojson'
class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

rasterized_buildings_sourceSD, buildings_label_sourceSD, crs_transformer_SD = create_buildings_raster_source(buildings_uri_SD, image_uri_SD, label_uri_SD, class_config, resolution=5)

# label_uri_GC = "../../data/SHP/Guatemala_PS.shp"
# image_uri_GC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
# buildings_uri_GC = '../../data/0/overture/GT_buildings3857.geojson'
# gpd.read_file(buildings_uri_GC)
# rasterized_buildings_sourceGC, buildings_label_sourceGC, crs_transformer_GC = create_buildings_raster_source(buildings_uri_GC, image_uri_GC, label_uri_GC, class_config, resolution=5)
# BuildingsScenceGC = Scene(
#         id='guatemalacity_buildings',
#         raster_source = rasterized_buildings_sourceGC,
#         label_source = buildings_label_sourceGC)
# buildingsGeoDatasetGC, train_buildings_datasetGC, val_buildings_datasetGC, test_buildings_datasetGC = create_datasets(
#     BuildingsScenceGC, imgsize=288, stride = 288, padding=0, val_ratio=0.3, test_ratio=0.1, seed=42)

BuildingsScenceSD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_sourceSD,
        label_source = buildings_label_sourceSD)

buildingsGeoDatasetSD, train_buildings_datasetSD, val_buildings_datasetSD, test_buildings_datasetSD = create_datasets(
    BuildingsScenceSD, imgsize=288, stride = 288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

# combined_train_dataset = ConcatDataset([train_buildings_datasetSD, train_buildings_datasetGC])
# combined_val_dataset = ConcatDataset([val_buildings_datasetSD, val_buildings_datasetGC])
# combined_val_dataset = ConcatDataset([test_buildings_datasetSD, test_buildings_datasetGC])

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildingsGeoDatasetSD.scene.label_source)
train_windows = train_buildings_datasetSD.windows
val_windows = val_buildings_datasetSD.windows
test_windows = test_buildings_datasetSD.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

batch_size=8
train_dl = DataLoader(train_buildings_datasetSD, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_buildings_datasetSD, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_buildings_datasetSD, batch_size=batch_size, shuffle=False)

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
model = BuildingsSimpleSS(weight_decay=0.001,
                            learning_rate=0.001,
                            gamma=0.01,
                            pos_weight=torch.tensor(5, device='mps'))
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
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=23-val_loss=0.3083.ckpt"
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/deeplab/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=18-val_loss=0.1848.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsSimpleSS.load_from_checkpoint(best_model_path)
best_model.eval()

#  Make predictions
buildingsGeoDataset, _, _, _ = create_datasets(BuildingsScenceSD, imgsize=288, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
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

# # Saving predictions as GEOJSON
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
# from rastervision.core.evaluation import SemanticSegmentationEvaluator

# evaluator = SemanticSegmentationEvaluator(class_config)
# gt_labels = BuildingsScenceSD.label_source.get_labels()
# evaluation = evaluator.evaluate_predictions(
#     ground_truth=gt_labels, predictions=pred_labels)
# eval_metrics_dict = evaluation.class_to_eval_item[0]
# f1_score = eval_metrics_dict.f1
# f1_score

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