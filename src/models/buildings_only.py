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
    create_datasets, create_building_scene,cities, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset,
    PolygonWindowGeoDataset, MergeDataset, vis_build)

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
buildGeoDataset_SD = PolygonWindowGeoDataset(buildings_sceneSD, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_SD, buil_val_ds_SD, buil_test_ds_SD = buildGeoDataset_SD.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# img_full = senitnel_create_full_image(buildGeoDataset_SD.scene.label_source)
# train_windows = buil_train_ds_SD.windows
# val_windows = buil_val_ds_SD.windows
# test_windows = buil_test_ds_SD.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# len(buil_train_ds_SD)

# x, y = vis_build.get_batch(buil_train_ds_SD, 2)
# vis_build.plot_batch(x, y, show=True)

# Guatemala City
buildings_sceneGC = create_building_scene('GuatemalaCity', cities['GuatemalaCity'])
buildGeoDataset_GC = PolygonWindowGeoDataset(buildings_sceneGC, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_GC, buil_val_ds_GC, buil_test_ds_GC = buildGeoDataset_GC.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_GC)

# img_full = senitnel_create_full_image(buil_train_ds_GC.scene.label_source)
# train_windows = buil_train_ds_GC.windows
# val_windows = buil_val_ds_GC.windows
# test_windows = buil_test_ds_GC.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Guatemala City - Sliding windows (Train in blue, Val in red, Test in green)')

# x, y = vis_build.get_batch(buildGeoDataset_GC, 5)
# vis_build.plot_batch(x, y, show=True)

# TegucigalpaHND
buildings_sceneTG = create_building_scene('TegucigalpaHND', cities['TegucigalpaHND'])
buildGeoDataset_TG = PolygonWindowGeoDataset(buildings_sceneTG, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_TG, buil_val_ds_TG, buil_test_ds_TG = buildGeoDataset_TG.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

len(buil_train_ds_TG)

# img_full = senitnel_create_full_image(buildGeoDataset_TG.scene.label_source)
# train_windows = buil_train_ds_TG.windows
# val_windows = buil_val_ds_TG.windows
# test_windows = buil_test_ds_TG.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='TegucigalpaHND - Sliding windows (Train in blue, Val in red, Test in green)')

# x, y = vis_build.get_batch(buildGeoDataset_TG, 5)
# vis_build.plot_batch(x, y, show=True)

# Managua
buildings_sceneMN = create_building_scene('Managua', cities['Managua'])
buildGeoDataset_MN = PolygonWindowGeoDataset(buildings_sceneMN, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_MN, buil_val_ds_MN, buil_test_ds_MN = buildGeoDataset_MN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_MN)

# img_full = senitnel_create_full_image(buildGeoDataset_MN.scene.label_source)
# train_windows = buil_train_ds_MN.windows
# val_windows = buil_val_ds_MN.windows
# test_windows = buil_test_ds_MN.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Managua Sliding windows')

# x, y = vis_build.get_batch(buildGeoDataset_MN, 5)
# vis_build.plot_batch(x, y, show=True)

# Panama
buildings_scenePN = create_building_scene('Panama', cities['Panama'])
buildGeoDataset_PN = PolygonWindowGeoDataset(buildings_scenePN,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_PN, buil_val_ds_PN, buil_test_ds_PN = buildGeoDataset_PN.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_PN)

# img_full = senitnel_create_full_image(buildGeoDataset_PN.scene.label_source)
# train_windows = buil_train_ds_PN.windows
# val_windows = buil_val_ds_PN.windows
# test_windows = buil_test_ds_PN.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Panama Sliding windows (Train in blue, Val in red, Test in green)')

# x, y = vis_build.get_batch(buildGeoDataset_PN, 5)
# vis_build.plot_batch(x, y, show=True)

# San Salvador
buildings_sceneSS = create_building_scene('SanSalvador_PS', cities['SanSalvador_PS'])
buildGeoDataset_SS = PolygonWindowGeoDataset(buildings_sceneSS,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_SS, buil_val_ds_SS, buil_test_ds_SS = buildGeoDataset_SS.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_SS)

# img_full = senitnel_create_full_image(buildGeoDataset_SS.scene.label_source)
# train_windows = buil_train_ds_SS.windows
# val_windows = buil_val_ds_SS.windows
# test_windows = buil_test_ds_SS.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='SanSalvador Sliding windows')

# x, y = vis_build.get_batch(buildGeoDataset_SS, 5)
# vis_build.plot_batch(x, y, show=True)

# SanJoseCRI - data too sparse and largely rural examples, not using for training
buildings_sceneSJ = create_building_scene('SanJoseCRI', cities['SanJoseCRI'])
buildGeoDataset_SJ = PolygonWindowGeoDataset(buildings_sceneSJ,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_SJ, buil_val_ds_SJ, buil_test_ds_SJ = buildGeoDataset_SJ.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_SJ)

# img_full = senitnel_create_full_image(buildGeoDataset_SJ.scene.label_source)
# train_windows = buil_train_ds_SJ.windows
# val_windows = buil_val_ds_SJ.windows
# test_windows = buil_test_ds_SJ.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='SanSalvador Sliding windows')

# x, y = vis_build.get_batch(buildGeoDataset_SJ, 5)
# vis_build.plot_batch(x, y, show=True)

# BelizeCity
buildings_sceneBL = create_building_scene('BelizeCity', cities['BelizeCity'])
buildGeoDataset_BL = PolygonWindowGeoDataset(buildings_sceneBL,window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_BL, buil_val_ds_BL, buil_test_ds_BL = buildGeoDataset_BL.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_BL)

# img_full = senitnel_create_full_image(buildGeoDataset_BL.scene.label_source)
# train_windows = buil_train_ds_BL.windows
# val_windows = buil_val_ds_BL.windows
# test_windows = buil_test_ds_BL.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Belize City Sliding windows')

# x, y = vis_build.get_batch(buildGeoDataset_BL, 4)
# vis_build.plot_batch(x, y, show=True)

# Belmopan - data mostly rural exluding from training
buildings_sceneBM = create_building_scene('Belmopan', cities['Belmopan'])
buildGeoDataset_BM = PolygonWindowGeoDataset(buildings_sceneBM, window_size=512,out_size=512,padding=0,transform_type=TransformType.noop,transform=None)
buil_train_ds_BM, buil_val_ds_BM, buil_test_ds_BM = buildGeoDataset_BM.split_train_val_test(val_ratio=0.2,test_ratio=0.1,seed=42)

# len(buil_train_ds_BM)

# img_full = senitnel_create_full_image(buildGeoDataset_BM.scene.label_source)
# train_windows = buil_train_ds_BM.windows
# val_windows = buil_val_ds_BM.windows
# test_windows = buil_test_ds_BM.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Belmopan City Sliding windows')

# x, y = vis_build.get_batch(buildGeoDataset_BM, 2)
# vis_build.plot_batch(x, y, show=True)

train_dataset = ConcatDataset([buil_train_ds_SD, buil_train_ds_GC, buil_train_ds_TG, buil_train_ds_MN, buil_train_ds_PN, buil_train_ds_SS, buil_train_ds_BL, buil_train_ds_SJ, buil_train_ds_BM])
val_dataset = ConcatDataset([buil_val_ds_SD, buil_val_ds_GC, buil_val_ds_TG, buil_val_ds_MN, buil_val_ds_PN, buil_val_ds_SS, buil_val_ds_BL, buil_val_ds_SJ, buil_val_ds_BM])
print(f"Train dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")

batch_size = 16

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Train the model
hyperparameters = {
    'model': 'DLV3',
    'train_cities': 'all',
    'batch_size': batch_size,
    'use_deeplnafrica': True,
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
    save_top_k=4,
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