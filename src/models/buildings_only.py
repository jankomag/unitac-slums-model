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
    
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (VectorSource, XarraySource,
                                    IdentityCRSTransformer, RasterioCRSTransformer,
                                    RasterioCRSTransformer)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
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

class CustomVectorOutputConfig(Config):
    """Config for vectorized semantic segmentation predictions."""
    class_id: int = Field(
        ...,
        description='The prediction class that is to be turned into vectors.'
    )
    denoise: int = Field(
        8,
        description='Diameter of the circular structural element used to '
        'remove high-frequency signals from the image. Smaller values will '
        'reduce less noise and make vectorization slower and more memory '
        'intensive (especially for large images). Larger values will remove '
        'more noise and make vectorization faster but might also remove '
        'legitimate detections.'
    )
    threshold: Optional[float] = Field(
        None,
        description='Probability threshold for creating the binary mask for '
        'the pixels of this class. Pixels will be considered to belong to '
        'this class if their probability for this class is >= ``threshold``. '
        'Defaults to ``None``, which is equivalent to (1 / num_classes).'
    )

    def vectorize(self, mask: np.ndarray) -> Iterator['Polygon']:
        """Vectorize binary mask representing the target class into polygons."""
        # Apply denoising if necessary
        if self.denoise > 0:
            kernel = np.ones((self.denoise, self.denoise), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to polygons
        for contour in contours:
            if contour.size >= 6:  # Minimum number of points for a valid polygon
                yield Polygon(contour.squeeze())

    def get_uri(self, root: str, class_config: Optional['ClassConfig'] = None) -> str:
        """Get the URI for saving the vector output."""
        if class_config is not None:
            class_name = class_config.get_name(self.class_id)
            uri = join(root, f'class-{self.class_id}-{class_name}.json')
        else:
            uri = join(root, f'class-{self.class_id}.json')
        return uri

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

# gdf = gpd.read_file(label_uri)
# gdf = gdf.to_crs('EPSG:3857')
# xmin, ymin, xmax, ymax = gdf.total_bounds
# pixel_polygon = Polygon([
#     (xmin, ymin),
#     (xmin, ymax),
#     (xmax, ymax),
#     (xmax, ymin),
#     (xmin, ymin)
# ])

BuildingsScenceSD = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_sourceSD,
        label_source = buildings_label_sourceSD)

# BuildingsScenceGC = Scene(
#         id='guatemalacity_buildings',
#         raster_source = rasterized_buildings_sourceGC,
#         label_source = buildings_label_sourceGC)

# buildingsGeoDatasetGC, train_buildings_datasetGC, val_buildings_datasetGC, test_buildings_datasetGC = create_datasets(
#     BuildingsScenceGC, imgsize=288, stride = 288, padding=0, val_ratio=0.3, test_ratio=0.1, seed=42)

buildingsGeoDatasetSD, train_buildings_datasetSD, val_buildings_datasetSD, test_buildings_datasetSD = create_datasets(
    SentinelScene, imgsize=288, stride = 288, padding=0, val_ratio=0.3, test_ratio=0.1, seed=42)

# combined_train_dataset = ConcatDataset([train_buildings_datasetSD, train_buildings_datasetGC])
# combined_val_dataset = ConcatDataset([val_buildings_datasetSD, val_buildings_datasetGC])
# combined_val_dataset = ConcatDataset([test_buildings_datasetSD, test_buildings_datasetGC])

# def create_full_image(source) -> np.ndarray:
#     extent = source.extent
#     chip = source.get_label_arr(extent)    
#     return chip

# img_full = create_full_image(buildingsGeoDatasetGC.scene.label_source)
# train_windows = train_buildings_datasetGC.windows
# val_windows = val_buildings_datasetGC.windows
# test_windows = test_buildings_datasetGC.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

batch_size=6
train_dl = DataLoader(train_buildings_datasetSD, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_buildings_datasetSD, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_buildings_datasetSD, batch_size=batch_size, shuffle=False)
train_dl.num_workers
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

# Model definition
class BuildingsOnlyDeeplabv3SegmentationModel(pl.LightningModule):
    def __init__(self,
                num_bands: int = 1,
                learning_rate: float = 1e-4,
                weight_decay: float = 1e-4,
                # pos_weight: torch.Tensor = torch.tensor([1.0, 1.0], device='mps'),
                pretrained_checkpoint: Optional[Path] = None) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # self.pos_weight = pos_weight
        self.segm_model = init_segm_model(num_bands)
        
        if pretrained_checkpoint:
            pretrained_dict = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
            model_dict = self.state_dict()

            # Filter out unnecessary keys and convert to float32
            pretrained_dict = {k: v.float() if v.dtype == torch.float64 else v for k, v in pretrained_dict.items() if k in model_dict and 'backbone.conv1.weight' not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            # Special handling for the first convolutional layer
            if num_bands == 1:
                conv1_weight = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']['segm_model.backbone.conv1.weight']
                new_weight = conv1_weight.mean(dim=1, keepdim=True).float()  # Ensure float32 dtype
                with torch.no_grad():
                    self.segm_model.backbone.conv1.weight[:, 0] = new_weight.squeeze(1)

        self.save_hyperparameters(ignore='pretrained_checkpoint')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.segm_model(x)['out'].squeeze(dim=1)
        # print(f"The shape of the model output before premutation {x.shape}")
        # x = x.permute(0, 2, 3, 1)
        # print(f"The shape of the model output after premutation  {x.shape}")
        return x

    def compute_mean_iou(self, preds, target):
        preds = preds.bool()
        target = target.bool()
        smooth = 1e-6
        intersection = (preds & target).float().sum((1, 2))
        union = (preds | target).float().sum((1, 2))
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean()

    def training_step(self, batch, batch_idx):
        img, groundtruth = batch
        segmentation = self(img)
        groundtruth = groundtruth.float()
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('train_loss', loss)
        self.log('train_mean_iou', mean_iou)

        return loss

    def validation_step(self, batch, batch_idx):
        img, groundtruth = batch
        groundtruth = groundtruth.float()
        segmentation = self(img)
        
        assert segmentation.shape == groundtruth.shape, f"Shapes mismatch: {segmentation.shape} vs {groundtruth.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, groundtruth)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, groundtruth)

        self.log('val_loss', loss)
        self.log('val_mean_iou', mean_iou.item())


    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch
        segmentation = self(img.to(device))

        informal_gt = groundtruth[:, 0, :, :].float().to(device)

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
        loss = loss_fn(segmentation, informal_gt)

        preds = torch.sigmoid(segmentation) > 0.5
        mean_iou = self.compute_mean_iou(preds, informal_gt)

        self.log('test_loss', loss)
        self.log('test_mean_iou', mean_iou)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.segm_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = MultiStepLR(optimizer, milestones=[6, 12], gamma=0.3)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
        
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'../../UNITAC-trained-models/buildings_only/'
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
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=6)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=1,
    max_epochs=120,
    num_sanity_val_steps=1
)

# Train the model
pretrained_checkpoint_path = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
model = BuildingsOnlyDeeplabv3SegmentationModel(num_bands=1)#, pretrained_checkpoint=pretrained_checkpoint_path)
model.to(device)

trainer.fit(model, train_dl, val_dl)

# Use best model for evaluation
# best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=09-val_loss=0.1590.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsOnlyDeeplabv3SegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

#  Make predictions
class PredictionsIterator:
    def __init__(self, model, dataset, device='cuda'):
        self.model = model
        self.dataset = dataset
        self.device = device
        
        self.predictions = []
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, _ = dataset[idx]
                image = image.unsqueeze(0).to(device)

                output = self.model(image)
                probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Store predictions along with window coordinates
                window = dataset.windows[idx]
                self.predictions.append((window, probabilities))

    def __iter__(self):
        return iter(self.predictions)
    
buildingsGeoDataset, _, _, _ = create_datasets(BuildingsScenceSD, imgsize=288, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
predictions_iterator = PredictionsIterator(best_model, buildingsGeoDataset, device=device)
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
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/buildings_model_only/2/',
    crs_transformer = crs_transformer_SD,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)

# Map interactive visualization
predspath = '/Users/janmagnuszewski/dev/slums-model-unitac/vectorised_model_predictions/buildings_model_only/2/vector_output/class-1-slums.json'
label_uri = "../../data/0/SantoDomingo3857.geojson"
extent_gdf = gpd.read_file(label_uri)
gdf = gpd.read_file(predspath)
m = folium.Map(location=[gdf.geometry[0].centroid.y, gdf.geometry[0].centroid.x], zoom_start=12)
folium.GeoJson(gdf).add_to(m) 
folium.GeoJson(extent_gdf, style_function=lambda x: {'color':'red'}).add_to(m)
m

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

predictions_iterator = PredictionsIterator(best_model, HTGeoDataset, device=device)
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
pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/buildings_model_only/HT/',
    crs_transformer = crs_transformer_HT,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels_HT)

predspath = '/Users/janmagnuszewski/dev/slums-model-unitac/vectorised_model_predictions/buildings_model_only/GT/vector_output/class-1-slums.json'
gdf = gpd.read_file(predspath)
m = folium.Map(location=[gdf.geometry[0].centroid.y, gdf.geometry[0].centroid.x], zoom_start=12)
folium.GeoJson(gdf).add_to(m) 
folium.GeoJson(extent_gdf, style_function=lambda x: {'color':'red'}).add_to(m)
m