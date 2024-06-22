import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
multiprocessing.set_start_method('fork')
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
import geopandas as gpd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import albumentations as A
from affine import Affine
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.features import rasterize
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from typing import TYPE_CHECKING
from pydantic import conint
from xarray import DataArray
import wandb
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.nn.functional as F

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pystac import Item
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from rastervision.core.box import Box
from rastervision.core.data import (
    RasterSource, RasterioSource, RasterTransformer, ClassConfig,
    GeoJSONVectorSourceConfig, GeoJSONVectorSource, MinMaxTransformer,
    MultiRasterSource, RasterizedSourceConfig, RasterizedSource, Scene,
    StatsTransformer, ClassInferenceTransformer, VectorSourceConfig,
    VectorSource, XarraySource, CRSTransformer, IdentityCRSTransformer,
    RasterioCRSTransformer, SemanticSegmentationLabelSource,
    LabelSource, LabelStore, SemanticSegmentationLabels,
    SemanticSegmentationLabelStore, SemanticSegmentationLabels,
    SemanticSegmentationLabelStore, pad_to_window_size
)

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer, CRSTransformer
    from rastervision.core.box import Box

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from src.data.dataloaders import (
    create_datasets, create_sentinel_raster_source,
    create_buildings_raster_source, show_windows
)

from rastervision.core.data.utils import all_equal, match_bboxes, geoms_to_bbox_coords
from rastervision.core.raster_stats import RasterStats
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt
from rastervision.pipeline.utils import repr_with_args
from rastervision.pytorch_learner.dataset.transform import (TransformType)
from rastervision.pipeline.config import (Config,Field)
from rastervision.pytorch_learner.dataset import SlidingWindowGeoDataset, TransformType

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
    
label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

sentinel_source_normalized, sentinel_label_raster_source = create_sentinel_raster_source(image_uri, label_uri, class_config)
rasterized_buildings_source, buildings_label_source = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)

raster_sources = [rasterized_buildings_source, sentinel_source_normalized]
raster_source_multi = MultiRasterSource(raster_sources=raster_sources, primary_source_idx=0, force_same_dtype=True)

MultiRasterScene = Scene(
        id='santodomingo_sentinel',
        raster_source = raster_source_multi,
        label_source = sentinel_label_raster_source)

MultiGeoDataset, train_ds, val_ds, test_ds = create_datasets(MultiRasterScene, imgsize=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

num_workers=11
batch_size=4 
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
# class CustomDeeplabv3SegmentationModel(pl.LightningModule):
#     def __init__(self,
#                  num_bands: int = 5,
#                  learning_rate: float = 1e-4,
#                  weight_decay: float = 1e-4,
#                  pos_weight: torch.Tensor = torch.tensor([1.0, 1.0]),
#                  pretrained_checkpoint: Optional[Path] = None) -> None:
#         super().__init__()

#         self.learning_rate = learning_rate
#         self.weight_decay = weight_decay
#         self.segm_model = init_segm_model(num_bands)

#         self.save_hyperparameters(ignore='pretrained_checkpoint')

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.segm_model(x)['out']
#         x = x.permute(0, 2, 3, 1)
#         return x
    
#     def compute_mean_iou(self, preds, target):
#         preds = preds.bool()
#         target = target.bool()
#         smooth = 1e-6
#         intersection = (preds & target).float().sum((1, 2))
#         union = (preds | target).float().sum((1, 2))
#         iou = (intersection + smooth) / (union + smooth)
#         return iou.mean()

#     def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
#         img, groundtruth = batch
#         segmentation = self(img)
#         groundtruth = groundtruth.float()

#         loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
#         loss = loss_fn(segmentation, groundtruth)

#         preds = torch.sigmoid(segmentation) > 0.5
#         mean_iou = self.compute_mean_iou(preds, groundtruth)

#         self.log('train_loss', loss)
#         self.log('train_mean_iou', mean_iou)

#         return loss

#     def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         img, groundtruth = batch
#         groundtruth = groundtruth.float()
#         segmentation = self(img)

#         loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
#         loss = loss_fn(segmentation, groundtruth)

#         preds = torch.sigmoid(segmentation) > 0.5
#         mean_iou = self.compute_mean_iou(preds, groundtruth)

#         self.log('val_loss', loss)
#         self.log('val_mean_iou', mean_iou)

#     def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
#         img, groundtruth = batch
#         segmentation = self(img)

#         informal_gt = groundtruth[:, 0, :, :].float()

#         loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
#         loss = loss_fn(segmentation, informal_gt)

#         preds = torch.sigmoid(segmentation) > 0.5
#         mean_iou = self.compute_mean_iou(preds, informal_gt)

#         self.log('test_loss', loss)
#         self.log('test_mean_iou', mean_iou)

#     def configure_optimizers(self) -> torch.optim.Optimizer:
#         optimizer = AdamW(
#             self.segm_model.parameters(),
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay
#         )

#         scheduler = MultiStepLR(optimizer, milestones=[6, 12], gamma=0.3)

#         return [optimizer], [scheduler]


model = CustomDeeplabv3SegmentationModel()
model.to(device)
model.eval()

for batch_idx, batch in enumerate(train_dl):
    features, labels = batch
    features = features.to(device)
    print(f"Buildings labels shape: {features.shape}")
    
    output = model(features)
    print("output shape: ", output.shape)
    break

output_dir = f'../../UNITAC-trained-models/multi_modal/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multiraster-5m')
wandb_logger = WandbLogger(project='UNITAC-multiraster-5m', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}_{image_size:02d}-{batch_size:02d}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min')
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=5,
    max_epochs=50,
    num_sanity_val_steps=2
)
# Train the model
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)