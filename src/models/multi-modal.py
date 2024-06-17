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
from src.data.dataloaders import create_full_image, show_windows, CustomSemanticSegmentationSlidingWindowGeoDataset

def customgeoms_to_raster(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool) -> np.ndarray:
    if len(df) == 0:
        return np.full(window.size, background_class_id, dtype=np.uint8)

    window_geom = window.to_shapely()

    # subset to shapes that intersect window
    df_int = df[df.intersects(window_geom)]
    # transform to window frame of reference
    shapes = df_int.translate(xoff=-window.xmin, yoff=-window.ymin)
    # class IDs of each shape
    class_ids = df_int['class_id']

    if len(shapes) > 0:
        raster = rasterize(
            shapes=list(zip(shapes, class_ids)),
            out_shape=window.size,
            fill=background_class_id,
            dtype='uint8',
            all_touched=all_touched)
    else:
        raster = np.full(window.size, background_class_id, dtype=np.uint8)

    return raster

def geoms_to_raster(df: gpd.GeoDataFrame, window: 'Box',
                    background_class_id: int, all_touched: bool) -> np.ndarray:
    if len(df) == 0:
        return np.full(window.size, background_class_id, dtype=np.uint8)

    window_geom = window.to_shapely()

    # subset to shapes that intersect window
    df_int = df[df.intersects(window_geom)]
    # transform to window frame of reference
    shapes = df_int.translate(xoff=-window.xmin, yoff=-window.ymin)
    # class IDs of each shape
    class_ids = df_int['class_id']

    if len(shapes) > 0:
        raster = rasterize(
            shapes=list(zip(shapes, class_ids)),
            out_shape=window.size,
            fill=background_class_id,
            dtype=np.uint8,
            all_touched=all_touched)
    else:
        raster = np.full(window.size, background_class_id, dtype=np.uint8)

    return raster

class CustomRasterizedSource(RasterSource):
    def __init__(self,
                 vector_source: 'VectorSource',
                 background_class_id: int,
                 bbox: Optional['Box'] = None,
                 all_touched: bool = False,
                 raster_transformers: List['RasterTransformer'] = []):
        self.vector_source = vector_source
        self.background_class_id = background_class_id
        self.all_touched = all_touched

        self.df = self.vector_source.get_dataframe()
        self.validate_labels(self.df)

        if bbox is None:
            bbox = self.vector_source.extent

        super().__init__(
            channel_order=[0],
            num_channels_raw=1,
            bbox=bbox,
            raster_transformers=raster_transformers)

    @property
    def dtype(self) -> np.dtype:
        return np.uint8

    @property
    def crs_transformer(self):
        return self.vector_source.crs_transformer

    def _get_chip(self,
                  window: 'Box',
                  out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        window = window.to_global_coords(self.bbox)
        chip = geoms_to_raster(
            self.df,
            window,
            background_class_id=self.background_class_id,
            all_touched=self.all_touched)

        if out_shape is not None:
            chip = self.resize(chip, out_shape)

        return np.expand_dims(chip, 2)

    def validate_labels(self, df: gpd.GeoDataFrame) -> None:
        geom_types = set(df.geom_type)
        if 'Point' in geom_types or 'LineString' in geom_types:
            raise ValueError('LineStrings and Points are not supported '
                             'in RasterizedSource. Use BufferTransformer '
                             'to buffer them into Polygons. '
                             f'Geom types found in data: {geom_types}')

        if len(df) > 0 and 'class_id' not in df.columns:
            raise ValueError('All label polygons must have a class_id.')

### Label source ###
label_uri = "../../data/1/UNITAC_data/SantoDomingo_PS_3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'

gdf = gpd.read_file(label_uri)
gdf = gdf.to_crs('EPSG:3857')
xmin, ymin, xmax, ymax = gdf.total_bounds

class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

crs_transformer = RasterioCRSTransformer.from_uri(image_uri)
crs_transformer.transform

affine_transform_buildings = Affine(2, 0, xmin,
                          0, -2, ymin)

crs_transformer_buildings = crs_transformer
crs_transformer_buildings.transform = affine_transform_buildings

label_vector_source = GeoJSONVectorSource(label_uri,
    crs_transformer_buildings,
    vector_transformers=[
        ClassInferenceTransformer(
            default_class_id=class_config.get_class_id('slums'))])

label_raster_source = CustomRasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
print(f"Loaded UNITAC CustomRasterizedSource: {label_raster_source.shape}")

label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)
print(f"Loaded UNITAC SemanticSegmentationLabelSource: {label_raster_source.shape}")

## Overture Buildings Data ###
geojson_uri = '../../data/0/overture/santodomingo_buildings.geojson'

crs_transformer_buildings = crs_transformer
crs_transformer_buildings.transform = affine_transform_buildings

buildings_vector_source = GeoJSONVectorSource(
    geojson_uri,
    crs_transformer_buildings,
    vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
print("Loaded buildings data")

rasterized_buildings_source = CustomRasterizedSource(
    buildings_vector_source,
    background_class_id=0)
print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")

chip = rasterized_buildings_source[:, :]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(chip, cmap="gray")
plt.show()

# Set up semantic segmentation training of just buildings and labels
buildings_only_SS_scene = Scene(
    id='santo_domingo_buildings_only',
    raster_source=rasterized_buildings_source,
    label_source = label_source)


# Creating training, validation, and testing datasets
imgszie = 256
batch_size = 8
buildingsGeoDataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
    scene=buildings_only_SS_scene,
    size=imgszie,
    stride=imgszie,
    out_size=imgszie,
    padding=50)

# Splitting dataset into train, validation, and test
train_ds, val_ds, test_ds = buildingsGeoDataset.split_train_val_test(val_ratio=0.2, test_ratio=0.1, seed=42)

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

# Create the full image from the raster source
img_full = create_full_image(buildingsGeoDataset.scene.label_source)
train_windows = train_ds.windows
val_windows = val_ds.windows
test_windows = test_ds.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) #, num_workers=3)
val_dl = DataLoader(val_ds, batch_size=batch_size)#, num_workers=3)
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

# Model definition - adapting deeplabv3
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

        # if pretrained_checkpoint:
        #     pretrained_dict = torch.load(pretrained_checkpoint, map_location='mps')['state_dict']
        #     model_dict = self.state_dict()
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #     model_dict.update(pretrained_dict)
        #     self.load_state_dict(model_dict)
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

        loss_fn = torch.nn.BCEWithLogitsLoss()#pos_weight=self.pos_weight)
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

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)

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
model = BuildingsOnlyDeeplabv3SegmentationModel(num_bands=1, pretrained_checkpoint=pretrained_checkpoint_path)
model.to(device)
model.train()

# Train the model
trainer.fit(model, train_dl, val_dl)

# Use best model for evaluation
best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/20240617_130750/buildings_image_size=00-batch_size=00-epoch=00-val_loss=0.5754.ckpt"
best_model_path = checkpoint_callback.best_model_path
best_model = BuildingsOnlyDeeplabv3SegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

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
    
# Example usage:
predictions_iterator = PredictionsIterator(best_model, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=buildingsGeoDataset.scene.extent,
    num_classes=len(class_config),
    smooth=True)

scores = pred_labels.get_score_arr(pred_labels.extent)
scores_building = scores[0]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
image = ax.imshow(scores_building)
ax.axis('off')
ax.set_title('infs Scores')
cbar = fig.colorbar(image, ax=ax)
plt.show()