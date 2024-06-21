import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
multiprocessing.set_start_method('fork')

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

gdf = gpd.read_file(label_uri)
gdf = gdf.to_crs('EPSG:3857')
xmin, ymin, xmax, ymax = gdf.total_bounds
pixel_polygon = Polygon([
    (xmin, ymin),
    (xmin, ymax),
    (xmax, ymax),
    (xmax, ymin),
    (xmin, ymin)
])

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

SentinelScene = Scene(
        id='santodomingo_sentinel',
        raster_source = sentinel_source_normalized,
        label_source = sentinel_label_raster_source,
        aoi_polygons=[pixel_polygon])

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScence, imgsize=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene, imgsize=144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
print(f"Loaded all dataset: {train_buildings_dataset}")

class MultiModalDataModule(LightningDataModule):
    def __init__(self, train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader):
        super().__init__()
        self.train_sentinel_loader = train_sentinel_loader
        self.train_buildings_loader = train_buildings_loader
        self.val_sentinel_loader = val_sentinel_loader
        self.val_buildings_loader = val_buildings_loader

    def train_dataloader(self):
        return zip(self.train_sentinel_loader, self.train_buildings_loader)

    def val_dataloader(self):
        return zip(self.val_sentinel_loader, self.val_buildings_loader)
data_module = MultiModalDataModule(train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader)

num_workers=11
batch_size= 4
train_sentinel_loader = DataLoader(train_sentinel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
train_buildings_loader = DataLoader(train_buildings_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_sentinel_loader = DataLoader(val_sentinel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_buildings_loader = DataLoader(val_buildings_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

assert len(train_sentinel_loader) == len(train_buildings_loader), "DataLoaders must have the same length"
assert len(val_sentinel_loader) == len(val_buildings_loader), "DataLoaders must have the same length"

# Other approach for dataloading # Concatenate datasets
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

train_dataset = ConcatDataset(train_sentinel_dataset, train_buildings_dataset)
val_dataset = ConcatDataset(val_sentinel_dataset, val_buildings_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Data module for PyTorch Lightning
class MultiModalDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

# Initialize the data module
data_module = MultiModalDataModule(train_loader, val_loader)

# def create_full_image(source) -> np.ndarray:
#     extent = source.extent
#     chip = source.get_label_arr(extent)    
#     return chip

# img_full = create_full_image(buildingsGeoDataset.scene.label_source)
# train_windows = train_buildings_dataset.windows
# val_windows = val_buildings_dataset.windows
# test_windows = test_buildings_dataset.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# def create_full_imagesent(source) -> np.ndarray:
#     extent = source.extent
#     chip = source._get_chip(extent)    
#     return chip

# img_full = create_full_imagesent(SentinelScene.label_source)
# train_windows = train_sentinel_dataset.windows
# val_windows = val_sentinel_dataset.windows
# test_windows = test_sentinel_dataset.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

def init_segm_model(num_bands: int = 4) -> torch.nn.Module:
    
    segm_model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
    
    # for 1 band (buildings layer only)
    if num_bands == 1:
        # Change the input convolution to accept 1 channel instead of 4
        weight = segm_model.backbone.conv1.weight.clone()
        segm_model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            segm_model.backbone.conv1.weight[:, 0] = weight.mean(dim=1, keepdim=True).squeeze(1)
            
    if num_bands == 4:
            # Initialise the new NIR dimension as for the red channel
            weight = segm_model.backbone.conv1.weight.clone()
            segm_model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            with torch.no_grad(): # avoid tracking this operation in the autograd
                segm_model.backbone.conv1.weight[:, 1:] = weight.clone()
                segm_model.backbone.conv1.weight[:, 0] = weight[:, 0].clone()

    return segm_model

# class BuildingsEncoder(nn.Module):
#     def __init__(self, pretrained_checkpoint=None):
#         super().__init__()
        
#         self.segm_model = init_segm_model(num_bands=1)
        
#         if pretrained_checkpoint:
#             checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
#             checkpoint = process_state_dict(checkpoint, "segm_model.", ["aux_classifier."])
#             checkpoint = {k: v.float() for k, v in checkpoint.items()}
            
#             # Adjust the first conv layer weights
#             checkpoint = adjust_first_conv_layer(checkpoint, 'backbone.conv1.weight', 'backbone.conv1.weight')
            
#             # Remove classifier layer weights if they don't match
#             if 'classifier.4.weight' in checkpoint and 'classifier.4.bias' in checkpoint:
#                 del checkpoint['classifier.4.weight']
#                 del checkpoint['classifier.4.bias']
            
#             model_dict = self.segm_model.state_dict()
#             model_dict.update(checkpoint)
#             self.segm_model.load_state_dict(model_dict)
            
#         self.additional_conv = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1)

#     def forward(self, x):
#         x = self.segm_model.backbone(x)['out']
#         return self.additional_conv(x)

# class JointEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.segm_model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
        
#     def forward(self, x):
#         # Assuming x is the combined features from sentinel and buildings encoders
#         x = self.segm_model.classifier(x)
#         x = F.interpolate(x, size=(288, 288), mode='bilinear', align_corners=False)
#         return x

# class SentinelEncoder(nn.Module):
#     def __init__(self, pretrained_checkpoint=None):
#         super().__init__()
        
#         self.segm_model = init_segm_model(num_bands=4)
        
#         if pretrained_checkpoint:
#             checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
#             checkpoint = process_state_dict(checkpoint, "segm_model.", ["aux_classifier."])
#             checkpoint = {k: v.float() for k, v in checkpoint.items()}
            
#             model_dict = self.segm_model.state_dict()
#             model_dict.update(checkpoint)
#             self.segm_model.load_state_dict(model_dict)
        
#     def forward(self, x):
#         return self.segm_model.backbone(x)['out']
    
# Function to strip the prefix from the keys in the state dict and exclude certain keys
def process_state_dict(state_dict, prefix, exclude_prefixes):
    processed_state_dict = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            if not any(new_key.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
                processed_state_dict[new_key] = state_dict[key]
    return processed_state_dict

# Function to adjust the weights of the first convolutional layer
def adjust_first_conv_layer(state_dict, old_key, new_key):
    weight = state_dict[old_key]
    new_weight = weight.mean(dim=1, keepdim=True)  # Average across the channel dimension
    state_dict[new_key] = new_weight
    del state_dict[old_key]
    return state_dict

class SentinelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Example encoder structure (adjust as needed)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class BuildingsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Example encoder structure (adjust as needed)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Example decoder structure (adjust as needed)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x):
        return self.decoder(x)

# Train the model
class MultiModalSegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.learning_rate = 1e-4
        self.weight_decay = 0
        
        self.sentinel_encoder = SentinelEncoder()
        self.buildings_encoder = BuildingsEncoder()
        self.decoder = Decoder()
             
    def forward(self, sentinel_data, buildings_data):
        sentinel_features = self.sentinel_encoder(sentinel_data)
        buildings_features = self.buildings_encoder(buildings_data)
        
        # Upsample sentinel features to match buildings features size
        sentinel_features = nn.functional.interpolate(sentinel_features, size=buildings_features.shape[2:])
        
        # Concatenate features along the channel dimension
        fused_features = torch.cat([sentinel_features, buildings_features], dim=1)
        
        # Decode the fused features
        segmentation = self.decoder(fused_features)
        
        return segmentation
    
    def training_step(self, batch, batch_idx):
        
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        segmentation = self(sentinel_data, buildings_data)
        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(segmentation, buildings_labels.float())
        
        self.log('train_loss', loss)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        segmentation = self(sentinel_data, buildings_data)        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fn(segmentation, buildings_labels.float())
        
        self.log('val_loss', val_loss, prog_bar=True)
        wandb.log({'val_loss': val_loss.item(), 'epoch': self.current_epoch})
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        segmentation = self(sentinel_data, buildings_data)        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        test_loss = loss_fn(segmentation, buildings_labels)
        self.log('test_loss', test_loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.parameters(),
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
    
model = MultiModalSegmentationModel()
model.to(device)
model.eval()
# data_module = MultiModalDataModule(train_loader, val_loader)

for batch_idx, batch in enumerate(data_module.train_dataloader()):
    sentinel_batch, buildings_batch = batch
    buildings_data, buildings_labels = buildings_batch
    sentinel_data, _ = sentinel_batch
    
    # Print shapes for verification
    sentinel_data = sentinel_data.to(device)
    buildings_data = buildings_data.to(device)

    print(f"Batch {batch_idx}:")
    print(f"Sentinel data shape: {sentinel_data.shape}")
    print(f"Buildings data shape: {buildings_data.shape}")
    print(f"Buildings labels shape: {buildings_labels.shape}")
    
    sent_model = SentinelEncoder()
    sent_encoded = sent_model(sentinel_data)
    buildings_model = BuildingsEncoder()
    buildings_encoded = buildings_model(buildings_data)
    print("Sentinel encoded shape: ", sent_encoded.shape)
    print("Buildings encoded shape: ", buildings_encoded.shape)
    
    # Pass the data through the model
    # segmentation = model(sentinel_data, buildings_data)
    # print(f"Segmentation output shape: {segmentation.shape}")
    
    break  # Exit after the first batch for brevity

output_dir = f'../../UNITAC-trained-models/multi_modal/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal')
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

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
trainer.fit(model, datamodule=data_module)

# Use best model for evaluation
best_model_path = checkpoint_callback.best_model_path
best_model = MultiModalSegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

class PredictionsIterator:
    def __init__(self, model, dataloader, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        self.predictions = []
        self.windows = []
        
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                sentinel_batch, buildings_batch = batch
                buildings_data, buildings_labels = buildings_batch
                sentinel_data, _ = sentinel_batch

                sentinel_data = sentinel_data.to(device)
                buildings_data = buildings_data.to(device)

                output = self.model(sentinel_data, buildings_data)
                probabilities = torch.sigmoid(output).cpu().numpy()
                
                # Store predictions along with window coordinates
                self.predictions.extend(probabilities)
                self.windows.extend([Box.from_tensor(window) for window in buildings_labels])  # Ensure windows are Box objects

    def __iter__(self):
        return iter(zip(self.windows, self.predictions))

predictions_iterator = PredictionsIterator(best_model, train_loader, device=device)
windows, predictions = zip(*predictions_iterator)

# Ensure windows are Box instances
windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=SentinelScene.extent,
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
    uri='../../vectorised_model_predictions/buildings_model_only/',
    crs_transformer = crs_transformer_buildings,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)

pred_labels.extent



# for batch_idx, batch in enumerate(data_module.train_dataloader()):
#     sentinel_batch, buildings_batch = batch
#     buildings_data, buildings_labels = buildings_batch
#     sentinel_data, _ = sentinel_batch
    
#     # Print shapes for verification
#     sentinel_data = sentinel_data.to(device)
#     buildings_data = buildings_data.to(device)

#     print(f"Batch {batch_idx}:")
#     print(f"Sentinel data shape: {sentinel_data.shape}")
#     print(f"Buildings data shape: {buildings_data.shape}")
#     print(f"Buildings labels shape: {buildings_labels.shape}")
    
#     # Pass the data through the model
#     segmentation = model(sentinel_data, buildings_data)
#     print(f"Segmentation output shape: {segmentation.shape}")
    
#     break  # Exit after the first batch for brevity








# Extra Classes to accommodate multi-modal data
class MultiSemanticSegmentationSlidingWindowGeoDataset(SlidingWindowGeoDataset):

    def __init__(self, scene, size, stride, out_size=None, padding=None, window_sizes=None):
        super().__init__(
            scene=scene,
            size=size,
            stride=stride,
            out_size=out_size,
            padding=padding,
            transform_type=TransformType.semantic_segmentation)
        self.window_sizes = window_sizes if window_sizes else [size]

    def split_train_val_test(self, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = None) -> Tuple['MultiSemanticSegmentationSlidingWindowGeoDataset', 'MultiSemanticSegmentationSlidingWindowGeoDataset', 'MultiSemanticSegmentationSlidingWindowGeoDataset']:
        """
        Split the dataset into training, validation, and test subsets.

        Args:
            val_ratio (float): Ratio of validation data to total data. Defaults to 0.2.
            test_ratio (float): Ratio of test data to total data. Defaults to 0.2.
            seed (int): Seed for the random number generator. Defaults to None.

        Returns:
            Tuple[MultiSemanticSegmentationSlidingWindowGeoDataset, MultiSemanticSegmentationSlidingWindowGeoDataset, MultiSemanticSegmentationSlidingWindowGeoDataset]: 
                Training, validation, and test subsets as MultiSemanticSegmentationSlidingWindowGeoDataset objects.
        """
        assert 0.0 < val_ratio < 1.0, "val_ratio should be between 0 and 1."
        assert 0.0 < test_ratio < 1.0, "test_ratio should be between 0 and 1."
        assert val_ratio + test_ratio < 1.0, "Sum of val_ratio and test_ratio should be less than 1."

        if seed is not None:
            random.seed(seed)

        # Calculate number of samples for validation, test, and training
        num_samples = len(self)
        num_val = int(val_ratio * num_samples)
        num_test = int(test_ratio * num_samples)
        num_train = num_samples - num_val - num_test

        # Create indices for train, validation, and test subsets
        indices = list(range(num_samples))
        random.shuffle(indices)  # Shuffle indices randomly

        val_indices = indices[:num_val]
        test_indices = indices[num_val:num_val + num_test]
        train_indices = indices[num_val + num_test:]

        # Create new datasets for training, validation, and test
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices)
        test_dataset = self._create_subset(test_indices)

        return train_dataset, val_dataset, test_dataset

    def _create_subset(self, indices):
        """
        Create a subset of the dataset using specified indices.

        Args:
            indices (list): List of indices to include in the subset.

        Returns:
            MultiSemanticSegmentationSlidingWindowGeoDataset: Subset of the dataset.
        """
        subset = MultiSemanticSegmentationSlidingWindowGeoDataset(
            scene=self.scene,
            size=self.size,
            stride=self.stride,
            out_size=self.out_size,
            padding=self.padding,
            window_sizes=self.window_sizes
        )

        # Initialize subset's windows based on provided indices
        subset.windows = [self.windows[i] for i in indices]

        return subset

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self):
            raise StopIteration()

        window = self.windows[idx]
        data_dict = super().__getitem__(idx)

        x_sentinel = data_dict['senitnel_data']
        y_sentinel = data_dict['senitnel_mask']

        x_buildings = data_dict['buildings_data']
        y_buildings = data_dict['buildings_mask']
        
        # Construct the final output dictionary
        output_dict = {
            'senitnel_data': x_sentinel,
            'senitnel_mask': y_sentinel,
            'buildings_data': x_buildings,
            'buildings_mask': y_buildings,
        }

        return output_dict


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self):
            raise StopIteration()

        window = self.windows[idx]
        data_dict = super().__getitem__(idx)

        x_sentinel = data_dict['senitnel_data']
        y_sentinel = data_dict['senitnel_mask']

        x_buildings = data_dict['buildings_data']
        y_buildings = data_dict['buildings_mask']
        
        # Construct the final output dictionary
        output_dict = {
            'senitnel_data': x_sentinel,
            'senitnel_mask': y_sentinel,
            'buildings_data': x_buildings,
            'buildings_mask': y_buildings,
        }

        return output_dict

class MultiScene:
    """The raster data and labels associated with an area of interest."""

    def __init__(self,
                 id: str,
                 raster_sources: List['RasterSource'],
                 label_sources: Optional[List['LabelSource']] = None,
                 label_store: Optional['LabelStore'] = None,
                 aoi_polygons: Optional[List['BaseGeometry']] = None):
        """Constructor.

        During initialization, `MultiScene` attempts to set the extents of the
        given `label_sources` and the `label_store` to be identical to the
        extent of the given `raster_sources`.

        Args:
            id: ID for this scene.
            raster_sources: List of sources of imagery for this scene.
            label_sources: List of sources of labels for this scene.
            label_store: Store of predictions for this scene.
            aoi: Optional list of AOI polygons in pixel coordinates.
        """
        if label_sources is not None:
            for label_source in label_sources:
                match_bboxes(raster_sources[0], label_source)

        if label_store is not None:
            match_bboxes(raster_sources[0], label_store)

        self.id = id
        self.raster_sources = raster_sources
        self.label_sources = label_sources if label_sources is not None else []

        if aoi_polygons is None:
            self.aoi_polygons = []
            self.aoi_polygons_bbox_coords = []
        else:
            for p in aoi_polygons:
                if p.geom_type not in ['Polygon', 'MultiPolygon']:
                    raise ValueError(
                        'Expected all AOI geometries to be Polygons or '
                        f'MultiPolygons. Found: {p.geom_type}.')
            bbox = raster_sources[0].bbox
            bbox_geom = bbox.to_shapely()
            self.aoi_polygons = [
                p for p in aoi_polygons if p.intersects(bbox_geom)
            ]
            self.aoi_polygons_bbox_coords = list(
                geoms_to_bbox_coords(self.aoi_polygons, bbox))

    @property
    def extent(self) -> 'Box':
        """Extent of the associated :class:`.RasterSource`."""
        return self.raster_sources[0].extent

    @property
    def bbox(self) -> 'Box':
        """Bounding box applied to the source data."""
        return self.raster_sources[0].bbox

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        """Return a dictionary with raster and label data for the given key."""
        raster_data = {f"raster_source_{i}": rs[key] for i, rs in enumerate(self.raster_sources)}
        label_data = {f"label_source_{i}": ls[key] for i, ls in enumerate(self.label_sources)}

        return {
            "raster_data": raster_data,
            "label_data": label_data
        }

raster_sources = [sentinel_source_normalized, rasterized_buildings_source]

multi_scene = MultiScene(
    id='santodomingo_multi',
    raster_sources=raster_sources,
    label_sources=[sentinel_label_raster_source, buildings_label_source]
)

multi_scene[:]

dataset = MultiSemanticSegmentationSlidingWindowGeoDataset(scene=multi_scene,
                                                           size=256, stride=128, 
                                                           window_sizes=[144, 288])

train_dataset, val_dataset, test_dataset = dataset.split_train_val_test(val_ratio=0.2, test_ratio=0.1, seed=42)


print(f"Sample: {train_dataset[:]}")
