import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
multiprocessing.set_start_method('fork')

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

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pystac import Item
from torch.utils.data import DataLoader, Dataset

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
# raster_source_multi = MultiRasterSource(raster_sources=raster_sources, primary_source_idx=0, force_same_dtype=True)

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

def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source.get_label_arr(extent)    
    return chip

img_full = create_full_image(buildingsGeoDataset.scene.label_source)
train_windows = train_buildings_dataset.windows
val_windows = val_buildings_dataset.windows
test_windows = test_buildings_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

def create_full_imagesent(source) -> np.ndarray:
    extent = source.extent
    chip = source._get_chip(extent)    
    return chip

img_full = create_full_imagesent(SentinelScene.label_source)
train_windows = train_sentinel_dataset.windows
val_windows = val_sentinel_dataset.windows
test_windows = test_sentinel_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

from pytorch_lightning import LightningDataModule

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

num_workers=4
batch_size= 3
train_sentinel_loader = DataLoader(train_sentinel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
train_buildings_loader = DataLoader(train_buildings_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_sentinel_loader = DataLoader(val_sentinel_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_buildings_loader = DataLoader(val_buildings_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

assert len(train_sentinel_loader) == len(train_buildings_loader), "DataLoaders must have the same length"
assert len(val_sentinel_loader) == len(val_buildings_loader), "DataLoaders must have the same length"

data_module = MultiModalDataModule(train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader)

# Testing train_dataloader()
print("Testing train_dataloader():")
for batch_idx, batch in enumerate(data_module.train_dataloader()):
    
    sentinel_batch, buildings_batch = batch
    
    buildings_data, buildings_labels = buildings_batch
    
    sentinel_data, _ = sentinel_batch
    
    print(f"Batch {batch_idx}:")
    print(f"Sentinel data shape: {sentinel_data.shape}")
    print(f"Buildings data shape: {buildings_data.shape}")
    print(f"Buildings labels shape: {buildings_labels.shape}")
    break  # Print only the first batch for brevity

print()

# Fine-tune the model
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")

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
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=1,
    max_epochs=100,
    num_sanity_val_steps=1
)

# Train the model
class MultiModalSegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.sentinel_encoder = _
        
        self.buildings_feature_extractor = _
            
    def forward(self, sentinel_data, buildings_data):
        sentinel_features = self.sentinel_encoder(sentinel_data)
        buildings_features = self.buildings_feature_extractor(buildings_data)
        
        # combine the features at same resolution
        combined_features = torch.cat((sentinel_features, buildings_features), dim=1)
        
        return combined_features
    
    def training_step(self, batch, batch_idx):
        
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        segmentation = self(sentinel_data, buildings_data)
        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(segmentation, buildings_labels)
        
        self.log('train_loss', loss)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        sentinel_batch, buildings_batch = batch
        buildings_data, buildings_labels = buildings_batch
        sentinel_data, _ = sentinel_batch
        
        segmentation = self(sentinel_data, buildings_data)        
        assert segmentation.shape == buildings_labels.shape, f"Shapes mismatch: {segmentation.shape} vs {buildings_labels.shape}"

        loss_fn = torch.nn.BCEWithLogitsLoss()
        val_loss = loss_fn(segmentation, buildings_labels)
        
        self.log('val_loss', val_loss, prog_bar=True)
        
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
    
model = MultiModalSegmentationModel()
model.to(device)
model.train()

# Train the model
trainer.fit(model, datamodule=data_module)










# Extra Classes to accommodate multi-modal data
import math
import random

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
