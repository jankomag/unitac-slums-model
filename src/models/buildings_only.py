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
label_uri = "../../data/0/SantoDomingo3857.geojson"
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
buildings_uri = '../../data/0/overture/santodomingo_buildings.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

rasterized_buildings_source, buildings_label_source = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)

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

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScence, imgsize=288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

chip = rasterized_buildings_source[:, :]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(chip, cmap="gray")
plt.show()

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

batch_size=6
train_dl = DataLoader(train_buildings_dataset, batch_size=batch_size, shuffle=True) #, num_workers=3)
val_dl = DataLoader(val_buildings_dataset, batch_size=batch_size)#, num_workers=3)

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
best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/buildings_only/buildings_runidrun_id=0_image_size=00-batch_size=00-epoch=09-val_loss=0.1590.ckpt"
# best_model_path = checkpoint_callback.best_model_path
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







### MAKING PREDICTIONS ON OTHER DATASETS ###
import duckdb
from tqdm import tqdm
import pandas as pd

con = duckdb.connect("../../data/0/data.db")
con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

class CustomGeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON data."""

    def __init__(self,
                 data: Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]],
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            data (Union[gpd.GeoDataFrame, List[gpd.GeoDataFrame]]): Input GeoDataFrame(s).
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            bbox (Optional[Box]): User-specified crop of the extent. If None,
                the full extent available in the source file is used.
        """
        self.data = data if isinstance(data, list) else [data]
        super().__init__(
            crs_transformer,
            vector_transformers=vector_transformers,
            bbox=bbox)

    def _get_geojson(self) -> dict:
        geojsons = [self._get_geojson_single(gdf) for gdf in self.data]
        geojson = self.merge_geojsons(geojsons)
        return geojson

    @staticmethod
    def merge_geojsons(geojsons: List[dict]) -> dict:
        # Implement your merging logic here if needed
        # For simplicity, let's assume a basic merge
        merged_geojson = {"type": "FeatureCollection", "features": []}
        for geojson in geojsons:
            merged_geojson["features"].extend(geojson["features"])
        return merged_geojson

    @staticmethod
    def _get_geojson_single(gdf: gpd.GeoDataFrame) -> dict:
        gdf = gdf.to_crs('epsg:4326')
        geojson = gdf.__geo_interface__
        return geojson

def create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5):
    gdf = gpd.read_file(buildings_uri)
    gdf = gdf.to_crs('EPSG:3857')
    xmin, ymin, xmax, ymax = gdf.total_bounds
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(resolution, 0, xmin, 0, -resolution, ymin)
    crs_transformer_buildings.transform = affine_transform_buildings

    buildings_vector_source = CustomGeoJSONVectorSource(
        buildings_uri,
        crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = CustomRasterizedSource(
        buildings_vector_source,
        background_class_id=0)

    print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")

    label_vector_source = GeoJSONVectorSource(label_uri,
        crs_transformer_buildings,
        vector_transformers=[
            ClassInferenceTransformer(
                default_class_id=class_config.get_class_id('slums'))])

    label_raster_source = CustomRasterizedSource(label_vector_source,background_class_id=class_config.null_class_id)
    buildings_label_source = SemanticSegmentationLabelSource(label_raster_source, class_config=class_config)

    return rasterized_buildings_source, buildings_label_source

def query_buildings_data(row, con):
    city_name = row['city_ascii']
    xmin, ymin, xmax, ymax = row['geometry'].bounds

    query = f"""
        SELECT *
        FROM buildings
        WHERE bbox.xmin > {xmin}
          AND bbox.xmax < {xmax}
          AND bbox.ymin > {ymin}
          AND bbox.ymax < {ymax};
    """
    buildings = pd.read_sql(query, con=con)

    if not buildings.empty:
        buildings = gpd.GeoDataFrame(buildings, geometry=gpd.GeoSeries.from_wkb(buildings.geometry.apply(bytes)), crs='EPSG:4326')
        buildings = buildings[['id', 'geometry']]
        buildings = buildings.to_crs("EPSG:3857")

    return buildings

def predict_and_save(buildings, image_uri, label_uri, class_config, crs_transformer_common, iso3_country_code, city_name, best_model, device):
    resolution = 5
    rasterized_buildings_source, buildings_label_source = create_buildings_raster_source(buildings, image_uri, label_uri, class_config, resolution=5)
    
    affine_transform_own = Affine(resolution, 0, xmin, 0, -resolution, ymin)
    crs_transformer_common.transform = affine_transform_own
    
    HT_eval_scene = Scene(
        id='HT_eval_scene',
        raster_source=rasterized_buildings_source,
        label_source=buildings_label_source)
    
    HTGeoDataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
        scene=HT_eval_scene,
        size=256,
        stride=256,
        out_size=256,
        padding=100)
    
    predictions_iterator = PredictionsIterator(best_model, HTGeoDataset, device=device)
    windows, predictions = zip(*predictions_iterator)

    pred_labels_HT = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
        extent=HTGeoDataset.scene.extent,
        num_classes=len(class_config),
        smooth=True)

    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5)
    
    pred_label_store = SemanticSegmentationLabelStore(
        uri=f'../../vectorised_model_predictions/buildings_model_only/{iso3_country_code}/{city_name}_{iso3_country_code}',
        crs_transformer=crs_transformer_common,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True)

    pred_label_store.save(pred_labels_HT)
    print(f"Saved buildings data for {city_name}")

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
gdf = gdf.head(2)

image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
label_uri = "../../data/0/SantoDomingo3857.geojson"
crs_transformer_common = RasterioCRSTransformer.from_uri(image_uri)
resolution = 5

for index, row in gdf.iterrows():
    buildings = query_buildings_data(row, con)
    if buildings is not None:
        predict_and_save(buildings, image_uri, label_uri, class_config, crs_transformer_common, row['iso3'], row['city_ascii'], best_model, device)















# Creating training, validation, and testing datasets
imgszie = 256
batch_size = 6

HTGeoDataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
    scene=HT_eval_scene,
    size=imgszie,
    stride=imgszie,
    out_size=imgszie,
    padding=100)

predictions_iterator = PredictionsIterator(best_model, HTGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels_HT = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=HTGeoDataset.scene.extent,
    num_classes=len(class_config),
    smooth=True)

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/buildings_model_only/HT/',
    crs_transformer = crs_transformer_HT,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels_HT)



# Creating training, validation, and testing datasets
imgszie = 256
batch_size = 6

HTGeoDataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
    scene=HT_eval_scene,
    size=imgszie,
    stride=imgszie,
    out_size=imgszie,
    padding=100)

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
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/buildings_model_only/HT/',
    crs_transformer = crs_transformer_HT,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels_HT)