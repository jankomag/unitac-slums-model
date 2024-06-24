import os
import sys
from datetime import datetime
from typing import Any, Optional, Tuple, Union, Sequence, Dict, Iterator, Literal, List
from shapely.geometry import Polygon

import multiprocessing
# multiprocessing.set_start_method('fork')
import cv2
import pytorch_lightning as pl
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from rastervision.core.box import Box
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from rasterio.features import rasterize
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from typing import TYPE_CHECKING
import wandb
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.data import ConcatDataset

from typing import Self
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
sys.path.append(parent_dir)

from src.models.model_definitions import MultiModalSegmentationModel, MultiModalPredictionsIterator
from deeplnafrica.deepLNAfrica import (Deeplabv3SegmentationModel, init_segm_model,
                                       CustomDeeplabv3SegmentationModel)
from src.data.dataloaders import (create_sentinel_raster_source, create_buildings_raster_source,
                                  create_datasets, show_windows, CustomStatsTransformer,
                                  CustomSemanticSegmentationSlidingWindowGeoDataset)
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.pytorch_learner import SemanticSegmentationVisualizer
from rastervision.core.data import (Scene, ClassConfig, RasterioCRSTransformer,
                                    RasterioSource, GeoJSONVectorSource,
                                    ClassInferenceTransformer, RasterizedSource,
                                    SemanticSegmentationLabelSource, VectorSource)
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.utils import listify_uris, merge_geojsons
from rastervision.pipeline.file_system import (
    get_local_path, json_to_file, make_dir, sync_to_dir, file_exists,
    download_if_needed, NotReadableError, get_tmp_dir)
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from rastervision.pytorch_learner.dataset.visualizer import Visualizer  # NOQA
from rastervision.pytorch_learner.utils import (
    color_to_triple, plot_channel_groups, channel_groups_to_imgs)

from typing import (TYPE_CHECKING, Sequence, Optional, List, Dict, Union,
                    Tuple, Any)
from abc import ABC, abstractmethod

from torch import Tensor
import albumentations as A

from rastervision.pipeline.file_system import make_dir
from rastervision.core.data import ClassConfig
from rastervision.pytorch_learner.utils import (
    deserialize_albumentation_transform, validate_albumentation_transform,
    MinMaxNormalize)
from rastervision.pytorch_learner.learner_config import (
    RGBTuple,
    ChannelInds,
    validate_channel_display_groups,
    get_default_channel_display_groups,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from matplotlib.figure import Figure

from typing import TYPE_CHECKING, Iterator, List, Optional
from os.path import join

from rastervision.pipeline.config import register_config, Config, Field
from rastervision.core.data.label_store import (LabelStoreConfig,
                                                SemanticSegmentationLabelStore)
from rastervision.core.data.utils import (denoise, mask_to_building_polygons,
                                          mask_to_polygons)

if TYPE_CHECKING:
    import numpy as np
    from shapely.geometry.base import BaseGeometry

    from rastervision.core.box import Box
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig
    
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

label_uriGC = "../../data/SHP/Guatemala_PS.shp"
image_uriGC = '../../data/0/sentinel_Gee/GTM_Chimaltenango_2023.tif'
buildings_uriGC = '../../data/0/overture/GT_buildings3857.geojson'

class_config = ClassConfig(names=['background', 'slums'], 
                                colors=['lightgray', 'darkred'],
                                null_class='background')

sentinel_source_normalized, sentinel_label_raster_source = create_sentinel_raster_source(image_uri, label_uri, class_config, clip_to_label_source=True)
rasterized_buildings_source, buildings_label_source, crs_transformer_buildings = create_buildings_raster_source(buildings_uri, image_uri, label_uri, class_config, resolution=5)    

sentinel_source_normalizedGC, sentinel_label_raster_sourceGC = create_sentinel_raster_source(image_uriGC, label_uriGC, class_config, clip_to_label_source=True)
rasterized_buildings_sourceGC, buildings_label_sourceGC, crs_transformer_buildingsGC = create_buildings_raster_source(buildings_uriGC, image_uriGC, label_uriGC, class_config, resolution=5)    

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
        label_source = sentinel_label_raster_source)
        # aoi_polygons=[pixel_polygon])
        
SentinelSceneGC = Scene(
        id='GC_sentinel',
        raster_source = sentinel_source_normalizedGC,
        label_source = sentinel_label_raster_sourceGC)

BuildingsScence = Scene(
        id='santodomingo_buildings',
        raster_source = rasterized_buildings_source,
        label_source = buildings_label_source)

BuildingsScenceGC = Scene(
        id='GC_buildings',
        raster_source = rasterized_buildings_sourceGC,
        label_source = buildings_label_sourceGC)

buildingsGeoDataset, train_buildings_dataset, val_buildings_dataset, test_buildings_dataset = create_datasets(BuildingsScence, imgsize=288, stride = 288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDataset, train_sentinel_dataset, val_sentinel_dataset, test_sentinel_dataset = create_datasets(SentinelScene, imgsize=144, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
buildingsGeoDatasetGC, train_buildings_datasetGC, val_buildings_datasetGC, test_buildings_datasetGC = create_datasets(BuildingsScenceGC, imgsize=288, stride = 288, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)
sentinelGeoDatasetGC, train_sentinel_datasetGC, val_sentinel_datasetGC, test_sentinel_datasetGC = create_datasets(SentinelSceneGC, imgsize=144, stride = 144, padding=0, val_ratio=0.2, test_ratio=0.1, seed=42)

SDGC_sentinel_train_ds = ConcatDataset([train_sentinel_dataset, train_sentinel_datasetGC])
SDGC_sentinel_val_ds = ConcatDataset([val_sentinel_dataset, val_sentinel_datasetGC])
SDGC_sentinel_test_ds = ConcatDataset([test_sentinel_dataset, test_sentinel_datasetGC])

SDGC_build_train_ds = ConcatDataset([train_buildings_dataset, train_buildings_datasetGC])
SDGC_build_val_ds = ConcatDataset([val_buildings_dataset, val_buildings_datasetGC])
SDGC_build_test_ds = ConcatDataset([test_buildings_dataset, test_buildings_datasetGC])

batch_size = 16

# num_workers = 11
# train_sentinel_loader = DataLoader(train_sentinel_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# train_buildings_loader = DataLoader(train_buildings_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# val_sentinel_loader = DataLoader(val_sentinel_dataset, batch_size=batch_size, shuffle=False) #, num_workers=num_workers)
# val_buildings_loader = DataLoader(val_buildings_dataset, batch_size=batch_size, shuffle=False)#, num_workers=num_workers)
# train_sentinel_loader.num_workers

# class MultiModalDataModule(LightningDataModule):
#     def __init__(self, train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader):
#         super().__init__()
#         self.train_sentinel_loader = train_sentinel_loader
#         self.train_buildings_loader = train_buildings_loader
#         self.val_sentinel_loader = val_sentinel_loader
#         self.val_buildings_loader = val_buildings_loader

#     def train_dataloader(self):
#         return zip(self.train_sentinel_loader, self.train_buildings_loader)

#     def val_dataloader(self):
#         return zip(self.val_sentinel_loader, self.val_buildings_loader)
    
# data_module = MultiModalDataModule(train_sentinel_loader, train_buildings_loader, val_sentinel_loader, val_buildings_loader)

# assert len(train_sentinel_loader) == len(train_buildings_loader), "DataLoaders must have the same length"
# assert len(val_sentinel_loader) == len(val_buildings_loader), "DataLoaders must have the same length"

# Other approach for dataloading # Concatenate datasets
class MergeDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
# when one city for training
# train_dataset = MergeDataset(train_sentinel_dataset, train_buildings_dataset)
# val_dataset = MergeDataset(val_sentinel_dataset, val_buildings_dataset)

train_dataset = MergeDataset(SDGC_sentinel_train_ds, SDGC_build_train_ds)
val_dataset = MergeDataset(SDGC_sentinel_val_ds, SDGC_build_val_ds)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

channel_display_groups_sent = {'RGB': (0,1,2), 'NIR': (3, )}
channel_display_groups_build = {'Buildings': (0,)}

vis_sent = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors,
    channel_display_groups=channel_display_groups_sent)

vis_build = SemanticSegmentationVisualizer(
    class_names=class_config.names, class_colors=class_config.colors,
    channel_display_groups=channel_display_groups_build)

x, y = vis_sent.get_batch(train_sentinel_dataset, 2)
vis_sent.plot_batch(x, y, show=True)

x, y = vis_build.get_batch(train_buildings_dataset, 2)
vis_build.plot_batch(x, y, show=True)

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

# Train the model
model = MultiModalSegmentationModel()
model.to(device)

for batch_idx, batch in enumerate(data_module.train_dataloader()):
    sentinel_batch, buildings_batch = batch
    buildings_data, buildings_labels = buildings_batch
    sentinel_data, _ = sentinel_batch
    
    sentinel_data = sentinel_data.to(device)
    buildings_data = buildings_data.to(device)

    print(f"Sentinel data shape: {sentinel_data.shape}")
    print(f"Buildings data shape: {buildings_data.shape}")
    print(f"Buildings labels shape: {buildings_labels.shape}")
    # Pass the data through the model
    model_out = model(batch)
    print(f"Segmentation output shape: {model_out.shape}")
    break  # Exit after the first batch for brevity

output_dir = f'../../UNITAC-trained-models/multi_modal/trained_SD_GC/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-multi-modal')
wandb_logger = WandbLogger(project='UNITAC-multi-modal', log_model=True)

# Loggers and callbacks
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='multimodal_runid{run_id}-{batch_size:02d}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=5,
    max_epochs=50,
    num_sanity_val_steps=3
)

# Train the model
trainer.fit(model, datamodule=data_module)

# # Use best model for evaluation
best_model_path = "/Users/janmagnuszewski/dev/slums-model-unitac/UNITAC-trained-models/multi_modal/multimodal_runidrun_id=0-batch_size=00-epoch=15-val_loss=0.2595.ckpt"
# best_model_path = checkpoint_callback.best_model_path
best_model = MultiModalSegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

predictions_iterator = MultiModalPredictionsIterator(best_model, sentinelGeoDataset, buildingsGeoDataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Ensure windows are Box instances
windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=BuildingsScence.extent,
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

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/multi-modal/SD_GC/',
    crs_transformer = crs_transformer_buildings,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)



### Make predictions on another city ###
# implements loading gdf - class CustomGeoJSONVectorSource
class CustomGeoJSONVectorSource(VectorSource):
    """A :class:`.VectorSource` for reading GeoJSON files or GeoDataFrames."""

    def __init__(self,
                 crs_transformer: 'CRSTransformer',
                 uris: Optional[Union[str, List[str]]] = None,
                 gdf: Optional[gpd.GeoDataFrame] = None,
                 vector_transformers: List['VectorTransformer'] = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            uris (Optional[Union[str, List[str]]]): URI(s) of the GeoJSON file(s).
            gdf (Optional[gpd.GeoDataFrame]): A GeoDataFrame with vector data.
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
            bbox (Optional[Box]): User-specified crop of the extent. If None,
                the full extent available in the source file is used.
        """
        self.uris = listify_uris(uris) if uris is not None else None
        self.gdf = gdf
        super().__init__(
            crs_transformer,
            vector_transformers=vector_transformers,
            bbox=bbox)

    def _get_geojson(self) -> dict:
        if self.gdf is not None:
            # Convert GeoDataFrame to GeoJSON
            df = self.gdf.to_crs('epsg:4326')
            geojson = df.__geo_interface__
        elif self.uris is not None:
            geojsons = [self._get_geojson_single(uri) for uri in self.uris]
            geojson = merge_geojsons(geojsons)
        else:
            raise ValueError("Either 'uris' or 'gdf' must be provided.")
        return geojson

    def _get_geojson_single(self, uri: str) -> dict:
        # download first so that it gets cached
        path = download_if_needed(uri)
        df: gpd.GeoDataFrame = gpd.read_file(path)
        df = df.to_crs('epsg:4326')
        geojson = df.__geo_interface__
        return geojson
    
image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
image_uriHT = '../../data/0/sentinel_Gee/HTI_Tabarre_2023.tif'

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
port_au_prince = gdf[gdf["city_ascii"] == "Tabarre"]
port_au_prince = port_au_prince.to_crs('EPSG:3857')
gdf_xmin, gdf_ymin, gdf_xmax, gdf_ymax = port_au_prince.total_bounds

with rasterio.open(image_uriHT) as src:
    bounds = src.bounds
    raster_xmin, raster_ymin, raster_xmax, raster_ymax = bounds.left, bounds.bottom, bounds.right, bounds.top

# Define the bounding box in EPSG:3857
common_xmin_3857 = max(gdf_xmin, raster_xmin)
common_ymin_3857 = max(gdf_ymin, raster_ymin)
common_xmax_3857 = min(gdf_xmax, raster_xmax)
common_ymax_3857 = min(gdf_ymax, raster_ymax)
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
common_xmin_4326, common_ymin_4326 = transformer.transform(common_xmin_3857, common_ymin_3857)
common_xmax_4326, common_ymax_4326 = transformer.transform(common_xmax_3857, common_ymax_3857)

import duckdb
import pandas as pd
con = duckdb.connect("../../data/0/data.db")
con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

def query_buildings_data(con, xmin, ymin, xmax, ymax):
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
        buildings['class_id'] = 1

    return buildings

buildings = query_buildings_data(con, common_xmin_4326, common_ymin_4326, common_xmax_4326, common_ymax_4326)
resolution = 5
crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_uri)
affine_transform_buildings = Affine(resolution, 0, common_xmin_3857, 0, -resolution, common_ymax_3857)
crs_transformer_buildings.transform = affine_transform_buildings

crs_transformer = RasterioCRSTransformer.from_uri(image_uriHT)
buildings_vector_source_crsimage = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])

buildings_vector_source_crsbuildings = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
rasterized_buildings_source = RasterizedSource(
    buildings_vector_source_crsbuildings,
    background_class_id=0)
    
print(f"Loaded Rasterised buildings data of size {rasterized_buildings_source.shape}, and dtype: {rasterized_buildings_source.dtype}")

# Define the bbox of buildings in the image crs 
buildings_extent = buildings_vector_source_crsimage.extent

### SENTINEL SOURCE ###
sentinel_source_unnormalized = RasterioSource(
    image_uriHT,
    allow_streaming=True)

# Calculate statistics transformer from the unnormalized source
calc_stats_transformer = CustomStatsTransformer.from_raster_sources(
    raster_sources=[sentinel_source_unnormalized],
    max_stds=3
)

# Define a normalized raster source using the calculated transformer
sentinel_sourceHT = RasterioSource(
    image_uriHT,
    allow_streaming=True,
    raster_transformers=[calc_stats_transformer],
    channel_order=[2, 1, 0, 3],
    bbox=buildings_extent
)
print(f"Loaded Sentinel data of size {sentinel_sourceHT.shape}, and dtype: {sentinel_sourceHT.dtype}")
# chip = sentinel_sourceHT[:, :, :]
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.imshow(chip)
# plt.show()

### SCENE ###    
HT_build_scene = Scene(
        id='portauprince_sent',
        raster_source = rasterized_buildings_source)

HT_sent_scene = Scene(
        id='portauprince_buildings',
        raster_source = sentinel_sourceHT)

build_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
        scene=HT_build_scene,
        size=288,
        stride=288,
        out_size=288,
        padding=0)

sent_ds = CustomSemanticSegmentationSlidingWindowGeoDataset(
        scene=HT_sent_scene,
        size=144,
        stride=144,
        out_size=144,
        padding=0)

# def create_full_image(source) -> np.ndarray:
#     extent = source.extent
#     chip = source._get_chip(extent)    
#     return chip

# img_full = create_full_image(build_ds.scene.raster_source)
# img_full.shape
# train_windows = build_ds.windows
# val_windows = build_ds.windows
# test_windows = build_ds.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# img_full = create_full_image(sentinel_sourceHT)
# img_full = img_full[:, :, 0]
# train_windows = sent_ds.windows
# val_windows = sent_ds.windows
# test_windows = sent_ds.windows
# window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
# show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

predictions_iterator = PredictionsIterator(best_model, sent_ds, build_ds, device=device)
windows, predictions = zip(*predictions_iterator)

# Ensure windows are Box instances
windows = [Box(*window.tolist()) if isinstance(window, torch.Tensor) else window for window in windows]

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=HT_build_scene.extent,
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

# Saving predictions as GEOJSON
vector_output_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.5)

crs_transformer_HT = RasterioCRSTransformer.from_uri(image_uriHT)
affine_transform_buildings = Affine(5, 0, common_xmin_3857, 0, -5, common_ymax_3857)
crs_transformer_HT.transform = affine_transform_buildings

pred_label_store = SemanticSegmentationLabelStore(
    uri='../../vectorised_model_predictions/multi-modal/SD_GC/Haiti_portauprincenostride/',
    crs_transformer = crs_transformer_HT,
    class_config = class_config,
    vector_outputs = [vector_output_config],
    discrete_output = True)

pred_label_store.save(pred_labels)