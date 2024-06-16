import sys
import torch
from affine import Affine
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd
from shapely.geometry import box
from rastervision.core.box import Box
import stackstac
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from torch.utils.data import DataLoader
from typing import List
from rasterio.features import rasterize
import pystac_client
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.loggers.wandb import WandbLogger

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
from deeplnafrica.deepLNAfrica import init_segm_model, CustomDeeplabv3SegmentationModel1Band
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
ax.imshow(chip)
plt.show()

# Set up semantic segmentation training of just buildings and labels
buildings_only_SS_scene = Scene(
    id='santo_domingo_buildings_only',
    raster_source=rasterized_buildings_source,
    label_source = label_raster_source)

# Creating training, validation, and testing datasets
imgszie = 512
buildingsGeoDataset = CustomSemanticSegmentationSlidingWindowGeoDataset(
    scene=buildings_only_SS_scene,
    size=imgszie,
    stride=imgszie,
    out_size=imgszie,
    padding=50)

# Splitting dataset into train, validation, and test
train_ds, val_ds, test_ds = buildingsGeoDataset.split_train_val_test(val_ratio=0.2, test_ratio=0.2, seed=42)

# Create the full image from the raster source
img_full = create_full_image(buildingsGeoDataset.scene.label_source)
train_windows = train_ds.windows
val_windows = val_ds.windows
test_windows = test_ds.windows
window_labels = (['train'] * len(train_windows) + 
                 ['val'] * len(val_windows) + 
                 ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

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
    
wandb_logger = WandbLogger(project='UNITAC-buildings-only', log_model=True)

# Reinitialise the model with pretrained weights
pretrained_checkpoint_path = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
model = CustomDeeplabv3SegmentationModel1Band(pretrained_checkpoint=pretrained_checkpoint_path, map_location=device)
model.to(device)

output_dir = '../../UNITAC-trained-models/deeplnafrica_finetuned_sentinel_only/'
make_dir(output_dir)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='finetuned_{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
    )

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=15)

# Define trainer
trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=1,
    max_epochs=50,
    num_sanity_val_steps=1
)

# Train the model
model.train()
trainer.fit(model, train_dl, val_dl)