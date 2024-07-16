import os
import torch
import geopandas as gpd
from tqdm import tqdm
from pyproj import Transformer
import rasterio
from rastervision.core.box import Box
from rastervision.core.data import ClassConfig, RasterioCRSTransformer
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore
from affine import Affine
from pystac_client import Client

import sys
import torch
from rastervision.core.data.label import SemanticSegmentationLabels

import numpy as np
from typing import Any, Optional, Tuple, Union, Sequence
from pyproj import Transformer
import matplotlib.pyplot as plt
import os

from typing import Iterator, Optional

# Project-specific imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(grandparent_dir)

from deeplnafrica.deepLNAfrica import init_segm_model
from src.features.dataloaders import (
    query_buildings_data, CustomGeoJSONVectorSource,MergeDataset,
    FixedStatsTransformer,
    show_windows, CustomRasterizedSource
)
from src.models.model_definitions import (MultiResolutionDeepLabV3, MultiResPredictionsIterator,check_nan_params, CustomInterpolateMultiResolutionDeepLabV3,
                                          MultiModalDataModule, create_predictions_and_ground_truth_plot, MultiResolutionFPN, CustomMultiResolutionDeepLabV3,
                                          CustomVectorOutputConfig, FeatureMapVisualization, MultiResolution128DeepLabV3)
from src.features.dataloaders import (cities, show_windows, buil_create_full_image,ensure_tuple, MultiInputCrossValidator, create_sentinel_mosaic,
                                  senitnel_create_full_image, CustomSlidingWindowGeoDataset, collate_multi_fn, get_sentinel_items,
                                  MergeDataset, show_single_tile_multi, get_label_source_from_merge_dataset, create_scenes_for_city, PolygonWindowGeoDataset)

from rastervision.core.raster_stats import RasterStats
from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ClassConfig, GeoJSONVectorSourceConfig, GeoJSONVectorSource,
                                    RasterioSource, RasterizedSourceConfig,
                                    RasterizedSource, Scene, StatsTransformer, ClassInferenceTransformer,
                                    VectorSourceConfig, VectorSource, XarraySource, CRSTransformer,
                                    IdentityCRSTransformer, RasterioCRSTransformer)
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_store import SemanticSegmentationLabelStore

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

from typing import Any, Optional, Tuple, Union, Sequence
from rastervision.core.data.raster_source import RasterSource

from typing import Tuple, Optional
import numpy as np
import torch
from rastervision.core.box import Box
from rastervision.pytorch_learner.dataset.transform import TransformType
from rastervision.core.data import Scene
from rastervision.pytorch_learner.learner_config import PosInt, NonNegInt

BANDS = [
    'blue', # B02
    'green', # B03
    'red', # B04
    'nir', # B08
]
URL = 'https://earth-search.aws.element84.com/v1'
catalog = Client.open(URL)

def ensure_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)

class PolygonWindowGeoDataset:
    def __init__(self,
                 scene: Scene,
                 city: str,
                 window_size: Union[int, Tuple[int, int]],
                 out_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0,
                 transform: Optional[A.Compose] = None,
                 transform_type: TransformType = TransformType.noop,
                 normalize: bool = False,
                 to_pytorch: bool = True,
                 return_window: bool = False,
                 within_aoi: bool = True):
        self.scene = scene
        self.city = city
        self.window_size = ensure_tuple(window_size)
        self.out_size = ensure_tuple(out_size)
        self.padding = (self.window_size[0] // 2, self.window_size[1] // 2)
        self.padding: Tuple[NonNegInt, NonNegInt] = ensure_tuple(self.padding)
        self.transform = transform
        self.transform_type = transform_type
        self.normalize = normalize
        self.to_pytorch = to_pytorch
        self.return_window = return_window
        self.within_aoi = within_aoi

        self.windows = self.get_polygon_windows()

        if self.transform is None:
            transform_func = TF_TYPE_TO_TF_FUNC[self.transform_type]
            if callable(transform_func):
                self.transform = transform_func(self.out_size)
            else:
                print(f"Warning: transform_func is not callable: {transform_func}")

    def get_polygon_windows(self):
        windows = []
        ymax = int(self.scene.extent.ymax)
        xmax = int(self.scene.extent.xmax)
        for y in range(0, ymax, self.window_size[0]):
            for x in range(0, xmax, self.window_size[1]):
                window = Box(y, x, min(y + self.window_size[0], ymax), min(x + self.window_size[1], xmax))
                if self.has_data(window):
                    windows.append(window)
        return windows

    def has_data(self, window):
        """
        Check if the given window contains any data.
        """
        try:
            data_arr = self.scene.raster_source.get_chip(window)
            return np.any(data_arr != 0)
        except Exception as e:
            print(f"Error in has_data for window {window}: {str(e)}")
            return False

    def __getitem__(self, index):
        window = self.windows[index]
        
        img = self.scene.raster_source.get_chip(window)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        
        if self.to_pytorch:
            img = torch.from_numpy(img).float()
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            else:
                img = img.permute(2, 0, 1)
        
        if self.return_window:
            return img, window
        else:
            return img

    def __len__(self):
        return len(self.windows)

    @property
    def extent(self):
        return self.scene.extent

    def get_labels(self):
        return None  # Since we're not using labels

    def get_image(self, window):
        return self.scene.raster_source.get_chip(window)

def make_buildings_raster(image_path, resolution=5):
    with rasterio.open(image_path) as src:
        bounds = src.bounds
        xmin3857, ymin3857, xmax3857, ymax3857 = bounds.left, bounds.bottom, bounds.right, bounds.top
    
    crs_transformer_buildings = RasterioCRSTransformer.from_uri(image_path)
    affine_transform_buildings = Affine(resolution, 0, xmin3857, 0, -resolution, ymax3857)
    crs_transformer_buildings.transform = affine_transform_buildings
    
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xmin4326, ymin4326 = transformer.transform(xmin3857, ymin3857)
    xmax4326, ymax4326 = transformer.transform(xmax3857, ymax3857)
    
    buildings = query_buildings_data(xmin4326, ymin4326, xmax4326, ymax4326)
    print(f"Buildings data loaded successfully with {len(buildings)} total buildings.")
    
    buildings_vector_source = CustomGeoJSONVectorSource(
        gdf = buildings,
        crs_transformer = crs_transformer_buildings,
        vector_transformers=[ClassInferenceTransformer(default_class_id=1)])
    
    rasterized_buildings_source = CustomRasterizedSource(
        buildings_vector_source,
        background_class_id=0,
        bbox=Box(xmin3857, ymin3857, xmax3857, ymax3857))
    
    return rasterized_buildings_source, crs_transformer_buildings

def make_sentinel_raster(image_uri):
    # Define the means and stds in NIR-RGB order
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    # Define a normalized raster source using the calculated transformer
    sentinel_source_normalized = RasterioSource(
        image_uri,
        allow_streaming=True,
        raster_transformers=[fixed_stats_transformer],
        channel_order=[3, 2, 1, 0]
    )

    print(f"Loaded Sentinel data of size {sentinel_source_normalized.shape}, and dtype: {sentinel_source_normalized.dtype}")
    return sentinel_source_normalized

def create_sentinel_scene(city_data):
    image_path = city_data['image_path']

    sentinel_source_normalized = make_sentinel_raster(image_path)
    
    sentinel_scene = Scene(
        id='scene_sentinel',
        raster_source=sentinel_source_normalized
    )
    return sentinel_scene

def create_building_scene(city_name, city_data):
    image_path = city_data['image_path']

    # Create Buildings scene
    rasterized_buildings_source, _ = make_buildings_raster(image_path, resolution=5)
    
    buildings_scene = Scene(
        id=f'{city_name}_buildings',
        raster_source=rasterized_buildings_source
    )

    return buildings_scene

def get_sentinel_items(bbox_geometry, bbox):
    items = catalog.search(
        intersects=bbox_geometry,
        collections=['sentinel-2-c1-l2a'],
        datetime='2024-01-01/2024-07-16',
        query={'eo:cloud_cover': {'lt': 20}},
        # Remove max_items=1 to get multiple items
    ).item_collection()
    
    if not items:
        print("No items found for this area.")
        return None
    # items = get_sentinel_items(bbox_geometry, bbox)

    return items

def create_sentinel_mosaic(items, bbox):
    # Create the initial XarraySource with temporal=True
    sentinel_source = XarraySource.from_stac(
        items,
        bbox_map_coords=tuple(bbox),
        stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
        allow_streaming=True,
        temporal=True
    )
    
    print(f"Initial data shape: {sentinel_source.data_array.shape}")
    print(f"Initial bbox: {sentinel_source.bbox}")
    print(f"Data coordinates: {sentinel_source.data_array.coords}")
    
    crs_transformer = sentinel_source.crs_transformer
    
    # Get the CRS of the data
    data_crs = sentinel_source.crs_transformer.image_crs
    
    # Use the original bbox of the data
    data_bbox = sentinel_source.bbox
    
    print(f"Data bbox: {data_bbox}")
    
    # Apply mosaic function to combine the temporal dimension
    mosaic_data = sentinel_source.data_array.median(dim='time')
    
    print(f"Mosaic data shape: {mosaic_data.shape}")
    print(f"Mosaic data coordinates: {mosaic_data.coords}")
    
    nir_rgb_means = [2581.270, 1298.905, 1144.928, 934.346]  # NIR, R, G, B
    nir_rgb_stds = [586.279, 458.048, 302.029, 244.423]  # NIR, R, G, B

    fixed_stats_transformer = FixedStatsTransformer(
        means=nir_rgb_means,
        stds=nir_rgb_stds)
    
    # Create the final normalized XarraySource
    normalized_mosaic_source = XarraySource(
        mosaic_data,
        crs_transformer=sentinel_source.crs_transformer,
        bbox=data_bbox,
        channel_order=[3,2,1,0],
        temporal=False,
        raster_transformers=[fixed_stats_transformer]
    )
    
    print(f"Created normalized mosaic of size {normalized_mosaic_source.shape}, and dtype: {normalized_mosaic_source.dtype}")
    print(f"Normalized mosaic bbox: {normalized_mosaic_source.bbox}")
    print(f"Data CRS: {data_crs}")
    # sentinel_source_SD, crs_transformer = create_sentinel_mosaic(items, bbox)

    return normalized_mosaic_source, crs_transformer

def display_mosaic(mosaic_source):
    print(f"Mosaic source shape: {mosaic_source.shape}")
    
    if mosaic_source.shape[0] == 0 or mosaic_source.shape[1] == 0:
        print("The mosaic has zero width or height. There might be no data for the specified region.")
        return
    
    chip = mosaic_source[:, :, [1, 2, 3]]  # RGB channels
    
    print(f"Chip shape: {chip.shape}")
    print(f"Chip min: {np.min(chip)}, max: {np.max(chip)}")
    
    # For normalized data, we might want to clip to a reasonable range
    vmin, vmax = -3, 3
    normalized_chip = np.clip(chip, vmin, vmax)
    
    # Scale to [0, 1] for display
    normalized_chip = (normalized_chip - vmin) / (vmax - vmin)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(normalized_chip, interpolation='nearest', aspect='equal')
    ax.set_title("Normalized Mosaic RGB")
    plt.colorbar(im, ax=ax)
    plt.show()
    
def load_model(model_path, device):
    model = CustomInterpolateMultiResolutionDeepLabV3()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.to(device)
    model.eval()
    return model

def make_predictions(model, dataset, device):
    predictions_iterator = MultiResPredictionsIterator(model, dataset, device=device)
    windows, predictions = zip(*predictions_iterator)
    return windows, predictions

def average_predictions(pred1, pred2):
    return [(p1 + p2) / 2 for p1, p2 in zip(pred1, pred2)]

class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = [
    os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv0_res256_BCH128_BKR5_epoch=08-val_loss=0.5143.ckpt'),
    os.path.join(grandparent_dir, 'UNITAC-trained-models/multi_modal/sel_CustomDLV3/multimodal_sel_cv1_res256_BCH128_BKR5_epoch=23-val_loss=0.3972.ckpt')
]
models = [load_model(path, device) for path in model_paths]

# Load GeoDataFrame
sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
# filter to counties where iso3 is HTI
gdf = gdf[gdf['iso3'] == 'HTI']
gdf = gdf.to_crs('EPSG:4326')
gdf = gdf.tail(1)

# Main loop
for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_ascii']
    country_code = row['iso3']
    xmin4326, ymin4326, xmax4326, ymax4326 = row.geometry.bounds

    # Create bbox and bbox_geometry for PySTAC query
    bbox = Box(ymin=ymin4326, xmin=xmin4326, ymax=ymax4326, xmax=xmax4326)
    bbox_geometry = {
        'type': 'Polygon',
        'coordinates': [
            [
                (xmin4326, ymin4326),
                (xmin4326, ymax4326),
                (xmax4326, ymax4326),
                (xmax4326, ymin4326),
                (xmin4326, ymin4326)
            ]
        ]
    }

    # Get Sentinel data
    items = get_sentinel_items(bbox_geometry, bbox)
    if items is None:
        print(f"No Sentinel data found for {city_name}. Skipping.")
        continue

    # Create Sentinel mosaic
    sentinel_source, crs_transformer = create_sentinel_mosaic(items, bbox)
    print(f"Created Sentinel mosaic for {city_name}, {country_code}.")
    display_mosaic(sentinel_source)

    # Query buildings data
    buildings = query_buildings_data(xmin4326, ymin4326, xmax4326, ymax4326)
    print(f"Got buildings for {city_name}, {country_code}, in the amoint of {len(buildings)}.")
    
    # Create scenes
    sentinel_scene = Scene(
        id=f'{city_name}_sentinel',
        raster_source=sentinel_source,
        label_source=None  # No labels for prediction
    )
    
    buildings_scene = create_building_scene(buildings, bbox, crs_transformer)

    # Create datasets
    sentinel_dataset = CustomSlidingWindowGeoDataset(sentinel_scene, city=city_name, size=256, stride=128, out_size=256, padding=256, transform_type=TransformType.noop, transform=None)
    buildings_dataset = CustomSlidingWindowGeoDataset(buildings_scene, city=city_name, size=512, stride=256, out_size=512, padding=512, transform_type=TransformType.noop, transform=None)

    # Create merged dataset
    merged_dataset = MergeDataset(sentinel_dataset, buildings_dataset)

    # Make predictions with both models and average
    windows = None
    avg_predictions = None

    for model in models:
        windows, predictions = make_predictions(model, merged_dataset, device)
        if avg_predictions is None:
            avg_predictions = predictions
        else:
            avg_predictions = average_predictions(avg_predictions, predictions)

    # Create SemanticSegmentationLabels from averaged predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        avg_predictions,
        extent=sentinel_scene.extent,
        num_classes=len(class_config),
        smooth=True
    )

    # Save predictions
    vector_output_config = CustomVectorOutputConfig(
        class_id=1,
        denoise=8,
        threshold=0.5
    )

    output_dir = f'../../vectorised_model_predictions/other_cities/{country_code}'
    os.makedirs(output_dir, exist_ok=True)

    pred_label_store = SemanticSegmentationLabelStore(
        uri=os.path.join(output_dir, f'{city_name}_{country_code}.geojson'),
        crs_transformer=crs_transformer,
        class_config=class_config,
        vector_outputs=[vector_output_config],
        discrete_output=True
    )

    pred_label_store.save(pred_labels)
    print(f"Saved predictions data for {city_name}, {country_code}")

print("Finished processing all cities.")