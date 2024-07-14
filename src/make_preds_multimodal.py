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
    create_datasets,FixedStatsTransformer,
    show_windows, CustomRasterizedSource
)
from src.models.model_definitions import (MultiResolutionDeepLabV3, CustomVectorOutputConfig)

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

best_model_path_dplv3 = os.path.join(grandparent_dir, "UNITAC-trained-models/multi_modal/all_DLV3/multimodal_cv1_BCH128_BKR7_epoch=06-val_loss=0.1827.ckpt")
best_model = MultiResolutionDeepLabV3(buil_channels=128, buil_kernel1=7)
checkpoint = torch.load(best_model_path_dplv3)
state_dict = checkpoint['state_dict']
best_model.load_state_dict(state_dict, strict=True)
best_model = best_model.to(device)
best_model.eval()

class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')

sica_cities = "/Users/janmagnuszewski/dev/slums-model-unitac/data/0/SICA_cities.parquet"
gdf = gpd.read_parquet(sica_cities)
gdf = gdf.to_crs('EPSG:3857')
gdf = gdf.tail(2)

# In your main loop
for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
    city_name = row['city_ascii']
    country_code = row['iso3']
    image_uri = f"../../data/0/sentinel_Gee/{country_code}_{city_name}_2023.tif"
    
    if not os.path.exists(image_uri):
        print(f"Warning: File {image_uri} does not exist. Skipping to next row.")
        continue
    
    city_data = {
        'image_path': image_uri,
    }
    
    # Create scenes
    sentinel_scene = create_sentinel_scene(city_data)
    buildings_scene = create_building_scene(city_name, city_data)
    
    # Create datasets
    sentinelGeoDataset = PolygonWindowGeoDataset(sentinel_scene, city=city_name, window_size=256, out_size=256, padding=0, transform_type=TransformType.noop, transform=None)
    buildingsGeoDataset = PolygonWindowGeoDataset(buildings_scene, city=city_name, window_size=512, out_size=512, padding=0, transform_type=TransformType.noop, transform=None)
    
    # Create merged dataset
    mergedds = MergeDataset(sentinelGeoDataset, buildingsGeoDataset)
    
    # Create prediction iterator
    predictions_iterator = MultiResPredictionsIterator(best_model, mergedds, device=device)
    windows, predictions = zip(*predictions_iterator)
    
    # Create SemanticSegmentationLabels from predictions
    pred_labels = SemanticSegmentationLabels.from_predictions(
        windows,
        predictions,
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
    
    crs_transformer = RasterioCRSTransformer.from_uri(image_uri)
    affine_transform_buildings = Affine(10, 0, common_xmin_3857, 0, -10, common_ymax_3857)
    crs_transformer.transform = affine_transform_buildings
    
    output_dir = f'../../vectorised_model_predictions/multi-modal/all_DLV3/{country_code}'
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



# # Merge geojson for cities
# def merge_geojson_files(country_directory, output_file):
#     # Create an empty GeoDataFrame with an appropriate schema
#     merged_gdf = gpd.GeoDataFrame()
    
#     # Traverse the directory structure
#     for city in os.listdir(country_directory):
#         city_path = os.path.join(country_directory, city)
#         vector_output_path = os.path.join(city_path, 'vector_output')
        
#         if os.path.isdir(vector_output_path):
#             # Find the .json file in the vector_output directory
#             for file in os.listdir(vector_output_path):
#                 if file.endswith('.json'):
#                     file_path = os.path.join(vector_output_path, file)
#                     # Load the GeoJSON file into a GeoDataFrame
#                     gdf = gpd.read_file(file_path)
#                     # Add the city name as an attribute to each feature
#                     gdf['city'] = city
#                     # Append to the merged GeoDataFrame
#                     merged_gdf = pd.concat([merged_gdf, gdf], ignore_index=True)
    
#     # Save the merged GeoDataFrame to a GeoJSON file
#     merged_gdf.to_file(output_file, driver='GeoJSON')
#     print(f'Merged GeoJSON file saved to {output_file}')

# # Specify the country directory and the output file path
# country_directory = '../vectorised_model_predictions/multi-modal/SD_GC/SLV/'
# output_file = os.path.join(country_directory, 'SLV_multimodal_SDGC.geojson')

# # Merge the GeoJSON files
# merge_geojson_files(country_directory, output_file)




# # From STAC
# BANDS = [
#     'blue', # B02
#     'green', # B03
#     'red', # B04
#     'nir', # B08
# ]

# URL = 'https://earth-search.aws.element84.com/v1'
# catalog = pystac_client.Client.open(URL)

# from stackstac import mosaic
# def mosaic_sentinel_images(items, bbox):
#     # Convert items to XarraySource
#     sentinel_source_unnormalized = XarraySource.from_stac(
#         items,
#         bbox_map_coords=tuple(bbox),
#         stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
#         allow_streaming=True,
#     )
    
#     # Mosaic the images
#     mosaic_image = mosaic(sentinel_source_unnormalized.data_array, reverse=True)
    
#     return mosaic_image

# def get_sentinel_item(bbox_geometry, bbox):
#     items = catalog.search(
#         intersects=bbox_geometry,
#         collections=['sentinel-2-c1-l2a'],
#         datetime='2023-01-01/2024-06-27',
#         query={'eo:cloud_cover': {'lt': 3}},
#         max_items=1,
#     ).item_collection()
    
#     if not items:
#         print("No items found for this city.")
#         return None
    
#     sentinel_source_unnormalized = XarraySource.from_stac(
#         items,
#         bbox_map_coords=tuple(bbox),
#         stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
#         allow_streaming=True,
#     )

#     stats_tf = CustomStatsTransformer.from_raster_sources([sentinel_source_unnormalized],max_stds=3)

#     sentinel_source = XarraySource.from_stac(
#         items,
#         bbox_map_coords=tuple(bbox),
#         raster_transformers=[stats_tf],
#         stackstac_args=dict(rescale=False, fill_value=0, assets=BANDS),
#         allow_streaming=True,
#         channel_order=[2, 1, 0, 3],
#     )
    
#     print(f"Loaded Sentinel data of size {sentinel_source.shape}, and dtype: {sentinel_source.dtype}")
    
#     chip = sentinel_source[:, :, [0, 1, 2]]
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#     ax.imshow(chip)
#     plt.show()
    
#     return sentinel_source



# # for index, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
#     city_name = row['city_ascii']
#     country_code = row['iso3']
#     print("Doing predictions for: ", city_name, country_code)
    
#     gdf_xmin, gdf_ymin, gdf_xmax, gdf_ymax = row['geometry'].bounds
    
#     transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
#     xmin_4326, ymin_4326 = transformer.transform(gdf_xmin, gdf_ymin)
#     xmax_4326, ymax_4326 = transformer.transform(gdf_xmax, gdf_ymax)

#     bbox = Box(ymin=ymin_4326, xmin=xmin_4326, ymax=ymax_4326, xmax=xmax_4326)
#     bbox_geometry = {
#         'type': 'Polygon',
#         'coordinates': [
#             [
#                 (xmin_4326, ymin_4326),
#                 (xmin_4326, ymax_4326),
#                 (xmax_4326, ymax_4326),
#                 (xmax_4326, ymin_4326),
#                 (xmin_4326, ymin_4326)
#             ]
#         ]
#     }
    
#     # Getting Sentinel data
#     try:
#         sentinel_source = get_sentinel_item(bbox_geometry, bbox)
#         if sentinel_source is None:
#             continue
#     except Exception as e:
#         print(f"An error occurred for {city_name}, {country_code}: {e}")
#         continue
    
#     # Getting Buildings data    
#     buildings = query_buildings_data(con, xmin_4326, ymin_4326, xmax_4326, ymax_4326)
#     print("Got buildings for: ", city_name, country_code)

#     rasterized_buildings_source, buildings_extent, crs_transformer_buildings = rasterise_buildings(image_uri, buildings, gdf_xmin, gdf_ymax)
    
#     buil_ds, sent_ds, build_scene = build_datasets(rasterized_buildings_source, sentinel_source)
#     print("Got datasets extent for: ", city_name, country_code)
    
#     save_predictions(best_model, sent_ds, buil_ds, build_scene, crs_transformer_buildings, country_code, city_name)
#     print(f"Saved predictions data for {city_name}, {country_code}")
