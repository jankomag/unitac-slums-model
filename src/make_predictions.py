import duckdb

con = duckdb.connect("../../data/0/data.db")
con.install_extension('httpfs')
con.install_extension('spatial')
con.load_extension('httpfs')
con.load_extension('spatial')
con.execute("SET s3_region='us-west-2'")
con.execute("SET azure_storage_connection_string = 'DefaultEndpointsProtocol=https;AccountName=overturemapswestus2;AccountKey=;EndpointSuffix=core.windows.net';")

# implements merging logc class CustomGeoJSONVectorSource
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


image_uri = '../../data/0/sentinel_Gee/HTI_Tabarre_2023.tif'
label_uri = "../../data/0/SantoDomingo3857.geojson"
crs_transformer_common = RasterioCRSTransformer.from_uri(image_uri)
resolution = 5

for index, row in gdf.iterrows():
    buildings = query_buildings_data(row, con)
    if buildings is not None:
        predict_and_save(buildings, image_uri, label_uri, class_config, crs_transformer_common, row['iso3'], row['city_ascii'], best_model, device)

