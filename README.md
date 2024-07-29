# Mapping precarious areas with open data 🌎

### Project in collaboration with UNITAC, developed as part of the final dissertation for the Urban Spatial Science MSC, CASA, UCL.

The project creates a multimodal deep learning model for combining Sentinel-2 imagery with higher resolution building footprints data from Overture Maps Foundation to improve mapping of precarious areas in Central American cities. The project builds on the pretrained Deeplabv3 model from the [DeepLNAfrica project](https://gitlab.renkulab.io/deeplnafrica/deepLNAfrica).

The model is defined [here](https://github.com/jankomag/unitac-slums-model/blob/main/src/models/model_definitions.py#L451), and training is done [here](https://github.com/jankomag/unitac-slums-model/blob/main/src/models/multimodal.py).

Jan Magnuszewski 2024, UCL
