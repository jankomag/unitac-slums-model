import os
from os.path import join
from subprocess import check_output

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()

import sys
from pathlib import Path
from typing import Iterator, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
import tempfile
import wandb
import numpy as np
import cv2
# import lightning.pytorch as pl

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger

import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import Polygon
from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, make_dir, zipdir, download_if_needed,
    download_or_copy, sync_from_dir, get_local_path, unzip, is_local,
    get_tmp_dir)
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels, RasterioCRSTransformer,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
from rastervision.pytorch_learner import (
    SolverConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner, SemanticSegmentationGeoDataConfig, SemanticSegmentationVisualizer,
)
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel, init_segm_model, CustomDeeplabv3SegmentationModel
from src.data.dataloaders import get_senitnel_dl_ds, eval_scene, get_training_sentinelOnly, create_full_image, show_windows
from rastervision.core.data.utils import get_polygons_from_uris

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
    
# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        print("MPS not available.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
# Load pretrained model
# model = Deeplabv3SegmentationModel(num_bands=4)
import sys
CKPT_PATH = Path('../../deeplnafrica/deeplnafrica_trained_models/')

pretrained_checkpoint_path = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/tanzania/all_wo_arusha/checkpoints/best-val-epoch=9-step=490-val_loss=0.2.ckpt"
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/south_africa/all_wo_embalenhle/checkpoints/best-val-epoch=5-step=4476-val_loss=0.5.ckpt" # 
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/kenya/all_wo_kawangware/checkpoints/best-val-epoch=8-step=261-val_loss=0.2.ckpt" #
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt" best so far
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/kenya/all_wo_eastleigh/checkpoints/best-val-epoch=37-step=1216-val_loss=0.1.ckpt"

# Path("../../deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt")
model = Deeplabv3SegmentationModel.load_from_checkpoint(pretrained_checkpoint_path, pretrained_checkpoint=None)

model.to(device)
model.eval()

# Load data
class_config = ClassConfig(names=['background', 'slums'], 
                           colors=['lightgray', 'darkred'],
                           null_class='background')
class_config.ensure_null_class()

data_cfg = SemanticSegmentationGeoDataConfig(class_config=class_config, num_workers=0)
solver_cfg = SolverConfig(batch_sz=4,lr=3e-2,class_loss_weights=[1., 1.])
learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)

image_size = 144
sentinel_dl, sentinel_train_ds, sentinel_val_ds = get_senitnel_dl_ds(batch_size=8, size=image_size, stride=int(image_size/2), out_size=image_size, padding=100)
scene = eval_scene()

# Define a learner
learner = SemanticSegmentationLearner(
    cfg=learner_cfg,
    output_dir='./model_predictions/',
    model=model,
    train_ds=sentinel_train_ds,
    valid_ds=sentinel_val_ds,
    training=False,
)

# Make predictions
predictions = learner.predict_dataset(
    sentinel_train_ds,
    raw_out=True,
    numpy_out=True,
    predict_kw=dict(out_shape=(image_size, image_size)),
    progress_bar=True)

pred_labels = SemanticSegmentationLabels.from_predictions(
    sentinel_train_ds.windows,
    predictions,
    smooth=True,
    extent=sentinel_train_ds.scene.extent,
    num_classes=len(class_config))

scores = pred_labels.get_score_arr(pred_labels.extent)

scores_building = scores[0]
scores_background = scores[1]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.tight_layout()
image = ax.imshow(scores_building) #cmap='plasma')
ax.axis('off')
ax.set_title('infs')
cbar = fig.colorbar(image, ax=ax)
plt.show()

# # Evaluate against labels:
# gt_labels = scene.label_source.get_labels()
# gt_extent = gt_labels.extent
# pred_extent = pred_labels.extent
# print(f"Ground truth extent: {gt_extent}")
# print(f"Prediction extent: {pred_extent}")

# evaluator = SemanticSegmentationEvaluator(class_config)
# evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels)

# evaluation.class_to_eval_item[0]
# evaluation.class_to_eval_item[1]

# # Discrete labels
# pred_labels_dis = SemanticSegmentationLabels.from_predictions(
#     sentinel_train_ds.windows,
#     predictions,
#     smooth=False,
#     extent=sentinel_train_ds.scene.extent,
#     num_classes=len(class_config))

# scores_dis = pred_labels.get_class_mask(window=sentinel_train_ds.windows[6],class_id=1,threshold=0.005)

# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# fig.tight_layout()
# image = ax.imshow(scores_dis, cmap='plasma')
# ax.axis('off')
# ax.set_title('infs')
# cbar = fig.colorbar(image, ax=ax)
# cbar.set_label('Score')
# plt.show()

# Fine-tune the model
image_size = 144
batch_size = 6

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'../../UNITAC-trained-models/deeplnafrica_finetuned_sentinel_only/{run_id}/'
os.makedirs(output_dir, exist_ok=True)

wandb.init(project='UNITAC-finetune-sentinel-only')
wandb_logger = WandbLogger(project='UNITAC-finetune-sentinel-only', log_model=True)

# Getting training data
full_dataset, train_dl, val_dl, train_dataset, val_dataset, test_dataset = get_training_sentinelOnly(batch_size=batch_size, imgsize=image_size, padding=100, seed=42)
def create_full_image(source) -> np.ndarray:
    extent = source.extent
    chip = source._get_chip(extent)    
    return chip

img_full = create_full_image(full_dataset.scene.label_source)
train_windows = train_dataset.windows
val_windows = val_dataset.windows
test_windows = test_dataset.windows
window_labels = (['train'] * len(train_windows) + ['val'] * len(val_windows) + ['test'] * len(test_windows))
show_windows(img_full, train_windows + val_windows + test_windows, window_labels, title='Sliding windows (Train in blue, Val in red, Test in green)')

# Reinitialise the model with pretrained weights
model = CustomDeeplabv3SegmentationModel.load_from_checkpoint(pretrained_checkpoint_path, pretrained_checkpoint=None)
# model = CustomDeeplabv3SegmentationModel()
model.to(device)

# Loggers and callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=output_dir,
    filename='finetuned_{image_size:02d}-{batch_size:02d}-{epoch:02d}-{val_loss:.4f}',
    save_top_k=1,
    mode='min',
    )

early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50)

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
model.train()
trainer.fit(model, train_dl, val_dl)

trainer.validate(model, val_dl)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
trainer.test(model, test_dl) 

# Use best model for evaluation
best_model_path = checkpoint_callback.best_model_path
best_model = CustomDeeplabv3SegmentationModel.load_from_checkpoint(best_model_path)
best_model.eval()

# Example usage:
predictions_iterator = PredictionsIterator(best_model, full_dataset, device=device)
windows, predictions = zip(*predictions_iterator)

# Create SemanticSegmentationLabels from predictions
pred_labels = SemanticSegmentationLabels.from_predictions(
    windows,
    predictions,
    extent=full_dataset.scene.extent,
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

# Save predictions to Geojson
vo_config = CustomVectorOutputConfig(
    class_id=1,
    denoise=8,
    threshold=0.3  # Adjust as needed
)

image_uri = '../../data/0/sentinel_Gee/DOM_Los_Minas_2024.tif'
crs_transformer = RasterioCRSTransformer.from_uri(image_uri)

pred_labels.save(
    uri='../vectorised_model_predictions/finetuned_sentinel_only/',
    crs_transformer=crs_transformer,
    class_config=class_config,
    vector_outputs=[vo_config],
    save_as_rgb=False  # Disable RGB output if not needed
)