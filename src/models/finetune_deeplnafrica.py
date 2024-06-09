import os
from subprocess import check_output

os.environ['GDAL_DATA'] = check_output('pip show rasterio | grep Location | awk \'{print $NF"/rasterio/gdal_data/"}\'', shell=True).decode().strip()

import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import tempfile
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer

import matplotlib.pyplot as plt
from rasterio.features import rasterize
from shapely.geometry import Polygon
from rastervision.pipeline.file_system import (
    sync_to_dir, json_to_file, make_dir, zipdir, download_if_needed,
    download_or_copy, sync_from_dir, get_local_path, unzip, is_local,
    get_tmp_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabels,
    VectorOutputConfig, Config, Field, SemanticSegmentationDiscreteLabels
)
from rastervision.pytorch_learner import (
    SolverConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationLearner, SemanticSegmentationGeoDataConfig,
)
from rastervision.core.data.utils import make_ss_scene
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from deeplnafrica.deepLNAfrica import Deeplabv3SegmentationModel
from src.data.dataloaders import get_senitnel_dl_ds, eval_scene, get_training_sentinelOnly
from rastervision.core.data.utils import get_polygons_from_uris

# Define device
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")
    print("MPS is available.")
    
# Load pretrained model
model = Deeplabv3SegmentationModel(num_bands=4)

import sys
CKPT_PATH = Path('../../deeplnafrica/deeplnafrica_trained_models/')

pretrained_checkpoint = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/tanzania/all_wo_arusha/checkpoints/best-val-epoch=9-step=490-val_loss=0.2.ckpt"
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/south_africa/all_wo_embalenhle/checkpoints/best-val-epoch=5-step=4476-val_loss=0.5.ckpt" # 
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/kenya/all_wo_kawangware/checkpoints/best-val-epoch=8-step=261-val_loss=0.2.ckpt" #
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt" best so far
# "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/kenya/all_wo_eastleigh/checkpoints/best-val-epoch=37-step=1216-val_loss=0.1.ckpt"

# Path("../../deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt")
model = Deeplabv3SegmentationModel.load_from_checkpoint(pretrained_checkpoint, pretrained_checkpoint=None)

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
image = ax.imshow(scores_background, cmap='plasma')
ax.axis('off')
ax.set_title('infs')
cbar = fig.colorbar(image, ax=ax)
cbar.set_label('Score')
plt.show()

# Evaluate against labels:
gt_labels = scene.label_source.get_labels()
gt_extent = gt_labels.extent
pred_extent = pred_labels.extent
print(f"Ground truth extent: {gt_extent}")
print(f"Prediction extent: {pred_extent}")

evaluator = SemanticSegmentationEvaluator(class_config)
evaluation = evaluator.evaluate_predictions(ground_truth=gt_labels, predictions=pred_labels)

evaluation.class_to_eval_item[0]
evaluation.class_to_eval_item[1]

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
image_size = 512
train_dl, val_dl, train_dataset, val_dataset = get_training_sentinelOnly(batch_size=8, size=image_size, stride=int(image_size / 2), out_size=image_size, padding=100)
model = Deeplabv3SegmentationModel.load_from_checkpoint(pretrained_checkpoint, pretrained_checkpoint=None)
model.to(device)

data_cfg = SemanticSegmentationGeoDataConfig(class_config=class_config, num_workers=0)
solver_cfg = SolverConfig(batch_sz=8,lr=3e-2,class_loss_weights=[1., 10.])

learner_cfg = SemanticSegmentationLearnerConfig(data=data_cfg, solver=solver_cfg)
scene = eval_scene()

# Define a learner
learner = SemanticSegmentationLearner(
    cfg=learner_cfg,
    output_dir='./model_predictions/',
    model=model,
    train_ds=train_dataset,
    valid_ds=val_dataset,
    training=True
)

model.train()
learner.train(epochs=10)

# Other approach
output_dir = '../lightning-demo/'
make_dir(output_dir)
fast_dev_run = False

default_root_dir = os.path.join(tempfile.gettempdir(), "experiments")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)
tb_logger = TensorBoardLogger(save_dir=output_dir, flush_secs=10)

trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=[tb_logger],
    min_epochs=1,
    max_epochs=50,
    num_sanity_val_steps=1
)

model.train()
trainer.fit(model, train_dl, val_dl)

# %load_ext tensorboard
# %tensorboard --bind_all --logdir "./train-demo/tb-logs" --reload_interval 10