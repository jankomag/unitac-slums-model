#### Adapting deeplabv3+
import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from lightning import LightningModule  # Assuming this is a custom module or part of your project
from datetime import datetime
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
import tempfile
import wandb
from torchgeo.models import ResNet50_Weights as TorchGeo_ResNet50_Weights
from torchgeo.models import resnet50 as torchgeo_resnet50

# Assuming you have defined your make_dir function somewhere
def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

output_dir = '../lightning-demo/'
make_dir(output_dir)
fast_dev_run = False

wandb_logger = WandbLogger(project='UNITAC-buildings-only', log_model=True)

class CustomSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomSegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for probability output
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4.0)

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        x = self.upsample(x)
        return x
    
import pytorch_lightning as pl

class SemanticSegmentationPL(pl.LightningModule):
    def __init__(self, deeplab, lr=1e-4):
        super(SemanticSegmentationPL, self).__init__()
        self.save_hyperparameters()
        self.deeplab = deeplab

    def forward(self, img):
        return self.deeplab(img)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        mask = mask.permute(0, 3, 1, 2)
        out = self.forward(img)
        loss = self.criterion(out, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask = mask.permute(0, 3, 1, 2)
        out = self.forward(img)
        loss = self.criterion(out, mask)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

class CustomResNet(nn.Module):
    def __init__(self, weights):
        super(CustomResNet, self).__init__()
        self.model = torchgeo_resnet50(weights=weights)
        self._output_stride = 32  # Typical value for ResNet in DeepLab models

    def forward(self, x):
        return self.model(x)

    @property
    def output_stride(self):
        return self._output_stride

    @output_stride.setter
    def output_stride(self, value):
        self._output_stride = value

# Weights transformation in first layer
torchgeo_model = CustomResNet(weights=TorchGeo_ResNet50_Weights.SENTINEL2_ALL_MOCO)
torchgeo_weights = torchgeo_model.state_dict()
torchgeo_conv1_weights = torchgeo_weights['model.conv1.weight']

# Initialize empty weights tensor for custom convolution
new_weights = torch.zeros(64, 1, 7, 7, device=torchgeo_conv1_weights.device)

# Fill with existing weights per channel
new_weights[:, 0, :, :] = torchgeo_conv1_weights.mean(dim=1)  # Average all weights from 13 channels to initialize weights for buildings channel

# Create a new backbone and load the ResNet50_Weights.SENTINEL2_ALL_MOCO weights into it
backbone = CustomResNet(weights=TorchGeo_ResNet50_Weights.SENTINEL2_ALL_MOCO)
backbone.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
backbone.model.conv1.weight.data = new_weights

# Initialize your custom segmentation head
segmentation_head = CustomSegmentationHead(2048, 2)  # Assuming the output channels from ResNet50 are 2048

# Initialize DeepLabV3Plus with custom encoder and segmentation head
deeplabv3plus = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=1,
    classes=2,
    activation=None  # No activation in the final layer of DeepLabV3Plus
)

# Assign the custom encoder and segmentation head to the DeepLabV3Plus model
deeplabv3plus.encoder = backbone
deeplabv3plus.segmentation_head = segmentation_head

# Initialize your Semantic Segmentation model
model = SemanticSegmentationPL(deeplabv3plus, lr=0.001)

# Checking if weights have been successfully loaded 
print((backbone.model.conv1.weight.data == model.deeplab.encoder.model.conv1.weight.data).all())  # first conv

trainer = Trainer(
    accelerator='auto',
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=[wandb_logger],
    min_epochs=1,
    max_epochs=50,
    num_sanity_val_steps=1
)

# Train the model
trainer.fit(model, train_dl, val_dl)


# Inspect a single batch of data from train_dl
for batch in train_dl:
    img, mask = batch
    print("Image shape:", img.shape)  # Check the shape of img
    print("Mask shape:", mask.shape)  # Check the shape of mask
    break  # Exit after printing the first batch