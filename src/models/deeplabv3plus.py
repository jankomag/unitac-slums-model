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
import pytorch_lightning as pl

device = torch.device('mps' if torch.has_mps else 'cpu')
device

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

output_dir = '../lightning-demo/'
make_dir(output_dir)
fast_dev_run = False

wandb_logger = WandbLogger(project='UNITAC-buildings-only', log_model=True)

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

# Initialize the ResNet50 model with weights pretrained on Sentinel-2 data
resnet_encoder_pretrainedsentinel = torchgeo_resnet50(weights=TorchGeo_ResNet50_Weights.SENTINEL2_ALL_MOCO)
resnet_encoder_pretrainedsentinel.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

torchgeo_weights = resnet_encoder_pretrainedsentinel.state_dict()
torchgeo_conv1_weights = torchgeo_weights['conv1.weight']

# Initialize empty weights tensor for custom convolution
new_weights = torch.zeros(64, 1, 7, 7, device='mps')

# Fill with existing weights
new_weights[:, 0, :, :] = torchgeo_conv1_weights.mean(dim=1)  # Average all weights from 13 channels to initialize weights for buildings channel
resnet_encoder_pretrainedsentinel.conv1.weight.data = new_weights
resnet_encoder_pretrainedsentinel.to(device='mps')

modules = list(resnet_encoder_pretrainedsentinel.children())[:-2]  # Remove last two layers
resnet_encoder_modified = nn.Sequential(*modules)

resnet_encoder_modified = resnet_encoder_modified.to(device)

# Create an input tensor and move it to the same device
input_tensor = torch.randn(4, 1, 288, 288).to(device)

# Define a function to get outputs at each stage from your custom encoder
def get_custom_encoder_outputs(encoder, x):
    outputs = []
    for module in encoder:
        x = module(x)
        outputs.append(x)
    return outputs

# Get the output shapes of the custom encoder
with torch.no_grad():
    custom_encoder_outputs = get_custom_encoder_outputs(resnet_encoder_modified, input_tensor)
    for i, output in enumerate(custom_encoder_outputs):
        print(f'Custom ResNet50 encoder output shape at stage {i}: {output.shape}')
        
# Initialize DeepLabV3Plus with custom encoder and segmentation head
deeplabv3plus = smp.DeepLabV3Plus(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=1,
    classes=2,
    activation=None
)

# Move the DeepLabV3Plus model to the specified device
deeplabv3plus = deeplabv3plus.to(device)

# Get the output shape of the original encoder
with torch.no_grad():
    original_encoder_outputs = deeplabv3plus.encoder(input_tensor)
    for i, output in enumerate(original_encoder_outputs):
        print(f'Original DeepLabV3+ ResNet50 encoder output shape at stage {i}: {output.shape}')    