# This file contains the code from the repository deepLNAfrica, which is used as a starting point for this analysis
# https://gitlab.renkulab.io/deeplnafrica/deepLNAfrica

import pytorch_lightning as pl
from typing import Iterator, Optional
from pathlib import Path
from torchmetrics import JaccardIndex, ConfusionMatrix

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

def init_segm_model(num_bands: int = 4) -> torch.nn.Module:
    segm_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)

    if num_bands == 4:
            # Initialise the new NIR dimension as for the red channel
            weight = segm_model.backbone.conv1.weight.clone()
            segm_model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            with torch.no_grad(): # avoid tracking this operation in the autograd
                segm_model.backbone.conv1.weight[:, 1:] = weight.clone()
                segm_model.backbone.conv1.weight[:, 0] = weight[:, 0].clone()

    elif num_bands > 4:
        segm_model.backbone.conv1 = torch.nn.Conv2d(num_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    segm_model.classifier = DeepLabHead(2048, 1)

    return segm_model

class Deeplabv3SegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning Module for supervised binary segmentation
    
    ...

    Attributes
    ----------
    num_bands : int
        Number of bands of the satellite image tile inputs
    learning_rate : float
        Learning rate for AdamW
    weight_decay : float
        Weight decay for AdamW 
    pos_weight : float
        Weight for positive class
    pretrained_checkpoint : Optional[Path]
        Optional checkpoint to pretrained model
    """
    def __init__(self, 
                 num_bands: int = 4, 
                 learning_rate: float = 1e-4, 
                 weight_decay: float = 0, 
                 pos_weight: float = 1.0, 
                 pretrained_checkpoint: Optional[Path] = None) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight

        self.segm_model = init_segm_model(num_bands)
        if pretrained_checkpoint:
            pretrained_weights = torch.load(pretrained_checkpoint)["state_dict"]
            missing_keys, _ = self.load_state_dict(pretrained_weights, strict=False)
            assert len(missing_keys) == 0, missing_keys
        
        self.save_hyperparameters(ignore='pretrained_checkpoint')
        self.setup_metrics()

    def setup_metrics(self) -> None:
        self.train_metric = JaccardIndex(task='binary')
        self.val_metric = JaccardIndex(task='binary')
        self.test_metric = JaccardIndex(task='binary')
        self.test_cm = ConfusionMatrix(task='binary')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.segm_model(x)['out'].squeeze(dim=1)

        return x

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        img, groundtruth = batch
        print(groundtruth.shape)

        segmentation = self(img)

        # GT contains informal settlements in channel 0 and 
        # formal/planned settlements in channel 1 (might be all 0s if not available)
        # weights = groundtruth[:, 1, :, :] # these weights correspond to the FORMAL/PLANNED settlements layer
        # weights *= 0.3
        # # Final Weights are 1.3 for planned settlements (if available) and 1 for everything else
        # weights = weights + torch.ones_like(weights, device=self.device)
        # informal_gt = groundtruth[:, 0, :, :].float()
        
        informal_gt = groundtruth[:, :, :, :].float()
        weights = groundtruth[:, :, :, :] 

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights, pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, informal_gt)

        self.log('train_loss', loss)
        self.train_metric(segmentation.sigmoid(), informal_gt.int())

        return loss

    def on_train_epoch_end(self) -> None:
        self.log('train_meanIOU', self.train_metric.compute(), prog_bar=True)
        self.train_metric.reset()

        weight_norm = 0
        num_params = 0
        for param in self.parameters():
            weight_norm += param.norm()
            num_params += 1
        mean_weight_norm = weight_norm / num_params
        self.log('mean_weight_norm', mean_weight_norm)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch

        segmentation = self(img)
       
        # GT contains informal settlements in channel 0 and 
        # formal/planned settlements in channel 1 (might be all 0s if not available)
        weights = groundtruth[:, 1, :, :] # these weights correspond to the FORMAL/PLANNED settlements layer
        weights *= 0.3
        # Final Weights are 1.3 for planned settlements (if available) and 1 for everything else
        weights = weights + torch.ones_like(weights, device=self.device)
        informal_gt = groundtruth[:, 0, :, :].float()

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights, pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, informal_gt)

        self.log('val_loss', loss, prog_bar=True)
        self.val_metric(segmentation.sigmoid(), informal_gt.int())

    def on_validation_epoch_end(self) -> None:
        self.log('val_meanIOU', self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        img, groundtruth = batch

        segmentation = self(img)
        
        # GT contains informal settlements in channel 0 and 
        # formal/planned settlements in channel 1 (might be all 0s if not available)
        weights = groundtruth[:, 1, :, :] # these weights correspond to the FORMAL/PLANNED settlements layer
        weights *= 0.3
        # Final Weights are 1.3 for planned settlements (if available) and 1 for everything else
        weights = weights + torch.ones_like(weights, device=self.device)
        informal_gt = groundtruth[:, 0, :, :].float()

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights, pos_weight=torch.tensor(self.pos_weight))
        loss = loss_fn(segmentation, informal_gt)

        self.log('test_loss', loss)
        self.test_metric(segmentation.sigmoid(), informal_gt.int())
        self.test_cm.update(segmentation.sigmoid().ravel(), informal_gt.ravel().squeeze().int())

    def on_test_epoch_end(self) -> None:
        self.log('test_meanIOU', self.test_metric.compute())
        test_cm = self.test_cm.compute()
        print(test_cm)

        self.test_metric.reset()
        self.test_cm.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = AdamW(
            self.segm_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = MultiStepLR(optimizer, milestones=[6, 12], gamma=0.3)

        return [optimizer], [scheduler]