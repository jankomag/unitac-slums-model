import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
from collections import OrderedDict

torch.set_default_dtype(torch.float32)

def init_segm_model(num_bands: int = 4) -> torch.nn.Module:
    
    segm_model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
    
    if num_bands == 1:
        # Change the input convolution to accept 1 channel instead of 4
        weight = segm_model.backbone.conv1.weight.clone()
        segm_model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            segm_model.backbone.conv1.weight[:, 0] = weight.mean(dim=1, keepdim=True).squeeze(1).float()
            
    if num_bands == 4:
            # Initialise the new NIR dimension as for the red channel
            weight = segm_model.backbone.conv1.weight.clone()
            segm_model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            with torch.no_grad(): # avoid tracking this operation in the autograd
                segm_model.backbone.conv1.weight[:, 1:] = weight.clone()
                segm_model.backbone.conv1.weight[:, 0] = weight[:, 0].clone()

    return segm_model

# Function to strip the prefix from the keys in the state dict and exclude certain keys
def process_state_dict(state_dict, prefix, exclude_prefixes):
    processed_state_dict = {}
    for key in state_dict.keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            if not any(new_key.startswith(exclude_prefix) for exclude_prefix in exclude_prefixes):
                processed_state_dict[new_key] = state_dict[key]
    return processed_state_dict

# Function to adjust the weights of the first convolutional layer
def adjust_first_conv_layer(state_dict, old_key, new_key):
    weight = state_dict[old_key]
    new_weight = weight.mean(dim=1, keepdim=True)  # Average across the channel dimension
    state_dict[new_key] = new_weight
    del state_dict[old_key]
    return state_dict

class BuildingsEncoder(nn.Module):
    def __init__(self, pretrained_checkpoint=None):
        super().__init__()
        
        self.segm_model = init_segm_model(num_bands=1)
        
        if pretrained_checkpoint:
            checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
            checkpoint = process_state_dict(checkpoint, "segm_model.", ["aux_classifier."])
            checkpoint = {k: v.float() for k, v in checkpoint.items()}
            
            for key in checkpoint:
                checkpoint[key] = checkpoint[key].to(dtype=torch.float32)
            
            # Adjust the first conv layer weights
            checkpoint = adjust_first_conv_layer(checkpoint, 'backbone.conv1.weight', 'backbone.conv1.weight')
            
            # Remove classifier layer weights if they don't match
            if 'classifier.4.weight' in checkpoint and 'classifier.4.bias' in checkpoint:
                del checkpoint['classifier.4.weight']
                del checkpoint['classifier.4.bias']
            
            model_dict = self.segm_model.state_dict()
            model_dict.update(checkpoint)
            self.segm_model.load_state_dict(model_dict)
            
        self.additional_conv = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.segm_model.backbone(x)['out']
        return self.additional_conv(x)

class SentinelEncoder(nn.Module):
    def __init__(self, pretrained_checkpoint=None):
        super().__init__()
        
        self.segm_model = init_segm_model(num_bands=4)
        
        self.backbone = segm_model.backbone

        new_layers = OrderedDict()
        for name, layer in backbone.named_children():
            if name == 'layer4':
                break
            new_layers[name] = layer

        self.backbone = nn.Sequential(new_layers)
        
        # if pretrained_checkpoint:
        #     checkpoint = torch.load(pretrained_checkpoint, map_location='cpu')['state_dict']
        #     checkpoint = process_state_dict(checkpoint, "segm_model.", ["aux_classifier."])
            
        #     for key in checkpoint:
        #         checkpoint[key] = checkpoint[key].to(dtype=torch.float32)
            
        #     model_dict = self.segm_model.state_dict()
        #     model_dict.update(checkpoint)
        #     self.segm_model.load_state_dict(model_dict)
        
    def forward(self, x):
        return self.backbone(x)

# Define the path to the pretrained checkpoint
pretrained_checkpoint_path = "/Users/janmagnuszewski/dev/slums-model-unitac/deeplnafrica/deeplnafrica_trained_models/all_countries/TAN_KEN_SA_UGA_SIE_SUD/checkpoints/best-val-epoch=44-step=1035-val_loss=0.2.ckpt"
BuildingsEncodermodel = BuildingsEncoder()
SentinelEncodermodel = SentinelEncoder()

# Random input
build_input = torch.randn(3, 1, 288, 288)
build_output = BuildingsEncodermodel(build_input)
build_output.shape

segm_model = init_segm_model(num_bands=1)
backbone = segm_model.backbone
out_conv = backbone.conv1(build_input) #['out']
out_conv = backbone.maxpool(out_conv) #['out']
out_conv = backbone.layer1(out_conv) #['out']
out_conv = backbone.layer2(out_conv) #['out']
out_conv = backbone.layer3(out_conv) #['out']
backbone.layer4[2]
# out_conv = backbone.layer4(out_conv) #['out']

out_conv.shape



sent_input= torch.randn(3, 4, 144, 144)
sent_output= SentinelEncodermodel(sent_input)
sent_output.shape

segm_model = init_segm_model(num_bands=4)
backbone = segm_model.backbone
out_conv = backbone.conv1(sent_input) #['out']
out_conv = backbone.maxpool(out_conv) #['out']
out_conv = backbone.layer1(out_conv) #['out']
out_conv = backbone.layer2(out_conv) #['out']
out_conv = backbone.layer3(out_conv) #['out']

out_conv.shape

# Assuming `segm_model` is your model with the backbone you described
backbone = segm_model.backbone

# Create an ordered dictionary of layers up to `layer3`
new_layers = OrderedDict()
for name, layer in backbone.named_children():
    if name == 'layer4':
        break
    new_layers[name] = layer

# Create a new IntermediateLayerGetter with the modified layers
new_backbone = nn.Sequential(new_layers)

sent_input= torch.randn(3, 4, 144, 144)
out_conv = new_backbone(sent_input)
out_conv.shape




print(f"Shape of build_output: {build_output.shape}")
print(f"Shape of sent_output: {sent_output.shape}")








# combine the features at same resolution
combined_features = build_output + sent_output

print(f"Shape of combined: {combined_features.shape}")

class JointEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.segm_model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=1)
        
    def forward(self, x):
        # Assuming x is the combined features from sentinel and buildings encoders
        x = self.segm_model.classifier(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x
    
model = JointEncoder()
input = torch.randn(3, 2048, 18, 18)
output = model(input).squeeze(1)

print(output.shape)

segm_model = init_segm_model(num_bands=1)
output = segm_model(original_input)

print(output['out'].shape)