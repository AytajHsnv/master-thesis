import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from model.dcn import DeformConv2d
classification_models = torchvision.models.list_models(module=torchvision.models)

# print(len(classification_models), "classification models:", classification_models)

# total_params = sum(p.numel() for p in models.vgg16().parameters())
# print(f"Number of parameters: {total_params}")
# print(f"model", models)
# print(VGG16_BN_Weights.IMAGENET1K_V1)
# Create a Deformable Convolution layer
deform_conv = DeformConv2d(3, 64, kernel_size=3, padding=1)

# Input tensor
x = torch.randn(1, 3, 64, 64)  # (batch_size, channels, height, width)

# Forward pass
output = deform_conv(x)

print("Output shape:", output.shape)