import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.functional as F
import torchvision
import math
from efficientnet_pytorch import EfficientNet

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels=3, num_classes=18):
        super(EfficientNet_MultiLabel, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.network = EfficientNet.from_pretrained('efficientnet-b4', in_channels=self.in_channels, num_classes=self.num_classes)

    def forward(self, x):
        
        x = self.network(x)

        return x
    
class convnext(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("convnext_base_384_in22ft1k", pretrained=True)
        self.num_classes = num_classes
        self.linear = nn.Linear(1000, self.num_classes)
    def forward(self,x):
        x = self.model(x)
        x = self.linear(x)
        return x