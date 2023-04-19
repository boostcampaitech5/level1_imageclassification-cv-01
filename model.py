import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.functional as F
import torchvision
import math
from efficientnet_pytorch import EfficientNet

# 주석 추가
class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

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