import torch
import torch.nn as nn
import torchvision.models as models

class ResNetPlantDisease(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetPlantDisease, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetSoilAnalysis(nn.Module):
    def __init__(self, num_features=4):
        super(EfficientNetSoilAnalysis, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_features)
    
    def forward(self, x):
        return self.backbone(x)