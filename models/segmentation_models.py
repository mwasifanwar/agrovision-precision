import torch
import torch.nn as nn
import torchvision.models as models

class CropSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CropSegmentationModel, self).__init__()
        
        self.encoder = models.resnet34(pretrained=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)