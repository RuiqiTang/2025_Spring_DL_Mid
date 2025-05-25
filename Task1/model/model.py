import torch 
import torch.nn as nn 
from torchvision import models

class ResNet18Model(nn.Module):
    def __init__(self, num_classes=101,pretrained=True):
        super().__init__()
        self.model=models.resnet18(pretrained=pretrained)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        self.num_classes=num_classes
        
    def forward(self,x):
        return self.model(x)