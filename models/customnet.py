import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # 3 input channels (RGB), 224x224 input size (due to Resize in transforms)
        
        # Feature Extraction Layers: Reduce spatial size (H*W) while increasing channels (C)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 -> 56

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 56 -> 28
        
        # Transition Layer: Converts 3D feature map to 1D vector
        self.flatten = nn.Flatten()
        
        # Classification Layer: Takes 256 channels * 28*28 features -> 200 classes
        self.fc1 = nn.Linear(256 * 28 * 28, 200) # 200 is the number of classes

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.pool1(x)

        x = self.conv2(x).relu()
        x = self.pool2(x)

        x = self.conv3(x).relu()
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        
        return x
