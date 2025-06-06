"""
UNET model for Mitotic ROI detection task on WSI

    Inspirations: https://github.com/aladdinpersson/Machine-Learning-Collection

    Original Paper: https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms.functional import crop
import torchvision.transforms.functional as TF
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
         super(DoubleConv, self).__init__()
         self.conv = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True),

             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True),
           #  nn.Dropout2d(p=0.2),

         )

    def forward(self, x):
        return self.conv(x)


class DoubleConvWithGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvWithGroupNorm, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels=3,  out_channels=1, features=[64, 128, 256, 512], use_group_norm=False):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ConvBlock = DoubleConvWithGroupNorm if use_group_norm else DoubleConv


        # 161 x  161, 80x80
        # Down part
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature
        # up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(ConvBlock(feature*2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Sigmoid for binary segmentation
        return self.sigmoid(self.final_conv(x))

def get_model_unet(in_channels=3, out_channels=1, features=[64, 128, 256, 512], use_group_norm=False, **kwargs):
    """Create and return a UNet model instance.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of feature dimensions for each level
        use_group_norm (bool): Whether to use Group Normalization instead of Batch Normalization
        **kwargs: Additional arguments to pass to the UNet constructor
    
    Returns:
        UNet: Configured UNet model instance
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        use_group_norm=use_group_norm,
        **kwargs
    )
    return model

def test():
    x = torch.randn((3, 1, 256, 256))
    model = UNet(in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256])
    print(model)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    print("test model with shape")
    test()
