import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Improved feature extractor with channel reduction to 64-D and adaptive GAP
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.name = "model_v1"

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)      # [B,32,54,28] for input [B,1,56,30]
        self.conv1_bd = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.2),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))     # [B,32,27,14]

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)     # [B,64,25,12]
        self.conv2_bd = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.2),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))     # [B,64,12,12]

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)    # [B,128,10,10]
        self.conv3_bd = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.2),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1))     # [B,128,5,10]

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)   # [B,256,3,8]
        self.conv4_bd = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(0.2),
        )

        # Channel reduction to 64 so pooled feature is exactly 64-D
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        x = self.pool1(self.conv1_bd(self.conv1(x)))
        x = self.pool2(self.conv2_bd(self.conv2(x)))
        x = self.pool3(self.conv3_bd(self.conv3(x)))
        x = self.conv4_bd(self.conv4(x))
        x = self.reduce(x)
        # Adaptive Global Average Pooling -> [B, 64, 1, 1] -> [B, 64]
        features = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return features  # [B, 64]


class AuxRegNeutron(nn.Module):
    def __init__(self, strength, **kwargs):
        super(AuxRegNeutron, self).__init__()
        self.name = "aux-architecture-1"
        self.strength = strength
        self.feature_extractor = FeatureExtractor()
        self.dense = nn.Linear(64, 2)  # match the 64-D features

    @staticmethod
    def regressor_loss(real_coords, fake_coords):
        # Smooth, robust loss (like logistic/softplus variant)
        diff = fake_coords - real_coords
        return torch.mean(diff + F.softplus(-2.0 * diff) - math.log(2.0))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B,1,H,W]
        features = self.feature_extractor(x)  # [B,64]
        out = self.dense(features)            # [B,2]
        return out
