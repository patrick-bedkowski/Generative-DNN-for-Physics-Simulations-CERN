import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AuxReg(nn.Module):
    def __init__(self, strength, output_dim=2, **kwargs):
        super(AuxReg, self).__init__()
        self.name = "regressor_v3_changed_loss_log_cosh"
        self.feature_extractor = FeatureExtractor()
        self.strength = strength
        # Fixed feature dimension from the feature extractor
        feature_dim = 64

        # Simplified regressor with 2 FC layers
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # Ensure input has channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        features = self.feature_extractor(x)
        coords = self.regressor(features)
        return coords

    @staticmethod
    def regressor_loss(real_coords, fake_coords):
        diff = fake_coords - real_coords
        return torch.mean(diff + F.softplus(-2.0 * diff) - math.log(2.0))


def Norm2d(C, groups=32):
    g = min(groups, C)
    # ensure groups divides channels
    while C % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=C)


# Improved feature extractor with residual blocks
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.name = "baseline_resnet_feature_extractor_v3_kernel_size_changed"

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU()
        )

        # First pooling (56x30 -> 28x15)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # First residual block (32->32 channels)
        self.res1 = ResidualBlock(32, 32, kernel_size=5, stride=2)

        # Second pooling (28x15 -> 14x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Second residual block (32->64 channels)
        self.res2 = ResidualBlock(32, 64, kernel_size=5, stride=2)

        # Final pooling (14x7 -> 7x3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.conv1(x)  # [B, 32, 56, 30]
        x = self.pool1(x)  # [B, 32, 28, 15]

        x = self.res1(x)  # [B, 32, 28, 15]
        x = self.pool2(x)  # [B, 32, 14, 7]

        x = self.res2(x)  # [B, 64, 14, 7]
        x = self.pool3(x)  # [B, 64, 7, 3]

        # Global average pooling
        features = x.mean([2, 3])  # [B, 64]
        return features


# Baseline residual block implementation
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=32):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm2d(out_channels, groups=groups),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            Norm2d(out_channels, groups=groups)
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                Norm2d(out_channels, groups=groups)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out
