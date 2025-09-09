import torch
import torch.nn as nn


class GeneratorNeutron(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength, **kwargs):
        self.name = "Generator-neutron-1-original-architecture"
        self.di_strength = di_strength
        self.in_strength = in_strength
        super(GeneratorNeutron, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),  # This should be 19 (10 + 9)
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128 * 13 * 13),
            nn.BatchNorm1d(128 * 13 * 13),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=(2, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(256, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, kernel_size=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 13, 13)  # reshape
        x = self.upsample(x)
        x = self.conv_layers(x)
        return x
