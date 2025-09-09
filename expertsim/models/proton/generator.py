import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength, **kwargs):
        super().__init__()
        self.name = "Generator-v5-bigkernel-res56x30"
        self.di_strength = di_strength
        self.in_strength = in_strength

        # Fully connected projection from latent+cond
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512 * 18 * 10),  # Starting map: 512x18x10
            nn.LayerNorm(512 * 18 * 10),
            nn.LeakyReLU(0.1, inplace=False)
        )

        self.conv_layers = nn.Sequential(
            # Stage 1: Upsample to 36x20
            nn.Upsample(scale_factor=(2, 2), mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=(4, 4), padding=(1, 1)),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.1, inplace=False),

            # Stage 2: Upsample directly to exactly 56x30
            nn.Upsample(size=(56, 30), mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=(4, 4), padding=(1, 1)),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=False),

            # Refinement at target resolution
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(32, 64),
            nn.LeakyReLU(0.1, inplace=False),

            nn.Conv2d(64, 1, kernel_size=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=False)
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 512, 18, 10)  # Starting spatial grid
        x = self.conv_layers(x)
        return x

