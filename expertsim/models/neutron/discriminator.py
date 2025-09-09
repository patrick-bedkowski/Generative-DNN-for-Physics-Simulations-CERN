import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class DiscriminatorNeutron(nn.Module):
    def __init__(self, cond_dim, **kwargs):
        super(DiscriminatorNeutron, self).__init__()
        self.name = "Discriminator-neutron-1-expert-hinge-SN"

        self.conv_layers = nn.Sequential(
            SN(nn.Conv2d(1, 32, kernel_size=3, padding=0)),   # 56x30 -> 54x28
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2)),                 # 54x28 -> 27x14

            SN(nn.Conv2d(32, 16, kernel_size=3, padding=0)),  # 27x14 -> 25x12
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2))                  # 25x12 -> 12x12
        )

        # Flatten will be 16 * 12 * 12 for 56x30 inputs
        flat_dim = 9 * 12 * 12

        self.fc1 = nn.Sequential(
            SN(nn.Linear(flat_dim + cond_dim, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=False),
        )

        self.fc2 = nn.Sequential(
            SN(nn.Linear(128, 64)),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=False),
        )

        # Final linear with spectral norm; no sigmoid for hinge loss
        self.fc3 = SN(nn.Linear(64, 1))

    def forward(self, img, cond):
        x = self.conv_layers(img)                 # [B,16,12,12] for 56x30 inputs
        x = x.view(x.size(0), -1)                 # [B, 16*12*12]
        x = torch.cat((x, cond), dim=1)           # concat conditioning
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)                    # raw logits for hinge loss
        return out, latent
