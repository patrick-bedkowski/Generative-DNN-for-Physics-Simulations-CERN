import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import torch.nn.utils as utils  # spectral_norm


class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_experts, mode='generator'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.Tensor(n_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(n_experts, out_features))
        self.reset_parameters()
        self.mode = mode

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.mode != 'generator':
            # Create fresh parameter copies for each forward pass to avoid version conflicts
            weight_copy = self.weight.clone()
            bias_copy = self.bias.clone()

            result = torch.einsum('bgi,goi->bgo', x, weight_copy)
            bias_expanded = bias_copy.unsqueeze(0).expand(result.shape[0], -1, -1).clone()
            return result + bias_expanded
        else:
            B_total, in_dim = x.size()
            x = x.view(self.n_experts, -1, in_dim)  # [n_experts, B, in_features]

            # Also create copies for generator mode for consistency
            weight_copy = self.weight.clone()
            bias_copy = self.bias.clone()

            out = torch.einsum("ebi,eoi->ebo", x, weight_copy) + bias_copy.unsqueeze(1)
            return out.view(B_total, self.out_features)


class DiscriminatorUnified(nn.Module):
    def __init__(self, cond_dim, n_experts=3, **kwargs):
        super().__init__()
        self.n_experts = n_experts

        self.conv1 = nn.Conv2d(n_experts, 32 * n_experts, kernel_size=3, groups=n_experts)
        self.gn1 = nn.GroupNorm(num_groups=n_experts, num_channels=32 * n_experts)
        self.conv2 = nn.Conv2d(32 * n_experts, 16 * n_experts, kernel_size=3, groups=n_experts)
        self.gn2 = nn.GroupNorm(num_groups=n_experts, num_channels=16 * n_experts)

        self.dropout = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d((2, 1))

        self.cond_fc = GroupedLinear(cond_dim, 64, n_experts, mode='discriminator')
        self.fc1 = GroupedLinear(16 * 12 * 12 + 64, 128, n_experts, mode='discriminator')
        self.fc2 = GroupedLinear(128, 64, n_experts, mode='discriminator')
        self.fc3 = GroupedLinear(64, 1, n_experts, mode='discriminator')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond, gumbel_weights=None):
        """
        x: [B, n_experts, 1, H, W] (each expert gets its own image)
        cond: [B, cond_dim]
        gumbel_weights: [B, n_experts] (only for routing — not used inside forward)
        """

        B, E, C, H, W = x.shape
        assert E == self.n_experts, "Mismatch in expert count"

        # Merge expert dim into channel dim → [B, E*C, H, W]
        x = x.view(B, E * C, H, W)

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.pool2(x)

        # Reshape to [B, E, -1]
        x = x.view(B, self.n_experts, -1)  # [B, E, 16*12*12]

        # Process condition for each expert
        cond_exp = cond.unsqueeze(1).expand(-1, self.n_experts, -1)  # [B, E, cond_dim]
        cond_embed = self.cond_fc(cond_exp)  # [B, E, 64]

        # Concatenate features + cond
        x = torch.cat([x, cond_embed], dim=2)  # [B, E, features + cond]

        x = self.fc1(x)  # [B, E, 128]
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)

        latent = self.fc2(x)  # [B, E, 64]
        x = F.leaky_relu(latent, 0.1)
        x = self.dropout(x)

        out = self.fc3(x)  # [B, E, 1]
        out = self.sigmoid(out)

        return out, latent


class Discriminator(nn.Module):
    def __init__(self, cond_dim, **kwargs):
        super(Discriminator, self).__init__()
        self.name = "Discriminator-5-hinge-spectralnorm"

        self.conv_layers = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(1, 32, kernel_size=(3, 3))),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(kernel_size=(2, 2)),

            utils.spectral_norm(nn.Conv2d(32, 16, kernel_size=(3, 3))),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1, inplace=False),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.fc1 = nn.Sequential(
            utils.spectral_norm(nn.Linear(16 * 12 * 12 + cond_dim, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1, inplace=False),
        )

        self.fc2 = nn.Sequential(
            utils.spectral_norm(nn.Linear(128, 64)),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1, inplace=False),
        )

        # final linear, SN applied, no sigmoid
        self.fc3 = utils.spectral_norm(nn.Linear(64, 1))

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)   # raw scores for hinge loss
        return out, latent
