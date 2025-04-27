import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Define the Generator model
class GeneratorNeutron(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength):
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

# class Generator(nn.Module):
#     def __init__(self, noise_dim, cond_dim):
#         self.name = "Generator-2-simplified-architecture"
#         super(Generator, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(noise_dim + cond_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 64 * 14 * 8),
#             nn.BatchNorm1d(64 * 14 * 8),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2)
#         )
#         # Upsampling to get the spatial dimension close to target size
#         self.upsample1 = nn.Upsample(scale_factor=(4, 2))
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=(3, 4), padding=1),  # Keep the same kernel size
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2),
#             nn.Conv2d(128, 64, kernel_size=(2, 4), padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2),
#             nn.Upsample(scale_factor=(1, 2)),
#             nn.Conv2d(64, 1, kernel_size=(2, 3), padding=(0, 2)),
#             nn.ReLU()
#         )
#
#     def forward(self, noise, cond):
#         x = torch.cat((noise, cond), dim=1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = x.view(-1, 64, 14, 8)
#         x = self.upsample1(x)
#         x = self.conv_layers(x)
#         return x


# # Define the Discriminator model
# class Discriminator(nn.Module):
#     def __init__(self, cond_dim, num_generators):
#         super(Discriminator, self).__init__()
#         self.num_generators = num_generators
#         self.name = "Discriminator-1-expert"
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(3, 3)),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             nn.Conv2d(32, 16, kernel_size=(3, 3)),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 1))
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 12 * 12 + cond_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc3 = nn.Linear(64, self.num_generators)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, img, cond):
#         x = self.conv_layers(img)
#         x = x.view(x.size(0), -1)
#         x = torch.cat((x, cond), dim=1)
#         x = self.fc1(x)
#         latent = self.fc2(x)
#         out = self.fc3(latent)
#         out = self.sigmoid(out)
#         return out, latent

class GeneratorUnified(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength, n_experts=3):
        self.name = "Generator-MultiOutput"
        self.di_strength = di_strength
        self.in_strength = in_strength
        self.n_experts = n_experts

        super(GeneratorUnified, self).__init__()
        # Keep the exact same architecture as your original
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128 * 20 * 10 * self.n_experts),
            nn.BatchNorm1d(128 * 20 * 10 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=(3, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128 * self.n_experts, 256 * self.n_experts, kernel_size=(2, 2)),
            nn.BatchNorm2d(256 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(256 * self.n_experts, 128 * self.n_experts, kernel_size=(2, 2)),
            nn.BatchNorm2d(128 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128 * self.n_experts, 64 * self.n_experts, kernel_size=(2, 2)),
            nn.BatchNorm2d(64 * self.n_experts),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64 * self.n_experts, 1 * self.n_experts, kernel_size=(2, 7)),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, cond):
        # Same forward pass as original
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128 * self.n_experts, 20, 10)
        x = self.upsample(x)
        x = self.conv_layers(x)  # Output shape: [batch_size, n_experts, H, W]

        # Reshape to separate the experts into a new dimension
        # Assuming the output shape is [batch_size, n_experts, H, W]
        batch_size, n_channels, height, width = x.shape

        # Reshape to [batch_size, n_experts, 1, H, W] to represent separate images
        # This keeps the exact same tensor, just viewed differently
        x = x.view(batch_size, self.n_experts, 1, height, width)

        return x


class DiscriminatorUnified(nn.Module):
    def __init__(self, cond_dim, n_experts):
        super(DiscriminatorUnified, self).__init__()
        self.name = "Discriminator-3-unified"
        self.n_experts = n_experts
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1*self.n_experts, 32*self.n_experts, kernel_size=(3, 3)),
            nn.BatchNorm2d(32*self.n_experts),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32*self.n_experts, 16*self.n_experts, kernel_size=(3, 3)),
            nn.BatchNorm2d(16*self.n_experts),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 12 * 12*self.n_experts + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(64, 1*self.n_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        # print("conv layers", x.shape)  # Debugging line to check the shape after conv layers
        x = x.view(x.size(0), -1)
        # print("conv view", x.shape)  # Debugging line to check the shape after conv layers
        x = torch.cat((x, cond), dim=1)
        # print("cat shape", x.shape)  # Debugging line to check the shape after conv layers
        x = self.fc1(x)
        # print("f1 shape", x.shape)  # Debugging line to check the shape after conv layers
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent


class AuxRegUnified(nn.Module):
    def __init__(self, n_experts):
        super(AuxRegUnified, self).__init__()
        self.name = "aux-architecture-2-unified"
        self.n_experts = n_experts
        # Feature extraction layers
        self.conv3 = nn.Conv2d(1*self.n_experts, 32*self.n_experts, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32*self.n_experts)
        self.leaky3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32*self.n_experts, 64*self.n_experts, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64*self.n_experts)
        self.leaky4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d((2, 1))

        self.conv5 = nn.Conv2d(64*self.n_experts, 128*self.n_experts, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128*self.n_experts)
        self.leaky5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d((2, 1))

        self.conv6 = nn.Conv2d(128*self.n_experts, 256*self.n_experts, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(256*self.n_experts)
        self.leaky6 = nn.LeakyReLU(0.1)

        # Dropout layers (separated from feature path)
        self.dropout = nn.Dropout(0.2)

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear((256 * 3 * 8)*self.n_experts, 2*self.n_experts)  # Update dimensions based on input size

    def forward(self, x):
        # Original forward pass with dropout
        x = self.pool3(self.dropout(self.leaky3(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout(self.leaky4(self.bn4(self.conv4(x)))))
        x = self.pool5(self.dropout(self.leaky5(self.bn5(self.conv5(x)))))
        x = self.dropout(self.leaky6(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        return self.dense(x)

    def get_features(self, img):
        x = self.pool3(self.leaky3(self.bn3(self.conv3(img))))
        x = self.pool4(self.leaky4(self.bn4(self.conv4(x))))
        x = self.pool5(self.leaky5(self.bn5(self.conv5(x))))
        x = self.leaky6(self.bn6(self.conv6(x)))
        features = x.mean([2, 3])  # Global average pooling
        return features  # [112, 256] E.g.


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength):
        self.name = "Generator-1-original-architecture"
        self.di_strength = di_strength
        self.in_strength = in_strength
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),  # This should be 19 (10 + 9)
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128 * 20 * 10),
            nn.BatchNorm1d(128 * 20 * 10),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=(3, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2, 2)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(256, 128, kernel_size=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, kernel_size=(2, 7)),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 20, 10)
        x = self.upsample(x)
        x = self.conv_layers(x)
        return x
# Define the Generator model


# # Define the Discriminator model
# class Discriminator(nn.Module):
#     def __init__(self, cond_dim):
#         super(Discriminator, self).__init__()
#         self.name = "Discriminator-1-expert"
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(3, 3)),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             nn.Conv2d(32, 16, kernel_size=(3, 3)),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 1))
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 12 * 12 + cond_dim, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc3 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, img, cond):
#         x = self.conv_layers(img)
#         x = x.view(x.size(0), -1)
#         x = torch.cat((x, cond), dim=1)
#         x = self.fc1(x)
#         latent = self.fc2(x)
#         out = self.fc3(latent)
#         out = self.sigmoid(out)
#         return out, latent
class Discriminator(nn.Module):
    def __init__(self, cond_dim):
        super(Discriminator, self).__init__()
        self.name = "Discriminator-3-expert-features"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 12 * 12 + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent

    # Disc features 2 architecture
    # def get_features(self, img):
    #     for layer in self.conv_layers:
    #         img = layer(img)
    #         if isinstance(layer, nn.Conv2d):
    #             last_conv_output = img  # e.g. torch.Size([28, 16, 25, 12])
    #     # Global average pooling on the last conv layer output
    #     features = last_conv_output.mean([2, 3])  # e.g. torch.Size([28, 16])
    #     return features #.reshape(features.size(0), -1)

    # Disc features 3 architecture
    def get_features(self, img):
        # Process through all conv layers (including final MaxPool)
        x = self.conv_layers(img)  # Shape: [batch, 16, H, W]
        features = x.mean([2, 3])  # Global average pooling → [batch, 16]
        return features


class DiscriminatorNeutron(nn.Module):
    def __init__(self, cond_dim):
        super(DiscriminatorNeutron, self).__init__()
        self.name = "Discriminator-neutron-1-expert"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(9 * 12 * 12 + cond_dim, 256),  # Increased from 128 to 256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),  # Increased from 64 to 512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(128, 1)  # Adjusted for new fc2 output
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        out = self.sigmoid(out)
        return out, latent


# File: models_pytorch.py (Modify Network Architectures)

class ParallelExpertWrapper(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.num_experts = len(experts)

        # Stack all parameters
        self.param_stacks = {}
        for name, param in experts[0].named_parameters():
            self.param_stacks[name] = nn.Parameter(torch.stack([getattr(e, name) for e in experts]))

        for name, buffer in experts[0].named_buffers():
            self.register_buffer(name + '_stack', torch.stack([getattr(e, name) for e in experts]))

    def forward(self, x, cond, expert_idx=None):
        if expert_idx is None:  # Batch processing
            # x shape: [batch_size, ...]
            # cond shape: [batch_size, cond_dim]
            return torch.vmap(self._forward_single)(x, cond)
        else:  # Single expert
            return self._forward_single(x, cond, expert_idx)

    def _forward_single(self, x, cond, expert_idx):
        params = {name: self.param_stacks[name][expert_idx] for name in self.param_stacks}
        buffers = {name: getattr(self, name + '_stack')[expert_idx] for name in self.param_stacks}
        return functional_call(self.experts[0], (params, buffers), (x, cond))


# Define the Router Network
class RouterNetwork(nn.Module):
    def __init__(self, cond_dim, num_generators):
        super(RouterNetwork, self).__init__()
        self.name = "router-architecture-1"
        self.num_generators = num_generators
        self.fc_layers = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, self.num_generators),
            nn.Softmax(dim=1)
        )

    def forward(self, cond):
        return self.fc_layers(cond)


class AttentionRouterNetwork(nn.Module):
    def __init__(self, cond_dim, num_experts, num_heads=4, hidden_dim=128):
        """
        cond_dim : dimensionality of the input condition vector.
        num_experts: number of generator/discriminator experts.
        num_heads : number of heads for the multi-head attention.
        hidden_dim : latent dimension to which conditions are projected.
        """
        super(AttentionRouterNetwork, self).__init__()
        self.cond_dim = cond_dim
        self.name = "AttentionRouterNetwork"
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        # Optionally project the condition vector into a hidden representation.
        self.query_proj = nn.Linear(cond_dim, hidden_dim)

        # Self-attention module (using batch_first = True so inputs are (B, S, D))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Expert keys: a learnable bank of expert embeddings (one per expert)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, hidden_dim))

        # A learnable temperature parameter to adjust the sharpness of the routing distribution.
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, cond):
        # Input: cond of shape (batch_size, cond_dim)
        # Project the condition to get a query representation
        query = self.query_proj(cond)  # shape: (B, hidden_dim)

        # Unsqueeze to add a sequence length (here, we'll use 1, treating each input as a sequence of length 1)
        query_seq = query.unsqueeze(1)  # shape: (B, 1, hidden_dim)

        # Apply self-attention.
        # Since our sequence length is 1, we simply get back a representation of shape (B, 1, hidden_dim)
        attn_output, _ = self.attention(query_seq, query_seq, query_seq)
        attn_output = attn_output.squeeze(1)  # shape: (B, hidden_dim)

        # Compute dot-product scores with each expert's key.
        # This yields a tensor of shape (B, num_experts)
        logits = torch.matmul(attn_output, self.expert_keys.T)

        # Scale the logits by the temperature to adjust their sharpness.
        logits = logits / self.temperature

        # Apply softmax to obtain routing probabilities
        routing_probs = F.softmax(logits, dim=-1)
        return routing_probs


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     """
#     Fixed bottleneck in the last layers. Added batchnor and dropout
#     """
#     def __init__(self, cond_dim, num_generators):
#         super(RouterNetwork, self).__init__()
#         self.name = "router-architecture-2"
#         self.num_generators = num_generators
#         self.fc_layers = nn.Sequential(
#             nn.Linear(cond_dim, 128),
#             nn.BatchNorm1d(128),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(64, 48),  # Smoother transition
#             nn.BatchNorm1d(48),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(48, 32),
#             nn.BatchNorm1d(32),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Linear(32, self.num_generators),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim):
#         super(RouterNetwork, self).__init__()
#         self.fc_layers = nn.Sequential(
#             # Hidden Layer 1
#             nn.Linear(cond_dim, 128),
#             nn.BatchNorm1d(128),  # Batch normalization
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),  # Dropout for regularization
#
#             # Hidden Layer 2
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),  # Batch normalization
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),  # Dropout for regularization
#
#             # Hidden Layer 3
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),  # Batch normalization
#             nn.LeakyReLU(0.1),
#
#             # Output Layer
#             nn.Linear(32, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim):
#         super(RouterNetwork, self).__init__()
#         self.fc_layers = nn.Sequential(
#             # Hidden Layer 1
#             nn.Linear(cond_dim, 32),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),
#
#             # Output Layer
#             nn.Linear(32, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


class AuxReg(nn.Module):
    def __init__(self):
        super(AuxReg, self).__init__()
        self.name = "aux-architecture-2-with_droppout_batchnorm"

        # Feature extraction layers
        self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.leaky3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)
        self.leaky4 = nn.LeakyReLU(0.1)
        self.pool4 = nn.MaxPool2d((2, 1))

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)
        self.leaky5 = nn.LeakyReLU(0.1)
        self.pool5 = nn.MaxPool2d((2, 1))

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn6 = nn.BatchNorm2d(256)
        self.leaky6 = nn.LeakyReLU(0.1)

        # Dropout layers (separated from feature path)
        self.dropout = nn.Dropout(0.2)

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 3 * 8, 2)  # Update dimensions based on input size

    def forward(self, x):
        # Original forward pass with dropout
        x = self.pool3(self.dropout(self.leaky3(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout(self.leaky4(self.bn4(self.conv4(x)))))
        x = self.pool5(self.dropout(self.leaky5(self.bn5(self.conv5(x)))))
        x = self.dropout(self.leaky6(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        return self.dense(x)

    def get_features(self, img):
        x = self.pool3(self.leaky3(self.bn3(self.conv3(img))))
        x = self.pool4(self.leaky4(self.bn4(self.conv4(x))))
        x = self.pool5(self.leaky5(self.bn5(self.conv5(x))))
        x = self.leaky6(self.bn6(self.conv6(x)))
        features = x.mean([2, 3])  # Global average pooling
        return features  # [112, 256] E.g.

    # def get_features(self, img):
    #     # Store original mode
    #     original_mode = self.training
    #
    #     # Feature extraction without dropout
    #     with torch.no_grad():
    #         self.eval()
    #         x = self.pool3(self.leaky3(self.bn3(self.conv3(img))))
    #         x = self.pool4(self.leaky4(self.bn4(self.conv4(x))))
    #         x = self.pool5(self.leaky5(self.bn5(self.conv5(x))))
    #         x = self.leaky6(self.bn6(self.conv6(x)))
    #         features = x.mean([2, 3])  # Global average pooling
    #
    #     # Restore original mode
    #     self.train(original_mode)
    #     return features  # [112, 256] E.g.


class AuxRegNeutron(nn.Module):
    def __init__(self):
        super(AuxRegNeutron, self).__init__()
        self.name = "aux-architecture-1"

        self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv3_bd = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4_bd = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5_bd = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv6 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6_bd = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 3 * 5, 2)  # Adjust the input dimension based on your input image size

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv3_bd(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv4_bd(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.conv5_bd(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.conv6_bd(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x

