import torch.nn as nn
import torch.nn.functional as F


# Define the Router Network
class RouterNetwork(nn.Module):
    def __init__(self, cond_dim, n_experts, **kwargs):
        super(RouterNetwork, self).__init__()
        self.name = "router-architecture-2"
        self.n_experts = n_experts
        self.fc_layers = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, self.n_experts)
        )

    def forward(self, cond, tau=1.0, hard=False):
        logits = self.fc_layers(cond)  # [B, E] raw scores
        gates = F.gumbel_softmax(logits, tau=tau, hard=hard)
        # gates now ∈ [0,1], sums to 1 per batch element;
        # if hard=True, a straight-through one-hot approximation
        return gates, logits
