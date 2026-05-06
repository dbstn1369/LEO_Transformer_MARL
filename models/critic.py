"""
Centralized value network Vω(s_t) — Mean-field variant.

During CTDE training the critic receives the *global* state and outputs
a scalar value estimate.  Uses mean-field pooling: average the neighbor
features across all agents then feed the compact summary vector to the MLP.

Input: flattened global state (n_agents × max_neighbors × DIM_IN)
  → reshape to (n_agents, max_neighbors, DIM_IN)
  → mean + max + std over all agent-neighbor pairs → 3*DIM_IN
  → MLP → scalar V(s)
"""

import torch
import torch.nn as nn

import config as cfg


class CentralizedCritic(nn.Module):
    """
    Mean-field centralized value function.
    Fixed 3×DIM_IN summary (mean + max + std) — well-conditioned for small MLP.
    """

    def __init__(
        self,
        n_agents    : int,
        max_neighbors: int = 4,
        dim_in      : int  = cfg.DIM_IN,
        hidden_dim  : int  = 128,
        n_layers    : int  = 3,
    ):
        super().__init__()
        self.n_agents     = n_agents
        self.max_nb       = max_neighbors
        self.dim_in       = dim_in

        input_dim = 3 * dim_in

        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        global_state : (B, n_agents × max_neighbors × dim_in) flattened

        Returns
        -------
        value : (B, 1)
        """
        B = global_state.shape[0]
        gs = global_state.view(B, self.n_agents, self.max_nb, self.dim_in)
        flat = gs.view(B, self.n_agents * self.max_nb, self.dim_in)
        mean_feat = flat.mean(dim=1)
        max_feat  = flat.max(dim=1).values
        std_feat  = flat.std(dim=1)
        pooled    = torch.cat([mean_feat, max_feat, std_feat], dim=-1)
        return self.net(pooled)
