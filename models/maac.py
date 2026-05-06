"""
MAAC baseline – Multi-Agent Actor-Critic WITHOUT Transformer.

The actor is a simple MLP that takes the concatenated neighbour
features as input (no attention).  This serves as the MAAC [16]
baseline from Section V-A of the paper.
"""

import torch
import torch.nn as nn
from typing import Tuple

import config as cfg


class MAACAgent(nn.Module):
    """
    MLP actor that maps flattened neighbour features → action logits.

    Parameters
    ----------
    max_neighbors : int    maximum expected number of neighbours (padding)
    dim_in        : int    feature dimension per neighbour
    hidden_dim    : int    MLP hidden size
    """

    def __init__(
        self,
        max_neighbors: int  = 8,
        dim_in       : int  = cfg.DIM_IN,
        hidden_dim   : int  = 128,
        n_layers     : int  = 3,
    ):
        super().__init__()
        self.max_nb   = max_neighbors
        self.dim_in   = dim_in

        input_dim = max_neighbors * dim_in
        layers    = []
        in_d      = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU()]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, max_neighbors))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        feats: torch.Tensor,   # (B, N, DIM_IN)
        mask : torch.Tensor,   # (B, N)
    ) -> torch.Tensor:         # (B, max_nb)  action logits
        B, N, _ = feats.shape
        # Pad or truncate to max_nb
        padded = torch.zeros(B, self.max_nb, self.dim_in, device=feats.device)
        n_use  = min(N, self.max_nb)
        padded[:, :n_use, :] = feats[:, :n_use, :]

        flat   = padded.view(B, -1)
        logits = self.net(flat)                # (B, max_nb)

        # Apply validity mask
        if mask is not None:
            padded_mask = torch.zeros(B, self.max_nb, device=feats.device)
            padded_mask[:, :n_use] = mask[:, :n_use]
            logits = logits.masked_fill(padded_mask == 0, -1e9)

        return logits

    def get_action_and_logprob(
        self,
        feats: torch.Tensor,
        mask : torch.Tensor,
        _mob : torch.Tensor = None,  # unused, for API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(feats, mask)
        all_masked = (logits == -1e9).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist   = torch.distributions.Categorical(logits=logits, validate_args=False)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        feats  : torch.Tensor,
        mask   : torch.Tensor,
        actions: torch.Tensor,
        _mob   : torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(feats, mask)
        all_masked = (logits == -1e9).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist   = torch.distributions.Categorical(logits=logits, validate_args=False)
        return dist.log_prob(actions), dist.entropy()
