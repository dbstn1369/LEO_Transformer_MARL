"""
GRLR baseline – Graph-RL Routing (Zhang et al.)

A distributed Actor-Critic scheme that uses a Graph Attention Network (GAT)
to aggregate neighbor features and incorporates ISL outage probability into
the state to account for link instability.

Key differences from the Proposed Transformer model:
  - Standard GAT self-attention over neighbors (no cross-attention with agent
    self-embedding, no velocity/distance bias in attention).
  - ISL outage probability P_out is computed from distance and velocity features
    and appended as an extra input feature.
  - Simpler MLP policy head on top of GAT output.

API matches MAACAgent: forward(feats, mask) → logits,
get_action_and_logprob(feats, mask), evaluate_actions(feats, mask, actions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config as cfg


class GATLayer(nn.Module):
    """Single-layer multi-head Graph Attention over neighbor features.

    Each neighbor attends to every other valid neighbor via standard
    dot-product attention (Velickovic et al., 2018 style, but using
    scaled dot-product for stability).

    Input:  (B, N, D)   neighbor features
    Output: (B, N, D)   attended features (residual + LayerNorm)
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, D)
        mask: (B, N)  1=valid, 0=padding
        """
        B, N, D = x.shape
        H, dk = self.n_heads, self.d_k

        q = self.W_q(x).view(B, N, H, dk).transpose(1, 2)  # (B, H, N, dk)
        k = self.W_k(x).view(B, N, H, dk).transpose(1, 2)
        v = self.W_v(x).view(B, N, H, dk).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)  # (B, H, N, N)

        # Mask: invalid neighbors should not attend or be attended to
        if mask is not None:
            # (B, 1, 1, N) – mask out keys
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(key_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, dk)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        # Residual + LayerNorm
        out = self.norm(x + self.dropout(out))
        return out


class GRLRAgent(nn.Module):
    """
    GRLR (Graph-RL Routing) actor network.

    Architecture:
      1. Compute ISL outage probability from distance/velocity features
         and append as extra feature → dim_in + 1
      2. Linear projection to hidden_dim
      3. Single GAT layer (multi-head self-attention over neighbors)
      4. MLP policy head → per-neighbor action logits

    Parameters
    ----------
    max_neighbors : int    maximum number of neighbors (padding size)
    dim_in        : int    input feature dim per neighbor (default: cfg.DIM_IN = 7)
    hidden_dim    : int    GAT / MLP hidden dimension
    n_heads       : int    number of GAT attention heads
    dropout       : float  dropout rate
    """

    def __init__(
        self,
        max_neighbors: int = 8,
        dim_in: int = cfg.DIM_IN,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_nb = max_neighbors
        self.dim_in = dim_in

        # +1 for outage probability feature
        self.input_proj = nn.Linear(dim_in + 1, hidden_dim)

        # Single GAT layer
        self.gat = GATLayer(hidden_dim, n_heads=n_heads, dropout=dropout)

        # Feedforward block after GAT (like a Transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        # Policy head: per-neighbor score
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _compute_outage_prob(feats: torch.Tensor) -> torch.Tensor:
        """Compute ISL outage probability from existing features.

        P_out = PER_DIST_MAX * (d/d_max)^2 + PER_VEL_MAX * (v/v_max)^2

        Feature 0 = capacity_norm ∈ [0,1] (inverse distance proxy: higher = closer)
            → distance_norm ≈ 1 - capacity_norm
        Feature 4 = rel_velocity_norm ∈ [0,1]

        Returns: (B, N, 1)
        """
        cap_norm = feats[..., 0:1]   # (B, N, 1)
        vel_norm = feats[..., 4:5]   # (B, N, 1)

        dist_norm = 1.0 - cap_norm   # higher distance → higher outage
        p_out = (cfg.PER_DIST_MAX * dist_norm ** 2
                 + cfg.PER_VEL_MAX * vel_norm ** 2)
        return p_out

    def forward(
        self,
        feats: torch.Tensor,  # (B, N, DIM_IN)
        mask: torch.Tensor,   # (B, N)
    ) -> torch.Tensor:        # (B, max_nb) action logits
        B, N, _ = feats.shape

        # Pad or truncate to max_nb
        padded = torch.zeros(B, self.max_nb, self.dim_in, device=feats.device)
        n_use = min(N, self.max_nb)
        padded[:, :n_use, :] = feats[:, :n_use, :]

        padded_mask = torch.zeros(B, self.max_nb, device=feats.device)
        if mask is not None:
            padded_mask[:, :n_use] = mask[:, :n_use]

        # Compute and append outage probability
        p_out = self._compute_outage_prob(padded)  # (B, max_nb, 1)
        x = torch.cat([padded, p_out], dim=-1)     # (B, max_nb, dim_in+1)

        # Project to hidden dim
        x = F.relu(self.input_proj(x))  # (B, max_nb, hidden_dim)

        # Zero out padded positions before GAT
        x = x * padded_mask.unsqueeze(-1)

        # GAT layer
        x = self.gat(x, padded_mask)  # (B, max_nb, hidden_dim)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        # Per-neighbor action logits
        logits = self.policy_head(x).squeeze(-1)  # (B, max_nb)

        # Mask invalid neighbors
        logits = logits.masked_fill(padded_mask == 0, -1e9)

        return logits

    def get_action_and_logprob(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor,
        _mob: torch.Tensor = None,  # unused, API compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(feats, mask)
        all_masked = (logits == -1e9).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor,
        actions: torch.Tensor,
        _mob: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(feats, mask)
        all_masked = (logits == -1e9).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        return dist.log_prob(actions), dist.entropy()
