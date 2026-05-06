"""
Transformer-based policy network (actor) — Cross-Attention architecture.

Architecture (Section IV-B of the paper):
  1. Neighbour embedding   e_j^(0) = W_e · z_{j,t}  (fixed across all layers)
  2. Agent self-embedding  e_i^(0) = W_s · z_i^self  (z_i^self = [Q_i/Q_max, H_i/H_max])
  3. L-layer Cross-Attention Encoder:
       Query  ← agent embedding e_i^(p-1)  (updated each layer)
       Key/V  ← fixed initial neighbour embeddings e_j^(0)
       Biases: − β·‖Δv_{ij}‖/v_max  (velocity)
               − w_d·d_{ij}/d_max    (distance)
       Validity mask applied (−1e9 for invalid neighbours)
  4. Bilinear action score: s_{ij} = e_i^(L)^T · W_a · e_j^(0) + b_a
  5. Policy: softmax over valid neighbours
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import config as cfg


# ── Cross-Attention sub-layer ────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with mobility + distance bias.

    Agent i is the single query; neighbours j provide keys and values
    from their FIXED initial embeddings e_j^(0).

    Attention score:
        A_{ij} = softmax_j( Q_i K_j^T / sqrt(d_k)
                             - β · mob_j              (velocity bias)
                             - (w_d · d_{ij} + b_d) ) (distance bias)
    mob_j : normalised relative velocity between AGENT i and neighbour j  (B, N)
    d_{ij}: normalised distance from agent to neighbour j                  (B, N)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Learnable scalar β for velocity bias (Eq.28)
        self.beta = nn.Parameter(torch.tensor(cfg.BETA_INIT))
        # Learnable distance weight w_d (Eq.28, no b_d offset)
        self.w_d  = nn.Parameter(torch.tensor(1.0))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        agent_e : torch.Tensor,   # (B, d_model)    agent embedding (query source)
        neigh_e0: torch.Tensor,   # (B, N, d_model)  fixed initial neighbour embeddings
        mob     : torch.Tensor,   # (B, N)           agent-to-neighbour rel-vel (normalised)
        dist    : torch.Tensor,   # (B, N)           agent-to-neighbour distance (normalised)
        mask    : torch.Tensor,   # (B, N)           validity mask, 1 = valid
    ) -> torch.Tensor:             # (B, d_model)    updated agent context
        B, N, _ = neigh_e0.shape
        scale = math.sqrt(self.d_k)

        # Q from agent embedding: (B, d_model) → (B, h, 1, d_k)
        Q = self.W_Q(agent_e).view(B, self.n_heads, self.d_k).unsqueeze(2)
        # K, V from fixed neighbour embeddings: (B, N, d_model) → (B, h, N, d_k)
        K = self.W_K(neigh_e0).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(neigh_e0).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product: (B, h, 1, N)
        logits = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Velocity bias: − softplus(β) · mob_j  →  column-wise (B, 1, 1, N)
        # softplus enforces β > 0 so high-vel neighbors are always penalized,
        # never rewarded (which would happen if β learns negative).
        logits = logits - F.softplus(self.beta) * mob.unsqueeze(1).unsqueeze(2)

        # Distance bias: − softplus(w_d) · d_{ij}/d_max  →  (B, 1, 1, N)  (Eq.28)
        dist_bias = (F.softplus(self.w_d) * dist).unsqueeze(1).unsqueeze(2)
        logits    = logits - dist_bias

        # Validity mask: set invalid neighbours to −∞
        if mask is not None:
            inv_mask = (1.0 - mask).bool().unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            logits   = logits.masked_fill(inv_mask, -1e9)

        attn = F.softmax(logits, dim=-1)             # (B, h, 1, N)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, V)                  # (B, h, 1, d_k)
        ctx = ctx.squeeze(2).contiguous().view(B, self.d_model)  # (B, d_model)
        return self.W_O(ctx)


class CrossAttentionLayer(nn.Module):
    """One Transformer cross-attention layer with residual + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn  = CrossAttention(d_model, n_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, agent_e, neigh_e0, mob, dist, mask):
        # Cross-attention: agent attends to neighbours (K/V fixed from neigh_e0)
        attn_out = self.attn(agent_e, neigh_e0, mob, dist, mask)
        agent_e  = self.norm1(agent_e + self.drop(attn_out))
        # FFN applied to updated agent embedding
        ffn_out  = self.ffn(agent_e)
        agent_e  = self.norm2(agent_e + self.drop(ffn_out))
        return agent_e   # (B, d_model)


# ── Full policy network ──────────────────────────────────────────────────────

class TransformerActor(nn.Module):
    """
    Cross-attention Transformer policy network π_θ(a_{i,t} | o_{i,t}).

    Forward inputs
    --------------
    feats     : (B, N, DIM_IN)   per-neighbour feature vectors z_{j,t}
    mob       : (B, N)           agent-to-neighbour relative velocity (normalised)
    mask      : (B, N)           1 = valid neighbour, 0 = padding / invalid
    dist      : (B, N)           agent-to-neighbour distances (normalised)
    self_feat : (B, DIM_SELF)    agent self-feature [Q_i/Q_max, H_i/H_max]
    """

    def __init__(
        self,
        dim_in       : int   = cfg.DIM_IN,
        dim_self     : int   = cfg.DIM_SELF,
        d_model      : int   = cfg.D_MODEL,
        n_heads      : int   = cfg.N_HEADS,
        d_ff         : int   = cfg.D_FF,
        n_layers     : int   = cfg.N_LAYERS,
        dropout      : float = cfg.DROPOUT,
        max_neighbors: int   = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_nb  = max_neighbors

        # Neighbour feature embedding W_e · z_{j,t}  (fixed across all layers)
        self.embed = nn.Linear(dim_in, d_model)

        # Agent self-embedding W_s · z_i^self
        self.self_embed = nn.Linear(dim_self, d_model)

        # Cross-attention encoder layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Bilinear action score: s_{ij} = e_i^(L)^T · W_a · e_j^(0) + b_a
        self.W_a = nn.Linear(d_model, d_model, bias=False)
        self.b_a = nn.Parameter(torch.zeros(1))

        # Direct score bypass: strong inductive bias toward hop_reduction (feature 6)
        self.direct_score = nn.Linear(dim_in, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.b_a)
        # Init hop_red bias for destination-directed routing
        nn.init.zeros_(self.direct_score.weight)
        self.direct_score.weight.data[0, 6] = 0.3   # direction hint: Proposed builds on this via attention

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        feats    : torch.Tensor,           # (B, N, DIM_IN)
        mob      : torch.Tensor,           # (B, N)
        mask     : torch.Tensor,           # (B, N)  float 0/1
        dist     : torch.Tensor = None,    # (B, N)  agent-to-neighbour distances
        self_feat: torch.Tensor = None,    # (B, DIM_SELF)
    ) -> torch.Tensor:                     # (B, N)  action logits (pre-softmax)
        B, N, _ = feats.shape

        # 1. Fixed neighbour embeddings e_j^(0)
        neigh_e0 = self.embed(feats)       # (B, N, d_model)

        # 2. Agent self-embedding e_i^(0)
        if self_feat is None:
            self_feat = torch.zeros(B, cfg.DIM_SELF, device=feats.device)
        agent_e = self.self_embed(self_feat)   # (B, d_model)

        if dist is None:
            dist = torch.zeros(B, N, device=feats.device)

        # 3. L-layer cross-attention: agent query evolves, neighbour K/V fixed
        for layer in self.layers:
            agent_e = layer(agent_e, neigh_e0, mob, dist, mask)
        # agent_e = e_i^(L): (B, d_model)

        # 4. Bilinear action score: s_{ij} = (W_a · e_i^(L)) · e_j^(0)^T + b_a
        agent_proj = self.W_a(agent_e)     # (B, d_model)
        attn_scores = torch.bmm(
            agent_proj.unsqueeze(1),       # (B, 1, d_model)
            neigh_e0.transpose(-1, -2)     # (B, d_model, N)
        ).squeeze(1) + self.b_a            # (B, N)

        # 4b. Direct score bypass (hop_reduction inductive bias)
        bypass = self.direct_score(feats).squeeze(-1)  # (B, N)
        scores = attn_scores + bypass

        # 4c. Final-layer Eq.27 bias on action logits.
        # Reuses attention-layer β, w_d (averaged across L layers, softplus-positive)
        # so the same physics-grounded biases also penalize unstable / long links
        # at the action-score stage, mirroring the edge-weight formulation in
        # evaluate.py. This (i) gives β, w_d gradient signal at both attention
        # AND final logits, preventing decay during training, and (ii) ensures
        # the policy's routing decisions remain link-quality-aware even when
        # attention layers concentrate on a small subset of neighbours.
        betas_layer = torch.stack([layer.attn.beta for layer in self.layers]).mean()
        w_ds_layer  = torch.stack([layer.attn.w_d  for layer in self.layers]).mean()
        scores = (scores
                  - F.softplus(betas_layer) * mob
                  - F.softplus(w_ds_layer)  * dist)

        # 5. Mask invalid neighbours
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        return scores   # caller applies softmax or samples

    def get_action_and_logprob(
        self,
        feats    : torch.Tensor,
        mob      : torch.Tensor,
        mask     : torch.Tensor,
        dist     : torch.Tensor = None,
        self_feat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action and return (action, log_prob, entropy). action: (B,)"""
        logits = self.forward(feats, mob, mask, dist, self_feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e9)
        all_masked = (logits < -1e8).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist_c = torch.distributions.Categorical(logits=logits, validate_args=False)
        action = dist_c.sample()
        logp   = dist_c.log_prob(action)
        ent    = dist_c.entropy()
        return action, logp, ent

    def evaluate_actions(
        self,
        feats    : torch.Tensor,
        mob      : torch.Tensor,
        mask     : torch.Tensor,
        actions  : torch.Tensor,
        dist     : torch.Tensor = None,
        self_feat: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy) for given actions. Used in PPO update."""
        logits = self.forward(feats, mob, mask, dist, self_feat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-1e9)
        all_masked = (logits < -1e8).all(dim=-1, keepdim=True)
        logits = torch.where(all_masked, torch.zeros_like(logits), logits)
        dist_c = torch.distributions.Categorical(logits=logits, validate_args=False)
        logp = dist_c.log_prob(actions)
        ent  = dist_c.entropy()
        return logp, ent
