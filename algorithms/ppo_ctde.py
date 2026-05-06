"""
PPO with Centralized Training / Decentralized Execution (CTDE).

Implements Algorithm 1 of the paper:
  - Per-slot rollout collection from all agents
  - GAE advantage estimation per agent (Eq. 34)
  - Clipped surrogate objective  (Eq. 35)
  - Value function loss          (Eq. 36)
  - Total objective              (Eq. 37)
  - Learning-rate annealing      (Eq. 38)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import config as cfg


# ── Transition buffer ───────────────────────────────────────────────────────────

@dataclass
class Transition:
    feats     : torch.Tensor      # (N, DIM_IN)   per-agent features
    mob       : torch.Tensor      # (N,)          agent-to-neighbour rel velocity
    mask      : torch.Tensor      # (N,)          validity mask
    action    : torch.Tensor      # ()            chosen action index
    logp      : torch.Tensor      # ()            log probability
    reward    : tuple             # (dly, cng, dir, stb) raw cost components
    active    : bool              # True if agent had actual routing activity this slot
    done      : bool
    slot_idx  : int               # index into the per-episode slot_gs_list
    dist      : torch.Tensor = None   # (N,)   agent-to-neighbour distances
    self_feat : torch.Tensor = None   # (DIM_SELF,) agent self-feature [Q_i/Q_max, H_i/H_max]


class RolloutBuffer:
    """Stores transitions for one rollout horizon."""

    def __init__(self):
        self.transitions: List[Transition] = []

    def push(self, t: Transition):
        self.transitions.append(t)

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)


# ── GAE ────────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards     : List[float],
    values      : torch.Tensor,   # (T+1,)
    dones       : List[bool],
    gamma       : float = cfg.GAMMA,
    lam         : float = cfg.GAE_LAMBDA,
) -> torch.Tensor:
    """
    Generalised Advantage Estimation (Eq. 34).
    Returns advantage tensor shape (T,).
    """
    T   = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nxt_v    = values[t + 1].item() if not dones[t] else 0.0
        delta    = rewards[t] + gamma * nxt_v - values[t].item()
        gae      = delta + gamma * lam * (0.0 if dones[t] else gae)
        adv[t]   = gae
    return adv


# ── PPO_CTDE ────────────────────────────────────────────────────────────────────

class PPO_CTDE:
    """
    CTDE-based PPO trainer.

    Supports both the Transformer actor (TransformerActor) and the
    MAAC baseline.  The critic is always the CentralizedCritic.
    """

    def __init__(
        self,
        actor,              # TransformerActor or MAACAgent
        critic,             # CentralizedCritic
        lr          : float = cfg.LEARNING_RATE,
        clip_eps    : float = cfg.PPO_CLIP_EPS,
        gamma       : float = cfg.GAMMA,
        gae_lambda  : float = cfg.GAE_LAMBDA,
        entropy_coef: float = cfg.ENTROPY_COEF,
        vf_coef     : float = cfg.VALUE_LOSS_COEF,
        grad_clip   : float = cfg.GRAD_CLIP,
        ppo_epochs  : int   = cfg.PPO_EPOCHS,
        batch_size  : int   = cfg.MINI_BATCH_SIZE,
        max_iter    : int   = cfg.N_EPISODES * cfg.HORIZON_SLOTS,
        lr_power    : float = cfg.LR_POWER,
        device      : str   = "cpu",
        is_transformer: bool = True,
    ):
        self.actor         = actor.to(device)
        self.critic        = critic.to(device)
        self.clip_eps      = clip_eps
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.entropy_coef  = entropy_coef
        self.vf_coef       = vf_coef
        self.grad_clip     = grad_clip
        self.ppo_epochs    = ppo_epochs
        self.batch_size    = batch_size
        self.max_iter      = max_iter
        self.lr_power      = lr_power
        self.device        = torch.device(device)
        self.is_transformer = is_transformer

        # Separate optimizers for actor and critic
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=lr * 2)

        self.lr0         = lr
        self.iter        = 0

        # Rollout buffers per agent (agent_id → list of transitions)
        self.buffers: Dict[int, RolloutBuffer] = {}

    # ── data collection ─────────────────────────────────────────────────────────

    def init_buffers(self, agent_ids: List[int]):
        self.buffers = {i: RolloutBuffer() for i in agent_ids}

    def store(
        self,
        agent_id    : int,
        feats       : np.ndarray,   # (N, DIM_IN)
        mob         : np.ndarray,   # (N,)
        mask        : np.ndarray,   # (N,)
        action      : int,
        logp        : float,
        reward      : tuple,        # (dly, cng, dir, stb) placeholder; updated after env.step
        active      : bool,         # True if agent had actual routing activity
        done        : bool,
        slot_idx    : int,          # index into per-episode slot_gs_list (no 324x duplication)
        dist        : np.ndarray = None,   # (N,) agent-to-neighbour distances
        self_feat   : np.ndarray = None,   # (DIM_SELF,) agent self-feature
    ):
        t = Transition(
            feats     = torch.tensor(feats, dtype=torch.float32),
            mob       = torch.tensor(mob,   dtype=torch.float32),
            mask      = torch.tensor(mask,  dtype=torch.float32),
            action    = torch.tensor(action, dtype=torch.long),
            logp      = torch.tensor(logp,   dtype=torch.float32),
            reward    = reward,
            active    = active,
            done      = done,
            slot_idx  = slot_idx,
            dist      = torch.tensor(dist,      dtype=torch.float32) if dist      is not None else None,
            self_feat = torch.tensor(self_feat, dtype=torch.float32) if self_feat is not None else None,
        )
        if agent_id not in self.buffers:
            self.buffers[agent_id] = RolloutBuffer()
        self.buffers[agent_id].push(t)

    # ── select action ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        feats: np.ndarray,   # (N, DIM_IN)
        mob  : np.ndarray,   # (N,)  agent-to-neighbour rel velocity
        mask : np.ndarray,   # (N,)
    ) -> Tuple[int, float]:
        """Return (action_index, log_prob)."""
        if feats.shape[0] == 0:
            return 0, 0.0

        f = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        m = torch.tensor(mob,   dtype=torch.float32).unsqueeze(0).to(self.device)
        v = torch.tensor(mask,  dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.is_transformer:
            action, logp, _ = self.actor.get_action_and_logprob(f, m, v)
        else:
            action, logp, _ = self.actor.get_action_and_logprob(f, v)

        return int(action.item()), float(logp.item())

    @torch.no_grad()
    def select_actions_batch(
        self,
        obs_list      : list,
        mob_list      : list,
        max_nb        : int,
        dist_list     : list = None,
        self_feat_list: list = None,
    ) -> Tuple[dict, dict]:
        """Batched action selection for all agents in one forward pass."""
        n = len(obs_list)
        feats_b = np.zeros((n, max_nb, cfg.DIM_IN),  dtype=np.float32)
        mob_b   = np.zeros((n, max_nb),               dtype=np.float32)
        mask_b  = np.zeros((n, max_nb),               dtype=np.float32)
        dist_b  = np.zeros((n, max_nb),               dtype=np.float32)
        sf_b    = np.zeros((n, cfg.DIM_SELF),         dtype=np.float32)
        active  = np.zeros(n, dtype=bool)

        for i, (obs_i, mob_i) in enumerate(zip(obs_list, mob_list)):
            k = obs_i["features"].shape[0]
            if k == 0:
                continue
            k_use = min(k, max_nb)
            feats_b[i, :k_use] = obs_i["features"][:k_use]
            if mob_i.shape[0] > 0:
                mob_b[i, :k_use] = mob_i[:k_use]
            mask_b[i, :k_use] = obs_i["mask"][:k_use]
            if dist_list is not None and dist_list[i] is not None:
                d = dist_list[i]
                dist_b[i, :min(len(d), max_nb)] = d[:k_use]
            if self_feat_list is not None and self_feat_list[i] is not None:
                sf_b[i] = self_feat_list[i]
            active[i] = True

        f  = torch.tensor(feats_b, dtype=torch.float32).to(self.device)
        m  = torch.tensor(mob_b,   dtype=torch.float32).to(self.device)
        v  = torch.tensor(mask_b,  dtype=torch.float32).to(self.device)
        d  = torch.tensor(dist_b,  dtype=torch.float32).to(self.device)
        sf = torch.tensor(sf_b,    dtype=torch.float32).to(self.device)

        if self.is_transformer:
            actions_t, logps_t, _ = self.actor.get_action_and_logprob(f, m, v, d, sf)
        else:
            actions_t, logps_t, _ = self.actor.get_action_and_logprob(f, v)

        actions_np = actions_t.cpu().numpy()
        logps_np   = logps_t.cpu().numpy()

        actions = {i: int(actions_np[i]) if active[i] else 0 for i in range(n)}
        logps   = {i: float(logps_np[i]) if active[i] else 0.0 for i in range(n)}
        return actions, logps

    # ── value estimate ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs = torch.tensor(global_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return float(self.critic(gs).item())

    # ── PPO update ────────────────────────────────────────────────────────────────

    def update(self, slot_gs_list: List[np.ndarray] = None) -> Dict[str, float]:
        """
        MAPPO-style update (Yu et al. 2021):
        - Per-agent GAE with own trajectory
        - Per-agent local critic V(o_i) (no global state)
        - Only ACTIVE transitions (agents actually routing)
        - Per-agent advantage normalization
        - Tighter PPO clip (0.1) for stability
        """
        # ── Step 1: Per-agent reward normalization ─────────────────────────────
        # Collect all raw reward components from active transitions for
        # episode-level min-max normalization (Eq. 20).
        all_active_raw = []
        for buf in self.buffers.values():
            for tr in buf.transitions:
                if tr.active:
                    all_active_raw.append(tr.reward)

        if len(all_active_raw) < 2:
            for buf in self.buffers.values():
                buf.clear()
            return {}

        raw_arr = np.array(all_active_raw, dtype=np.float64)  # (N_active, 4)
        rhos = [cfg.RHO_DELAY, cfg.RHO_CONGEST, cfg.RHO_ROUTING, cfg.RHO_STAB]

        # Robust normalization: use 5/95 percentiles to prevent outlier
        # (e.g., rare drop with huge penalty) from dominating the normalization.
        comp_mins = np.percentile(raw_arr, 5, axis=0)   # (4,)
        comp_maxs = np.percentile(raw_arr, 95, axis=0)  # (4,)
        comp_spans = comp_maxs - comp_mins              # (4,)

        def composite_reward(raw_tuple):
            """Compute normalized composite reward (Eq. 20-21) with outlier clipping."""
            r = 0.0
            for c, rho in enumerate(rhos):
                if rho == 0 or comp_spans[c] < cfg.DELTA_NORM:
                    continue
                norm_c = (raw_tuple[c] - comp_mins[c]) / (comp_spans[c] + cfg.DELTA_NORM)
                # Clip to [0, 1] — outliers beyond 5/95 percentile are clamped
                norm_c = max(0.0, min(1.0, norm_c))
                r -= rho * norm_c
            return r

        # ── Step 2: Per-agent GAE with global critic ──────────────────────────
        if slot_gs_list is not None and len(slot_gs_list) > 0:
            gs_table = torch.from_numpy(
                np.stack(slot_gs_list).astype(np.float32)
            ).to(self.device)
        else:
            gs_table = torch.zeros(1, 1, device=self.device)

        batch_feats, batch_mob, batch_mask, batch_dist = [], [], [], []
        batch_self_feat = []
        batch_actions, batch_logps_old = [], []
        batch_advantages, batch_returns = [], []
        batch_slot_idx = []

        for agent_id, buf in self.buffers.items():
            active_transitions = [tr for tr in buf.transitions if tr.active]
            if len(active_transitions) < 1:
                continue

            agent_rewards = [composite_reward(tr.reward) for tr in active_transitions]
            agent_dones = [tr.done for tr in active_transitions]

            T_agent = len(active_transitions)
            with torch.no_grad():
                vals = []
                for tr in active_transitions:
                    si = min(tr.slot_idx, gs_table.shape[0] - 1)
                    gs = gs_table[si].unsqueeze(0)
                    vals.append(self.critic(gs).item())
                if active_transitions[-1].done:
                    vals.append(0.0)
                else:
                    si = min(active_transitions[-1].slot_idx, gs_table.shape[0] - 1)
                    gs = gs_table[si].unsqueeze(0)
                    vals.append(self.critic(gs).item())

            values_t = torch.tensor(vals, dtype=torch.float32)

            agent_adv = compute_gae(
                agent_rewards, values_t, agent_dones,
                self.gamma, self.gae_lambda
            )
            agent_returns = agent_adv + values_t[:T_agent]

            if len(agent_adv) > 1 and agent_adv.std() > 1e-8:
                agent_adv = (agent_adv - agent_adv.mean()) / (agent_adv.std() + 1e-8)

            for i, tr in enumerate(active_transitions):
                batch_feats.append(tr.feats)
                batch_mob.append(tr.mob)
                batch_mask.append(tr.mask)
                batch_dist.append(tr.dist)
                batch_self_feat.append(
                    tr.self_feat if tr.self_feat is not None
                    else torch.zeros(cfg.DIM_SELF)
                )
                batch_actions.append(tr.action)
                batch_logps_old.append(tr.logp)
                batch_advantages.append(agent_adv[i])
                batch_returns.append(agent_returns[i])
                batch_slot_idx.append(tr.slot_idx)

        T = len(batch_actions)
        if T < 2:
            for buf in self.buffers.values():
                buf.clear()
            return {}

        # Stack tensors
        adv = torch.stack(batch_advantages).to(self.device)
        returns = torch.stack(batch_returns).to(self.device)
        old_logps = torch.stack(batch_logps_old).to(self.device)
        actions_t = torch.stack(batch_actions).to(self.device)
        slot_idx_t = torch.tensor(batch_slot_idx, dtype=torch.long)

        # Global advantage normalization on top of per-agent norm
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── Step 3: PPO mini-batch updates ────────────────────────────────────
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_upd = 0

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                b = idx[start: start + self.batch_size]
                if len(b) < 2:
                    continue

                b_feats = [batch_feats[i] for i in b]
                b_mob   = [batch_mob[i]   for i in b]
                b_mask  = [batch_mask[i]  for i in b]
                b_dist  = [batch_dist[i]  for i in b]
                max_n   = max(f.shape[0] for f in b_feats)

                def pad2d(t, n):
                    if t.shape[0] == n:
                        return t
                    p = torch.zeros(n - t.shape[0], t.shape[1])
                    return torch.cat([t, p], dim=0)

                def pad1d(t, n):
                    if t is None or t.shape[0] == n:
                        return t if t is not None else torch.zeros(n)
                    p = torch.zeros(n - t.shape[0])
                    return torch.cat([t, p], dim=0)

                bf  = torch.stack([pad2d(f, max_n) for f in b_feats]).to(self.device)
                bm  = torch.stack([pad1d(m, max_n) for m in b_mob]).to(self.device)
                bv  = torch.stack([pad1d(v, max_n) for v in b_mask]).to(self.device)
                bd  = torch.stack([pad1d(d, max_n) for d in b_dist]).to(self.device)
                bsf = torch.stack([batch_self_feat[i] for i in b]).to(self.device)
                ba  = actions_t[b]
                bo  = old_logps[b]
                bA  = adv[b]
                bR  = returns[b]

                # Evaluate actions with current policy
                if self.is_transformer:
                    new_logp, ent = self.actor.evaluate_actions(bf, bm, bv, ba, bd, bsf)
                else:
                    new_logp, ent = self.actor.evaluate_actions(bf, bv, ba)

                # Probability ratio (Eq. 35)
                ratio = torch.exp(new_logp - bo)

                # Clipped surrogate loss
                clip1  = ratio * bA
                clip2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * bA
                pol_loss = -torch.min(clip1, clip2).mean()

                # Value function loss (Eq. 36) — mean-field critic V(s)
                b_gs   = gs_table[slot_idx_t[b].clamp(max=gs_table.shape[0]-1).to(self.device)]
                pred_v = self.critic(b_gs).squeeze(-1)
                vf_loss  = ((pred_v - bR) ** 2).mean()

                # Total loss (Eq. 37)
                loss = pol_loss + self.vf_coef * vf_loss - self.entropy_coef * ent.mean()

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  self.grad_clip)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
                self.opt_actor.step()
                self.opt_critic.step()

                stats["policy_loss"] += pol_loss.item()
                stats["value_loss"]  += vf_loss.item()
                stats["entropy"]     += ent.mean().item()
                n_upd += 1

        # Learning-rate annealing (Eq. 38)
        self.iter += 1
        new_lr = self.lr0 * (1 - self.iter / max(self.max_iter, 1)) ** self.lr_power
        for g in self.opt_actor.param_groups:
            g["lr"] = max(new_lr, 1e-6)

        # Clear buffers
        for buf in self.buffers.values():
            buf.clear()

        if n_upd > 0:
            stats = {k: v / n_upd for k, v in stats.items()}
        return stats
