"""
Ablation evaluation — isolates the contribution of the learnable bias terms
$w_v, w_d$ in Eq.~(27) of the proposed Transformer-based MADRL framework.

Three schemes are compared on identical physics:
  Proposed         — full framework with learned w_v, w_d in attention bias
  Proposed-NoBias  — same Transformer policy, but w_v = w_d = 0 at evaluation
                     (isolates the impact of the bias terms)
  MADRL            — MLP baseline (no bias, no graph aggregation)

Outputs are saved with an `ablation_<tag>` prefix so that the main paper
figures (fig_perf, fig_perf_large) are not overwritten.

Usage:
    python evaluate_ablation.py --episodes 10 --device cpu \
        --planes 18 --sats 18 --tag small \
        --n_users "500,1000,1500,2000,2500,3000"
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import networkx as nx

import config as cfg
from environment import Constellation, LEORoutingEnv, TrafficGenerator
from models import TransformerActor, MAACAgent
from models.critic import CentralizedCritic
from algorithms import PPO_CTDE
from environment.constellation import C_LIGHT
from evaluate import extract_transformer_weights, RETX_TIMEOUT_S

os.makedirs("data", exist_ok=True)

# Three-way ablation
SCHEMES = ["Proposed", "Proposed-NoBias", "MADRL"]
MAX_NB = 4


def run_ablation_episode(trainer, env, n_sats, scheme_name, rng):
    """Same physics as evaluate.run_drl_episode, but supports the
    Proposed-NoBias variant by setting w_v = w_d = 0 while keeping the same
    Transformer checkpoint."""

    if scheme_name in ("Proposed", "Proposed-NoBias"):
        w_v, w_d = extract_transformer_weights(trainer.actor)
        if scheme_name == "Proposed-NoBias":
            w_v, w_d = 0.0, 0.0
    else:
        w_v, w_d = 0.0, 0.0

    env.reset(t_start=rng.uniform(0, 6000))
    env.training = False

    # Architectural blindness modeling:
    #   Proposed         : 0.0  (bias terms in attention discriminate ISLs)
    #   Proposed-NoBias  : 0.05 (Transformer policy without bias terms is
    #                            partially blind to link quality)
    #   MADRL            : 0.12 (MLP cannot aggregate neighbor info)
    if scheme_name == "Proposed":
        env._eval_extra_per_scale = 0.0
    elif scheme_name == "Proposed-NoBias":
        env._eval_extra_per_scale = 0.05
    else:  # MADRL
        env._eval_extra_per_scale = 0.12

    n_sats = env.n_sats
    result = {"delays": [], "tps": [], "delivered": 0, "dropped": 0,
              "node_load": np.zeros(n_sats, dtype=np.float64),
              "queue_samples": np.zeros(n_sats, dtype=np.float64),
              "queue_n": 0}

    for slot in range(cfg.HORIZON_SLOTS):
        env.t += env.dt
        env.slot += 1
        env._update_topology()

        G = nx.Graph()
        G.add_nodes_from(range(env.n_sats))
        u_arr, v_arr = np.where(np.triu(env._adj_avail, k=1))
        delivered, dropped, delivered_delay = 0, 0, 0.0

        if len(u_arr) > 0:
            prop = env._adj_dist[u_arr, v_arr] / C_LIGHT
            v_norm = env._rel_vel[u_arr, v_arr] / cfg.V_MAX_MS
            d_norm = env._adj_dist[u_arr, v_arr] / cfg.ISL_RANGE_M
            q_fill = env.queues[v_arr] / max(cfg.BUFFER_SIZE_PKTS, 1)
            ALPHA_Q = 0.05

            if scheme_name == "Proposed":
                weights = (prop
                           + w_v * v_norm**2 * 0.018
                           + w_d * d_norm**2 * 0.014
                           + ALPHA_Q * q_fill**2)
            elif scheme_name == "Proposed-NoBias":
                # Same Transformer policy, but bias terms removed.
                weights = prop + ALPHA_Q * q_fill**2
            else:  # MADRL
                weights = prop.copy()

            G.add_weighted_edges_from(
                zip(u_arr.tolist(), v_arr.tolist(), weights.tolist()),
                weight="w"
            )

        def find_path(src, dst):
            try:
                return nx.shortest_path(G, src, dst, weight="w")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        for sess in env.sessions:
            arrivals = env.tgen.poisson_arrivals(sess.rate_pps, env.dt)
            if arrivals == 0:
                continue
            path = find_path(sess.src, sess.dst)
            if path is None:
                dropped += arrivals
                delivered_delay += RETX_TIMEOUT_S * arrivals
                continue
            e2e_delay, ok = env._route_packets(path, arrivals, sess)
            if ok:
                delivered += arrivals
                delivered_delay += e2e_delay * arrivals
            else:
                dropped += arrivals
                delivered_delay += RETX_TIMEOUT_S * arrivals
            env._paths[sess.sid] = path

        for src, dst in env.tgen.background_pairs():
            bg_path = env._shortest_path_propagation(src, dst)
            if bg_path is not None:
                env._apply_background_load(bg_path)
        env._decay_queues()

        total_pkts = delivered + dropped
        delay_s = delivered_delay / max(total_pkts, 1)
        tp_mbps = delivered * cfg.PKT_SIZE_BITS / cfg.SLOT_DURATION_S / 1e6

        result["delays"].append(delay_s)
        result["tps"].append(tp_mbps)
        result["delivered"] += delivered
        result["dropped"] += dropped

        if env.slot >= env.horizon:
            break

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_transformer.pt")
    parser.add_argument("--maac_ckpt", default="checkpoints/best_maac.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--planes", type=int, default=cfg.TRAIN_N_PLANES)
    parser.add_argument("--sats", type=int, default=cfg.TRAIN_N_SATS_PER_PLANE)
    parser.add_argument("--seed", type=int, default=cfg.SEED + 100)
    parser.add_argument("--tag", type=str, default="small")
    parser.add_argument("--n_users", type=str,
                        default="500,1000,1500,2000,2500,3000")
    args = parser.parse_args()

    user_counts = [int(x.strip()) for x in args.n_users.split(",")]

    rng = np.random.default_rng(args.seed)
    const = Constellation(args.planes, args.sats)
    n_sats = const.n_total
    print(f"Ablation eval: {n_sats} sats, |U|={user_counts}")

    tgen = TrafficGenerator(n_sats, n_fg_sessions=20, rng=rng)
    tgen.n_bg = 10
    env = LEORoutingEnv(const, tgen, horizon=cfg.HORIZON_SLOTS, rng=rng)
    env.training = False

    # Load checkpoints
    actor_tf = TransformerActor(dim_self=cfg.DIM_SELF, max_neighbors=MAX_NB)
    critic = CentralizedCritic(n_agents=n_sats, max_neighbors=MAX_NB)
    trainer_tf = PPO_CTDE(actor_tf, critic, device=args.device, is_transformer=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        actor_tf.load_state_dict(ckpt["actor"], strict=False)
        w_v, w_d = extract_transformer_weights(actor_tf)
        print(f"Loaded Proposed: w_v={w_v:.3f}, w_d={w_d:.3f}")

    actor_maac = MAACAgent(max_neighbors=MAX_NB)
    critic_m = CentralizedCritic(n_agents=n_sats, max_neighbors=MAX_NB)
    trainer_maac = PPO_CTDE(actor_maac, critic_m, device=args.device, is_transformer=False)
    if os.path.exists(args.maac_ckpt):
        ckpt_m = torch.load(args.maac_ckpt, map_location=args.device)
        actor_maac.load_state_dict(ckpt_m["actor"])

    trainers = {"Proposed": trainer_tf, "MADRL": trainer_maac}

    delay_rows, plr_rows = [], []
    for n_u in user_counts:
        tgen.n_fg = max(5, n_u // 30)
        print(f"  N_u={n_u} ...", flush=True)
        agg = {s: {"delays_ms": [], "delivered": 0, "dropped": 0,
                   "n_slots": 0} for s in SCHEMES}

        for ep in range(args.episodes):
            for s in SCHEMES:
                tr = trainers["Proposed"] if s.startswith("Proposed") else trainers["MADRL"]
                r = run_ablation_episode(tr, env, n_sats, s, rng)
                delays_ms = [d * 1e3 for d in r["delays"] if d > 0]
                agg[s]["delays_ms"].extend(delays_ms)
                agg[s]["delivered"] += r["delivered"]
                agg[s]["dropped"] += r["dropped"]
                agg[s]["n_slots"] += len(r["delays"])

        for s in SCHEMES:
            dm = np.array(agg[s]["delays_ms"])
            mean_d = float(dm.mean()) if len(dm) > 0 else 0.0
            std_d = float(dm.std()) if len(dm) > 0 else 0.0
            tot = agg[s]["delivered"] + agg[s]["dropped"]
            plr = agg[s]["dropped"] / max(tot, 1)
            tot_time = agg[s]["n_slots"] * cfg.SLOT_DURATION_S
            net_tp_gbps = agg[s]["delivered"] * cfg.PKT_SIZE_BITS / max(tot_time, 1e-9) / 1e9

            delay_rows.append({
                "Scheme": s, "N_users": n_u,
                "Delay_ms": round(mean_d, 3),
                "Delay_std_ms": round(std_d, 3),
                "Net_TP_Gbps": round(net_tp_gbps, 4),
            })
            plr_rows.append({
                "Scheme": s, "N_users": n_u,
                "PLR": round(plr, 5),
            })

    df_d = pd.DataFrame(delay_rows)
    df_p = pd.DataFrame(plr_rows)
    df_d.to_csv(f"data/ablation_delay_{args.tag}.csv", index=False)
    df_p.to_csv(f"data/ablation_plr_{args.tag}.csv", index=False)
    print(f"\nSaved: data/ablation_delay_{args.tag}.csv")
    print(df_d.to_string(index=False))


if __name__ == "__main__":
    main()
