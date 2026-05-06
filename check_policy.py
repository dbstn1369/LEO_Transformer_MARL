"""
Quick policy diagnostic: check if Transformer has learned to use hop_reduction feature.
- Run inference with artificial observations where feature_6 (hop_reduction) varies
- If policy prefers action with higher hop_reduction → learning direction routing
- If policy is uniform regardless → not using direction feature
"""
import numpy as np
import torch
import config as cfg
from models import TransformerActor

MAX_NB = 4

def load_actor(path, device="cpu"):
    actor = TransformerActor(
        dim_in=cfg.DIM_IN, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        d_ff=cfg.D_FF, n_layers=cfg.N_LAYERS, dropout=0.0, max_neighbors=MAX_NB
    )
    ckpt = torch.load(path, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor

@torch.no_grad()
def get_probs(actor, feats):
    """feats: (MAX_NB, DIM_IN) numpy"""
    f = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # (1, 4, 7)
    mob = torch.zeros(1, MAX_NB, MAX_NB)
    mask = torch.ones(1, MAX_NB)
    dist = torch.zeros(1, MAX_NB)
    logits = actor(f, mob, mask, dist)
    return torch.softmax(logits, dim=-1).squeeze(0).numpy()

def make_obs(hop_reduction_values):
    """Make a fake observation where feature 6 (hop_reduction) varies."""
    feats = np.zeros((MAX_NB, cfg.DIM_IN), dtype=np.float32)
    # All features neutral except hop_reduction
    feats[:, 0] = 0.5  # capacity
    feats[:, 1] = 1.0  # link_avail
    feats[:, 2] = 0.3  # prop_delay
    feats[:, 3] = 0.1  # queue_delay
    feats[:, 4] = 0.2  # rel_velocity
    feats[:, 5] = 0.3  # dist_norm
    feats[:, 6] = hop_reduction_values  # hop_reduction
    return feats

def main():
    import os
    if not os.path.exists("checkpoints/best_transformer.pt"):
        print("No checkpoint found.")
        return

    actor = load_actor("checkpoints/best_transformer.pt")
    print("=== Policy Direction-Awareness Check ===")
    print(f"DIM_IN = {cfg.DIM_IN}")

    # Test 1: All neighbors neutral (hop_reduction=0.5)
    feats = make_obs([0.5, 0.5, 0.5, 0.5])
    probs = get_probs(actor, feats)
    print(f"\nAll hop_reduction=0.5 (neutral): {probs}")

    # Test 2: Action 0 is best direction (hop_reduction=0.833 = delta_H=+1)
    feats = make_obs([0.833, 0.167, 0.5, 0.5])
    probs = get_probs(actor, feats)
    print(f"Action 0 best (0.833), action 1 worst (0.167): {probs}")
    print(f"  → Policy prefers action 0: {probs[0]:.3f} (should be highest if learning direction)")

    # Test 3: Action 2 is best direction
    feats = make_obs([0.5, 0.5, 0.833, 0.167])
    probs = get_probs(actor, feats)
    print(f"Action 2 best (0.833), action 3 worst (0.167): {probs}")
    print(f"  → Policy prefers action 2: {probs[2]:.3f} (should be highest)")

    # Test 4: All pointing toward destination (all=0.833)
    feats = make_obs([0.833, 0.833, 0.833, 0.833])
    probs = get_probs(actor, feats)
    print(f"\nAll hop_reduction=0.833 (all toward dst): {probs}")

    # Test 5: Only quality difference (hop_reduction=0.5 but queue varies)
    feats_qual = np.zeros((MAX_NB, cfg.DIM_IN), dtype=np.float32)
    feats_qual[:, 0] = [0.9, 0.5, 0.5, 0.5]  # action 0 has high capacity
    feats_qual[:, 1] = 1.0
    feats_qual[:, 2] = 0.3
    feats_qual[:, 3] = [0.0, 0.5, 0.5, 0.5]  # action 0 has low queue
    feats_qual[:, 4] = 0.2
    feats_qual[:, 5] = 0.3
    feats_qual[:, 6] = 0.5  # neutral direction
    probs_qual = get_probs(actor, feats_qual)
    print(f"\nAction 0 high capacity+low queue, direction neutral: {probs_qual}")
    print(f"  → Policy prefers action 0 for quality: {probs_qual[0]:.3f}")

    print("\n=== Summary ===")
    feats_dir = make_obs([0.833, 0.167, 0.5, 0.5])
    probs_dir = get_probs(actor, feats_dir)
    direction_sensitivity = probs_dir[0] - probs_dir[1]
    print(f"Direction sensitivity (action0 vs action1): {direction_sensitivity:.4f}")
    print(f"  0.0 = no direction learning; 0.5+ = strong direction preference")

    if direction_sensitivity > 0.1:
        print("  ✓ Policy IS learning direction routing!")
    elif direction_sensitivity > 0.05:
        print("  ~ Policy is weakly learning direction routing")
    else:
        print("  ✗ Policy NOT learning direction routing (need more training)")

if __name__ == "__main__":
    main()
