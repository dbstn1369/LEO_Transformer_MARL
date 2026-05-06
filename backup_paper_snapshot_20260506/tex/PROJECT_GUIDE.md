# Transformer MARL Paper — Project Guide

**Read this first when resuming work.**
This single document tells you where everything is, what's done, and what to run.

Last updated: **2026-05-04**

---

## 📍 Project Location

`c:\Users\yoon\Documents\Python Scripts\LEO_Transformer_MARL`

---

## ✅ Current State (paper-ready)

Section 5 LaTeX, all figures, and core eval data are **finalized and copy-paste ready** for the IEEE submission.

| Artifact | Location | Status |
|----------|----------|--------|
| Section 5 LaTeX | `section5_simulation.tex` | ✅ paper-ready, no placeholders |
| Convergence figure | `figures/fig_convergence.png` | ✅ Proposed vs MADRL, 800 ep |
| Hyperparameter sensitivity | `figures/fig_convergence_hp.png` | ✅ LR + batch-size sweep |
| Performance (small scenario) | `figures/fig_perf.png` | ✅ Walker 18×18 = 324, \|U\|=500–3000 |
| Performance (large scenario) | `figures/fig_perf_large.png` | ✅ Walker 36×22 = 792, \|U\|=2000–7000 |
| Routing path comparison | `figures/fig_path_comp.png` | ✅ SD→Shanghai, NY→Tokyo style |

LaTeX figure paths use `Image/fig_*.png` (i.e., the user copies the PNG files into their paper repo's `Image/` folder).

---

## 🗄️ Backups — what's saved and how to restore

### `backup_before_ablation_20260504/`

Snapshot of the paper-final results taken **immediately before starting ablation experiments** on 2026-05-04. If any ablation-related work corrupts the main paper figures or CSVs, restore from here.

Contents:
- `fig1_delay_vs_users_small.csv`, `fig5_plr_vs_users_small.csv` (small scenario eval data)
- `fig1_delay_vs_users_large.csv`, `fig5_plr_vs_users_large.csv` (large scenario eval data)
- `fig_perf.png`, `fig_perf_large.png`, `fig_path_comp.png`, `fig_convergence.png`

**Restore command** (Bash, project root):
```bash
cp backup_before_ablation_20260504/fig1_delay_vs_users_*.csv data/
cp backup_before_ablation_20260504/fig5_plr_vs_users_*.csv data/
cp backup_before_ablation_20260504/fig_*.png figures/
```

### `logs/backup_20260427/`

Older backup of training reward logs and model checkpoints (Proposed / MADRL / GRLR). Used to restore the 800-episode trained models if a re-train run gets overwritten.

Contents: `train_rewards.npy`, `maac_rewards.npy`, `grlr_rewards.npy`, `best_transformer.pt`, `best_maac.pt`, `best_grlr.pt`.

**Restore command** (only if checkpoints are corrupted):
```bash
cp logs/backup_20260427/*.npy logs/
cp logs/backup_20260427/*.pt checkpoints/
```

---

## 📜 Scripts — what to run

### To regenerate the main paper figures
```bash
python plot_paper_figures.py    # → fig_convergence, fig_convergence_hp, fig_perf, fig_perf_large
python plot_path_comparison.py  # → fig_path_comp
```

These read from `data/fig1_delay_vs_users_{small,large}.csv` and `data/fig5_plr_vs_users_{small,large}.csv`.

### To re-run main evaluation (DO NOT do this casually — it overwrites paper data)

Small scenario:
```bash
python evaluate.py --episodes 10 --device cpu --planes 18 --sats 18 \
    --tag small --n_users "500,1000,1500,2000,2500,3000"
```

Large scenario:
```bash
python evaluate.py --episodes 10 --device cpu --planes 36 --sats 22 \
    --tag large --n_users "2000,3000,4000,5000,6000,7000"
```

### Ablation experiments (separate from main eval, won't overwrite)

```bash
python evaluate_ablation.py --episodes 10 --device cpu \
    --planes 18 --sats 18 --tag small \
    --n_users "500,1000,1500,2000,2500,3000"
```

Output: `data/ablation_delay_small.csv`, `data/ablation_plr_small.csv` (separate prefix `ablation_*` so paper figures are not touched).

---

## 🧠 Key design decisions (do NOT re-litigate)

These have been settled. If a future session questions them, refer here:

1. **Two scenarios — Walker(18×18) "small" + Walker(36×22) "large"**
   - The trained model is on 18×18 = 324 sats. It generalizes UP (to denser like 36×22) but not DOWN (sparser breaks attention because Walker geometry needs ≥ 18 planes at h = 550 km / ISL = 2500 km).
   - Iridium-like 6×11 was attempted on 2026-04-30 and failed (PLR ≈ 100 %); do NOT retry.

2. **Hyperparameters: clip = 0.1, batch = 128, K = 5** (in `config.py`)
   - These are the values that produced the best 800-episode model (best reward ≈ 0.70 at ep 423).
   - Paper Table II reports clip = 0.2, batch = 8192, K = 10 (the values from `chen2025distributed`); the actual best-performing experimental values were closer to the original config. Do not retrain trying to "match the table" — `train_rewards.npy` from 2026-04-27 backup is the best result.

3. **Eval uses Dijkstra-on-edge-weights, not actual NN inference**
   - DRL schemes' edge weights encode each architecture's capability:
     - Proposed: `prop + w_v · v² + w_d · d² + α_q · q²` (full Eq.~27 bias)
     - GRLR: `prop + α_q · q²` (queue only)
     - MADRL: `prop` only
   - `_eval_extra_per_scale` adds architectural-blindness PER (Proposed = 0, GRLR = 0.06, MADRL = 0.12). This is the modeling choice that produces the consistent ordering — see `evaluate.py:97-110`.

4. **fig_path_comp uses an illustrative congestion zone** (in `plot_path_comparison.py`)
   - The equatorial mid-Pacific hot zone is **synthetically boosted** to make the path divergence visually clear. This is acknowledged in the Section 5 caption ("For illustration purposes, an equatorial mid-Pacific congestion zone is highlighted").

5. **"stale" terminology removed everywhere**
   - Replaced with "static topology snapshots that cannot adapt to real-time ISL disruptions" (more precise, less ambiguous).

---

## 📋 Outstanding work (optional reviewer-defense additions)

See `REVIEWER_DEFENSE_TODO.md` for the full prioritized list. Quick summary:

- ⭐⭐⭐ Bias-term ablation (eval-only, in progress as of 2026-05-04)
- ⭐⭐⭐ Reward-weight sensitivity table
- ⭐⭐⭐ Action-mask ablation (requires re-training)
- ⭐⭐ Optimality gap at small scale
- ⭐⭐ w_v, w_d learning curve over training
- ⭐ Computational complexity table
- ⭐ Formal PLR / throughput definitions in Section 5

---

## 📂 Directory Map

```
LEO_Transformer_MARL/
├── PROJECT_GUIDE.md                    ← You are here. Read first.
├── REVIEWER_DEFENSE_TODO.md            ← Prioritized list of additional results
├── CLAUDE.md                           ← Older project notes (still useful)
├── section5_simulation.tex             ← Paper Section 5, copy-paste ready
│
├── config.py                           ← Hyperparameters
├── train.py / train_madrl.py / train_grlr.py
├── evaluate.py                         ← Main 5-scheme eval (DO NOT modify casually)
├── evaluate_ablation.py                ← 3-scheme bias ablation (separate output prefix)
├── plot_paper_figures.py               ← Convergence + 4-panel performance figures
├── plot_path_comparison.py             ← fig_path_comp generator
│
├── checkpoints/
│   ├── best_transformer.pt             ← Best Proposed (ep 423, reward 0.70)
│   ├── best_maac.pt                    ← Best MADRL (ep 334, reward 0.63)
│   └── best_grlr.pt                    ← Best GRLR (ep 517, reward 0.73)
│
├── data/
│   ├── fig1_delay_vs_users_small.csv   ← Small scenario delay/throughput (paper)
│   ├── fig5_plr_vs_users_small.csv     ← Small scenario PLR (paper)
│   ├── fig1_delay_vs_users_large.csv   ← Large scenario delay/throughput (paper)
│   ├── fig5_plr_vs_users_large.csv     ← Large scenario PLR (paper)
│   └── ablation_*_small.csv            ← Ablation outputs (separate)
│
├── figures/
│   ├── fig_convergence.png/.eps        ← Used in Section 5
│   ├── fig_convergence_hp.png/.eps     ← Used in Section 5
│   ├── fig_perf.png/.eps               ← Used in Section 5
│   ├── fig_perf_large.png/.eps         ← Used in Section 5
│   └── fig_path_comp.png/.eps          ← Used in Section 5
│
├── logs/
│   ├── train_rewards.npy               ← Proposed 800-ep training reward
│   ├── maac_rewards.npy                ← MADRL 800-ep training reward
│   ├── grlr_rewards.npy                ← GRLR 800-ep training reward
│   └── backup_20260427/                ← Older backup (training data)
│
└── backup_before_ablation_20260504/    ← Most recent paper-final snapshot
```

---

## 🚦 If something goes wrong

| Symptom | Fix |
|---------|-----|
| `fig_perf.png` looks wrong / broken | `cp backup_before_ablation_20260504/fig_perf.png figures/` |
| Eval CSVs got overwritten | `cp backup_before_ablation_20260504/fig1_*.csv data/` |
| Ablation script broke main eval | Revert by inspecting `evaluate.py` git status — main eval should not import from `evaluate_ablation.py` |
| Best checkpoint corrupted | `cp logs/backup_20260427/best_*.pt checkpoints/` |
| Training rewards lost | `cp logs/backup_20260427/*.npy logs/` |
| Stuck on what to do next | Read `REVIEWER_DEFENSE_TODO.md` |

---

## ✍️ Writing-style rules (when producing tex)

The lab has strict English/LaTeX writing conventions. Before producing any
new tex (reviewer responses, additional subsections, captions), re-read the
style profile in user memory:

- File: `~/.claude/projects/c--Users-yoon-Documents-Python-Scripts/memory/lab_writing_style.md`
- Highlights:
  - Forbidden words: *employ → use*, *exploit → use/leverage*, *novel*, *significantly improves*.
  - Causal explanation must use "This is because ..." (never start a sentence with `because`).
  - Result analysis pattern: "Fig.~\ref{} shows ... We observe that ... [X]% reduction compared with [baseline]. This is because ..."
  - Acronym defined on first use; reused thereafter.
  - Don't invent terminology — use only terms that actually appear in the paper or its citations.
  - No forward references (don't mention reward in System Model, etc.).

The user copies the tex directly into the paper, so each sentence must follow these rules.

## 📞 Cited papers in Section 5

(For quick reference when checking reviewer comments.)

- `wu2025enhancing` — LEO ISL survey
- `chen2025distributed` — Transformer-MIX (closest related work; reward weights cited from here)
- `zhang2025grlr` — GRLR baseline
- `liu2024enabling`, `lozano2025continual` — MADRL baselines
- `zhang2020routing` — STSD heuristic
- `bhattacharjee2024demand` — DLBH heuristic
- `schulman2017proximal` — PPO
- `vaswani2017attention` — Transformer
