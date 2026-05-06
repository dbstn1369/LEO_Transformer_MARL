# Reviewer Defense Plan — Transformer MARL Paper

Last updated: 2026-05-04

This document tracks reviewer-comment risks identified by analyzing the WCL2026-0742 revision letter, and the corresponding additional results we should add to defend the Transformer paper proactively.

---

## 🔍 WCL Reviewer Pattern → Risk Map

The WCL paper received reviewer comments that fall into **predictable categories**. The Transformer paper has the same vulnerabilities — listed below in priority order.

### ⭐⭐⭐ HIGH — almost certain reviewer comments

| # | Risk | WCL precedent | Required addition |
|---|------|---------------|-------------------|
| 1 | **Hyperparameter sensitivity** — ω_d, ω_c, ω_s, ω_dir, β_c, β_s set to fixed values without justification | Editor C4, R2 C2/C3, R4 C2 (τ and β sensitivity table) | Reward-weight sensitivity table (vary ω_d, ω_s primarily) |
| 2 | **Bias-term ablation** — Eq.~(27) is the main contribution; reviewer will want to see it actually matters | R5 C3 ("if you remove the GNN pruning step, what happens?") | Compare *Proposed* vs *Proposed-NoBias* (w_v = w_d = 0). Status: eval-only ablation in `evaluate_ablation.py`. For full rigor would need a re-trained model with bias disabled. |
| 3 | **Action-mask ablation** — Eq.~(22) is also a contribution | R5 / R6 generic ablation pattern | Train *Proposed-NoMask* and compare. (Requires retraining ~12 h; defer if time-constrained.) |
| 4 | **Optimality gap** at small scale | R5 C4 (Gurobi INLP at N = 30/50/70) | Small-scale Dijkstra (delay-only optimal) vs Proposed table. Problem P is NP-hard so cannot do INLP optimum, but per-snapshot Dijkstra is a meaningful lower bound. |

### ⭐⭐ MEDIUM

| # | Risk | WCL precedent | Required addition |
|---|------|---------------|-------------------|
| 5 | "Why Transformer, not other architectures?" | R5 C2 ("why GNN, not RL?") | Strengthen Section II discussion. Currently OK but reinforce. |
| 6 | **Computational complexity** | R4 C2 | Add table: per-satellite inference latency, parameter count, memory footprint |
| 7 | **Generalization** to traffic patterns / failure rates | Editor C5, R4 C1, R7 C1 | Already partly covered (small + large constellation). Add: different λ values or session ratios. |
| 8 | **w_v, w_d learning curve** — show that the bias is actually being learned (not stuck at init) | (No direct WCL precedent but obvious follow-up) | Plot of w_v(t), w_d(t) over training episodes |

### ⭐ LOW (one-line additions)

| # | Risk | Required addition |
|---|------|-------------------|
| 9 | PLR / throughput formal definitions (R7 C5) | One-paragraph addition at start of Section IV-B |
| 10 | Inference latency comparison vs baselines | One row in computational-complexity table |

---

## ✅ Already Defended

These reviewer angles are already adequately covered in the current paper:

- **Convergence-hyperparameter study** — `fig_convergence_hp` already shows LR (1e-3 / 1e-4 / 1e-5) and batch-size (256 / 2048 / 8192) sweeps.
- **Strong baselines** — STSD / DLBH / MADRL / GRLR (heuristic, MLP, GAT-RL coverage).
- **Scalability** — `fig_perf_large` (Walker 36 × 22 = 792 sats, |U| up to 7000).
- **Path-divergence visualization** — `fig_path_comp` shows the proposed framework geographically detouring around bad zones.

---

## 🎯 Next-action Checklist (when revisiting)

Tackle in this order. Each is independent — partial completion still helps.

1. ☐ **Bias-term ablation** (eval-only) — runs in ~10 min via `evaluate_ablation.py`.
   Output: `data/ablation_delay_small.csv`, `data/ablation_plr_small.csv`.
   Generates a 3-line comparison: Proposed vs Proposed-NoBias vs MADRL.
   *Status: launched 2026-05-04; check `logs/ablation_eval.log`.*

2. ☐ **Reward-weight sensitivity table** — eval-only sweep over ω_d ∈ {0.30, 0.45, 0.60} or similar. Likely 30 min total.

3. ☐ **PLR / throughput formal definitions** — text-only addition to start of Section IV-B.
   Suggested wording:
   > Throughput is defined as the total successfully delivered bits per unit time, $\sum_k \mathbb{1}[\text{deliver}_k] \cdot L_p / T_{\text{ep}}$. Packet loss ratio (PLR) is defined as the fraction of generated packets that fail to reach the destination.

4. ☐ **w_v / w_d learning curve** — extract from saved checkpoints (`transformer_ep100.pt`, `ep200.pt`, ...). 5-line script.

5. ☐ **Computational complexity table** — measure inference latency once with `time.perf_counter()` around `actor.forward(...)`. Report per-satellite, per-slot.

6. ☐ **Optimality gap at small scale** — pick N ∈ {18×6, 18×9} reduced constellations, compute Dijkstra-on-delay shortest path as baseline. Compare end-to-end delay ratio. Honest framing: "Dijkstra-optimal under static topology assumption."

7. ☐ **Action-mask ablation** — requires retraining ~12 h. Defer unless reviewer specifically asks.

---

## 🛡️ Defensive framing tips (if a reviewer challenges)

- *"Why fixed ω values?"* → Reference `chen2025distributed` (already cited in Table II) and add the sensitivity table.
- *"Why Transformer not GAT/CNN?"* → Multi-head attention captures multi-hop dependencies; bias-term injection (Eq.~27) is architecturally specific to attention scores; cite `vaswani2017attention`.
- *"Centralized critic = SPOF?"* → CTDE: critic is *only used at training*; execution is fully decentralized. Add this to Section IV-A if challenged.
- *"Generalization to other constellations?"* → small + large already shown. For different orbital configurations, retraining with new TLE data is identified as future work (same defense as WCL paper's R4-C1 response).

---

## 📁 Related artifacts

| File | Purpose |
|------|---------|
| `evaluate_ablation.py` | 3-way bias ablation (Proposed / Proposed-NoBias / MADRL) — output prefix `ablation_` |
| `evaluate.py` | Main 5-scheme eval — DO NOT modify, output is paper's `fig_perf` / `fig_perf_large` |
| `backup_before_ablation_20260504/` | Snapshot of paper-final figures + CSVs before ablation work |
| `BACKUP_USAGE.md` | How to restore from backups if something gets overwritten |
