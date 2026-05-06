# LEO Transformer MARL

Implementation of:
> **"Transformer-Based Multi-Agent Reinforcement Learning Routing for Dynamic Links in LEO Satellite Networks"**
> Yoonsoo Choi, Jaeseong Park, SuKyoung Lee

---

## Project Structure

```
LEO_Transformer_MARL/
├── config.py                  # All hyperparameters (Table I)
├── train.py                   # Train Proposed (Transformer-MADRL)
├── train_maac.py              # Train MAAC baseline
├── evaluate.py                # Compare ALL schemes + generate figures
│
├── environment/
│   ├── constellation.py       # Walker Delta + ISL/GSL model (Sec. III-A,B,C)
│   ├── leo_env.py             # Dec-POMDP environment (Sec. III-D, IV-A)
│   └── traffic.py             # Poisson traffic generator (Sec. III-A)
│
├── models/
│   ├── transformer_actor.py   # Mobility-aware Transformer policy (Sec. IV-B)
│   ├── critic.py              # Centralized value network (CTDE)
│   └── maac.py                # MAAC baseline (no Transformer) [16]
│
├── algorithms/
│   └── ppo_ctde.py            # PPO + CTDE + GAE (Sec. IV-C,D, Algorithm 1)
│
├── routing/
│   ├── stsd.py                # STSD benchmark [20]
│   └── dlrh.py                # DLRH benchmark [14]
│
└── utils/
    └── metrics.py             # E2E delay, throughput, jitter, PDR
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Proposed scheme
```bash
python train.py --episodes 2000 --planes 12 --sats 11
```

### 3. Train MAAC baseline
```bash
python train_maac.py --episodes 2000 --planes 12 --sats 11
```

### 4. Compare all schemes
```bash
python evaluate.py --episodes 100
```

Figures are saved to `./figures/`.

---

## Key Hyperparameters (Table I)

| Parameter           | Value   |
|---------------------|---------|
| ISL range           | 2500 km |
| Slot duration       | 100 ms  |
| Buffer size         | 1000 pkts |
| Bandwidth           | 1 GHz   |
| PPO clip ε          | 0.2     |
| Learning rate       | 1e-4    |
| Transformer d_model | 128     |
| Attention heads     | 4       |

---

## Schemes Compared

| Scheme   | Description                                      | Reference |
|----------|--------------------------------------------------|-----------|
| Proposed | Transformer-MADRL with mobility-aware attention  | This work |
| MAAC     | MLP actor-critic, no Transformer                 | [16]      |
| STSD     | Static Topology Shortest Delay (Dijkstra)        | [20]      |
| DLRH     | Demand-Aware Load Balancing Heuristic            | [14]      |

---

## Output Figures

| File                         | Content                        |
|------------------------------|--------------------------------|
| `figures/fig1_e2e_delay.eps` | E2E delay time series          |
| `figures/fig2_throughput_cdf.eps` | CDF of TCP throughput     |
| `figures/fig3_stability.eps` | Jitter & path switch frequency |
| `figures/fig4_convergence.eps` | Training convergence curve   |
