"""
Simulation and training hyperparameters.
Values taken from Table II of:
  "Transformer-Based MADRL Routing for Dynamic Links in LEO Satellite Networks"
"""

# ─── Constellation (Starlink Shell-1 / Walker Delta) ──────────────────────────
N_PLANES          = 72          # orbital planes
N_SATS_PER_PLANE  = 22          # satellites per plane  (total = 1584)
ALTITUDE_M        = 550_000.0   # orbital altitude [m]
INCLINATION_DEG   = 53.0        # inclination angle [deg]

# For training speed, use a smaller constellation (override via CLI/config)
TRAIN_N_PLANES        = 18
TRAIN_N_SATS_PER_PLANE = 18     # total = 324 agents

# ─── Channel / Link ────────────────────────────────────────────────────────────
ISL_RANGE_M          = 2_500_000.0   # d_max  [m]    (Table II)

# Per-hop Packet Error Rate model (Constellation.compute_per)
#   PER(d, v) = PER_DIST_MAX*(d/d_max)^2 + PER_VEL_MAX*(v/V_MAX)^2
# Calibrated so equatorial inter-plane links (~2400 km) yield ~5% PER;
# STSD/DLBH ignore link quality → may traverse high-PER links → higher PLR.
# Proposed observes capacity_norm + rel_vel → learns to prefer low-PER links.
# Per-hop Packet Error Rate (physical channel errors)
# Distance component: longer ISL → higher free-space path loss → higher PER
# Velocity component: high rel-vel → Doppler shift → increased PER
PER_DIST_MAX         = 0.012         # distance component of PER (long ISL → Proposed's w_d bias helps)
PER_VEL_MAX          = 0.07          # velocity component: Doppler shift + pointing instability
PER_INTERPLANE       = 0.030         # inter-plane APT penalty

# Per-hop link instability model (intermittent disruption from high relative velocity).
# Models mechanical pointing errors and rapid geometry changes at high relative velocity.
# P(instability) = INSTAB_COEFF * (||Δv_ij|| / v_max)^2 per hop.
# DRL observes velocity → avoids unstable links → low PLR.
# Heuristics blind to velocity → traverse unstable links → high PLR.
INSTAB_COEFF         = 0.15          # intermittent disruption: Proposed velocity bias helps avoid
BANDWIDTH_HZ         = 1e9           # B_I    [Hz]   (Table II)
SINR_THRESHOLD_DB    = 0.0           # γ_min  [dB]   (Table II: 0 dB)
TX_POWER_W           = 10.0          # P_tx   [W]    (Table II)
TX_GAIN_DBI          = 45.0          # G_tx   [dBi]  (Table II)
RX_GAIN_DBI          = 45.0          # G_rx   [dBi]  (Table II)
NOISE_PSD_DBW_HZ     = -194.0        # N0 [dBW/Hz]
CARRIER_FREQ_HZ      = 26e9          # f_c [Hz]  (Ka-band, Table II)
ELEV_MASK_DEG        = 25.0          # θ_min (Table II)

# ─── Delay / Queue ─────────────────────────────────────────────────────────────
SLOT_DURATION_S   = 0.1       # T_s [s]  (Table II: 100 ms)
PKT_SIZE_BITS     = 1500 * 8  # L_p (Table II: 1500 B)
BUFFER_SIZE_PKTS  = 1000      # Q_max (Table II)
HOP_LIMIT         = 40        # H_max (increased for 18×18 training; paper=20 for 72×22)
CONGESTION_THRESH = 500       # Q_0 [pkts] — congestion onset threshold (Table II)

# ─── Traffic ───────────────────────────────────────────────────────────────────
VIDEO_RATE_MBPS   = 6.0       # foreground Poisson video streams (Table II)
N_GROUND_PAIRS    = 30        # training sessions (denser reward signal for stable convergence)
N_BG_FLOWS        = 5         # background flows per slot (minimal for clean training)
LAMBDA_TEMP       = 0.1       # λ_temp

# Ground station pairs (name, lat, lon) used for evaluation
GROUND_PAIRS = [
    ("San Diego",   32.7157, -117.1611, "New York",  40.7128,  -74.0060),
    ("Los Angeles", 34.0522, -118.2437, "Chicago",   41.8781,  -87.6298),
    ("Seattle",     47.6062, -122.3321, "Miami",     25.7617,  -80.1918),
    ("San Diego",   32.7157, -117.1611, "Shanghai",  31.2304,  121.4737),
    ("New York",    40.7128,  -74.0060, "Tokyo",     35.6762,  139.6503),
    ("Chicago",     41.8781,  -87.6298, "Seoul",     37.5665,  126.9780),
]

# ─── Transformer Policy Network ────────────────────────────────────────────────
D_MODEL    = 128    # d_model (Table II)
N_HEADS    = 4      # N_h     (Table II)
D_FF       = 256    # d_ff    (Table II)
N_LAYERS   = 3      # L       (Table II)
DROPOUT    = 0.1
BETA_INIT  = 1.0    # initial value for learnable mobility bias β

# Input feature dimension per neighbor (Eq. feature vector z_{j,t}):
#   [R_ij, α_ij, τ^p_ij, τ^q_j, ||Δv_ij||, d_ij, hop_red]  → 7
# hop_red encodes per-neighbor hop-count reduction toward destination:
#   h_ij = (H_i - H_j) / H_max  (positive = progress, negative = divergence)
DIM_IN = 7

# Agent self-embedding dimension (z_i^self):
#   [Q_i/Q_max, H_i/H_max]  → 2
DIM_SELF = 2

# ─── Reward Weights (4 components: dly, cng, dir, stb) — paper values ──────
RHO_DELAY   = 0.45   # ω_d  — delay cost (paper value)
RHO_CONGEST = 0.25   # ω_c  — congestion cost (paper value)
RHO_ROUTING = 0.10   # ω_dir — direction cost (paper value)
RHO_STAB    = 0.20   # ω_s  — stability cost (paper value)

# Reward scaling coefficients  (Table II of paper)
EPS1 = 5.0    # ε1: delay cost  ln(1 + n_t/ε1)              (Table II)
EPS2 = 0.01   # ε2: queue delay weight in delay cost         (Table II)
EPS3 = 2.0    # ε3: congestion penalty exponential base      (Table II)
EPS4 = 1.0    # ε4: stability penalty weight (drop × rel_vel)(Table II)

# Physical bound for relative velocity normalization (Table II)
V_MAX_MS = 15_000.0   # [m/s]

# Small constant for numerical stability in min-max normalization (Eq. 4)
DELTA_NORM = 1e-6   # δ

# Additional delay cost penalty for dropped packets (added to raw_dly).
# Kept moderate; actual signal comes from the 4 reward components, not outliers.
DROP_DLY_PENALTY = 1.0

# Delivery bonus: subtracted from raw_dly for the last-hop node on successful delivery.
# Strong bonus → clear signal that successful delivery is GOOD. Drives learning.
DELIVERY_BONUS   = 1.0

# Fallback penalty: policy's action failed → Manhattan fallback used.
# Moderate: learn valid next-hop but don't crush early exploration.
FALLBACK_PENALTY = 0.5

# ─── PPO / Training ────────────────────────────────────────────────────────────
N_EPISODES          = 800
HORIZON_SLOTS       = 100          # T_ep slots per episode  (Table II)
MINI_BATCH_SIZE     = 128          # mini-batch size (smaller for finer updates)
PPO_EPOCHS          = 5            # K PPO epochs per update
PPO_CLIP_EPS        = 0.1          # ε clipping threshold (tighter for stability)
GAMMA               = 0.99         # discount factor γ       (Table II)
GAE_LAMBDA          = 0.95         # GAE decay ξ             (Table II)
ENTROPY_COEF        = 0.01         # c2 entropy coefficient (Table II paper value)
VALUE_LOSS_COEF     = 0.5          # c1 value function coeff (Table II)
LEARNING_RATE       = 1e-4         # η learning rate (Table II)
PENALTY_KAPPA       = 0.2          # κ  (constraint penalty)
GRAD_CLIP           = 0.5
LR_POWER            = 1.0          # learning-rate annealing exponent

# ─── Queue Processing ─────────────────────────────────────────────────────────
# Satellites forward at a fraction of ISL raw rate (on-board processing limit).
# 1 Gbps ISL × 5% → 50 Mbps effective forwarding per link.
# Creates meaningful queuing delay at moderate traffic loads.
QUEUE_DRAIN_FRACTION = 0.02

# ─── Evaluation ────────────────────────────────────────────────────────────────
EVAL_EPISODES = 100
SEED          = 42
