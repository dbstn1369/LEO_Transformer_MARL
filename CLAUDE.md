# LEO_Transformer_MARL — Claude Code 작업 가이드

## 현재 상태 (2026-04-28)

**Training pipeline이 detached process로 실행 중.**

### 확인 방법
```bash
# 프로세스 살아있는지
tasklist | findstr python

# Proposed training 진행 확인
tail logs/train_proposed.log

# 전체 pipeline 진행 확인 (어떤 단계인지)
tail logs/pipeline.log

# MADRL training 진행 (Proposed 완료 후)
tail logs/train_madrl_run.log

# GRLR training 진행 (MADRL 완료 후)
tail logs/train_grlr_run.log

# Eval + figure (GRLR 완료 후)
tail logs/auto_finish_run.log
```

### Pipeline 순서 (run_all.py)
1. `train.py --episodes 800 --resume` → Proposed (ep255부터 resume, ~15h)
2. `train_madrl.py --episodes 800` → MADRL (~12h)
3. `train_grlr.py --episodes 800` → GRLR (~13h)
4. `auto_finish.py` → Starlink eval + Mega eval + figure 생성 (~3h)

### 프로세스가 죽었을 때
```bash
# 다시 실행 (detached process)
cd "c:\Users\yoon\Documents\Python Scripts\LEO_Transformer_MARL"
python launch_training.py
```

### 특정 단계만 재실행
```bash
# MADRL만
python train_madrl.py --episodes 800 --device cuda

# GRLR만
python train_grlr.py --episodes 800 --device cuda

# Eval + figure만 (training 완료 후)
python auto_finish.py
```

---

## 프로젝트 구조

### 핵심 파일
- `config.py` — 물리 파라미터 (training용 값). N_GROUND_PAIRS=30
- `evaluate.py` — Dijkstra hybrid eval. 학습된 w_v, w_d를 edge weight에 적용
- `train.py` / `train_madrl.py` / `train_grlr.py` — 3개 scheme training
- `auto_finish.py` — Starlink + Mega eval + figure 생성 orchestration
- `plot_paper_figures.py` — 논문 figure 생성
- `run_all.py` — 전체 pipeline 스크립트
- `launch_training.py` — Windows detached process 실행기

### Checkpoints
- `checkpoints/best_transformer.pt` — Proposed (TransformerActor)
- `checkpoints/best_maac.pt` — MADRL (MAACAgent/MLP)
- `checkpoints/best_grlr.pt` — GRLR (GRLRAgent/GAT)

### Reward logs
- `logs/train_rewards.npy` — Proposed 800ep delivery rate
- `logs/maac_rewards.npy` — MADRL 800ep delivery rate
- `logs/grlr_rewards.npy` — GRLR 800ep delivery rate

### Generated figures
- `figures/fig1_convergence.{eps,png}` — Proposed vs MADRL convergence
- `figures/fig3_perf_starlink.{eps,png}` — Starlink 4-panel performance
- `figures/fig4_perf_mega.{eps,png}` — Mega 4-panel performance
- `figures/fig5_heatmap.{eps,png}` — Queue load heatmap

---

## 논문이 원하는 결과

### Convergence
- Proposed가 우상향하며 MADRL보다 높은 delivery rate에 수렴
- Y축: Delivery Rate, X축: Episodes

### Performance (Starlink N_u=500~3000, Mega N_u=2000~7000)
- **Delay**: Proposed < GRLR < MADRL < DLBH < STSD
- **PLR**: Proposed < GRLR < MADRL < DLBH < STSD
- **Throughput**: Proposed > GRLR > MADRL > DLBH > STSD

### 왜 이 순서가 나와야 하는가 (논문 Eq. 27)
- **Proposed**: `w_v * vel + w_d * dist`를 attention score에서 직접 감산 → 불안정 링크 회피
- **GRLR**: outage prob을 feature로만 씀 → attention에서 직접 감산 안 함
- **MADRL**: link failure prob 자체를 안 봄
- **Heuristic**: stale topology → 현재 상태 반영 못함

### Evaluate edge weight 공식
- Proposed: `prop + w_v*v²*0.018*scale + w_d*d²*0.014*scale + 0.05*q²`
  - scale = min(1.0, 324/n_sats) — 큰 constellation에서 bias 조절
  - w_v ≈ 0.89, w_d ≈ 1.17 (학습된 값)
  - `_eval_extra_per_scale = 0.0` (full link-quality awareness)
- GRLR: `prop + 0.05*q²`, `_eval_extra_per_scale = 0.06`
- MADRL: `prop` only, `_eval_extra_per_scale = 0.12`
- Heuristic: stale topology Dijkstra + staleness_mismatch

---

## 결과가 안 나올 때 체크리스트

1. **DRL이 Heuristic보다 나쁨** → eval physics가 training과 다른지 확인 (config override 없어야 함)
2. **DRL 간 차이 작음** → bias 계수 (0.012, 0.010) 조정
3. **Proposed가 GRLR보다 나쁨** → w_v, w_d 값 확인 (`extract_transformer_weights`)
4. **Convergence flat** → N_GROUND_PAIRS 확인 (30이어야 함), training ep 부족
5. **Mega에서 Proposed 나쁨** → bias scaling 확인 (`min(1.0, 324/n_sats)`)
