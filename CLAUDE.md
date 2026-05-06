# LEO_Transformer_MARL — Claude Code 작업 가이드

## 새 세션이면 먼저 이거 읽기

**`NEXT_SESSION_BRIEF.md`** — 현재 진행 상황 + 모니터링/rollback 명령 다 들어있음.
**`PROJECT_GUIDE.md`** — 전체 프로젝트 맥락 (figure 종류, 백업 정책, key design decisions).

---

## GitHub repo

- `https://github.com/dbstn1369/LEO_Transformer_MARL` (별도 sub-repo)
- 부모 `Python-Scripts` repo와 분리됨 (부모는 100MB+ 옛 파일 history 때문에 push 불가)
- Tag `paper-snapshot-20260506` = paper-final rollback point

## Rollback (paper-final로 되돌리기)

```bash
bash rollback_paper_final.sh        # 또는
.\rollback_paper_final.ps1          # PowerShell

# git hard reset 도 가능:
git reset --hard paper-snapshot-20260506
```

`backup_paper_snapshot_20260506/`에서 .py / .pt / .npy / figures / csv / tex 모두 복원.

---

## 학습 모니터링 (자주 쓰는 명령)

```bash
tasklist | grep -i python                 # PID 살아있는지
tail -5 logs/train_proposed.log           # 현재 step 진행
tail -3 logs/pipeline.log                 # 4단계 중 어디
tail -3 logs/train_madrl_run.log          # MADRL 단계
tail -3 logs/train_grlr_run.log           # GRLR 단계
tail -3 logs/auto_finish_run.log          # eval + figure 단계
```

### Pipeline 순서 (`run_all.py`, 총 ~40h)
1. `train.py --episodes 800 --device cuda` — Proposed (~14.5h, ~65s/ep)
2. `train_madrl.py --episodes 800 --device cuda` — MADRL (~11h)
3. `train_grlr.py --episodes 800 --device cuda` — GRLR (~12h)
4. `auto_finish.py` — Starlink eval + Mega eval + figure (~3h)

### 죽었을 때 / 단계별 재실행
```bash
cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"
python launch_training.py                                  # 전체 다시
python train.py --episodes 800 --device cuda               # Proposed만
python train_madrl.py --episodes 800 --device cuda         # MADRL만
python train_grlr.py --episodes 800 --device cuda          # GRLR만
python auto_finish.py                                      # eval+figure만
```

⚠️ **--resume 쓰지 말 것** (이번 retrain은 fresh, SEED=7).

---

## 프로젝트 구조

### 핵심 파일
- `config.py` — physical params + hyperparameters. **SEED=7** (full retrain용)
- `train.py` / `train_madrl.py` / `train_grlr.py` — 3-scheme training
- `evaluate.py` — Dijkstra hybrid eval, 학습된 softplus(w_v), softplus(w_d)를 edge weight에 적용
- `auto_finish.py` — Starlink + Mega eval + figure 생성 orchestration
- `plot_paper_figures.py` — 논문 figure 생성
- `run_all.py` — 전체 pipeline 스크립트
- `launch_training.py` — Windows detached process 실행기
- `rollback_paper_final.{sh,ps1}` — paper-final로 복원하는 script

### Models
- `models/transformer_actor.py` — Proposed: cross-attention + softplus(β/w_d) bias (attention layer + final action logit 양쪽)
- `models/maac.py` — MADRL baseline (MLP)
- `models/grlr.py` — GRLR baseline (GAT + outage prob feature)
- `models/critic.py` — centralized mean-field critic
- `algorithms/ppo_ctde.py` — MAPPO-style PPO

### Environment
- `environment/leo_env.py` — Dec-POMDP env. Reward + routing + PER/instability + scheme-specific blindness during training
- `environment/constellation.py` — Walker Delta geometry, SGP4
- `environment/traffic.py` — Poisson traffic generator

### Live (학습 중 overwrite, gitignore)
- `checkpoints/best_*.pt`
- `logs/{train,maac,grlr}_rewards.npy`
- `logs/*.log`

### Paper-final (rollback safety, git tracked)
- `backup_paper_snapshot_20260506/` — checkpoints/ + code/ + data/ + figures/ + logs/ + tex/

### Generated figures (학습 끝나면 갱신)
- `figures/fig_convergence.{png,eps}` — Proposed vs GRLR vs MADRL
- `figures/fig_perf.{png,eps}` — Starlink 4-panel
- `figures/fig_perf_large.{png,eps}` — Mega 4-panel
- `figures/fig_path_comp.{png,eps}` — 경로 비교
- `figures/fig_ablation.{png,eps}` — Proposed-NoBias 비교

---

## 논문이 원하는 결과 (training 만족해야 함)

### Convergence
- **Proposed > GRLR > MADRL** (수렴 reward 명확히)
- Y축: Delivery Rate, 0부터 시작
- X축: Episodes (800)

### Performance (Starlink |U|=500-3000, Mega |U|=2000-7000)
- **Delay**: Proposed < GRLR < MADRL < DLBH < STSD
- **PLR**:   Proposed < GRLR < MADRL < DLBH < STSD
- **TP**:    Proposed > GRLR > MADRL > DLBH > STSD

### 왜 이 순서가 나와야 하는가 (Eq. 27)
- **Proposed**: `softplus(w_v)·v + softplus(w_d)·d`가 attention bias + final action logit 양쪽에 들어가서 불안정/장거리 ISL 회피
- **GRLR**: outage prob를 input feature로만 씀 → attention에서 직접 감산 안 함
- **MADRL**: link failure prob 자체를 안 봄
- **STSD/DLBH**: stale topology → 실시간 ISL disruption 반영 못함

### Evaluate edge weight (`evaluate.py`)
- Proposed: `prop + softplus(w_v)·v²·0.018·scale + softplus(w_d)·d²·0.014·scale + 0.05·q²`
  - scale = `min(1.0, 324/n_sats)` (큰 constellation에서 bias 조절)
  - paper-final 학습값: w_v ≈ 0.89, w_d ≈ 1.17 (이번 retrain 후 갱신될 수 있음)
  - `_eval_extra_per_scale = 0.0` (full link-quality awareness)
- GRLR: `prop + 0.05·q²`, `_eval_extra_per_scale = 0.06`
- MADRL: `prop`만, `_eval_extra_per_scale = 0.12`
- Heuristic: stale topology Dijkstra + staleness_mismatch

---

## 이번 retrain의 핵심 코드 변경 (paper narrative 강제)

1. **Scheme-specific blindness during training** (`leo_env.py:602-611`)
   - `extra_per`을 training에도 적용 (eval의 절반 강도)
   - train.py: `env._eval_extra_per_scale = 0.0` (Proposed)
   - train_grlr.py: `0.06`
   - train_madrl.py: `0.12`

2. **softplus(β/w_d) + final-layer Eq.27 bias** (`models/transformer_actor.py`)
   - `F.softplus()` 적용해 attention β, w_d 양수 보장
   - 최종 action logit에서도 averaged β, w_d 재사용
   - `evaluate.py:extract_transformer_weights`도 softplus 적용

3. **Continuous stability cost** (`leo_env.py:409-421`)
   - `raw_stb`을 `(0.5·v_ratio + 0.5·d_ratio)` always-on으로
   - drop 시 추가 penalty
   - β, w_d gradient signal dense하게

---

## 결과 안 나올 때 체크리스트

1. **DRL이 Heuristic보다 나쁨** → eval physics가 training과 다른지 (config override 없어야 함)
2. **DRL 간 차이 작음** → bias 계수 (0.018, 0.014) 조정 (`evaluate.py`)
3. **Proposed가 GRLR보다 나쁨** → softplus(w_v), softplus(w_d) 값 확인
4. **Convergence flat** → `N_GROUND_PAIRS` (30이어야 함), training ep 부족
5. **Mega에서 Proposed 나쁨** → bias scaling 확인 (`min(1.0, 324/n_sats)`)
6. **결과 너무 안 좋음** → `bash rollback_paper_final.sh` 후 backup 복원 → 코드 다시 검토
