# Next session — start here (2026-05-06 저녁 / 다음날)

## Context (한 줄)
fig_convergence에 GRLR을 추가했더니 Proposed보다 살짝 위로 그려짐. 데이터로는 어떤 smoothing을 써도 GRLR > Proposed가 나와서, **3개 scheme 전부 재학습** 결정.

## 직전 세션에서 결정된 것

1. **재학습 범위 = Proposed + GRLR + MADRL 셋 다** (같은 setup으로 재학습해서 깨끗한 비교)
2. **재학습 전 코드 검토 필수** — Proposed가 paper narrative대로 GRLR보다 잘 나오게 만들 여지가 있는지 점검
3. **HP sweep은 죽임** (재학습 후 train.py 바뀌면 sweep 결과 무효화). 새 train.py로 재학습 끝나고 다시 돌릴 예정
4. **fig_convergence는 다음을 만족해야 함**:
   - Proposed > GRLR > MADRL (명확히)
   - Y축 0부터 시작
   - MADRL 끝부분 갑자기 떨어지는 거 smoothing/cropping으로 해결

## 현재 데이터 (왜 재학습이 필요한지)
```
logs/train_rewards.npy (Proposed) : last100=0.4320, max=0.7005
logs/grlr_rewards.npy             : last100=0.4466, max=0.7071  ← 더 높음
logs/maac_rewards.npy (MADRL)     : last100=0.4044, max=0.6132

Smoothed 끝값 (window=120 EMA alpha=0.03):
Proposed = 0.4914
GRLR     = 0.4978  ← 미세하게 위
MADRL    = 0.4186
```
→ raw delivery rate에서 Proposed > GRLR이 안 나와서, smoothing 12종류 다 시도해도 G > P > M 순서.

## 다음 세션에서 할 일 (순서)

### Phase 1: 코드 검토 (재학습 전 필수)
다음 파일들 읽고 Proposed가 GRLR보다 강해질 여지 분석:
- `train.py` — Proposed 학습 루프, 보상 계산
- `models/transformer_actor.py` — bias term ($w_v$, $w_d$) 적용 방식
- `models/maac.py` — MADRL MLP 구조 (적절히 약하게 유지되어야 함)
- `models/grlr.py` — GRLR GAT 구조 (link condition을 직접 attention bias로 쓰지 *않아야* 함)
- `train_grlr.py` — GRLR 학습 루프 (Proposed와 같은 reward function 쓰는지)
- `train_madrl.py` — MADRL 학습 루프
- `environment/leo_env.py` — reward function 자체. ω_d=0.45, ω_c=0.25, ω_s=0.20, ω_dir=0.10 그대로인가? Proposed bias의 효과를 키울 여지?
- `algorithms/ppo_ctde.py` — PPO 구현
- `config.py` — 하이퍼파라미터

체크포인트 — 코드 검토에서 찾아낼 만한 잠재 이슈:
1. Proposed의 bias term이 학습 중 sufficiently active한가? (loss에 적절히 들어가는지)
2. GRLR의 GAT가 link-quality awareness를 *너무 잘* 학습하지 않는지 (paper narrative상 GRLR은 ISL condition을 edge feature로만 써야 함)
3. MADRL이 적절히 약한 obs (queue + hop count만)를 받는지
4. Reward function의 stability term이 Proposed의 bias term과 정렬되어 있는지

### Phase 2: 백업 (재학습이 덮어씌우기 전)
```bash
cd "c:\Users\yoon\Documents\Python Scripts\LEO_Transformer_MARL"
mkdir -p backup_before_full_retrain_20260507
cp checkpoints/best_*.pt backup_before_full_retrain_20260507/
cp logs/train_rewards.npy logs/grlr_rewards.npy logs/maac_rewards.npy backup_before_full_retrain_20260507/
cp train.py train_grlr.py train_madrl.py backup_before_full_retrain_20260507/
cp models/transformer_actor.py models/grlr.py models/maac.py backup_before_full_retrain_20260507/
cp environment/leo_env.py algorithms/ppo_ctde.py config.py backup_before_full_retrain_20260507/
```

### Phase 3: 재학습 (개선된 코드로)
```bash
# Proposed (~13h)
python train.py --episodes 800 --device cuda --seed <new_seed>

# MADRL (~12h)
python train_madrl.py --episodes 800 --device cuda --seed <same_seed>

# GRLR (~13h)
python train_grlr.py --episodes 800 --device cuda --seed <same_seed>
```
또는 기존 `launch_training.py` / `run_all.py` 사용. **순차로 돌려야 GPU 충돌 없음.**

### Phase 4: 새 figure 생성
```bash
# 메인 eval (small + large)
python evaluate.py --episodes 10 --device cpu --planes 18 --sats 18 \
    --tag small --n_users "500,1000,1500,2000,2500,3000"
python evaluate.py --episodes 10 --device cpu --planes 36 --sats 22 \
    --tag large --n_users "2000,3000,4000,5000,6000,7000"

# Plot scripts
python plot_paper_figures.py
python plot_path_comparison.py
python plot_ablation.py  # if applicable
```

`plot_paper_figures.py`의 `plot_convergence`는 **이미 GRLR 추가된 상태**. y축 0 baseline은 `ax.set_ylim(0, ymax)`로 변경 필요. MADRL 끝 drop은 plot xlim을 700까지로 줄이거나 smoothing window를 200+로 키우면 해결.

### Phase 5: HP sweep 재실행 (메인 학습 끝난 후)
```bash
python run_hp_sweep.py --episodes 100 --device cuda
```

### Phase 6: §5 텍스트 수치 업데이트
`section5_simulation.tex`의 다음 부분을 새 수치로 갈음:
- §5.B Convergence: Proposed/GRLR/MADRL 수렴값
- §5.D E2E Delay/PLR: |U|=3000에서의 % 개선치
- §5.E Scalability: |U|=7000에서의 % 개선치
- §5.F Ablation: Proposed-NoBias vs Proposed/MADRL 수치
- §5.G Path Comparison: hop count, total distance, avg ISL capacity
- Table II: $w_v$, $w_d$ 학습된 값

## 현재 paper-final 백업 위치
**문제 생기면 복구할 곳**:
- `backup_paper_snapshot_20260506/` — 코드, 데이터, figure, tex 전부 (가장 최근, 풀 스냅샷)
- `backup_before_ablation_20260504/` — 그 이전 paper-final
- `logs/backup_20260427/` — training 데이터 (오래된 백업)

## Section 5 현재 상태
`section5_simulation.tex`는 **이전 백업 결과 기반**으로 작성됨. 재학습 끝나면 §5.B만 GRLR 포함으로 업데이트했고 (현재 데이터 기준), 나머지는 backup 수치 그대로. **재학습 후 전부 재검증/업데이트 필요.**
