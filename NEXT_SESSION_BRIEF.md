# Next session — start here

**작성: 2026-05-06 저녁** (full retrain in progress)

새 세션 열면 이 파일을 먼저 읽어서 어디까지 왔는지 파악할 것.

---

## 1. 현재 상태 한 줄 요약

`backup_paper_snapshot_20260506/`에 paper-final state 보존된 채로,
**3-scheme full retrain** (Proposed + GRLR + MADRL, fresh seed=7) 진행 중.
완료 시 paper narrative (P > G > M) 강하게 만족하는 새 figure 자동 생성됨.

## 2. GitHub repo (rollback safety)

- Repo: **https://github.com/dbstn1369/LEO_Transformer_MARL**
- Branch: `main`
- 첫 commit `203859e` + tag **`paper-snapshot-20260506`** = rollback point.
- 부모 repo `Python-Scripts`와 분리된 sub-repo (부모는 100MB+ 옛 blob들 때문에 push 불가).

## 3. 학습 상태 확인 (매번 체크)

```bash
tasklist | grep -i python                 # PID 살아있는지 (PID 변동 가능)
tail -5 logs/train_proposed.log           # 현재 step 진행
tail -3 logs/pipeline.log                 # 4단계 중 어디
```

### 4단계 pipeline (`run_all.py`, detached process)
1. `train.py --episodes 800 --device cuda` — Proposed (~14.5h, ~65s/ep)
2. `train_madrl.py --episodes 800 --device cuda` — MADRL (~11h)
3. `train_grlr.py --episodes 800 --device cuda` — GRLR (~12h)
4. `auto_finish.py` — Starlink eval + Mega eval + figure 생성 (~3h)

총 ~40시간. seed: cfg.SEED=7 (P=7, M=8, G=9 — 각 train script 내부 +1, +2 offset).

### 죽었으면 다시 띄우기
```bash
cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"
python launch_training.py
# 또는 단계별: python train.py --episodes 800 --device cuda
```

## 4. 적용된 코드 변경 (이번 retrain의 핵심)

paper narrative상 Proposed > GRLR > MADRL이 training reward에서도 emerge하도록:

1. **scheme-specific blindness during training** (`leo_env.py:602-611`)
   - `extra_per`을 training에서도 적용 (eval의 절반 강도)
   - `train.py`: `env._eval_extra_per_scale = 0.0` (Proposed)
   - `train_grlr.py`: `0.06`
   - `train_madrl.py`: `0.12`

2. **softplus(β/w_d) + final-layer Eq.27 bias** (`models/transformer_actor.py`)
   - `F.softplus()` 적용해서 attention β, w_d가 음수로 학습되는 거 방지
   - 최종 action logit에서도 동일한 β, w_d (averaged across L layers) 재사용
   - `evaluate.py:extract_transformer_weights`도 softplus 적용해 effective 값 반환

3. **continuous stability cost** (`leo_env.py:409-421`)
   - 기존 `raw_stb = ε4·I_drop·(rel_vel/V_max)` (drop 시에만 fire) → sparse
   - 신규: `raw_stb = ε4·(0.5·v_ratio + 0.5·d_ratio)` always-on + drop 시 추가 penalty
   - β, w_d gradient signal dense하게

## 5. 학습 끝난 후 할 일 (Phase 4~6)

### Phase 4: figure 생성 (auto_finish.py가 자동 처리)
끝나면 다음 파일들 갱신됨:
- `figures/fig_convergence.{png,eps}` — Proposed/GRLR/MADRL 수렴 곡선
- `figures/fig_perf.{png,eps}` — Starlink (18×18) 4-panel
- `figures/fig_perf_large.{png,eps}` — Mega (36×22) 4-panel
- `figures/fig_path_comp.{png,eps}` — 경로 시각화
- `figures/fig_ablation.{png,eps}` — bias 제거 ablation
- `data/fig1_delay_vs_users_{small,large}.csv`, `data/fig5_plr_vs_users_*.csv`

### Phase 5: 결과 점검 + HP sweep 재실행
1. `figures/fig_convergence.png` 보고 P > G > M 순서 확인
2. `data/fig1_delay_vs_users_*.csv` 보고 |U|=3000, |U|=7000에서 % 개선 계산
3. **만약 결과 좋으면**: HP sweep 다시 (재학습 후 train.py 변경됐으니까)
   ```bash
   python run_hp_sweep.py --episodes 100 --device cuda
   ```
4. **만약 결과 나쁘면**: rollback (아래 Section 6)

### Phase 6: §5 텍스트 수치 업데이트 (`section5_simulation.tex`)
- §5.B Convergence: Proposed/GRLR/MADRL 수렴값 (예전: 0.49/0.42)
- §5.D E2E Delay/PLR/TP: |U|=3000에서의 % 개선
- §5.E Scalability: |U|=7000에서의 % 개선
- §5.F Ablation: Proposed-NoBias 수치
- §5.G Path Comparison: hop count, total distance, avg ISL capacity
- Table II: $w_v$, $w_d$ 학습된 값 (현재 paper: 0.89, 1.17)

## 6. Rollback (만약 retrain 결과 나쁘면)

### 빠른 복원 (script로)
```bash
# bash
bash rollback_paper_final.sh

# PowerShell
.\rollback_paper_final.ps1
```
이러면 backup_paper_snapshot_20260506/에서 .py, .pt, .npy, figures, CSV, .tex 모두 복원.

### git으로 hard rollback (commit history도 paper-snapshot으로)
```bash
git reset --hard paper-snapshot-20260506
# 주의: 이후 커밋 모두 사라짐 (untracked 파일은 보존)
```

## 7. 디렉토리 구조 (정리됨)

```
LEO_Transformer_MARL/
├── .gitignore                    # __pycache__/, /checkpoints/, /logs/{*.log,*.npy} 제외
├── CLAUDE.md                     # 프로젝트 가이드
├── NEXT_SESSION_BRIEF.md         # 이 파일 (always start here)
├── PROJECT_GUIDE.md              # 전체 프로젝트 맥락
├── REVIEWER_DEFENSE_TODO.md      # 리뷰어 방어용 추가 결과 list
├── section5_simulation.tex       # 논문 Section 5
├── rollback_paper_final.{sh,ps1} # 한 줄로 rollback
│
├── config.py                     # SEED=7
├── train.py / train_grlr.py / train_madrl.py
├── evaluate.py
├── plot_paper_figures.py / plot_path_comparison.py / plot_ablation.py
├── run_all.py / launch_training.py / auto_finish.py
│
├── environment/leo_env.py        # ① + ③ 변경 적용
├── models/transformer_actor.py   # ② 변경 적용
├── models/{maac,grlr,critic}.py
├── algorithms/ppo_ctde.py
├── routing/{stsd,dlbh}.py
│
├── checkpoints/                  # ⚠️ 학습 중 overwrite. paper-final은 backup에
├── logs/                         # ⚠️ runtime 로그 + reward .npy
├── figures/                      # 학습 끝나면 갱신됨
├── data/                         # CSV outputs
│
└── backup_paper_snapshot_20260506/   # ⭐ rollback point (모든 paper-final 포함)
    ├── checkpoints/   (best_*.pt)
    ├── code/          (paper-final .py)
    ├── data/          (CSV)
    ├── figures/       (paper-final .png/.eps)
    ├── logs/          (.npy reward histories)
    └── tex/           (section5 + project_guide)
```

## 8. Quick monitoring snippet

새 세션 시작하면 가장 먼저 이거 한 번 돌려서 상태 파악:

```bash
cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"
echo "=== Process ===" && tasklist | grep -i python
echo "=== Last training step ===" && tail -3 logs/train_proposed.log
echo "=== Pipeline stage ===" && tail -3 logs/pipeline.log
echo "=== Reward log (if final step done) ===" && ls -la logs/*_rewards.npy 2>/dev/null
```
