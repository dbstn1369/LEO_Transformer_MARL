# 연구실 Lab Style Profile
## SuKyoung Lee Lab @ Yonsei University
### 분석 논문: IPO (TSC 2024), PRA (IoT J 2023), Public Bus UAV (TITS 2024), DSFL (TIV 2025), V2V-FL (TITS 2026)

---

## 1. 섹션 구조 (Section Structure)

### 표준 섹션 순서
```
Abstract → Introduction → Related Works → System Model (Use Case 포함 가능)
→ Problem Formulation → Proposed Algorithm → Performance Evaluation → Conclusion
```

### 섹션별 내용 규칙

#### Abstract
- **들어가는 것**: 배경 1~2문장 → 문제점 → 제안 방법 → 핵심 기술 요소 → 결과 요약
- **들어가지 않는 것**: 수식, 인용, 구체적 수치(퍼센트 제외)
- **길이**: 보통 8~12문장, 200~250 단어

#### Introduction
- **1단락**: 기술/응용 배경 ("With the rapid development of...", "Owing to their...")
- **2단락**: 핵심 문제 제기 ("However, [문제]. [결과]. Moreover, [심화].")
- **3단락**: 기존 연구 리뷰 + 한계 (Reference A did X. Reference B studied Y. However, these did not consider Z.)
- **4단락**: 본 논문 제안 ("Therefore, in this paper, we propose...")
- **5단락 (마지막)**: Contributions (bullet points: "We propose...", "We design...", "We develop...", "We evaluate...")
- **마지막 문장**: "The remainder of this paper is organized as follows: ..."
- **들어가지 않는 것**: 수식, 증명, 상세 파라미터값

#### Related Work
- **구조**: A. 소주제1, B. 소주제2, C. 소주제3 (보통 2~4개 subsection)
- **각 subsection**: 기존 연구 나열 → 마지막 단락은 반드시 한계 지적
- **마지막 단락 전환 표현**: "Motivated by the studies discussed above, we..."  또는 "These problems and challenges motivate us to..."
- **들어가지 않는 것**: 본 논문 수식, 성능 비교 수치

#### System Model (= Use Case + System Description)
- **첫 문장**: "We consider a [네트워크 타입] consisting of [구성요소]..."
- **구조**: A. Network Model → B. Communication Model → C. Computation Model → D. (필요시 추가)
- **변수 도입**: `Let $x$ denote ...` 또는 수식 뒤 `where $x$ denotes ...`
- **들어가지 않는 것**: Optimization problem 정의, 알고리즘 세부사항

#### Problem Formulation
- **구조**: Utility/Cost 모델 정의 → 최적화 문제 공식화 (min/max) → Constraint 설명
- **제약 조건 설명**: "Constraint C1 ensures that...", "The constraint in (X) indicates..."
- **들어가지 않는 것**: 알고리즘 해법, 증명

#### Proposed Algorithm / Method
- **구조**: 문제 분해 → 서브문제별 해법 → 알고리즘 pseudocode → 복잡도 분석
- **Theorem/Lemma**: 존재하면 Proof 포함 (□ 또는 ■ 마무리)

#### Performance Evaluation
- **A. Simulation Setup**: 환경 → 파라미터 테이블 → Benchmark 방법 정의
- **B. Simulation Results**: Fig 순서대로 분석
- **들어가지 않는 것**: 새로운 방법 제안, 수식 도출

#### Conclusion
- 과거형 + 현재형 혼용 ("we proposed", "results show")
- 미래 연구: "In future work, we plan to..." 또는 "Future work will..."
- **들어가지 않는 것**: 새로운 기여 사항, 수식

---

## 2. 문장 및 표현 패턴 (Sentence & Expression Patterns)

### 2-1. 도입부 표현
```
"With the rapid development of [기술], [분야] has emerged as a key technology for..."
"Owing to their [장점], [기술]s have attracted significant interest for..."
"has emerged as a promising solution"
"[기술] has been widely [recognized/adopted] as [역할]"
```

### 2-2. 문제 제기 패턴 (3단 구조)
```
[문제 상황]. This causes [결과]. Consequently, [심화된 문제].
```
실제 예시: "This causes edge servers to process computation tasks offloaded from other devices as well as UAVs. Consequently, UAV tasks offloaded to such an overloaded edge server can suffer a queuing delay."

### 2-3. 역접 전환
- **Nevertheless**: 앞 문장과 대조가 강할 때 (장점을 인정한 뒤 단점)
- **However**: 일반적 역접, 한계 지적

### 2-4. 기존 연구 나열 패턴
```
[Author] et al. [X] proposed/investigated/studied [내용].
[Author] et al. [Y] designed/presented [내용].
However, [these studies / none of these works] did not consider [한계].
```

### 2-5. 제안 방법 도입
```
"Therefore, in this paper, we propose a [이름] [방법/프레임워크/메커니즘] that [목적]."
"Specifically, we [세부 내용]."
"To address this, we formulate [문제] to [목적], and solve it using [해법]."
```

### 2-6. Related Work 마무리 전환 (필수)
```
"Motivated by the studies discussed above, we propose..."
"Motivated by the previous studies discussed above, we investigate..."
"These problems and challenges motivate us to design..."
"This motivates us to [방향]."
```

### 2-7. 결과 분석 패턴
```
"Fig. X shows/illustrates [내용]."
"We observe that [결과]."  또는  "It can be observed that [결과]."
"[수치]% [개선] compared with/to [기준선]"
"This is because [이유]."
"In contrast, [반대 결과]."
```

### 2-8. 인과관계 표현
```
"This is because [이유]."   (결과 → 이유, 가장 빈번)
"because [원인] causes [결과]"
"due to [원인]"
"As [원인] increases, [결과] also increases for all schemes due to..."
"This [결과] occurred because [원인]."
```

### 2-9. 성능 비교 표현
```
"outperforms [기준선] in terms of [지표]"
"achieves [X]% [reduction/improvement] compared with [기준선]"
"reduces [지표] by [X]%, [Y]%, and [Z]% compared with A, B, and C, respectively"
"The proposed scheme achieves the lowest [지표] while [부가 조건]."
```

---

## 3. 수식 표기 규칙 (Mathematical Notation)

### 3-1. 집합 (Sets)
- **이탤릭 대문자**: $\mathcal{N}$, $\mathcal{S}$, $\mathcal{M}$, $\mathcal{K}$, $\mathcal{I}$
- **또는 단순 대문자**: $M$, $K$, $I$, $D$ (집합이면서 원소 수로도 쓸 때 $|M|$)
- 원소: $m \in \mathcal{M}$, $k \in \mathcal{K}$

### 3-2. 벡터 / 행렬
- **벡터**: bold 소문자 ($\mathbf{w}$, $\mathbf{q}$, $\mathbf{v}$)
- **행렬**: bold 대문자 ($\mathbf{W}$, $\mathbf{H}$)
- **스칼라**: 이탤릭 소문자 ($x$, $f$, $p$, $\delta$)

### 3-3. 인덱스 변수 규칙
| 인덱스 | 용도 |
|--------|------|
| $m$ | satellite, server |
| $k$ | vehicle, round (FL round) |
| $i$ | device, vehicle, node |
| $j$ | neighboring device, relay |
| $r$ | FL round |
| $n$ | task, model type |
| $t$ | time step |
| $l$ | layer (DNN) |

### 3-4. 주요 기호 관례
- **지연(delay)**: $\tau$ (상첨자로 종류 구분: $\tau^{\text{GSL}}$, $\tau^{\text{wait}}$, $\tau^{\text{agg}}$)
- **데이터 크기**: $|w|$, $|D|$, $S_m$ (비트 단위)
- **CPU 주파수**: $f$ (Hz 또는 cycles/sec)
- **전송률**: $R$ (Shannon 정리 기반, $B \log_2(1 + \text{SINR})$)
- **에너지**: $E$, $e$ (소문자 = 특정 태스크의 에너지, 대문자 = 총 에너지)
- **이진 변수**: $s_m \in \{0, 1\}$, $\eta_{i,j} \in \{0, 1\}$, $a_i \in \{0, 1\}$
- **연속 변수**: $x$, $\rho$, $\delta$ (비율/분율이면 $[0, 1]$ 범위)

### 3-5. 최적화 문제 형식 (필수 준수)
```latex
\min_{\{s_m^{(r)}\}} \quad \alpha F(\mathbf{w}) + (1-\alpha) \sum_{r} \tau^{(r)}
\text{s.t.} \quad \text{C1: } s_m^{(r)} \in \{0,1\}, \; \forall m, r
\quad\quad\quad \text{C2: } \sum_m s_m^{(r)} \leq |M|
```
- **목적함수**: `\min` / `\max` + 변수 하첨자
- **제약**: `\text{s.t.}` 후 C1:, C2:, ... 순서로 명명
- **전칭 기호**: $\forall m \in \mathcal{M}$, $\forall r \in \mathcal{R}$

### 3-6. 변수 정의 스타일
**첫 등장 시**: `Let $x$ denote [설명].`  
**수식 직후**: `where $x$ denotes [설명], $y$ represents [설명].`  
**재사용 시**: 재정의 불필요, 기존 기호 그대로 사용

---

## 4. 금지 사항 (Forbidden Patterns)

### 4-1. 금지 동사
| 금지 | 대체 |
|------|------|
| employ | use |
| incorporate | include / integrate |
| exploit | use / leverage |
| prohibitive | not tractable for |

### 4-2. 금지 표현
```
❌ "significantly improves" → ✅ "reduces ... by X%"
❌ "greatly enhances" → ✅ "achieves higher ... by X%"
❌ "The results show that our method achieves better performance."
✅ "The proposed scheme reduces the total delay by 17.1% compared to FP."
❌ "Using Eqs. X and Y, the result is obtained." (자명한 재서술)
❌ ", enabling parallel execution" (dangling participial phrase)
✅ ", which enables parallel execution"
❌ Forward reference (System Model에서 reward function 언급)
```

### 4-3. 문장 구조 금지
- `because`를 문두에 두지 않음 → `This is because ...` 형태
- 복수 개념을 한 문장에 억지로 넣는 복잡한 문장
- 한계 문장을 독립 단락으로 분리 (본문 단락 안에 포함)

---

## 5. 자주 쓰이는 관용구 (Fixed Phrases)

### 논문 전반
```
"has emerged as a promising solution"
"have been proposed to"
"to address this [problem/issue]"
"without the need for"
"with the aim of/to"
"in terms of [metric]"
"making it challenging to"
"noting that"
"as mentioned in Section [X]"
"as in [Ref]"
"referred to as"
```

### System Model
```
"We consider a [system] consisting of [elements]..."
"Let $x$ denote..."
"where $x$ denotes..."
"is given by" / "is expressed as" / "can be written as" / "is defined as"
"can be calculated as" / "is computed as"
"Using Eq. (X) and (Y), we obtain..."
"From (X), we derive..."
"Constraint CX ensures that..."
"is subject to"
"satisfying the constraint"
```

### Performance Evaluation
```
"We evaluate the performance of the proposed scheme in terms of [metric]..."
"Referring to [Refs], the remaining parameters are set as..."
"The simulation results demonstrate/show that"
"outperforms the benchmark [methods/schemes]"
"[X]% [reduction/improvement] on average compared with [A], [B], and [C], respectively"
"as shown in Fig. X" / "as depicted in Fig. X"
"This improvement is achieved because..."
"In contrast, [반대 결과]."
```

### Conclusion
```
"In this paper, we propose[d] a [방법], considering [요소]."
"We formulate[d] [문제] to [목적], and [해법]."
"Simulation results show that the proposed [방법] outperforms the benchmark schemes."
"In future work, we plan to..."  또는  "Future work will..."
```

---

## 6. 섹션별 시작 문장 패턴

| 섹션 | 전형적 첫 문장 |
|------|---------------|
| Introduction | "With the rapid development of..." 또는 "Owing to their..." |
| Related Work (섹션) | "In this section, we review..." (생략하기도 함, 바로 subsection으로) |
| System Model | "In this section, we [introduce/present/describe] the [use case and] system model." |
| 수식 정의 도입 | "For convenience, Table I lists the notations frequently used throughout..." |
| Problem Formulation | "In this section, we formulate the [문제] as [형식]." |
| Algorithm 섹션 | "In this section, we [present/develop/propose] the [알고리즘 이름]." |
| Performance Evaluation | "In this section, [we present/the] simulation results are presented to evaluate the performance of the proposed [방법]." |
| Conclusion | "In this paper, we proposed [방법] that [목적]." |

---

## 7. 수식 참조 방식

- 수식: `(X)` — "as in (3)", "from (9)", "using (12) and (13)"
- 그림: `Fig. X` — "as shown in Fig. 3(a)", "as depicted in Fig. 2"
- 표: `Table I`, `Table II` — "as summarized in Table I", "as listed in Table II"
- 알고리즘: `Algorithm 1`, `Algorithm 2`
- 섹션: `Section III`, `Section IV-B`

---

## 8. 약어 도입 규칙
- **첫 등장**: 풀네임 (약어) — "deep reinforcement learning (DRL)"
- **이후**: 약어만 사용
- **제목/Abstract**: 약어를 써도 되나 풀네임 병기 권장
- **벤치마크 방법**: 괄호로 약어 정의 후 본문에서 약어로 통일

---

## 9. 용어 일관성 원칙 (Terminology Consistency — CRITICAL)

### 9-1. 용어 선택 기준
- **반드시 실제 논문에서 사용된 용어만 사용** — AI가 창작하거나 일반화한 용어 절대 금지
- 새 용어를 쓰고 싶다면 해당 용어가 IEEE 논문에서 실제로 쓰이는지 확인
- 의심스러우면 더 일반적이고 단순한 표현으로 교체

금지 예시:
```
❌ "proactive routing" — 실제 논문에서 쓰는지 확인 전 사용 금지
❌ "topology-aware scheduler" — AI 창작 가능성 있으면 출처 확인
❌ "semantic congestion signal" — 정의 없이 사용 금지
```

### 9-2. 정의 없는 기호/용어 절대 금지
- 수식에 등장하는 모든 기호는 반드시 `Let $x$ denote...` 또는 `where $x$ denotes...`로 정의
- 모든 약어는 첫 등장 시 풀네임(약어) 형태로 도입
- 이전 섹션에서 정의된 기호도 섹션이 멀면 간략히 재언급
- 정의 전에 기호를 수식에 쓰는 것 절대 금지 (Forward reference 금지)

### 9-3. 형용사/부사 사용 원칙
- **문장 의미에 실질적으로 영향을 주지 않는 형용사/부사는 쓰지 않음**
- "significantly", "greatly", "effectively", "efficiently" 등: 수치 뒷받침 없으면 삭제
- "various", "diverse", "numerous": 구체적 나열이 가능하면 나열, 아니면 삭제
- "novel": 스스로 novelty를 주장하는 표현이므로 금지
- 허용: 의미를 바꾸는 형용사 ("binary variable", "non-convex problem", "long-term constraint")

```
❌ "We propose an efficient and novel framework that significantly reduces delay."
✅ "We propose a framework that reduces the total delay by 24.87% compared with FP."
```

### 9-4. 직관적 전달 원칙
- 해당 분야를 처음 접하는 리뷰어가 System Model을 읽을 때 의미가 바로 전달되어야 함
- 추상적 표현 대신 구체적 대상/동작으로 서술

```
❌ "The agent interacts with the environment to maximize the long-term reward."
✅ "Satellite m selects PSs to minimize the total FL delay across rounds r ∈ R."
```

### 9-5. System Model 내 정의 순서 원칙 (CRITICAL)
- 독자가 모델을 읽어 내려가면서 필요한 개념이 등장하는 순서로 정의
- 예: 전송률 $R$ 정의 → 전송 지연 $\tau^{\text{tx}}$ 정의 (역순 절대 금지)
- 최적화 변수는 Problem Formulation에서 처음 도입 (System Model에서 미리 쓰면 안 됨)
- 각 subsection은 이전 subsection의 결과를 자연스럽게 이어받아야 함

올바른 순서 예시 (V2V-FL 논문 기준):
```
A. Network Model       → 전체 구조, 기본 집합 정의
B. Computing Model     → 로컬 처리 시간, CPU 주파수 (A에서 정의한 노드 사용)
C. Communication Model → 전송률, 채널 모델 (B의 노드 쌍으로 전송 계산)
D. Mobility Model      → 체류 시간, V2V 범위 (C의 링크가 언제 유효한지)
E. FL Model            → loss function, 집계 방식 (B,C,D 모두 필요)
```

---

## 10. 인용 스타일
- `[X]` 형태 (숫자)
- 범위: `[1]–[5]`
- 복수: `[3], [8], [10]`
- 문장 내 인용: "Chen et al. [11] proposed..."
- 방법 채용 시: "as in [X]", "referring to [X]", "based on [X]"
