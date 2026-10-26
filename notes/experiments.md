# experiments.md — 실험 기록

학습을 돌릴 때마다 한 줄씩 추가한다.

---

## 논문 근거 정리 (NeurIPS 2023, SAIT MLFF Benchmark)

실험 설계의 기준점. 추측이 아니라 논문에 명시된 내용만 적는다.

### 평가 지표
- **EF metric = 원자당 에너지 RMSE(meV/atom) + axis-wise 힘 RMSE(meV/Å)** — 단순 합산
- 우리가 쓰는 계산 방식과 동일함 (논문 Table C.1)

### MACE 벤치마크 성적 (SiN, V100 GPU, 200 epoch)
| loss | Train | **Test** | OOD |
|---|---|---|---|
| MSE-based | 186.7 | **190.3** | 252.0 |
| MAE-based | 147.1 | **155.1** | 217.7 |
→ SiN에서는 MAE-based가 명확히 우수 ("substantial superiority of the MAE-based loss")

### 논문의 MACE 학습 설정
- optimizer Adam, **lr = 0.01**, batch_size 16, weight decay 5e-7, ema_decay 0.99
- **200 epoch**, ReduceLROnPlateau 스케줄러 사용
- cutoff 6Å, 이웃 원자 최대 50개로 제한

### 힘 가중치(λf) 탐색 결과 — 논문 4.2.2절, Figure 2
MACE와 GemNet-dT를 λf = 0.1 / 0.3 / 1 / 3 / 10으로 실제 학습해 비교함.
- "force errors are **dominant** in the terms of EF metrics" — 힘이 EF의 대부분을 차지
- **λf < 1 → 오차 증가** (실험으로 확인됨)
- λf = 10 → 항상 좋아지지는 않으며, λf > 10은 더 나쁠 수 있음
- **결론: `1 ≤ λf < 10` 구간 권장**

> **중요**: MACE 라이브러리 기본값 `forces_weight=100`은 논문이 시험한 최대치(10)의 10배로,
> **검증 범위 밖의 값**이다. v1까지 이 값을 썼던 것이 힘 발산의 원인 후보.

---

## 실험 기록

| 날짜 | 모델설정 | train/valid 크기 | energy RMSE | force RMSE | 비고 |
|---|---|---|---|---|---|
| 2026-08-29 | mace-mp small, fine-tuning 없음 (zero-shot) | -/4518 | 67.16 meV/atom (4.8467 eV) | 0.5488 eV/Å | Phase 3 baseline. CPU로 실행 (float64 하드캐스팅 이슈로 MPS 불가) |
| 2026-08-29 | mace_v1: mace-mp small fine-tuning, 1 epoch, lr=0.01(기본값), batch_size=4, multiheads_finetuning=False | 17992/4518 | 53.7 meV/atom (개선) | **2841.4 meV/Å (5배 악화)**, relative F RMSE 144% | **실패로 판단.** 에너지는 좋아졌지만 힘 오차가 zero-shot보다 훨씬 나빠짐. 원인은 epoch 부족(논문은 200 epoch)으로 추정 |
| 2026-08-29 | **A/B 실험 A**: loss=weighted(기본), 5 epoch, batch=4, lr=0.01 | 1997/801 (서브셋) | **68.2 meV/atom** | **344.5 meV/Å** | EF=413. epoch 2에서 최저(344), epoch 4에 565로 급등 → lr 불안정 신호 |
| 2026-08-29 | **A/B 실험 B**: loss=huber(delta=0.01), 나머지 A와 동일 | 1997/801 (서브셋) | 87.8 meV/atom | 578.9 meV/Å | EF=667. **5 epoch 구간에서는 A에 완패.** 단 huber는 초반 수렴이 느린 특성이 있어 장기 학습에서 역전 가능성 있음(논문: MAE-based가 200 epoch 후 더 우수) |
| 2026-08-29 | **Colab v1**: T4 GPU, loss=huber(delta=0.01), lr=0.01, forces_weight=100, batch=16, 50 epoch 목표 → **33 epoch에서 중단** | 17992/4518 | 최저 16.3, 대체로 17~20 (안정) | epoch14 **337.1 최저** → epoch28 **1187.6 발산** | **실패.** 1 epoch=9분55초. 에너지는 전 구간 우수·안정. 힘이 epoch14 이후 발산. 원인: huber_delta=0.01이 너무 작아 오차가 0.01 초과 시 벌점이 포화 → 모델이 어려운 구조를 포기. huber loss는 계속 감소하는데 RMSE는 폭발하는 괴리 발생. **추가 교훈: --keep_checkpoints 미사용으로 epoch14 최고 모델이 덮어쓰기 삭제됨** (MACE는 valid loss 개선 시에만 저장하며 이전 것을 지움) |
| 2026-08-30 | **Colab v2** (진행 예정): loss=weighted, lr=0.005, **forces_weight=1**, patience=10, scheduler_patience=3, max 40 epoch, keep_checkpoints | 17992/4518 | | | v1의 알려진 문제를 한꺼번에 수정. **변수를 4개 동시에 바꿨으므로 어느 것이 결정적이었는지는 구분 불가** |

---

## 현재 최고 기록

| 항목 | 값 | 출처 |
|---|---|---|
| 에너지 RMSE | 19.5 meV/atom | Colab v1, epoch 14 |
| 힘 RMSE | 337.1 meV/Å | Colab v1, epoch 14 |
| **EF metric** | **356.6** | (모델 파일은 삭제되어 없음) |
| 목표 | **155~190** | 논문 MACE |

힘이 EF의 95%를 차지하므로, **개선 여지는 사실상 힘 하나에 있다.**

---

## 다음 실험 계획 (v2 결과를 본 뒤 하나씩)

우선순위 순. **한 번에 하나씩만 바꾼다.**

### 1순위 — 힘 가중치 λf 탐색
논문 권장 구간 `1 ≤ λf < 10` 안에서 비교. v2가 λf=1이므로:
- [ ] λf = 3 (권장 구간 중앙)
- [ ] λf = 5

v2에서 힘 오차가 나빠졌다면 올리고, 안정적이라면 유지.

### 2순위 — MAE 계열 loss 재도전
논문에서 SiN에 대해 MAE-based가 MSE-based보다 명확히 우수했다 (155.1 vs 190.3).
v1의 huber 실패는 loss 종류가 아니라 **delta 값(0.01)이 너무 작았던 것**이 원인일 수 있다.
- [ ] `--loss=huber --huber_delta=0.1`
- [ ] `--loss=huber --huber_delta=1.0`
- [ ] `--loss=l1l2energyforces` (MAE 계열 대안)

판단 기준: loss는 내려가는데 RMSE가 오르는 괴리가 다시 생기는지 확인.

### 3순위 — Stage Two (SWA)
`--swa` 옵션. 학습 막바지에 학습률을 낮추고 에너지 가중치를 올려 마무리하는 기능.
논문도 언급하며, MACE 공식 권장 사항.
- [ ] v2 설정 + `--swa --start_swa=<전체의 80% 지점>`

### 4순위 — 모델 크기
- [ ] `--foundation_model="medium"` (L=1, 방향 정보 포함 → 힘 예측에 유리할 가능성)
- 주의: 학습 시간과 메모리가 크게 늘어남. T4 16GB에서 batch_size 조정 필요할 수 있음

### 5순위 — 극단 힘 구조 처리
train.xyz의 0.79%가 힘 20 eV/Å 초과, 0.06%가 100 초과.
- [ ] 상위 0.1%(힘 73 초과) 구조를 train에서만 제외하고 비교
- **주의**: valid/test에도 같은 비율로 존재하므로 제거가 오히려 손해일 수 있음.
  loss 조정으로 해결되지 않을 때만 시도.

### 보류 — 이미 검증했거나 근거가 없는 것
- ~~forces_weight=100~~ : 논문 검증 범위(10) 밖. v1 실패 원인 후보. 되돌리지 않는다.
- ~~lr=0.05 이상~~ : 0.01에서 이미 발산. 더 올릴 근거 없음.
- ~~MPS(애플 GPU)~~ : CPU보다 느림이 실측됨. Colab 사용 중이므로 무의미.
