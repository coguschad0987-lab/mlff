# CLAUDE.md — MLFF 프로젝트 작업 규칙

이 파일은 Claude Code가 이 저장소에서 작업할 때 항상 먼저 읽는 지침이다.

---

## 1. 프로젝트 개요

**목표**: 2023 Samsung AI Challenge (SAIT 주최, DACON 운영)의 "반도체 소재 시뮬레이션을 위한 Machine Learning Force Field 알고리즘 개발" 대회를 개인 학습용으로 재현한다.

**최종 산출물**:
- (ext)xyz 원자구조 → 에너지(eV) + 힘(eV/Å)을 예측하는 MLFF 모델
- 학습/검증 스크립트, 평가 리포트(RMSE, parity plot)
- ASE 기반 MD 시뮬레이션으로 물리적 타당성 정성 평가
- 테스트셋 예측 파일 생성 파이프라인

**데이터**: SAMD23 (SAIT 공개 반도체 벤치마크 — SiN, HfO₂)
- 참조 저장소: https://github.com/SAITPublic/MLFF-Framework
- 논문: NeurIPS 2023 Datasets & Benchmarks, "Benchmark of MLFF for Semiconductor Simulations"

**핵심 접근**: 밑바닥부터 학습하지 않는다. MACE 파운데이션 모델(mace-mp)을 zero-shot → fine-tuning 하는 경로를 기본으로 한다.

---

## 2. 사용자 프로필 (매우 중요)

- 신소재공학 전공자. **DFT, MD, 포텐셜 에너지 표면, 결정구조 등 물리·소재 도메인 지식은 충분하다.**
- **비개발자다.** 파이썬/터미널/git에 익숙하지 않다.
- 따라서 Claude Code는 다음을 지킨다:
  - **명령어를 실행하기 전에 "이 명령이 무엇을 하는지" 한국어로 한두 문장 설명**한다.
  - 코드를 작성하면, 각 블록이 무슨 역할인지 주석 또는 설명을 붙인다.
  - 물리/화학 개념은 설명을 생략해도 된다. 프로그래밍 개념은 설명한다.
  - 에러가 나면 원문 로그를 그대로 붙여넣지 말고, **원인 → 해결책 → 실행할 명령** 순서로 정리해 준다.
  - 전문 용어 남발 금지. 단, 도메인 용어(RMSE, cutoff, NVT 등)는 그대로 써도 된다.

---

## 3. 하드웨어 / 환경 제약 (반드시 준수)

**기기**: MacBook Air M5, 16GB 통합 메모리, macOS, Apple Silicon (arm64)

| 항목 | 상태 |
|---|---|
| CUDA GPU | **없음** |
| PyTorch MPS (Apple GPU) | 사용 가능 |
| float64 on MPS | **불가능** — MPS는 float64 미지원 |
| torch-scatter / PyG | Apple Silicon 사전빌드 wheel 없음 → 소스 컴파일 필요, **가급적 회피** |
| 메모리 | CPU/GPU가 16GB를 공유. 스왑 발생 시 극심하게 느려짐 |

### 반드시 지킬 규칙
1. **학습 시**: `--device=mps --default_dtype=float32`
2. **정밀 지오메트리 최적화 / 포논 계산**: `--device=cpu`, float64
3. `torch-scatter`가 필요한 패키지(NequIP 등)는 설치 시도하지 말 것. **MACE와 CHGNet을 우선 사용**한다 (둘 다 torch-scatter 불필요).
4. 연산자 미구현 에러가 나면 터미널에서 `export PYTORCH_ENABLE_MPS_FALLBACK=1` 설정 후 재실행 (파이썬 실행 **전**에 설정해야 함).
5. **한 번의 학습 실행이 30분을 넘길 것 같으면 먼저 사용자에게 알리고 확인받는다.** 노트북 발열/배터리 문제.
6. 메모리 부족(OOM)이 나면 순서대로 낮춘다: `batch_size` → `num_channels` → `max_L` → `r_max`

---

## 4. 기술 스택

```
Python 3.11
PyTorch (MPS 백엔드)
mace-torch          # 주력 모델
ase                 # 원자 구조 입출력, MD, calculator
numpy, matplotlib   # 분석·시각화
pandas              # 결과 정리
```

선택적으로 추가 가능: `chgnet` (MPS 친화적, 비교군), `pymatgen` (구조 조작)

**설치 시 주의**: PyTorch는 Mac에서 CUDA 인덱스 URL 없이 그냥 `pip install torch` 로 설치한다.

---

## 5. 디렉토리 구조

```
mlff-project/
├── CLAUDE.md              # 이 파일
├── PLAN.md                # 단계별 실행 계획 (진행상황 체크)
├── README.md              # 프로젝트 요약 (마지막에 작성)
├── .venv/                 # 가상환경 (git 제외)
├── data/
│   ├── raw/               # 다운로드 원본 (건드리지 않음)
│   ├── processed/         # train.xyz / valid.xyz / test.xyz
│   └── DATA_NOTES.md      # 데이터 출처·단위·구조 메모
├── configs/               # 학습 설정 파일 (yaml 또는 sh)
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_split_data.py
│   ├── 03_zeroshot_eval.py
│   ├── 04_train.sh
│   ├── 05_evaluate.py
│   ├── 06_run_md.py
│   └── 07_predict_test.py
├── models/                # 학습된 체크포인트 (git 제외)
├── results/
│   ├── figures/           # parity plot, 학습곡선, RDF
│   └── logs/              # 학습 로그
└── notes/                 # 실험 기록
```

---

## 6. 작업 규칙

### 진행 방식
- **PLAN.md의 순서를 따른다.** 단계를 건너뛰지 않는다.
- 한 번에 **하나의 Phase만** 진행한다. 끝나면 결과를 보고하고 다음으로 넘어갈지 물어본다.
- Phase가 끝나면 PLAN.md의 체크박스를 업데이트한다.

### 코드 작성
- 스크립트는 **짧고 하나의 일만** 하게 만든다. 거대한 만능 스크립트 금지.
- 모든 스크립트 맨 위에 "이 스크립트가 하는 일" 주석을 3줄 이내로 쓴다.
- 하드코딩된 경로 대신 상단에 상수로 모아둔다.
- 랜덤 시드는 항상 고정한다 (`seed=42`).

### 실험 기록
- 학습을 돌릴 때마다 `notes/experiments.md`에 한 줄 추가:
  `날짜 | 모델설정 | train/valid 크기 | energy RMSE | force RMSE | 비고`
- 이게 나중에 "뭘 바꿨더니 좋아졌는지" 추적하는 유일한 근거다.

### 데이터 취급
- `data/raw/`는 **읽기 전용처럼** 다룬다. 절대 덮어쓰지 않는다.
- 데이터를 다운로드하기 전에 **먼저 URL이 실제로 유효한지 확인**하고, 용량을 사용자에게 알린다.
- train/valid/test 분할은 **궤적(trajectory) 단위**로 한다. 같은 MD 궤적의 인접 프레임이 train과 test에 나뉘어 들어가면 성능이 과대평가된다.

### 하지 말 것
- 사용자 확인 없이 `pip install` 대량 실행
- 사용자 확인 없이 1시간 이상 걸리는 학습 시작
- 잘 안 되는 것을 "성공했다"고 넘어가기 — 실패는 실패라고 명확히 보고
- 임의로 다른 모델/프레임워크로 갈아타기 (먼저 제안하고 승인받는다)

---

## 7. 도메인 규칙 (틀리면 조용히 망하는 것들)

| 항목 | 규칙 |
|---|---|
| 에너지 단위 | **eV** (구조 전체 기준). 비교할 때는 원자당 에너지 meV/atom으로 환산 |
| 힘 단위 | **eV/Å** |
| E0s (고립원자 기준에너지) | 반드시 설정. 모르면 `--E0s="average"` 로 시작. 미설정 시 에너지 학습이 통째로 망가진다 |
| 힘의 정의 | F = −∂E/∂r. autograd로 얻는다 (conservative). 힘을 직접 예측하는 방식은 이 프로젝트에서 쓰지 않는다 |
| xyz 키 이름 | `atoms.info['energy']`, `atoms.arrays['forces']` — MACE 학습 시 `--energy_key`, `--forces_key`로 실제 키와 맞춰야 함 |
| 주기경계조건(PBC) | 반도체 결정 구조이므로 PBC가 켜져 있어야 정상. 데이터 로드 후 `atoms.pbc` 확인 |
| 평가 지표 | 에너지 RMSE + 힘 RMSE (대회 방식). 여기에 MD 안정성 정성평가 추가 |

---

## 8. 자주 쓰는 명령어

```bash
# 가상환경 활성화
source .venv/bin/activate

# MPS 사용 가능 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# MPS 폴백 (미구현 연산 대비)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 데이터 빠른 확인
python -c "from ase.io import read; f=read('data/processed/train.xyz', ':'); print(len(f), f[0])"
```

MACE 학습 기본형 (노트북용 경량 설정):
```bash
mace_run_train \
  --name="mace_v1" \
  --train_file="data/processed/train.xyz" \
  --valid_file="data/processed/valid.xyz" \
  --E0s="average" \
  --model="MACE" \
  --num_channels=128 \
  --max_L=0 \
  --r_max=5.0 \
  --batch_size=4 \
  --max_num_epochs=50 \
  --ema --ema_decay=0.99 \
  --energy_weight=1 --forces_weight=100 \
  --default_dtype=float32 \
  --device=mps \
  --seed=42
```

---

## 9. 알려진 함정 체크리스트

작업 중 문제가 생기면 이 목록부터 확인한다.

- [ ] `Cannot convert a MPS Tensor to float64` → `--default_dtype=float32` 로 변경
- [ ] 에너지 예측이 터무니없이 큰 값 → E0s 미설정 또는 잘못 설정
- [ ] 힘 RMSE는 좋은데 에너지 RMSE가 나쁨 → `--energy_weight` 상향, 학습 후반 stage-two 사용
- [ ] valid 성능이 비현실적으로 좋음 → 데이터 누수 의심. 궤적 단위 분할 확인
- [ ] MD 돌리면 원자가 폭주/구조 붕괴 → 모델 정확도가 아니라 **학습 데이터 커버리지** 문제일 가능성이 큼
- [ ] 학습이 극단적으로 느림 → 스왑 발생 여부 확인, batch_size 축소
- [ ] `torch_scatter` 설치 실패 → 그 패키지를 쓰지 않는다. MACE로 우회

---

## 10. 막혔을 때 참고

- MACE 공식 문서: https://mace-docs.readthedocs.io
- MACE GitHub / Discussions: https://github.com/ACEsuit/mace
- ASE 문서: https://ase-lib.org
- SAIT MLFF-Framework: https://github.com/SAITPublic/MLFF-Framework
- 커뮤니티: https://matsci.org
