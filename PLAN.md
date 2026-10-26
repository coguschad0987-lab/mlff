# PLAN.md — 단계별 실행 계획

CLAUDE.md의 규칙에 따라 한 번에 하나의 Phase만 진행한다. Phase가 끝나면 체크박스를 갱신하고 사용자에게 결과를 보고한 뒤 다음 단계로 넘어갈지 확인받는다.

---

## Phase 0. 환경 설정 ✅ (2026-08-28 완료)
- [x] `.venv` 가상환경 생성 및 활성화 (Homebrew로 Python 3.11 설치 후 생성)
- [x] Python 3.11 확인 — 3.11.16
- [x] PyTorch 설치 (`pip install torch`, CUDA 인덱스 URL 없이) — 2.13.0
- [x] `python -c "import torch; print(torch.backends.mps.is_available())"` 로 MPS 확인 — True
- [x] `mace-torch`, `ase`, `numpy`, `matplotlib`, `pandas` 설치 — 모두 정상 임포트 확인, `requirements.txt`로 고정

## Phase 1. 데이터 확보 및 탐색 — `scripts/01_explore_data.py` ✅ (2026-08-29 완료)
- [x] ~~SAMD23 다운로드~~ → 사용자가 DACON 대회 실제 데이터 보유 (`/Users/immack2/Desktop/open/`), `data/raw/`로 복사
- [x] 구조 개수, 원자 수 분포, PBC 여부 확인 — train 22,510개(16~510 atoms), test 4,101개, 원소 N/Si만, PBC 전체 True
- [x] 에너지/힘 키 및 접근 방식 확인 — `atoms.get_potential_energy()` / `atoms.get_forces()` (SinglePointCalculator 경유)
- [x] `data/DATA_NOTES.md`에 출처·단위·구조·제출형식 메모 작성 (SiN만 포함, HfO₂ 없음을 명시)

## Phase 2. 데이터 분할 — `scripts/02_split_data.py` ✅ (2026-08-29 완료)
- [x] train.xyz를 33개 블록(같은 원자 수 연속 구간)으로 나누고, 블록 내부를 청크(연속 프레임) 단위로 쪼개 train/valid에 배정 (seed=42, chunk 경계 유지로 누수 방지)
- [x] `data/processed/train.xyz`(17,992개), `valid.xyz`(4,518개) 생성, `test.xyz`(4,101개)는 그대로 복사
- [x] 시스템 크기(원자 수) 20종 전부 train/valid 양쪽에 존재 확인, valid 비율 20.1%

## Phase 3. Zero-shot 평가 — `scripts/03_zeroshot_eval.py` ✅ (2026-08-29 완료)
- [x] mace-mp small 파운데이션 모델을 fine-tuning 없이 valid.xyz(4,518개)에 그대로 평가
  - 체크포인트가 float64 하드캐스팅을 포함해 MPS 실행 불가 → CPU로 실행 (1회성 추론이라 수 분 내 완료)
- [x] energy RMSE 67.16 meV/atom, force RMSE 0.5488 eV/Å 기록 (`notes/experiments.md`) — fine-tuning 대비 baseline

## Phase 4. Fine-tuning 학습 — `scripts/04_train.sh`
- [x] MPS 학습 가능 여부 확인 → mace-torch 0.3.16의 `ScaleShiftMACE.forward()`에 하드코딩된 `.double()` 캐스팅 때문에
  **모든** mace_run_train 학습(scratch/fine-tuning 무관)이 `--device=mps`에서 크래시함을 확인.
  `scripts/patch_mace_mps.py`로 로컬 mace-torch 패키지를 1줄 패치(float64 캐스팅 제거)해서 해결.
  **주의**: venv를 새로 만들거나 mace-torch를 재설치하면 `python scripts/patch_mace_mps.py`를 다시 실행해야 함.
- [x] MPS/CPU 벤치마크 결과 CPU가 더 빠름을 확인 → `--device=cpu`로 설정
- [x] E0s 설정 (`--E0s="average"`)
- [x] 30분 이상 소요 예상됨을 사전에 알리고 확인받음 (1 epoch, 전체 데이터, 약 40분 예상 → 실제 약 66분 소요)
- [x] `notes/experiments.md`에 실행 기록 추가 (mace_v1)
- [ ] **재시도 필요**: 1차 시도(lr=0.01 기본값)는 energy RMSE 53.7 meV/atom으로 개선됐지만 force RMSE가 2841 meV/Å로 zero-shot(548.8) 대비 5배 악화 → 실패로 판단. learning rate를 낮춰(예: 0.001 이하) 재학습 필요

## Phase 5. 평가 — `scripts/05_evaluate.py`
- [ ] energy RMSE (meV/atom 환산 포함), force RMSE 계산
- [ ] parity plot 생성 → `results/figures/`

## Phase 6. MD 시뮬레이션 — `scripts/06_run_md.py`
- [ ] ASE 기반 NVT MD 실행 (정밀 계산은 `--device=cpu`, float64)
- [ ] 구조 붕괴/원자 폭주 여부로 물리적 타당성 정성 평가
- [ ] RDF 등 결과 → `results/figures/`

## Phase 7. 테스트셋 예측 — `scripts/07_predict_test.py`
- [ ] 최종 모델로 test.xyz(4,101개 구조)의 에너지·힘 예측
- [ ] `sample_submission.csv`와 동일한 형식(ID, energy, force)으로 결과 CSV 생성
  - `force` 컬럼은 `str(forces_array)` 형태의 문자열로 저장

## Phase 8. 정리
- [ ] `README.md` 작성 (프로젝트 요약, 최종 결과)
