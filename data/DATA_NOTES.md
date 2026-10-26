# DATA_NOTES.md — 데이터 출처·단위·구조 메모

## 출처
- **CLAUDE.md 1절의 전제(SAITPublic/MLFF-Framework에서 직접 다운로드)와 다름.**
  사용자가 이미 2023 Samsung AI Challenge(DACON 운영) 대회의 실제 train/test 분할 파일을 보유하고 있어 이를 그대로 사용한다.
- 원본 위치: `/Users/immack2/Desktop/open/{train.xyz, test.xyz, sample_submission.csv}`
- 프로젝트 내 복사본: `data/raw/` (읽기 전용으로 취급, 수정 금지)
- 다운로드 일자: 2023-08-18 (파일 mtime 기준, 대회 당시 배포본으로 추정)
- 용량: train.xyz 139MB, test.xyz 32MB, sample_submission.csv 3.6MB

## 구조 정보
- 물질계: **SiN 계열만** (HfO₂ 없음 — CLAUDE.md 1절 설명과 달리 이 배포본에는 SiN만 포함)
- 원소: N, Si
- train.xyz: 구조(프레임) 22,510개, 원자 수 16~510 (평균 57.1)
- test.xyz: 구조(프레임) 4,101개, 원자 수 16~510 (평균 72.0)
- PBC: 전체 구조 True/True/True 확인됨

## 단위 및 값 범위 (train.xyz 기준)
- 에너지: eV (구조 전체 기준). 범위 -3011.50 ~ -69.60 eV
  - 원자당 환산 시 -8.2646 ~ -4.3111 eV/atom (물리적으로 타당한 범위)
- 힘: eV/Å. 관측된 최대 성분값 277.51 eV/Å
  - 매우 큰 값이지만 평형에서 먼 구조(고에너지 샘플)가 포함된 벤치마크 데이터셋의 특성일 수 있음 → 이상치 여부는 학습 단계에서 재확인
- xyz 파일 자체의 comment 라인 키는 `energy=`, `free_energy=`, `Properties=...forces:R:3`
  - **주의**: ASE로 읽으면 이 값들은 `atoms.info`/`atoms.arrays`가 아니라 `atoms.calc`(SinglePointCalculator)에 들어간다.
    `atoms.get_potential_energy()`, `atoms.get_forces()`로 접근해야 함 (CLAUDE.md 7절의 설명은 원본 파일의 키 이름 기준이고, ASE 객체 접근 방식은 이 방식이 맞음 — MACE의 `--energy_key=energy --forces_key=forces`도 내부적으로 이 comment 라인 키를 그대로 사용)

## 분할 (Phase 2에서 진행 예정)
- train/valid/test 분할 기준: 궤적(trajectory) 단위 — 단, 이 xyz 파일에는 명시적 궤적 ID가 없어 Phase 2에서 궤적을 어떻게 식별할지 별도로 정해야 함
- test.xyz는 이미 대회가 정한 홀드아웃 세트이므로 건드리지 않음. train.xyz만 train/valid로 추가 분할

## 제출 형식 (sample_submission.csv)
- 컬럼: `ID`, `energy`, `force`
- 행 개수: 4,101 (test.xyz 구조 1개당 1행)
- `energy`: 구조 전체 에너지(eV) 스칼라
- `force`: `str(numpy_array)` 형태의 문자열 (예: `"[[0. 0. 0.]\n [0. 0. 0.] ...]"`, shape=(원자 수, 3))
  → 07_predict_test.py에서 `str(forces_array)`로 그대로 생성하면 형식이 맞음
