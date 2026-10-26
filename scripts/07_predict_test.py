# 이 스크립트가 하는 일:
# 학습된 MACE 모델로 test.xyz(4,101개 구조)의 에너지·힘을 예측한다.
# sample_submission.csv와 동일한 형식(ID, energy, force)으로 결과 CSV를 만든다.

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator

# 프로젝트 루트를 import 경로에 추가해 tools/ 모듈을 쓸 수 있게 한다
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.submission_format import format_force, assert_no_ellipsis

TEST_PATH = "data/processed/test.xyz"
SUBMISSION_TEMPLATE = "data/raw/sample_submission.csv"
DEVICE = "cpu"  # ScaleShiftMACE의 float64 하드캐스팅이 MPS에서 막혀 있어 CPU로 추론

# 모델과 출력 경로는 인자로 받는다. Kaggle/Colab에서 받은 모델로 바꿔 끼울 때
# 파일을 직접 수정하지 않아도 되고, 기존 제출 파일을 실수로 덮어쓰지 않는다.
parser = argparse.ArgumentParser(description="학습된 MACE 모델로 test 예측 CSV를 만든다")
parser.add_argument("--model", default="models/mace_v1.model", help="사용할 .model 파일")
parser.add_argument("--out", required=True, help="저장할 제출 CSV 경로")
args = parser.parse_args()

MODEL_PATH = args.model
OUTPUT_PATH = args.out

if Path(OUTPUT_PATH).exists():
    raise SystemExit(f"이미 존재하는 파일이다. 덮어쓰지 않는다: {OUTPUT_PATH}")

frames = read(TEST_PATH, ":")
print(f"test 구조 {len(frames)}개 로드")

calc = MACECalculator(model_paths=MODEL_PATH, device=DEVICE, default_dtype="float32")

ids, energies, forces_str = [], [], []
template_ids = pd.read_csv(SUBMISSION_TEMPLATE)["ID"].tolist()

for i, atoms in enumerate(frames):
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    ids.append(template_ids[i])
    energies.append(energy)
    # str(forces)는 원소가 1000개를 넘으면 가운데를 '...'로 잘라버린다.
    # format_force()는 threshold=inf라 크기와 무관하게 항상 전체를 출력한다.
    forces_str.append(format_force(forces))

    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(frames)} 완료")

submission = pd.DataFrame({"ID": ids, "energy": energies, "force": forces_str})

# 저장 직전 안전장치: 생략(...)된 행이 하나라도 있으면 여기서 멈춘다
assert_no_ellipsis(submission)

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\n저장 완료: {OUTPUT_PATH} ({len(submission)}행)")
