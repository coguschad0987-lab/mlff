"""손상된 특정 구조만 골라 다시 예측하고 preds/<ID>.npy 로 저장한다.

전체 4101개를 다시 돌리지 않고 필요한 것만 계산해 시간을 아낀다.
모델과 입력이 같으므로 결과는 원래 예측과 동일하다.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_PATH = ROOT / "data/processed/test.xyz"
TEMPLATE = ROOT / "data/raw/sample_submission.csv"
MODEL_PATH = ROOT / "models/mace_v1.model"
OUT_DIR = ROOT / "preds"

target_ids = sys.argv[1:]
if not target_ids:
    print("사용법: python tools/repredict_ids.py TEST_1543 TEST_1544 ...")
    sys.exit(2)

template_ids = pd.read_csv(TEMPLATE)["ID"].astype(str).tolist()
pos = {sid: i for i, sid in enumerate(template_ids)}
missing = [s for s in target_ids if s not in pos]
if missing:
    raise SystemExit(f"템플릿에 없는 ID: {missing}")

# 필요한 구조만 xyz에서 읽어온다 (인덱스 순서대로)
wanted = sorted(target_ids, key=lambda s: pos[s])
idx_list = [pos[s] for s in wanted]
print(f"재예측 대상 {len(wanted)}개 (test.xyz 인덱스 {idx_list[0]}~{idx_list[-1]})")

frames = read(str(TEST_PATH), ":")
calc = MACECalculator(model_paths=str(MODEL_PATH), device="cpu", default_dtype="float32")

OUT_DIR.mkdir(exist_ok=True)
energies = {}
t0 = time.time()
for k, (sid, idx) in enumerate(zip(wanted, idx_list), 1):
    atoms = frames[idx]
    atoms.calc = calc
    energies[sid] = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    np.save(OUT_DIR / f"{sid}.npy", forces)
    print(f"  [{k}/{len(wanted)}] {sid}  원자수={len(atoms)}  "
          f"경과={time.time() - t0:.1f}s", flush=True)

pd.Series(energies, name="energy").to_csv(OUT_DIR / "energies.csv", index_label="ID")
print(f"\n완료: {OUT_DIR} 에 {len(wanted)}개 저장 (총 {time.time() - t0:.1f}s)")
