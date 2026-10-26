# 이 스크립트가 하는 일:
# mace-mp 파운데이션 모델(small)을 fine-tuning 없이 valid.xyz에 그대로 적용해 zero-shot baseline을 구한다.
# energy RMSE(eV, meV/atom), force RMSE(eV/Å)를 계산해 이후 fine-tuning 성능과 비교할 기준점으로 삼는다.
#
# 주의: 이 mace-mp 체크포인트(2023-12 버전)는 모델 내부에서 에너지 계산 시 float64로
# 강제 캐스팅한다. MPS는 float64를 지원하지 않아 --device=mps로는 실행이 불가능하므로
# (PYTORCH_ENABLE_MPS_FALLBACK로도 해결 안 되는 하드웨어 제약) 이 스크립트는 CPU로 돌린다.
# 1회성 추론이라 CPU로도 수 분 내에 끝난다. 체크포인트 자체는 미리 float32로 변환해
# models/mace_mp_small_float32.model로 저장해 두었다.

import numpy as np
from ase.io import read
from mace.calculators import mace_mp

VALID_PATH = "data/processed/valid.xyz"
MODEL_PATH = "models/mace_mp_small_float32.model"
DEVICE = "cpu"

frames = read(VALID_PATH, ":")
print(f"valid 구조 {len(frames)}개 로드")

calc = mace_mp(model=MODEL_PATH, device=DEVICE, default_dtype="float32")

energy_errors = []
energy_errors_per_atom = []
force_errors = []

for i, atoms in enumerate(frames):
    true_energy = atoms.get_potential_energy()
    true_forces = atoms.get_forces()
    n = len(atoms)

    atoms.calc = calc
    pred_energy = atoms.get_potential_energy()
    pred_forces = atoms.get_forces()

    energy_errors.append(pred_energy - true_energy)
    energy_errors_per_atom.append((pred_energy - true_energy) / n)
    force_errors.append((pred_forces - true_forces).flatten())

    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(frames)} 완료")

energy_errors = np.array(energy_errors)
energy_errors_per_atom = np.array(energy_errors_per_atom)
force_errors = np.concatenate(force_errors)

energy_rmse = np.sqrt(np.mean(energy_errors ** 2))
energy_rmse_per_atom = np.sqrt(np.mean(energy_errors_per_atom ** 2))
force_rmse = np.sqrt(np.mean(force_errors ** 2))

print("\n=== Zero-shot (mace-mp small, fine-tuning 없음) ===")
print(f"Energy RMSE: {energy_rmse:.4f} eV  ({energy_rmse_per_atom * 1000:.2f} meV/atom)")
print(f"Force RMSE: {force_rmse:.4f} eV/Å")
