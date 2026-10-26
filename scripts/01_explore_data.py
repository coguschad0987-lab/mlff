# 이 스크립트가 하는 일:
# data/raw/의 train.xyz, test.xyz, sample_submission.csv를 읽어 구조 개수,
# 원자 수 분포, 원소 종류, 에너지/힘 범위, PBC 여부를 확인하고 요약을 출력한다.

from ase.io import read
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
TRAIN_PATH = f"{RAW_DIR}/train.xyz"
TEST_PATH = f"{RAW_DIR}/test.xyz"
SUBMISSION_PATH = f"{RAW_DIR}/sample_submission.csv"


def summarize(path, label, has_labels=True):
    print(f"\n=== {label} ({path}) ===")
    frames = read(path, ":")
    print(f"구조(프레임) 개수: {len(frames)}")

    natoms = np.array([len(a) for a in frames])
    print(f"원자 수: 최소 {natoms.min()}, 최대 {natoms.max()}, 평균 {natoms.mean():.1f}")

    elements = set()
    for a in frames:
        elements.update(a.get_chemical_symbols())
    print(f"등장 원소: {sorted(elements)}")

    pbc_all = all(tuple(a.pbc) == (True, True, True) for a in frames)
    print(f"모든 구조 PBC=True,True,True: {pbc_all}")

    if has_labels:
        energies = np.array([a.get_potential_energy() for a in frames])
        energies_per_atom = energies / natoms
        print(f"에너지(eV, 구조 전체): 최소 {energies.min():.2f}, 최대 {energies.max():.2f}")
        print(f"에너지(eV/atom): 최소 {energies_per_atom.min():.4f}, 최대 {energies_per_atom.max():.4f}")

        force_max_per_frame = np.array([np.abs(a.get_forces()).max() for a in frames])
        print(f"힘(eV/Å) 최대 성분값: 전체 프레임 중 최댓값 {force_max_per_frame.max():.2f}")

    return frames


train_frames = summarize(TRAIN_PATH, "train.xyz", has_labels=True)
test_frames = summarize(TEST_PATH, "test.xyz", has_labels=False)

print(f"\n=== {SUBMISSION_PATH} ===")
sub = pd.read_csv(SUBMISSION_PATH)
print(f"행 개수: {len(sub)} (test.xyz 구조 개수와 일치해야 함: {len(test_frames)})")
print(f"컬럼: {list(sub.columns)}")
print("첫 행 예시:")
print(sub.iloc[0])
