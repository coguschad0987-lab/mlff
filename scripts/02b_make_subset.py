# 이 스크립트가 하는 일:
# train/valid에서 원자 수 비율을 유지한 채 일부만 뽑아 작은 실험용 파일을 만든다.
# loss 함수 A/B 비교처럼 빠른 반복이 필요한 실험에 쓴다 (전체 데이터는 1 epoch에 66분).

from collections import defaultdict

import numpy as np
from ase.io import read, write

SEED = 42
N_TRAIN = 2000
N_VALID = 800

TRAIN_SRC = "data/processed/train.xyz"
VALID_SRC = "data/processed/valid.xyz"
TRAIN_OUT = "data/processed/train_sub2000.xyz"
VALID_OUT = "data/processed/valid_sub800.xyz"


def stratified_sample(frames, n_target, rng):
    # 원자 수별로 묶은 뒤, 각 그룹에서 원래 비율만큼 뽑는다
    groups = defaultdict(list)
    for i, a in enumerate(frames):
        groups[len(a)].append(i)

    picked = []
    for size, idxs in sorted(groups.items()):
        share = len(idxs) / len(frames)
        n_pick = max(1, round(n_target * share))
        n_pick = min(n_pick, len(idxs))
        chosen = rng.choice(idxs, size=n_pick, replace=False)
        picked.extend(chosen.tolist())

    picked.sort()
    return picked


rng = np.random.default_rng(SEED)

for src, out, n_target, label in [
    (TRAIN_SRC, TRAIN_OUT, N_TRAIN, "train"),
    (VALID_SRC, VALID_OUT, N_VALID, "valid"),
]:
    frames = read(src, ":")
    idxs = stratified_sample(frames, n_target, rng)
    write(out, [frames[i] for i in idxs])

    sizes = defaultdict(int)
    for i in idxs:
        sizes[len(frames[i])] += 1
    print(f"{label}: {len(frames)} -> {len(idxs)}개 ({out})")
    print(f"  원자 수 종류: {len(sizes)}가지 유지")
