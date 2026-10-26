# 이 스크립트가 하는 일:
# train.xyz 안에서 원자 수가 같은 프레임들이 뭉쳐 있는 "블록"을 찾고,
# 블록마다 연속 프레임 청크(chunk) 단위로 잘라 train/valid에 배정한다 (같은 청크는 절대 쪼개지 않음).
# test.xyz는 대회 홀드아웃이므로 라벨 없이 그대로 복사한다.

import shutil
from collections import defaultdict

import numpy as np
from ase.io import read, write

SEED = 42
VALID_RATIO = 0.2
CHUNK_SIZE = 50  # 기본 청크 길이(연속 프레임 개수)
SMALL_BLOCK_THRESHOLD = CHUNK_SIZE * 2  # 이보다 작은 블록은 청크를 더 잘게 나눔

TRAIN_SRC = "data/raw/train.xyz"
TEST_SRC = "data/raw/test.xyz"
TRAIN_OUT = "data/processed/train.xyz"
VALID_OUT = "data/processed/valid.xyz"
TEST_OUT = "data/processed/test.xyz"


def find_blocks(natoms):
    # 같은 natoms 값이 연속되는 구간을 (natoms, start, end) 리스트로 반환
    blocks = []
    start = 0
    cur = natoms[0]
    for i in range(1, len(natoms)):
        if natoms[i] != cur:
            blocks.append((cur, start, i))
            start = i
            cur = natoms[i]
    blocks.append((cur, start, len(natoms)))
    return blocks


def split_block_into_chunks(size, start, end):
    length = end - start
    chunk_size = CHUNK_SIZE if length >= SMALL_BLOCK_THRESHOLD else max(1, length // 5)
    chunks = []
    cs = start
    while cs < end:
        ce = min(cs + chunk_size, end)
        chunks.append((cs, ce))
        cs = ce
    return chunks


frames = read(TRAIN_SRC, ":")
natoms = [len(a) for a in frames]
blocks = find_blocks(natoms)

rng = np.random.default_rng(SEED)
train_idx, valid_idx = [], []

for size, s, e in blocks:
    chunks = split_block_into_chunks(size, s, e)
    n_chunks = len(chunks)
    n_valid_chunks = max(1, round(n_chunks * VALID_RATIO)) if n_chunks > 1 else 0
    order = rng.permutation(n_chunks)
    valid_positions = set(order[:n_valid_chunks].tolist())
    for ci, (cs, ce) in enumerate(chunks):
        target = valid_idx if ci in valid_positions else train_idx
        target.extend(range(cs, ce))

train_idx.sort()
valid_idx.sort()

write(TRAIN_OUT, [frames[i] for i in train_idx])
write(VALID_OUT, [frames[i] for i in valid_idx])
shutil.copy(TEST_SRC, TEST_OUT)

print(f"블록 개수: {len(blocks)}")
print(f"train: {len(train_idx)} 프레임, valid: {len(valid_idx)} 프레임")
print(f"valid 비율: {len(valid_idx) / (len(train_idx) + len(valid_idx)):.3f}")

# 시스템 크기(원자 수)별로 train/valid에 고르게 들어갔는지 확인
size_train = defaultdict(int)
size_valid = defaultdict(int)
for i in train_idx:
    size_train[natoms[i]] += 1
for i in valid_idx:
    size_valid[natoms[i]] += 1

print("\n원자 수 | train 프레임 | valid 프레임")
for size in sorted(set(natoms)):
    print(f"{size:>7} | {size_train[size]:>12} | {size_valid[size]:>12}")
