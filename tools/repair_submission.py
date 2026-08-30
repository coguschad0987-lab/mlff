"""이미 만들어진 제출 CSV에서 생략(...)된 행을 복구한다.

numpy가 잘라버린 값은 CSV 안에 남아 있지 않으므로 진짜 복구는 재예측뿐이다.
이 스크립트는 두 가지 모드를 제공한다.

1. --forces <dir>: ID.npy 형태로 저장해 둔 원본 force 배열이 있으면 그것으로 교체.
2. 없으면 sample_submission의 원자 수에 맞춰 0으로 채워 최소한 업로드는 가능하게 만든다
   (해당 구조의 force 점수는 포기하는 임시 조치).

사용법:
    python tools/repair_submission.py in.csv sample_submission.csv out.csv [--forces preds/]
"""

import argparse
import os

import numpy as np
import pandas as pd

from submission_format import ELLIPSIS_PATTERN, format_force, parse_force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("sample")
    parser.add_argument("output")
    parser.add_argument("--forces", help="ID.npy 파일들이 들어있는 디렉터리")
    args = parser.parse_args()

    df = pd.read_csv(args.submission)
    sample = pd.read_csv(args.sample).set_index("ID")

    broken = df["force"].astype(str).str.contains(ELLIPSIS_PATTERN, regex=True)
    print(f"복구 대상 {int(broken.sum())}개 행")

    filled_with_zeros = []
    for i in df.index[broken]:
        structure_id = df.at[i, "ID"]
        n_atoms = len(parse_force(sample.at[structure_id, "force"]))

        npy_path = os.path.join(args.forces, f"{structure_id}.npy") if args.forces else None
        if npy_path and os.path.exists(npy_path):
            force = np.load(npy_path).reshape(-1, 3)
            if len(force) != n_atoms:
                raise ValueError(f"{structure_id}: 원자 수 {len(force)} != {n_atoms}")
        else:
            force = np.zeros((n_atoms, 3))
            filled_with_zeros.append(structure_id)

        df.at[i, "force"] = format_force(force)

    # 나머지 행도 동일 포맷으로 재직렬화해 파일 전체 포맷을 통일한다.
    for i in df.index[~broken]:
        df.at[i, "force"] = format_force(parse_force(df.at[i, "force"]))

    df.to_csv(args.output, index=False)
    print(f"저장: {args.output}")
    if filled_with_zeros:
        print(f"경고: 0으로 채운 구조 {len(filled_with_zeros)}개 -> {filled_with_zeros}")
        print("정확한 점수를 원하면 이 ID들만 재예측해서 --forces로 다시 넣으세요.")


if __name__ == "__main__":
    main()
