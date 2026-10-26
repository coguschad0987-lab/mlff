"""손상된(생략된) 제출 CSV를 고쳐 새 CSV를 만든다.

사용법:
    python tools/repair_submission.py <입력.csv> <sample_submission.csv> <출력.csv> [--forces preds/]

--forces 를 주면 preds/<ID>.npy 에서 재예측한 (N,3) force 배열을 읽어 채운다.
--forces 를 생략하면 손상된 구조의 force를 0으로 채운다(업로드 확인용, 실제 제출 비권장).
멀쩡한 행은 손대지 않고 그대로 둔다.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.submission_format import format_force, assert_no_ellipsis  # noqa: E402


def n_atoms(text: str) -> int:
    return sum(1 for line in str(text).strip().splitlines() if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("template_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--forces", default=None,
                    help="재예측 force가 담긴 디렉터리 (<ID>.npy)")
    args = ap.parse_args()

    sub = pd.read_csv(args.input_csv)
    tmpl = pd.read_csv(args.template_csv)

    # 템플릿에서 구조별 정답 원자 수를 얻는다
    expected = tmpl["force"].map(n_atoms)

    damaged = sub.index[
        sub["force"].astype(str).str.contains(r"\.\.\.", regex=True)
        | (sub["force"].map(n_atoms).values != expected.values)
    ]
    print(f"손상된 행 {len(damaged)}개를 고친다.")

    filled_zero, filled_pred = [], []
    for i in damaged:
        sid = str(sub.at[i, "ID"])
        n = int(expected.iloc[i])

        npy = os.path.join(args.forces, f"{sid}.npy") if args.forces else None
        if npy and os.path.exists(npy):
            arr = np.load(npy)
            if arr.shape != (n, 3):
                raise ValueError(f"{sid}: npy 모양 {arr.shape} != 기대 {(n, 3)}")
            filled_pred.append(sid)
        else:
            arr = np.zeros((n, 3))
            filled_zero.append(sid)

        sub.at[i, "force"] = format_force(arr)

    print(f"  재예측 값으로 채움 : {len(filled_pred)}개")
    print(f"  0으로 채움         : {len(filled_zero)}개")
    if filled_zero:
        print(f"  (0으로 채운 ID: {filled_zero[:10]}{' ...' if len(filled_zero) > 10 else ''})")
        print("  주의: 0 채움은 업로드 확인용이다. 실제 제출에는 권장하지 않는다.")

    assert_no_ellipsis(sub)
    sub.to_csv(args.output_csv, index=False)
    print(f"\n저장 완료: {args.output_csv} ({len(sub)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
