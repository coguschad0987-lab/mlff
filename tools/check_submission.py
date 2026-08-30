"""제출 파일 검증: 생략(...) 여부와 원자 수가 sample_submission과 일치하는지 확인.

사용법:
    python tools/check_submission.py submission_mace_v2.csv sample_submission.csv
"""

import sys

import numpy as np
import pandas as pd

from submission_format import ELLIPSIS_PATTERN, parse_force


def main(submission_path, sample_path):
    df = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)

    ok = True

    ellipsis_rows = df[df["force"].astype(str).str.contains(ELLIPSIS_PATTERN, regex=True)]
    print(f"생략(...) 포함 행: {len(ellipsis_rows)}")
    if len(ellipsis_rows):
        print("  예시:", ellipsis_rows["ID"].tolist()[:10])
        ok = False

    if list(df["ID"]) != list(sample["ID"]):
        print("ID 순서/구성이 sample_submission과 다릅니다.")
        ok = False

    mismatched = []
    for row, sample_row in zip(df.itertuples(), sample.itertuples()):
        expected = len(parse_force(sample_row.force))
        try:
            actual = len(parse_force(row.force))
        except ValueError:
            mismatched.append((row.ID, "parse 실패", expected))
            continue
        if actual != expected:
            mismatched.append((row.ID, actual, expected))

    print(f"원자 수 불일치 행: {len(mismatched)}")
    for item in mismatched[:10]:
        print("  ", item)
    if mismatched:
        ok = False

    if not np.isfinite(df["energy"].to_numpy(dtype=float)).all():
        print("energy에 NaN/inf가 있습니다.")
        ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
