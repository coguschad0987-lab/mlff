"""제출 직전 검증 스크립트.

사용법:
    python tools/check_submission.py <제출파일.csv> <sample_submission.csv>

검사 항목:
    1. 행 수와 ID가 sample_submission과 완전히 일치하는가
    2. force 열에 생략(...)이 있는가
    3. 각 구조의 원자 수가 sample_submission의 원자 수와 같은가
    4. energy가 유한한 숫자인가 (NaN/inf 금지)
"""

import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.submission_format import parse_force  # noqa: E402


def n_atoms(text: str) -> int:
    """force 문자열의 줄 수 = 원자 수. 파싱보다 훨씬 빠르다."""
    return sum(1 for line in str(text).strip().splitlines() if line.strip())


def main(sub_path: str, template_path: str) -> int:
    sub = pd.read_csv(sub_path)
    tmpl = pd.read_csv(template_path)

    print(f"제출 파일 : {sub_path}  ({len(sub)}행)")
    print(f"템플릿    : {template_path}  ({len(tmpl)}행)")
    print("-" * 56)

    failures = []

    # 1. 행 수 / ID 일치
    if len(sub) != len(tmpl):
        failures.append(f"행 수 불일치: {len(sub)} != {len(tmpl)}")
    elif not (sub["ID"].astype(str).values == tmpl["ID"].astype(str).values).all():
        failures.append("ID 순서 또는 값이 템플릿과 다르다")
    print("행 수/ID 일치:", "FAIL" if failures else "OK")

    # 2. 생략(...) 검사
    ellipsis = sub[sub["force"].astype(str).str.contains(r"\.\.\.", regex=True)]
    print(f"생략(...) 포함 행: {len(ellipsis)}")
    if len(ellipsis):
        failures.append(f"생략된 행 {len(ellipsis)}개: {ellipsis['ID'].tolist()[:10]}")

    # 3. 원자 수 일치
    sub_counts = sub["force"].map(n_atoms)
    tmpl_counts = tmpl["force"].map(n_atoms)
    mismatch = sub_counts.values != tmpl_counts.values
    print(f"원자 수 불일치 행: {int(mismatch.sum())}")
    if mismatch.any():
        bad_ids = sub.loc[mismatch, "ID"].tolist()
        failures.append(f"원자 수 불일치 {int(mismatch.sum())}개: {bad_ids[:10]}")

    # 4. energy 유효성
    energy = pd.to_numeric(sub["energy"], errors="coerce")
    bad_energy = ~np.isfinite(energy.values)
    print(f"energy 비정상(NaN/inf) 행: {int(bad_energy.sum())}")
    if bad_energy.any():
        failures.append(f"energy 비정상 {int(bad_energy.sum())}개")

    # 5. 표본 파싱 검사 (앞/중간/뒤 각각 확인)
    probe = [0, len(sub) // 2, len(sub) - 1]
    for i in probe:
        try:
            arr = parse_force(sub["force"].iloc[i])
            if arr.shape[0] != tmpl_counts.iloc[i]:
                failures.append(f"{sub['ID'].iloc[i]} 파싱 원자 수 불일치")
        except Exception as exc:
            failures.append(f"{sub['ID'].iloc[i]} 파싱 실패: {exc}")
    print(f"표본 파싱 검사: {len(probe)}건 수행")

    print("-" * 56)
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
