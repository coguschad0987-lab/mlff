"""제출 CSV의 force 문자열을 안전하게 만들고 검증하는 도구 모음.

문제의 배경:
    numpy 배열을 str()/f-string/repr()로 문자열로 바꾸면, 원소 수가 1000개를
    넘는 순간 numpy가 가운데를 '...'로 줄여버린다. 384원자 구조는 384*3=1152개라
    이 한계를 넘어 값이 실제로 잘려나갔다.

해결:
    np.array2string(threshold=np.inf)를 쓰면 크기와 무관하게 항상 전체를 출력한다.
"""

import numpy as np

# sample_submission.csv와 동일한 서식으로 맞추기 위한 설정
_PRECISION = 8


def format_force(arr) -> str:
    """(N, 3) force 배열을 생략 없는 문자열로 변환한다.

    threshold=inf  -> 원소가 아무리 많아도 '...'로 줄이지 않음
    max_line_width -> 한 줄을 임의로 접지 않음 (행 단위 줄바꿈만 유지)
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"force 배열은 (N, 3) 모양이어야 한다. 받은 모양: {arr.shape}")

    return np.array2string(
        arr,
        threshold=np.inf,       # 핵심: 생략 금지
        max_line_width=np.inf,  # 핵심: 줄 접기 금지
        precision=_PRECISION,
        suppress_small=False,
        separator=" ",
        floatmode="maxprec",
    )


def parse_force(text: str) -> np.ndarray:
    """format_force()가 만든 문자열(또는 numpy 기본 출력)을 다시 배열로 되돌린다.

    검증·비교용. '...'가 들어 있으면 복구가 불가능하므로 에러를 낸다.
    """
    if "..." in text:
        raise ValueError("생략(...)된 문자열은 배열로 복원할 수 없다.")

    rows = []
    for line in text.strip().strip("[]").splitlines():
        line = line.strip().strip("[]").strip()
        if not line:
            continue
        rows.append([float(v) for v in line.replace("[", " ").replace("]", " ").split()])

    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"파싱 결과가 (N, 3)이 아니다: {arr.shape}")
    return arr


def assert_no_ellipsis(df, column: str = "force") -> None:
    """저장 직전 호출하는 안전장치. 생략된 행이 하나라도 있으면 즉시 중단시킨다."""
    bad = df[df[column].astype(str).str.contains(r"\.\.\.", regex=True)]
    if len(bad) > 0:
        ids = bad["ID"].tolist() if "ID" in bad.columns else bad.index.tolist()
        raise AssertionError(
            f"force 열에 생략(...)된 행이 {len(bad)}개 있다. 저장을 중단한다.\n"
            f"해당 ID: {ids[:10]}{' ...' if len(ids) > 10 else ''}"
        )
