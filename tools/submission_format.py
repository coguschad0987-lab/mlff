"""DACON 236133 제출 파일용 force/energy 문자열 변환 유틸.

핵심: numpy 기본 str()/repr()은 원소가 1000개를 넘으면 가운데를 '...'로 생략한다.
384원자 구조는 384*3 = 1152개라 임계값을 넘어 잘리고, 채점 파서가 이를 읽지 못한다.
아래 format_force()는 threshold=inf 로 항상 전체를 출력한다.
"""

import numpy as np

ELLIPSIS_PATTERN = r"\.\.\."


def format_force(force):
    """(N, 3) 힘 배열을 sample_submission과 동일한 포맷의 문자열로 변환한다."""
    arr = np.asarray(force, dtype=float).reshape(-1, 3)
    return np.array2string(
        arr,
        threshold=np.inf,
        max_line_width=np.inf,
        separator=" ",
        formatter={"float_kind": lambda v: f"{v:.8f}"},
    )


def parse_force(text):
    """format_force()가 만든 문자열을 다시 (N, 3) 배열로 되돌린다."""
    values = np.fromstring(text.replace("[", " ").replace("]", " "), sep=" ")
    return values.reshape(-1, 3)


def assert_no_ellipsis(df, column="force"):
    """제출 직전 검사: 생략(...)이 남아 있으면 예외를 던진다."""
    bad = df[df[column].astype(str).str.contains(ELLIPSIS_PATTERN, regex=True)]
    if len(bad):
        raise ValueError(
            f"{len(bad)}개 행에 생략된 배열이 있습니다: {bad['ID'].tolist()[:10]}"
        )
