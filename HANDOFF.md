# DACON 236133 제출 실패 — 로컬(VS Code) 이어받기 문서

이 문서는 클라우드 세션에서 진행한 진단 결과와, 로컬 맥에서 마무리해야 할 작업을
정리한 것이다. 로컬 Claude Code 세션에서 이 파일을 그대로 읽히면 된다.

## 0. 로컬에서 시작하는 법

```bash
cd /Users/immack2/mlff-project
git fetch origin claude/dacon-array-ellipsis-fix-yfxkyx
git checkout claude/dacon-array-ellipsis-fix-yfxkyx
claude
```

첫 프롬프트 예시: `HANDOFF.md 읽고 남은 작업 이어서 해줘`

> 클라우드 세션은 `/Users/immack2/mlff-project`에 접근할 수 없어 07_predict_test.py를
> 직접 열지 못했다. 로컬 세션에서는 접근 가능하므로 3번 작업부터 바로 하면 된다.

## 1. 문제 원인 (확정)

`submission_mace_v1.csv`가 DACON 심사 페이지에서 업로드되지 않은 이유.

numpy는 배열을 `str()` / `f"{arr}"` / `repr()`로 문자열화할 때 원소 수가
1000개(`np.get_printoptions()["threshold"]`)를 넘으면 가운데를 `...`로 생략한다.

| 구조 크기 | 값 개수 (원자수 × 3) | 결과 |
|---|---|---|
| 105원자 | 315 | 정상 |
| 384원자 | 1152 | **생략 발생** |

채점기가 `...`가 낀 문자열을 배열로 파싱하지 못해 거부됐다.

## 2. 진단 결과 — 손실 규모 (중요)

v1 CSV를 열어 확인한 결과, 잘린 행은 `...`만 낀 게 아니라 **앞뒤 3원자씩 6줄만 남고
나머지 값이 파일에서 실제로 사라진** 상태였다.

- 영향 행: **28개** / 4101개
- 소실된 원자행: **11,056개** (전체 295,234개의 약 3.7%)
- 남아있는 값: 구조당 6행뿐

즉 **CSV만으로는 복구가 불가능하다.** 아래 28개 ID는 반드시 재예측해야 한다.

```
TEST_1543 TEST_1544 TEST_1545 TEST_1546 TEST_1547 TEST_1548 TEST_1549
TEST_1550 TEST_1551 TEST_1552 TEST_1553 TEST_1554 TEST_1555
TEST_2699 TEST_2700 TEST_2701 TEST_2702 TEST_2703
TEST_2818 TEST_2819
TEST_3478 TEST_3479 TEST_3480 TEST_3481
TEST_4029 TEST_4030 TEST_4031 TEST_4032
```

## 3. 해야 할 일 ① — 07_predict_test.py 수정

force를 문자열로 만드는 줄을 찾는다. 아래 중 하나 형태일 것이다.

```python
force_str = str(force_array)
force_str = f"{force_array}"
force_str = repr(force_array)
rows.append({"ID": sid, "energy": e, "force": str(f)})
```

다음으로 교체한다.

```python
from tools.submission_format import format_force, assert_no_ellipsis

force_str = format_force(force_array)   # threshold=inf 라 항상 전체 출력

# ... 저장 직전
assert_no_ellipsis(submission_df)       # 재발 방지 가드
submission_df.to_csv("submission_mace_v2.csv", index=False)
```

`tools/submission_format.py`의 `format_force()`는 내부적으로
`np.array2string(arr, threshold=np.inf, max_line_width=np.inf, ...)`를 쓴다.

## 4. 해야 할 일 ② — 재예측 후 CSV 생성

수정한 스크립트로 **전체를 다시 예측해 v2를 만드는 것이 가장 깔끔하다.**

시간을 아끼려면 위 28개 ID만 재예측해서 `preds/TEST_1543.npy` 형태로 저장한 뒤,
기존 v1을 패치해도 된다.

```bash
python tools/repair_submission.py submission_mace_v1.csv sample_submission.csv \
    submission_mace_v2.csv --forces preds/
```

`--forces`를 생략하면 28개 구조의 force를 0으로 채운다. 업로드는 되지만 해당
구조의 force 점수는 포기하는 임시 조치이므로, 실제 제출에는 권하지 않는다.

## 5. 해야 할 일 ③ — 제출 전 검증

```bash
python tools/check_submission.py submission_mace_v2.csv sample_submission.csv
```

`생략(...) 포함 행: 0`, `원자 수 불일치 행: 0`, `PASS`가 나와야 한다.

한 줄로 빠르게 확인하려면:

```python
import pandas as pd
df = pd.read_csv("submission_mace_v2.csv")
print(df["force"].str.contains(r"\.\.\.").sum())   # 0 이어야 정상
```

## 6. 클라우드 세션에서 이미 해둔 것

브랜치 `claude/dacon-array-ellipsis-fix-yfxkyx`에 다음이 커밋되어 있다.

| 파일 | 역할 |
|---|---|
| `tools/submission_format.py` | `format_force()`, `parse_force()`, `assert_no_ellipsis()` |
| `tools/check_submission.py` | 제출 전 검증 (생략 / 원자 수 / energy 유한성) |
| `tools/repair_submission.py` | 기존 CSV 패치 (재예측 npy 주입 또는 0 채움) |
| `tools/README.md` | 수정 방법 요약 |

또한 v1을 0으로 채워 만든 임시 `submission_mace_v2.csv`를 채팅으로 전달했다.
검증은 통과하지만 28개 구조의 force가 0이므로 **업로드 동작 확인용**이다.

## 7. 남은 체크리스트

- [ ] 07_predict_test.py의 force 문자열 변환부를 `format_force()`로 교체
- [ ] 저장 직전 `assert_no_ellipsis()` 가드 추가
- [ ] 재예측하여 submission_mace_v2.csv 생성 (전체 또는 28개 ID)
- [ ] `tools/check_submission.py`로 PASS 확인
- [ ] DACON 재업로드
- [ ] 변경사항 커밋 & 푸시
