# 제출 파일 생략(...) 문제 해결

## 원인

numpy는 배열을 `str()` / `f"{arr}"` / `repr()`로 문자열화할 때 원소가
1000개(`np.get_printoptions()["threshold"]`)를 넘으면 가운데를 `...`로 생략한다.

- 105원자 구조 → 315개 값 → 정상
- 384원자 구조 → 1152개 값 → **생략 발생**

`submission_mace_v1.csv`에서는 28개 구조가 이렇게 잘렸고, 남은 값은 앞뒤 3원자씩
6줄뿐이었다(총 11,056 원자행 소실). DACON 채점기가 이 문자열을 배열로 되돌리지
못해 업로드가 거부된다.

## 07_predict_test.py 수정

force를 문자열로 만드는 줄을 찾아서 (아래 중 하나일 것)

```python
force_str = str(force_array)
force_str = f"{force_array}"
force_str = repr(force_array)
rows.append({"ID": sid, "energy": e, "force": str(f)})
```

다음으로 바꾼다.

```python
from tools.submission_format import format_force

force_str = format_force(force_array)
```

`format_force()`는 `np.array2string(..., threshold=np.inf, max_line_width=np.inf)`를
쓰므로 원자 수와 무관하게 항상 전체를 출력한다.

저장 직전에 검사도 함께 넣어 두면 같은 사고가 재발하지 않는다.

```python
from tools.submission_format import assert_no_ellipsis

assert_no_ellipsis(submission_df)
submission_df.to_csv("submission_mace_v2.csv", index=False)
```

## 검증

```bash
python tools/check_submission.py submission_mace_v2.csv sample_submission.csv
```

생략 행 0개, 원자 수 불일치 0개, `PASS`가 나와야 한다.

## 기존 CSV 응급 복구

잘린 값은 CSV에 남아있지 않으므로 정확한 복구는 재예측뿐이다. 재예측한 force를
`preds/TEST_1543.npy` 형태로 저장해 두었다면:

```bash
python tools/repair_submission.py submission_mace_v1.csv sample_submission.csv \
    submission_mace_v2.csv --forces preds/
```

`--forces`를 생략하면 해당 구조의 force를 0으로 채운다. 업로드는 되지만 그 28개
구조의 force 점수는 포기하는 임시 조치다.
