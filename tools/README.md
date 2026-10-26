# tools/ — 제출 파일 안전장치

## 왜 필요했나

numpy 배열을 `str()` / `f"{arr}"` / `repr()` 로 문자열로 바꾸면,
원소 수가 1000개(`np.get_printoptions()["threshold"]`)를 넘는 순간
numpy가 가운데를 `...` 로 줄여버린다.

| 구조 크기 | 값 개수 (원자수 × 3) | 결과 |
|---|---|---|
| 105원자 | 315 | 정상 |
| 384원자 | 1152 | **생략 발생 — 값이 실제로 소실** |

`submission_mace_v1.csv`에서 28개 구조가 이렇게 손상되어 DACON 업로드가 거부됐다.
손상된 행은 앞뒤 3원자씩 6줄만 남아 있어 **CSV만으로는 복구가 불가능**하다.

## 파일

| 파일 | 역할 |
|---|---|
| `submission_format.py` | `format_force()` — 생략 없이 문자열화 / `parse_force()` / `assert_no_ellipsis()` |
| `check_submission.py` | 제출 전 검증 (생략 / 원자 수 / ID / energy) |
| `repredict_ids.py` | 특정 ID만 다시 예측해 `preds/<ID>.npy` 로 저장 |
| `repair_submission.py` | 기존 CSV의 손상된 행만 교체 |

## 사용법

예측 스크립트에서:

```python
from tools.submission_format import format_force, assert_no_ellipsis

forces_str.append(format_force(forces))   # str(forces) 대신

assert_no_ellipsis(submission)            # 저장 직전 안전장치
submission.to_csv(OUTPUT_PATH, index=False)
```

손상된 CSV 고치기:

```bash
python tools/repredict_ids.py TEST_1543 TEST_1544 ...          # preds/ 생성
python tools/repair_submission.py \
    results/submission_mace_v1.csv data/raw/sample_submission.csv \
    results/submission_mace_v2.csv --forces preds/
```

Kaggle/Colab에서 받은 모델로 새 제출 파일 만들기
(모델·출력 경로를 인자로 준다. 기존 파일이 있으면 덮어쓰지 않고 멈춘다):

```bash
python scripts/07_predict_test.py \
    --model models/mace_kaggle_v2.model \
    --out results/submission_mace_v3.csv
```

제출 전 검증 (`PASS` 가 나와야 한다):

```bash
python tools/check_submission.py results/submission_mace_v2.csv data/raw/sample_submission.csv
```
