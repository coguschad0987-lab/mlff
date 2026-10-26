# 이 스크립트가 하는 일:
# mace-torch 라이브러리(models.py)에 있는 하드코딩된 .double() 캐스팅을 제거해
# ScaleShiftMACE 학습을 MPS(Apple GPU)에서 돌릴 수 있게 만든다.
# venv를 새로 만들거나 mace-torch를 재설치하면 패치가 사라지므로 그때마다 다시 실행한다.

from pathlib import Path

import mace.modules.models as models_module

TARGET_FILE = Path(models_module.__file__)
OLD_LINE = "node_energy = node_e0.clone().double() + node_inter_es.clone().double()"
NEW_LINE = "node_energy = node_e0.clone() + node_inter_es.clone()"

text = TARGET_FILE.read_text()

if NEW_LINE in text:
    print(f"이미 패치되어 있음: {TARGET_FILE}")
elif OLD_LINE in text:
    TARGET_FILE.write_text(text.replace(OLD_LINE, NEW_LINE))
    print(f"패치 완료: {TARGET_FILE}")
else:
    raise RuntimeError(
        "예상한 코드를 찾지 못했습니다. mace-torch 버전이 바뀌어 코드가 달라졌을 수 있으니 "
        f"{TARGET_FILE}의 forward() 함수를 직접 확인하세요."
    )
