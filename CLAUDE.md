## 실행 환경

패키지 관리는 **uv**를 사용한다. 모든 Python 명령은 `uv run`으로 실행할 것.

```powershell
uv run python scripts/xxx.py
uv run yolo ...
```

`uv run`은 가상환경 활성화 없이 `.venv`를 자동으로 사용한다.
(`.venv\Scripts\activate` 후 직접 실행해도 되지만, 셸 세션 간 활성화 상태가 유지되지 않으므로 `uv run`을 기본으로 한다.)

## 의존성

- 의존성은 `pyproject.toml`에 선언하고 `uv.lock`으로 고정한다.
- 추가·변경은 `uv add <패키지>` / `uv remove <패키지>`로 한다.
- **`pip install`을 직접 쓰지 말 것** — `uv.lock`과 실제 환경이 어긋나 팀원 간 환경 재현이 깨진다.
- `torch`·`torchvision`은 `[tool.uv.sources]`에서 PyTorch CUDA 인덱스(`cu130`)에 고정되어 있다. 이 설정을 임의로 제거하면 CPU 빌드가 설치되어 GPU 학습이 동작하지 않는다.
