## 먼저 읽을 것 ★

**세션을 시작하면 [`docs/STATUS.md`](docs/STATUS.md) 를 먼저 읽는다.** 현재 상태·막힌 것·
반복된 함정·문서 지도가 한 장에 있다. 나머지 문서는 **필요할 때만** 연다.

문서가 크므로 통째로 읽지 말 것 — `grep -n "^#" <파일>` 로 목차를 보고
`sed -n 'A,Bp'` 로 해당 절만 연다. 각 문서 맨 위 `0. 결론 요약`이 그 문서의 요약이다.

작업을 끝내면 **`docs/STATUS.md` 를 먼저 갱신**하고, 상세는 해당 주제 문서에 남긴다.
종결된 실험·결정의 원문은 `docs/archive/` 로 옮긴다 (삭제하지 않는다).

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
