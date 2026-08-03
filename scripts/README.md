# scripts 인덱스

> **스크립트를 열기 전에 여기서 찾을 것.** 각 파일의 상세 근거는 모듈 docstring 에 있다.
> 모든 실행은 `uv run python scripts/<이름>.py`. 산출물은 전부 `outputs/`(git 비추적).

## 공용 모듈 (import 해서 쓴다 — 단독 실행 아님)

| 파일 | 역할 |
|---|---|
| `lowlight.py` | ② arm 구현 + 레지스트리. `build("D1A1+bf")` → `bgr→bgr`. A1/A2(톤커브) · D1/R1/L1(하이라이트 압축) · `+bf`(공간 노이즈) · `+ts`/`+td`/`+a3`(**A3 시간축, 연속 프레임 전용**) |
| `metrics.py` | ② 목적 축 지표 — 글레어·대비·노이즈. **전부 절대 기준**(상대 정의는 폐기됨). `python metrics.py` 로 자기검증 실행 |
| `nightowls_yolo.py` | NightOwls 라벨 → YOLO 변환 공용. 프레임 4종 분류(`ignore`-only 제외) · 좌표 클램프 · recording 키. **평가와 학습이 GT 를 똑같이 만들도록** 여기 한 곳에만 둔다 |
| `emphasize.py` | ④ 강조 렌더 + ③→④ `Detection` 인터페이스. 단독 실행 시 데모 |

## ③ 탐지 파이프라인 (순서대로)

| 파일 | 하는 일 | 주요 옵션 |
|---|---|---|
| `stairnet_to_bbox.py` | C1 — StairNet 선분 라벨 → YOLO BBox (계단 전체 1박스) | — |
| `aihub_to_yolo.py` | AIHub 인도보행(CVAT XML) → YOLO. **`bollard` 유일 소스** · 29종 중 3클래스만 담는다 · **연속 프레임이라 블록 분할** | `--pole-as-bollard` · `--val-ratio` |
| `extract_nightowls.py` | NightOwls zip 선택 해제 (전량 53.5GiB 대신 ~14GiB) | `--run` 없으면 계획만 출력 |
| `build_detect_dataset.py` | 통합 데이터셋 생성. LoLI(low 이미지+**high 라벨**) + NightOwls + StairNet + AIHub | `--nightowls` · `--aihub` · `--loli-n 0`(train 제외) · `--dst` |
| `train_detect.py` | YOLO11n 학습 | `--data` · `--name` · `--cache`(**Windows 는 `none` 유지**) |
| `eval_nightowls.py` | 야간 정직 평가 | `--recordings 34 --drop-unlabeled-person` ← **판정 표준 조합** |
| `compare_detect.py` | ★ before/after 가중치를 **같은 자로** 일괄 판정 + `stairs` 오탐 | `--runs a,b,c` |
| `arm_detect_eval.py` | ★ C7 — **② arm 을 앞단에 붙였을 때 ③ 가 좋아지는가**. `표시/탐지 분리` 판정의 근거 | `--arms none D1A1+bf` |
| `eval_stairs_night.py` | `stairs` 를 **야간/주간 갈라서** 평가 (개발 val 은 섞여 있어 야간 성능이 묻힌다) | `--arm D1A1+bf` |
| `pipeline_demo.py` | ②→③→④ **end-to-end 동영상**. C9 통합 사전검증 + 시연 소재 | `--start` · `--detect-every` · `--fused` |
| `export_onnx.py` | ★ 가중치 → **ONNX 배포 패키지**(onnx + metadata.json + README + zip). PT↔ONNX 정합성 검증 후 실패 시 종료코드 1 | `--imgsz` · `--opset` · `--nms` · `--check-n` |

## ② 저조도 실험 하네스

| 파일 | 하는 일 |
|---|---|
| `probe_classical.py` | 고전 기법 속도 프로브 (arm 이전 단계 탐색) |
| `compare_lowlight.py` | arm 비교 — 속도·PSNR/SSIM·노이즈 (`--dataset lol\|loli`) |
| `night_eval.py` | 야간 표본 목적 축 평가 (`--profile-only` 로 밝기·포화 프로파일만) |
| `resolution_sweep.py` | C6 — **해상도 × arm** 속도 게이트와 목적 축 동시 측정 |
| `temporal_eval.py` | W1 — A3 시간축. 플리커 증폭률 · 시간축 σ · **고스팅(디테일 유지율)** · 파라미터 스윕 |

## 기타

| 파일 | 하는 일 |
|---|---|
| `inspect_datasets.py` | `data/` 배치 후 무결성 검증 (데이터 추가 시마다) |
| `label_stats.py` | YOLO 라벨 박스 통계 — 라벨 **규칙 역산**(StairNet) + 자체 촬영분 **검수**(`C3`). 경계 접촉률·작은 박스 비율이 판정 지표 (→ [labeling_stairs.md](../docs/labeling_stairs.md)) |

---

## 반드시 지킬 것

- **원본 데이터 미수정.** 이미지는 하드링크하거나 `outputs/` 에 변환본을 만든다.
- **NightOwls rec 34·38 학습 금지** — 34 는 판정용 held-out, 38 은 ② 시간축 전용.
  `build_detect_dataset.py` 가 실행을 거부한다.
- **평가 옵션을 바꾸면 before/after 비교가 조용히 깨진다.** 판정은 `compare_detect.py`
  로 한 번에 돌려서 옵션이 어긋날 여지를 없앤다.
- **A3(`+ts`/`+td`/`+a3`) arm 은 연속 프레임 전용.** 정지영상 하네스에 넣지 말 것
  (`arm.is_temporal` 로 구분, 시퀀스 경계에서 `arm.reset()`).
