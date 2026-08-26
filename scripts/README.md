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
| `aihub_subset_to_yolo.py` | ★ `C4e` S2 — **이미 YOLO 인** AIHub 서브셋(`bammasil_aihub_subset`, 113,163장)을 학습 레이아웃으로 옮기는 **다리**. `aihub_to_yolo.py` 는 CVAT XML 전용이라 이걸 못 받는다. **세션 단위 분할**(파일명이 곧 세션 키 · 누수 0 검사) · **서브샘플**(전량은 `C4` 붕괴 재현) · **배경 음성 10%**. 기본값이 `person_night` 를 빼는 이유는 머리말 참고 | `--src` · `--dst`(🔴 원본과 같은 드라이브) · `--max-images` · `--bg-ratio` · `--dry-run` |
| `aihub_pack_for_colab.py` | ★ AIHub 다운로드분 → **Colab 으로 옮길 수 있는** 데이터셋(YOLO 변환 + **640 리사이즈** + zip, 30GB→2~3GB). **AIHub 해외 차단** 때문에 다운로드는 국내에서만 되므로 필요하다. 장애물/노면 XML 을 **내용으로 자동 분류**하고, 노면 `stairs` 는 **StairNet 대조 게이트**로 학습 투입/평가 전용을 가른다. ★ **정찰**(8/5) — 전량에서 "볼라드 N장"이 나왔을 때 **몇 블록(촬영 세션)인지**와 **저조도가 야간인지 그늘인지**를 같이 낸다 | `--src`(필수) · `--dry-run`(분포·정찰만) · `--recon-sample`(0=전수) · `--zip` · `--stairs auto\|train\|eval` |
| `extract_nightowls.py` | NightOwls zip 선택 해제 (전량 53.5GiB 대신 ~14GiB) | `--run` 없으면 계획만 출력 |
| `build_detect_dataset.py` | 통합 데이터셋 생성. LoLI(low 이미지+**high 라벨**) + NightOwls + StairNet + AIHub | `--nightowls` · `--aihub` · `--loli-n 0`(train 제외) · `--dst` |
| `train_detect.py` | YOLO11n 학습 | `--data` · `--name` · `--cache`(**Windows 는 `none` 유지**) |
| `eval_nightowls.py` | 야간 정직 평가 | `--recordings 34 --drop-unlabeled-person` ← **판정 표준 조합** |
| `compare_detect.py` | ★ before/after 가중치를 **같은 자로** 일괄 판정 + `stairs` 오탐 | `--runs a,b,c` |
| `arm_detect_eval.py` | ★ C7 — **② arm 을 앞단에 붙였을 때 ③ 가 좋아지는가**. `표시/탐지 분리` 판정의 근거 | `--arms none D1A1+bf` |
| `eval_real_night.py` | ★ `C4e` S0b — **자체 실촬영 야간 소재에서 오탐을 센다(라벨 불요)**. 음성 5장은 계단·볼라드가 없고 사람은 7장 전부 0명이라 예측이 곧 오탐이다. **rec34(대시캠)가 못 가르는 FP 축**을 보행 시점에서 잰다. 🔴 **8/25 수정** — 예측을 한 장씩 돌리고 `--letterbox` 로 전처리를 고정한다(기본 `square`). 그 전에는 음성/양성이 **다른 자**로 재어졌다 (→ 함정 18) | `--runs`\|`--weights` · `--conf` · `--letterbox` · `--device` · `--videos` |
| `eval_own_night.py` | ★ **`C5` 판정** — 자체 촬영 야간분(**라벨 있음**)에서 후보를 **한 표로** 가른다. 5축 — mAP · 운영점 recall/precision(**+종합**) · 음성 프레임 오탐 · **GT 폭 구간별 recall**(+전체) · **볼라드 박스별 conf**. `ignore` 박스는 감점 제외. ⚠️ **`--letterbox` 기본 `square`(배포 ONNX 와 같은 자)** · 예측은 한 장씩(→ 함정 18) | `--src` · `--runs`\|`--weights` · `--class-conf` · `--letterbox` · `--device cpu` · `--skip-map` |
| `track_eval.py` | ★ `C4e` S1(E1) — **탐지 프레임 스킵의 대가**를 잰다. 주기 × 보간(none/hold/track) × EMA 를 **오라클(매 프레임 탐지) 대비**로 판정. GT 불요. ⚠️ 트래커는 **순수 파이썬 IoU** — `model.track()` 은 `lap` 이 없어 이 환경에서 못 돌고, 배포 ONNX 는 NMS 도 그래프 밖이라 앱 코드 레벨이 맞다 | `--detect-every 1,2,3` · `--interp none,hold,track` · `--smooth-alpha` · `--videos` |
| `eval_stairs_night.py` | `stairs` 를 **야간/주간 갈라서** 평가 (개발 val 은 섞여 있어 야간 성능이 묻힌다) | `--arm D1A1+bf` |
| `pipeline_demo.py` | ②→③→④ **end-to-end 동영상**. C9 통합 사전검증 + 시연 소재 | `--start` · `--detect-every` · `--fused` |
| `export_onnx.py` | ★ 가중치 → **ONNX 배포 패키지**(onnx + metadata.json + README + zip). PT↔ONNX 정합성 검증 후 실패 시 종료코드 1 | `--imgsz` · `--opset` · `--nms` · `--check-n` |
| `quantize_onnx.py` | ★ `C10a` — 배포 ONNX → **양자화 패키지**(onnx + metadata + README + 캘리브 manifest). static PTQ · QDQ · 캘리브는 **야간 실촬영 159프레임**. 🔴 **`Conv` 만 양자화한다** — 전부 하면 머리의 `Concat` 이 박스좌표(0~640)와 점수(0~1)를 한 텐서로 합치는 탓에 **검출이 0 이 된다**(→ 함정 20). `--bits 4` 도 굽히지만 **붕괴해서 기각**됐다 | `--bits 8\|4` · `--preset generic,qnn` · `--block-size` · `--calibrate-method` · `--name` |

## ② 저조도 실험 하네스

| 파일 | 하는 일 |
|---|---|
| `probe_classical.py` | 고전 기법 속도 프로브 (arm 이전 단계 탐색) |
| `compare_lowlight.py` | arm 비교 — 속도·PSNR/SSIM·노이즈 (`--dataset lol\|loli`) |
| `night_eval.py` | 야간 표본 목적 축 평가 (`--profile-only` 로 밝기·포화 프로파일만) |
| `darken.py` | 실촬영 소재를 **더 저조도로** 합성 (이미지+동영상). 선형 도메인 노출 감소 + 산탄/리드 노이즈. `--gain`. **정성 확인용** — 합성본이라 정량 근거로 쓰지 말 것 |
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
