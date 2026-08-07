# 오류 사례 — 주간 보도 환경 기준 (bollard_val_2146, 6988 bollard boxes)

- 가중치 `outputs/detect/c4d_26n_640/weights/best.pt`
- 평가셋 `outputs/datasets/bollard_val_2146/data.yaml` · 이미지 2,146장
- conf 0.25 · IoU 0.5 · imgsz 640
- 사유별 최대 30장 저장 (**개수는 전량 집계**, 이미지만 상한)

> 이미지 안 캡션은 ASCII 로만 표시된다 — `cv2.putText` 가 Hershey 폰트라
> 한글을 못 그린다. 한글 설명은 이 파일에 둔다.

## FN — 미탐 (미매칭 GT 객체 단위)

`fn/{nearmiss,lowconf,merged,blind}/{class}_{i}_{stem}_gt{인덱스}_bestiou{값}.jpg`
놓친 GT 만 **굵은 노란 점선**, 맞춘 GT·예측은 얇은 회색.

| 클래스 | FN | 사유별 |
|---|---|---|
| person | 1,396 | nearmiss 87 (6%) · lowconf 712 (51%) · merged 239 (17%) · blind 358 (26%) |
| bollard | 2,331 | nearmiss 125 (5%) · lowconf 1140 (49%) · merged 94 (4%) · blind 972 (42%) |

- `nearmiss` 0.3 ≤ bestIoU < 0.5 — 위치는 맞으나 박스 부정확
- `lowconf` 같은 자리에 conf<0.25 예측이 IoU≥0.5 로 있다
- `merged` 가장 겹치는 예측을 **다른 GT 가 선점** — 인접 객체 병합·중복 GT
- `blind` 위 어디에도 안 걸림

## FP — 오탐 (미매칭 예측 객체 단위)

`fp/{dup,nearmiss,confusion,ghost}/{class}_{i}_{stem}_fp{인덱스}_conf{값}_bestiou{값}.jpg`
해당 오탐만 **굵은 빨간 실선**, 나머지는 얇은 회색.

| 클래스 | FP | 사유별 |
|---|---|---|
| person | 735 | dup 176 (24%) · nearmiss 95 (13%) · confusion 3 (0%) · ghost 461 (63%) |
| stairs | 1 | ghost 1 (100%) |
| bollard | 925 | dup 235 (25%) · nearmiss 134 (14%) · confusion 2 (0%) · ghost 554 (60%) |

- `dup` 같은 클래스 GT 와 IoU≥0.5 인데 그 GT 는 이미 매칭됨 (중복 예측)
- `nearmiss` 0.3 ≤ IoU < 0.5
- `confusion` 다른 클래스 GT 와 IoU≥0.3
- `ghost` GT 와 사실상 안 겹침

박스가 20개를 넘는 이미지는 `_crop.jpg` 확대본을 같이 둔다.

## conf 별 recall 과 FP 비용

`conf_sweep.csv` 참조. 운영 conf 를 낮출 때 recall 이 얼마 오르고 FP 가 얼마 느는지.
