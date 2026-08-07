# PROVENANCE — bollard_val_2146 / bollard_dark_20

생성 2026-08-07 13:08 · commit `93272ac` · `scripts/build_bollard_val_2146.py`

## 이 셋은 무엇인가

| 셋 | 이미지 | 볼라드박스 | 성격 |
|----|--------|-----------|------|
| `bollard_val_2146` | 2,146 | **6,988** | **주간 보도 환경 held-out** |
| `bollard_dark_20` | 20 | **40** | 실측 야간(평균휘도<65) · 참고용 |

`bollard_val_2146` 에는 person 4,736박스 · stairs 0박스도 들어 있다.

## ⚠️ 이름에 "night" 를 쓰지 않는 이유

선별 조건은 manifest 의 `batch=="Bbox_15_new" AND is_night==1 AND n_bollard>0` 이지만,
**`is_night` 은 무효한 필드다.**

`is_night = (19 <= hour or hour <= 5)` 이고, 그 `hour` 는
`~/bammasil/scripts/bollard_night_all.py:37` 이
`datetime.fromtimestamp(f.stat().st_mtime).hour` — **파일 mtime** 에서 뽑는다
(zip 경로는 `:58` 의 아카이브 기록시각). 촬영시각이 아니라 추출·복사 시각이다.

근거 (→ `~/bammasil/outputs/luminance_audit.md` 6절):
- `Bbox_15_new/Bbox_0910` 한 세션의 mtime 범위가 **584.2일** — 초당 2~3프레임 연속 영상에서 불가능
- 프레임 순서대로 mtime 이 **역행**한다 (B102152 가 B102151 보다 2분 28초 이르다)
- **EXIF 촬영시각이 아예 없다** (태그 수 0)

그래서 이 조건으로 고른 2,146장의 실제 휘도를 다시 쟀다:

    평균휘도  min=41.9  p5=80.1  중앙값=108.9  p95=133.3  max=160.0
    평균휘도 < 65 (실측 야간) : 20장 (0.9%) / 볼라드 40박스
    → **99.1% 가 주간이다.**

### 임계 민감도

    luma < 55  :    8장 /   15박스
    luma < 60  :   14장 /   27박스
    luma < 62  :   16장 /   32박스
    luma < 65  :   20장 /   40박스
    luma < 68  :   28장 /   60박스
    luma < 70  :   33장 /   81박스
    luma < 75  :   56장 /  131박스

기본값 **65** 는 `bollard_dark_20` 의 이름·박스 수(20장/40박스)에 맞춘 것이다.
`luminance_audit.md` 3절의 일반 임계 **60** 을 쓰면 14장/27박스가 되고, 결론은 바뀌지 않는다
(어느 쪽이든 표본이 너무 작아 순위 판정 불가).

AIHub 인도보행 전체로도 실측 야간은 220/37,961 = **0.58%** 다.
**AIHub 인도보행은 야간 데이터셋이 아니다.** 야간 볼라드 성능은 이 셋으로 측정할 수 없고,
자체 야간 촬영분(`C2`)이 필요하다.

## 누수 확인

- 학습셋 `detect_v3_full/images/train` 에 `Bbox_15_new` 는 **0장** — 미포함 ✅
- 학습에 쓰인 AIHub 배치는 29개로 `Bbox_15_new` 를 제외한 전부다
- 따라서 두 셋 모두 **이미지·세션 양쪽에서 held-out** 이다
  (`bollard_dark_20` 은 `bollard_val_2146` 의 부분집합이다)

## 구성

- 세션(sequence) 39개 · 배치 1개(`Bbox_15_new`)
- 해상도 640×360 (AIHub 원본의 640 리사이즈본)
- 이미지·라벨 모두 `outputs/datasets/aihub_yolo_v3/` 에서 **하드링크**. 라벨 재생성 없음.
- `/mnt/d` 를 읽지 않는다 — 2,146장 전량이 이미 홈(ext4)에 있다.

## 재현

```bash
./.venv/bin/python scripts/build_bollard_val_2146.py --dark-max-luma 65
```

입력: `/home/kihun/bammasil/outputs/manifest/bollard_night_images.csv`
      `outputs/datasets/aihub_yolo_v3/{images,labels}/val/`

각 셋의 `manifest.csv` 에 이미지별 `luma`·`is_dark`·클래스별 박스 수가 들어 있다.
