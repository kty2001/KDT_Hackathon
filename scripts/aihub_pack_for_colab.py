"""AIHub 인도보행 다운로드분 → **Colab 으로 옮길 수 있는 크기**의 YOLO 데이터셋.

★ 왜 이 스크립트가 필요한가 — **AI Hub 는 해외 IP 다운로드를 차단한다** (2026-08-04 실측)

    Colab 런타임에서 `aihubshell` 을 돌리면 이렇게 끝난다.

        Download failed with HTTP status 502.
        AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다.

    Colab 은 미국·유럽에 뜨므로 이 경로는 원천적으로 막혀 있다. 그래서 **다운로드는
    국내에서** 하고, 학습만 Colab 에서 한다. 문제는 원천이 태스크당 10GB 씩이라
    그대로는 Drive 로 못 옮긴다는 것인데 — **학습 입력이 어차피 640 이라 미리 줄이면
    정보 손실 없이 10배 이상 작아진다.** 1920×1080 3만 장 30GB → 640×360 2~3GB.

★ 그리고 이 데이터셋은 **라벨과 원천이 안 갈려 있다** (2026-08-04 실측)

    `aihubshell -mode l -datasetkey 189` 의 실제 트리는 이렇다.

        뎁스프리딕션        Depth_001~005.zip     19GB …
        바운딩박스          Bbox_1_new.zip        10GB … (30개)
        서피스마스킹        Surface_1.zip         10GB … (5개)
        폴리곤세그멘테이션  Polygon_1_new.zip     10GB … (14개)

    **태스크별로만 나뉘고 라벨/원천 구분이 없다.** data.md 3-1-3 이 계획한 "라벨 전량을
    먼저 받아 분포를 실측한 뒤 원천을 고른다" 는 순서가 성립하지 않는다. 대신 **태스크별
    1번 zip 만 먼저 받아 정찰**한다 — 목적(원천 600GB 를 헛되이 받지 않기)은 같다.

    받을 것:  바운딩박스(person·bollard) · 서피스마스킹(stairs)
    안 받을 것: 폴리곤세그멘테이션(같은 장애물 29종이라 얻을 게 없다) · 뎁스(무관)

무엇을 하는가
    1. 받은 zip/디렉토리를 훑어 **CVAT XML 을 태스크별로 자동 분류**한다
       (장애물 29종이 보이면 bbox · 노면 20종이 보이면 surface)
    2. 장애물 → `person`(0) · `bollard`(2)   — `aihub_to_yolo.parse_xml` 재사용
    3. 노면 `caution_zone > stairs` 폴리곤 → bbox → `stairs`(1)
    4. ★ **게이트** — 노면 stairs 의 면적·종횡비를 StairNet 과 대조해
       **학습 투입 / 평가 전용**을 가른다 (→ detection.md 8-3)
    5. ★ **정찰** — 저조도 프레임을 절대 휘도로 분류하고 **촬영 블록 수**를 센다
       (아래 "정찰이 답하는 두 질문")
    6. 이미지를 긴 변 640 으로 줄여 JPEG 재인코딩
    7. **파일명 블록 분할**(무작위 금지 — 연속 프레임이다) · data.yaml · stats.json
    8. `--zip` 이면 Drive 에 올릴 zip 까지 만든다

★ 정찰이 답하는 두 질문 (2026-08-05 추가)

    전량 다운로드분에서 "볼라드 3,500장 · 저조도 600장" 같은 장수가 나왔을 때,
    **그 숫자만으로는 학습 계획을 세울 수 없다.** 두 가지가 빠져 있다.

    ① **몇 장면인가** — 이 데이터는 연속 영상 프레임이다. 3,500장이 클립 20개일 수도
       있고 300개일 수도 있는데, val 을 가르는 단위는 장이 아니라 **블록**이다.
       무작위로 나누면 인접 프레임 쌍둥이가 train/val 에 갈려 mAP 가 부풀어 오른다
       (NightOwls 를 recording 단위로 나눈 것과 같은 이유 → detection.md 6-2).

    ② **"저조도"가 야간인가 그늘인가** — 이건 `C2` 자체 야간 촬영의 존폐가 걸린
       질문이다. 어둡기만 한 것과 **광원이 있는 야간**은 다른 문제이고, 우리가
       풀려는 것은 후자다. 그래서 밝기와 광원을 **별개 축**으로 잰다.
       실측 근거는 아래 `REF_LUMA` 주석에 있다.

사용:
    # 1) 국내에서 AIHub 로그인 → 바운딩박스 1번·서피스마스킹 1번 zip 다운로드·압축해제
    # 2) 변환·리사이즈·패키징
    uv run python scripts/aihub_pack_for_colab.py --src D:\\datasets\\AIHub인도보행 --zip
    uv run python scripts/aihub_pack_for_colab.py --src ... --no-surface   # 장애물만
    uv run python scripts/aihub_pack_for_colab.py --src ... --dry-run      # 분포만 센다

    # 3) outputs/datasets/aihub_colab.zip 을 Drive/bammasil/datasets/ 에 올린다
    # 4) notebooks/colab_aihub_train.ipynb 실행
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aihub_to_yolo import LABEL_MAP, POLE_LIKE, parse_xml  # noqa: E402
from metrics import LAMP_MIN, luma  # noqa: E402
from nightowls_yolo import CLASS_NAMES, STAIRS_ID, write_data_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "outputs/datasets/aihub_colab"

# ⚠️ 한국어 Windows 콘솔은 cp949 라 `⚠️`·`★` 에서 UnicodeEncodeError 로 **죽는다.**
# 30GB 를 다 훑고 마지막 출력에서 죽으면 그 실행이 통째로 날아가므로, 인코딩은
# 그대로 두고 못 쓰는 글자만 흘린다. (`chcp 65001` 을 요구하지 않는 쪽을 택했다 —
# 이 스크립트는 국내 PC 에서 팀원이 한 번 돌리는 것이다.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 노면 20종 중 우리가 찾는 것. `caution_zone` 아래 5종 중 하나라 라벨명이 그대로
# `stairs` 일 수도, `caution_zone` + 속성일 수도 있어 양쪽을 다 본다.
STAIRS_KEYS = ("stairs", "stair")

# 태스크 판별용 지문 — 두 스키마에만 있는 이름들
BBOX_MARKERS = {"bollard", "tree_trunk", "traffic_sign", "pole", "car"}
SURFACE_MARKERS = {"sidewalk", "roadway", "caution_zone", "braille_guide_blocks",
                   "bike_lane", "alley"}

# StairNet 실측 (→ detection.md 8-1) — 게이트의 비교 기준
STAIRNET_AREA_MEDIAN = {"train": 0.475, "val": 0.383}

# ─────────────────────────── 정찰 임계 — 보유 데이터로 보정했다 (2026-08-05 실측)
#
# 640 환산 · 표본 296~300장씩. 밝기와 광원이 **따로 논다**는 것이 요점이다.
#
#     표본                       평균휘도 중앙   암부(<=80)   포화(>=235)   성분수
#     AIHub 샘플 (문서상 주간)      113.4  (p5 94.4)   32%       3.58%       152
#     LOL low   (실제 저조도)        13.8  (p95 31.0)  100%      **0.00%**     0
#     LOL high  (정상 노출)         115.3               25%       0.05%        10
#
# 밝기 축: 주간 p5(94.4) 와 저조도 p95(31.0) 사이가 넓게 비어 있어 절대 임계를
#   안전하게 놓을 수 있다. 백분위로 정의하지 않는 이유는 STATUS 3장 함정 2 그대로다 —
#   상대 정의는 영상이 바뀌면 재는 대상 자체가 달라진다.
#
# ★ 광원 축이 밝기와 별개인 이유: **LOL low 는 포화 광원이 0.00% 다.** 어둡지만
#   광원이 없는 영상이고, 이는 LoLI-Street 가 "글레어를 학습·측정 불가"였던 것과
#   같은 부류다 (→ data.md 2-1). 우리가 풀려는 야간은 가로등·헤드라이트·간판이
#   있는 장면이므로, "어둡다"만 보고 야간이라 부르면 안 된다.
#   ⚠️ 보유 데이터에 **진짜 야간 가로등 표본이 없다**(7/29 확인). 그래서 이 축은
#   자동 판정을 좁게만 하고(광원 유무), 야간/그늘 판별은 사람이 눈으로 확정한다.
LUMA_DAY_MIN = 90.0        # 이상이면 주간   (AIHub 주간 p5 = 94.4 바로 아래)
LUMA_LOWLIGHT_MAX = 40.0   # 미만이면 저조도 (LOL low p95 = 31.0 위로 여유)
LAMP_PRESENT_PCT = 0.02    # 포화화소 비율이 이 이상이면 '광원이 있다'
                           # (LOL low p95 0.01 · AIHub 주간 p5 0.06 사이)

REF_LUMA = {
    "LOL low (실저조도·광원없음)": {"lamp": 0.00, "big": 0.00, "ncc": 0},
    "AIHub 샘플 (주간·하늘)":      {"lamp": 3.58, "big": 1.32, "ncc": 152},
}

# 파일명 번호가 이 이상 뛰면 다른 촬영 블록으로 본다. AIHub 는 연속 프레임이
# 1씩 증가하고(`MP_SEL_B027451`~`B027750`), 라벨 없는 프레임이 빠져 몇 칸씩
# 건너뛸 수 있어 여유를 준다.
BLOCK_GAP = 30


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="압축 해제한 AIHub 다운로드분의 루트 (하위를 전부 훑는다)")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--imgsz", type=int, default=640,
                   help="긴 변 기준. 학습 입력과 같게 두면 정보 손실이 없다")
    p.add_argument("--quality", type=int, default=90, help="JPEG 품질")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--max-images", type=int, default=0,
                   help="장수 상한. **균등 간격으로 솎아 낸다**(앞부분만 자르지 않는다). "
                        "0=전부. Drive 여유가 빠듯할 때 쓴다 — 장당 ~80KB")
    p.add_argument("--no-surface", action="store_true",
                   help="노면 stairs 를 아예 다루지 않는다")
    p.add_argument("--stairs", choices=["auto", "train", "eval"], default="auto",
                   help="노면 stairs 처리. auto=게이트 판정에 따름 · "
                        "train=학습 투입 · eval=평가 전용 분리")
    p.add_argument("--pole-as-bollard", action="store_true",
                   help="pole·tree_trunk 도 클래스 2 로 (제품 정의 — 회의 안건 7)")
    p.add_argument("--workers", type=int, default=8, help="리사이즈·정찰 스레드")
    p.add_argument("--recon-sample", type=int, default=2000,
                   help="밝기 정찰 표본 수 (균등 추출). **0=전수** — 저조도가 잡히면 "
                        "블록 수를 알기 위해 0 으로 다시 돌린다")
    p.add_argument("--block-gap", type=int, default=BLOCK_GAP,
                   help="파일명 번호가 이보다 크게 뛰면 다른 촬영 블록으로 센다")
    p.add_argument("--dry-run", action="store_true",
                   help="분포·게이트·정찰만 보고 파일은 안 쓴다")
    p.add_argument("--zip", action="store_true", help="Drive 업로드용 zip 까지 만든다")
    return p.parse_args()


# ────────────────────────────────────────────────────────────── 분류·파싱

def _has_stairs(*vals) -> bool:
    return any(v is not None and any(k in str(v).lower() for k in STAIRS_KEYS)
               for v in vals)


def classify_xml(xml_path: Path) -> str:
    """XML 하나를 훑어 'bbox' / 'surface' / 'unknown' 판별.

    파일명·디렉토리명으로 가르지 않는다 — AIHub 는 배포마다 폴더 이름을 바꾼다.
    **안에 들어 있는 라벨명**이 스키마의 지문이라 그것으로 가른다.
    """
    labels = set()
    try:
        for _, el in ET.iterparse(xml_path, events=("start",)):
            if el.tag in ("box", "polygon", "polyline", "mask"):
                if el.get("label"):
                    labels.add(el.get("label").lower())
            if len(labels) >= 12:
                break
    except ET.ParseError:
        return "unknown"
    if labels & SURFACE_MARKERS:
        return "surface"
    if labels & BBOX_MARKERS:
        return "bbox"
    return "unknown"


def parse_surface_stairs(xml_path: Path):
    """노면 CVAT XML → (이미지명, W, H, [(x1,y1,x2,y2)]). stairs 만 담는다.

    폴리곤 → bbox 는 min/max 다. `C1`(StairNet 선분→bbox)에서 이미 한 변환이라
    새로운 위험이 없다.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return
    for im in root.findall("image"):
        name = im.get("name")
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        if not (name and W and H):
            continue
        boxes = []
        for shape in ("polygon", "polyline", "box"):
            for el in im.findall(shape):
                attrs = [a.text for a in el.findall("attribute")]
                if not _has_stairs(el.get("label"), *attrs):
                    continue
                if shape == "box":
                    boxes.append((float(el.get("xtl")), float(el.get("ytl")),
                                  float(el.get("xbr")), float(el.get("ybr"))))
                    continue
                pts = [tuple(map(float, q.split(",")))
                       for q in (el.get("points") or "").split(";") if "," in q]
                if len(pts) >= 3:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if boxes:
            yield name, W, H, boxes


def to_yolo_rows(cls: int, boxes, W: float, H: float) -> list[str]:
    rows = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                    f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return rows


# ────────────────────────────────────────────────────────────── 게이트

def stairs_gate(records) -> tuple[bool, dict]:
    """노면 stairs 가 우리 '계단 전체 1박스' 정의와 같은 것인지 가른다.

    노면 라벨은 `grating`·`manhole` 과 나란한 **"노면 구역"** 이다. 폴리곤이 디딤면만
    덮었다면 bbox 가 StairNet 보다 작고 납작해진다. 그러면 학습에 섞지 않고 **평가
    전용**으로 쓴다 — 정의가 섞이면 두 소스를 다 망친다 (→ detection.md 8-3).
    """
    areas, ratios = [], []
    for _, W, H, boxes in records:
        for x1, y1, x2, y2 in boxes:
            w, h = (x2 - x1) / W, (y2 - y1) / H
            if w > 0 and h > 0:
                areas.append(w * h)
                ratios.append(w / h)
    if not areas:
        return False, {}

    def q(v, p):
        return st.quantiles(v, n=100)[p - 1] if len(v) > 2 else float("nan")

    stat = {
        "n_box": len(areas),
        "area_p25": q(areas, 25), "area_median": st.median(areas), "area_p75": q(areas, 75),
        "ratio_median": st.median(ratios),
        "stairnet_area_median": STAIRNET_AREA_MEDIAN,
    }
    return st.median(areas) >= 0.15, stat


# ────────────────────────────────────────────────────── 정찰 (저조도 · 블록)

def filename_blocks(names, gap: int = BLOCK_GAP) -> list[tuple[str, int, int, int]]:
    """파일명을 **연속 촬영 구간(≈세션)** 으로 묶는다 → [(접두어, 시작, 끝, 장수)].

    왜 세는가: 이 데이터는 연속 영상 프레임이라 **장수가 장면 수가 아니다.**
    "볼라드 3,500장" 이 클립 20개면 실효 표본은 20 에 가깝고, val 을 무작위로
    가르면 인접 프레임 쌍둥이가 양쪽에 갈려 수치가 부풀어 오른다. 나누는 단위는
    언제나 블록이다 (→ detection.md 6-2 의 recording 분리와 같은 논리).
    """
    keyed = []
    for n in names:
        m = re.match(r"^(.*?)(\d+)$", Path(n).stem)
        keyed.append((m.group(1), int(m.group(2))) if m else (Path(n).stem, -1))

    blocks: list[list] = []
    for pre, num in sorted(keyed):
        # 번호가 없는 파일(-1)은 이어 붙일 근거가 없으므로 각자 단독 블록이 된다.
        if blocks and blocks[-1][0] == pre and num >= 0 and 0 <= num - blocks[-1][2] <= gap:
            blocks[-1][2] = num
            blocks[-1][3] += 1
        else:
            blocks.append([pre, num, num, 1])
    return [tuple(b) for b in blocks]


def probe_luma(path: Path, side: int = 640) -> dict | None:
    """한 장의 밝기·광원 프로파일. **학습 입력과 같은 640 환산**에서 잰다.

    원본 1920×1080 에서 재면 광원 코어가 리사이즈로 뭉개지기 전 값이라, 정작
    모델이 보는 것과 다른 영상을 재게 된다.

    반환 — mean(평균휘도) · dark(휘도<=80 %) · crush(<3 %) · lamp(>=235 %)
           big(최대 광원 성분 면적 %) · ncc(광원 성분 개수)

    `big`·`ncc` 로 **하늘과 점광원을 가른다.** 주간 하늘은 큰 덩어리 하나이고
    가로등은 작은 점 여러 개다 — 같은 `lamp%` 라도 뜻이 정반대다.
    """
    try:
        # ⚠️ `cv2.imread` 는 Windows 에서 비ASCII 경로를 못 연다. AIHub 압축 해제본
        # 경로에 한글이 들어가면(`AIHub인도보행영상…`) 전부 조용히 None 이 된다.
        buf = np.fromfile(str(path), dtype=np.uint8)
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except OSError:
        return None
    if im is None:
        return None

    h, w = im.shape[:2]
    s = side / max(h, w)
    if s < 1.0:
        im = cv2.resize(im, (max(1, round(w * s)), max(1, round(h * s))),
                        interpolation=cv2.INTER_AREA)
    L = luma(im)
    n = L.size

    lamp = (L >= LAMP_MIN).astype(np.uint8)
    npx = int(lamp.sum())
    big, ncc = 0.0, 0
    if npx:
        _, _, stats, _ = cv2.connectedComponentsWithStats(lamp, 8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        big, ncc = float(areas.max()) / n * 100.0, int(len(areas))

    return {"mean": float(L.mean()),
            "dark": 100.0 * float((L <= 80).mean()),
            "crush": 100.0 * float((L < 3).mean()),
            "lamp": 100.0 * npx / n,
            "big": big, "ncc": ncc}


def band_of(r: dict) -> str:
    if r["mean"] >= LUMA_DAY_MIN:
        return "주간"
    if r["mean"] < LUMA_LOWLIGHT_MAX:
        return "저조도"
    return "박명·그늘"


def recon(obstacle: dict, img_index: dict, args) -> dict:
    """★ 정찰 — 촬영 블록 수와 저조도 분포를 낸다. 학습 계획의 입력이다.

    ⚠️ 표본으로 재면 **저조도 블록 수는 알 수 없다** — 표본에 걸린 몇 장이 한
    클립인지 여러 클립인지 구분되지 않기 때문이다. 저조도가 나오면 전수(`0`)로
    다시 돌린다. 그래서 기본값은 "있는지 없는지"를 싸게 보는 크기로 둔다.
    """
    stat: dict = {}
    names = sorted(obstacle)
    if not names:
        return stat

    bollard = {n for n, rows in obstacle.items() if any(r.startswith("2 ") for r in rows)}

    print("\n[정찰] 촬영 블록 — **연속 프레임이라 장수는 장면 수가 아니다**")
    for label, keys in (("라벨 프레임", names), ("bollard 프레임", sorted(bollard))):
        if not keys:
            continue
        bl = filename_blocks(keys, args.block_gap)
        med = st.median(b[3] for b in bl)
        print(f"  {label:16} {len(keys):>7,}장 → 블록 {len(bl):>5,}개 "
              f"(중앙 {med:,.0f}장/블록)")
        stat[label] = {"프레임": len(keys), "블록": len(bl), "중앙_장당블록": med}
    print("  ★ train/val 을 가르는 단위는 장이 아니라 **블록**이다 (→ detection.md 6-2)")

    # ── 밝기·광원
    picks = names
    if 0 < args.recon_sample < len(picks):
        step = len(picks) / args.recon_sample
        picks = [picks[int(i * step)] for i in range(args.recon_sample)]
    paths = [(n, img_index.get(Path(n).name)) for n in picks]
    paths = [(n, p) for n, p in paths if p is not None]
    if not paths:
        print("\n[정찰] 밝기 — 원천 이미지를 못 찾아 건너뛴다")
        return stat

    print(f"\n[정찰] 밝기 — 640 환산 · 표본 {len(paths):,}장 "
          f"({len(paths) / len(names):.0%})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        probed = list(ex.map(lambda t: probe_luma(t[1], args.imgsz), paths))
    rows = [(n, r) for (n, _), r in zip(paths, probed) if r]
    if not rows:
        print("  ⚠️ 한 장도 못 읽었다 — 경로·포맷을 확인할 것")
        return stat

    bands: dict[str, list] = {}
    for n, r in rows:
        bands.setdefault(band_of(r), []).append((n, r))

    stat["밝기"] = {"표본": len(rows)}
    for band in ("주간", "박명·그늘", "저조도"):
        got = bands.get(band, [])
        nb = sum(1 for n, _ in got if n in bollard)
        rng = {"주간": f">={LUMA_DAY_MIN:.0f}",
               "박명·그늘": f"{LUMA_LOWLIGHT_MAX:.0f}~{LUMA_DAY_MIN:.0f}",
               "저조도": f"<{LUMA_LOWLIGHT_MAX:.0f}"}[band]
        print(f"  {band:10} (평균휘도 {rng:>8})  {len(got):>7,}장 "
              f"{len(got) / len(rows):>6.1%}  · bollard 포함 {nb:,}장")
        stat["밝기"][band] = {"표본장수": len(got), "bollard": nb,
                              "비율": len(got) / len(rows)}

    # ── 어두운 밴드의 광원 프로파일 — 여기서 야간/그늘이 갈린다
    for band in ("저조도", "박명·그늘"):
        got = bands.get(band, [])
        if len(got) < 5:
            continue
        med = {k: st.median(r[k] for _, r in got) for k in ("lamp", "big", "ncc")}
        print(f"\n  ── {band} {len(got):,}장의 광원 프로파일 — **진짜 야간인가**")
        print(f"     포화화소(>={LAMP_MIN:.0f}) 중앙 {med['lamp']:.3f}% · "
              f"최대성분 {med['big']:.3f}% · 성분수 {med['ncc']:.0f}")
        for name, ref in REF_LUMA.items():
            print(f"     대조 — {name:24} {ref['lamp']:.2f}% / "
                  f"{ref['big']:.2f}% / {ref['ncc']}")
        if med["lamp"] < LAMP_PRESENT_PCT:
            print("     → **포화 광원이 없다.** LoLI-Street·LOL low 와 같은 부류라")
            print("        글레어를 학습할 수도 측정할 수도 없다 (→ data.md 2-1).")
            print("        ⚠️ `C2` 자체 야간 촬영을 대체하지 못한다.")
        elif med["big"] > 0.5:
            print("     → 광원 화소가 **큰 덩어리**다. 가로등이 아니라 하늘·반사면일")
            print("        공산이 크다 — 주간 그늘일 수 있으니 눈으로 확인할 것.")
        else:
            print("     → **점광원이 있다.** 야간일 가능성이 높다 — 확정되면")
            print("        `C2` 우선순위와 문서의 '주간 전용' 기술을 다시 잡아야 한다")
            print("        (STATUS 2장 5 · data.md 3-1 · aihub_to_yolo.py 헤더).")
        print("     ⚠️ 자동 판정은 참고값이다. 이 밴드에서 몇 장을 꺼내 눈으로 확정할 것")
        stat.setdefault("광원", {})[band] = med

    if bands.get("저조도") and len(rows) < len(names):
        print(f"\n  ⚠️ 저조도가 잡혔다 — 지금은 표본이라 **몇 클립인지 모른다.**")
        print("     `--recon-sample 0` 으로 전수 측정해야 저조도 블록 수가 나온다.")
    return stat


# ────────────────────────────────────────────────────────────── 출력

def make_portable(yaml_path: Path) -> None:
    """`data.yaml` 에서 `path:` 를 **지운다** — zip 이 다른 머신에서 풀려도 돌게.

    ⚠️ 2026-08-04 실측한 함정. ultralytics 는 `path:` 를 이렇게 푼다.

        절대경로  → 그대로 쓴다
        상대경로  → **`settings['datasets_dir']` 기준** (Colab 은 `/content/datasets`)
        없음      → **yaml 파일이 있는 디렉토리 기준** ← 우리가 원하는 것

    즉 상대경로를 적어 두면 Colab 에서
    `images not found, missing path '/content/datasets/outputs/datasets/...'`
    로 죽고, 절대경로를 적으면 이번엔 **Windows 경로가 박혀** 역시 못 쓴다.
    zip 으로 옮기는 데이터셋은 **`path:` 를 아예 빼는 것**이 유일하게 이식성 있는 답이다.
    """
    lines = [ln for ln in yaml_path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("path:")]
    lines.insert(1, "# path: 를 일부러 뺐다 — ultralytics 가 **이 파일의 위치**를 기준으로")
    lines.insert(2, "# 잡으므로 어느 머신에서 풀든 동작한다 (→ aihub_pack_for_colab.py)")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resize_one(args_tuple) -> bool:
    from PIL import Image

    src, dst, imgsz, quality = args_tuple
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = imgsz / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def emit(records: dict, dst: Path, args, prefix: str, img_index: dict,
         val_ratio: float, max_images: int = 0) -> dict:
    """{이미지명: [YOLO 행]} → images/labels 트리. 반환: split 별 (이미지, 박스) 수."""
    names = sorted(records)                     # ★ 무작위 금지 — 연속 프레임이다

    # 상한이 걸리면 **균등 간격**으로 솎는다. 앞에서 자르면 특정 촬영 구간만 남지만,
    # 균등 추출은 전 구간을 유지하면서 인접 프레임(≈16fps 의 쌍둥이)을 걷어내
    # 오히려 같은 장수당 정보량이 늘어난다. 정렬 순서는 그대로라 블록 분할은 유효하다.
    if 0 < max_images < len(names):
        step = len(names) / max_images
        names = [names[int(i * step)] for i in range(max_images)]
        print(f"  ⚠️ 장수 상한 {max_images:,} — {step:.1f} 프레임마다 하나씩 균등 추출")

    n_val = int(len(names) * val_ratio)
    splits = {"train": names[:len(names) - n_val], "val": names[len(names) - n_val:]}

    out = {}
    for split, keys in splits.items():
        img_dir, lab_dir = dst / "images" / split, dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        jobs, pending = [], []
        for name in keys:
            src = img_index.get(Path(name).name)
            if src is None:
                continue
            stem = f"{prefix}_{Path(name).stem}"
            jobs.append((src, img_dir / f"{stem}.jpg", args.imgsz, args.quality))
            pending.append((stem, records[name]))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            ok = list(ex.map(resize_one, jobs))

        n_img = n_box = 0
        for good, (stem, rows) in zip(ok, pending):
            if not good:
                continue
            (lab_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            n_img += 1
            n_box += len(rows)
        out[split] = (n_img, n_box)
    return out


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"경로가 없다: {args.src}")

    mapping = dict(LABEL_MAP)
    if args.pole_as_bollard:
        mapping.update(POLE_LIKE)

    print("=" * 78)
    print("AIHub 인도보행 → Colab 용 YOLO 데이터셋")
    print(f"   원본 {args.src}")
    print(f"   출력 {args.dst}  ·  긴 변 {args.imgsz}px JPEG q{args.quality}")
    print(f"   매핑 {mapping}")
    print("=" * 78)

    # 1) 이미지 인덱싱 + XML 분류
    img_index: dict[str, Path] = {}
    xmls: list[Path] = []
    for p in args.src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXT:
            img_index.setdefault(p.name, p)
        elif p.suffix.lower() == ".xml":
            xmls.append(p)

    kinds = Counter()
    by_kind: dict[str, list[Path]] = {"bbox": [], "surface": [], "unknown": []}
    for x in xmls:
        k = classify_xml(x)
        kinds[k] += 1
        by_kind[k].append(x)

    print(f"\n이미지 {len(img_index):,}장 · XML {len(xmls):,}개")
    print(f"  장애물(bbox) {kinds['bbox']:,} · 노면(surface) {kinds['surface']:,} "
          f"· 미상 {kinds['unknown']:,}")
    if not img_index:
        raise SystemExit("이미지를 못 찾았다 — zip 압축을 풀었는지 확인할 것")
    if kinds["unknown"]:
        print("  ⚠️ 미상 XML 은 건너뛴다. 스키마가 바뀌었는지 하나 열어 볼 것")

    # 2) 장애물 → person · bollard
    obstacle: dict[str, list[str]] = {}
    dropped = Counter()
    n_person = n_bollard = 0
    for x in by_kind["bbox"]:
        recs, _, drop = parse_xml(x, mapping, 0.0)
        dropped += drop
        for name, W, H, rows in recs:
            if not rows:
                continue
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
            obstacle.setdefault(name, []).extend(lines)
            for c, *_ in rows:
                n_person += c == 0
                n_bollard += c == 2

    print(f"\n[장애물] 프레임 {len(obstacle):,}장 · person {n_person:,} · bollard {n_bollard:,}")
    print("  버린 클래스 상위: "
          + ", ".join(f"{k} {v:,}" for k, v in dropped.most_common(6)))
    if 0 < n_bollard < 1000:
        print(f"  ⚠️ bollard {n_bollard:,}개는 적다 — zip 을 더 받을지 판단할 것")

    # 3) 노면 → stairs (+ 게이트)
    surface_records, gate_stat, use_train = [], {}, False
    if not args.no_surface:
        for x in by_kind["surface"]:
            surface_records += list(parse_surface_stairs(x))
        n_sbox = sum(len(r[3]) for r in surface_records)
        print(f"\n[노면] stairs 프레임 {len(surface_records):,}장 · 박스 {n_sbox:,}개")

        if surface_records:
            use_train, gate_stat = stairs_gate(surface_records)
            print(f"\n  ── 게이트: StairNet 대조 ({gate_stat['n_box']:,} 박스)")
            print(f"     면적 비율  p25 {gate_stat['area_p25']:.3f} · "
                  f"중앙 {gate_stat['area_median']:.3f} · p75 {gate_stat['area_p75']:.3f}")
            print(f"     종횡비 중앙 {gate_stat['ratio_median']:.2f}")
            print(f"     StairNet 면적 중앙 — train {STAIRNET_AREA_MEDIAN['train']} · "
                  f"val {STAIRNET_AREA_MEDIAN['val']}")
            if args.stairs != "auto":
                use_train = args.stairs == "train"
                print(f"     → --stairs {args.stairs} 로 수동 지정됨")
            elif use_train:
                print("     → 계단 덩어리를 덮는 쪽. **학습 투입**")
            else:
                print("     → StairNet 보다 훨씬 작다(디딤면만?). **평가 전용으로 분리**")
            print("     ⚠️ 자동 판정은 참고값이다. 예측 몇 장을 눈으로 보고 확정할 것")
        elif by_kind["surface"]:
            print("  ⚠️ 노면 XML 은 있는데 stairs 가 0 이다 — 라벨명이 다를 수 있다")
        else:
            png = [p for p in args.src.rglob("*.png") if "mask" in p.name.lower()]
            if png:
                print(f"  ⚠️ 마스크로 보이는 PNG {len(png):,}개 — 노면이 **PNG 마스크 포맷**이면")
                print("     신규 변환 코드가 필요하다 (data.md 3-1-2 미확인 항목의 답)")

    # 4) 정찰 — 블록 수 · 저조도 분포 (학습 계획의 입력이다)
    recon_stat = recon(obstacle, img_index, args)

    if args.dry_run:
        print("\n--dry-run — 파일은 쓰지 않았다.")
        return

    # 5) 출력
    stairs_in_train = bool(surface_records) and use_train
    if stairs_in_train:
        for name, W, H, boxes in surface_records:
            rows = to_yolo_rows(STAIRS_ID, boxes, W, H)
            if rows:
                obstacle.setdefault(name, []).extend(rows)

    print()
    res = emit(obstacle, args.dst, args, "aihub", img_index, args.val_ratio,
               args.max_images)
    n_out = sum(v[0] for v in res.values())
    for split, (i, b) in res.items():
        print(f"[{split}] 이미지 {i:,}장 · 박스 {b:,}개")
    print(f"  예상 zip 크기 ≈ {n_out * 80 / 2**20:.2f} GiB  (장당 ~80KB 실측)")

    # 정찰이 저조도를 실제로 세었으므로 "주간 전용"을 단정하지 않는다 — 샘플 297장
    # 육안이 근거였던 8/3 의 기술은 전량에서 깨질 수 있다 (→ recon 의 광원 프로파일).
    lowlight_pct = recon_stat.get("밝기", {}).get("저조도", {}).get("비율", 0.0)
    domain = ("⚠️ 주간 전용" if lowlight_pct < 0.005
              else f"⚠️ 저조도 {lowlight_pct:.1%} 포함 — 야간 여부는 광원 프로파일 확인")
    data_yaml = write_data_yaml(
        args.dst, "scripts/aihub_pack_for_colab.py", "images/train", "images/val",
        extra=f"AIHub 인도보행 · 긴 변 {args.imgsz}px · {domain}")
    make_portable(data_yaml)

    eval_res = {}
    if surface_records and not stairs_in_train:
        eval_dst = args.dst.parent / f"{args.dst.name}_stairs_eval"
        recs = {name: to_yolo_rows(STAIRS_ID, boxes, W, H)
                for name, W, H, boxes in surface_records}
        recs = {k: v for k, v in recs.items() if v}
        eval_res = emit(recs, eval_dst, args, "aihubsurf", img_index, 1.0)
        make_portable(write_data_yaml(
            eval_dst, "scripts/aihub_pack_for_colab.py", "images/val", "images/val",
            extra="노면 stairs — **평가 전용**. 학습에 넣지 말 것"))
        print(f"\n[평가 전용] {eval_dst} — val {eval_res.get('val', (0, 0))[0]:,}장")

    stats = {
        "생성": "scripts/aihub_pack_for_colab.py",
        "원본": str(args.src),
        "클래스": {str(k): v for k, v in CLASS_NAMES.items()},
        "imgsz": args.imgsz,
        "장애물": {"프레임": len(obstacle), "person": n_person, "bollard": n_bollard,
                   "버린_클래스": dict(dropped.most_common(10))},
        "노면_stairs": {"프레임": len(surface_records),
                        "박스": sum(len(r[3]) for r in surface_records),
                        "학습투입": stairs_in_train, "게이트": gate_stat},
        "정찰": recon_stat,
        "분할": {k: {"이미지": v[0], "박스": v[1]} for k, v in res.items()},
    }
    (args.dst / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n산출물: {args.dst}")
    print(f"  data.yaml · stats.json (분포 실측 — **문서에 옮길 것**)")

    if args.zip:
        import shutil
        z = shutil.make_archive(str(args.dst), "zip", root_dir=args.dst)
        print(f"\nzip: {z}  ({Path(z).stat().st_size / 2**30:.2f} GiB)")
        print("  → Drive 의 bammasil/datasets/ 에 올린 뒤")
        print("     notebooks/colab_aihub_train.ipynb 를 실행한다")

    if lowlight_pct < 0.005:
        print("\n⚠️ 주간 전용이다. 야간 성능은 이 데이터로 판정되지 않는다 —")
        print("   held-out(NightOwls rec 34) 또는 자체 촬영분(C5)에서 한다.")
    else:
        print(f"\n⚠️ 저조도가 {lowlight_pct:.1%} 섞여 있다. 위 광원 프로파일로 야간 여부를")
        print("   먼저 확정할 것 — 야간이면 그 블록은 **학습에 다 넣지 말고** 일부를")
        print("   held-out 으로 뗀다. 안 그러면 야간 볼라드를 잴 자가 없어진다.")


if __name__ == "__main__":
    main()
