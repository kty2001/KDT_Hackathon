"""AIHub 인도보행영상 (CVAT XML) → YOLO 라벨. **3클래스 스키마** 로 좁혀 담는다.

    0 person   1 stairs   2 bollard        (2026-08-03 확정)

AIHub 원본은 29종이지만 우리는 3종만 쓴다. `stairs` 는 AIHub 에 **없으므로**
(스키마·박스 모두 0건) 이 스크립트가 내는 것은 `person` 과 `bollard` 뿐이다.

★ 나머지 26종을 그냥 버려도 되는가 — 된다 (실측 근거)
    라벨에서 빠진 것은 "배경"으로 학습된다. NightOwls `ignore` 함정(사람이 있는데
    라벨이 비어 "사람=배경"을 가르친 것 → STATUS 3장 함정 4)이 재현될까 봐 쟀는데,
    **`bollard` 와 혼동될 만한 것이 없다.**

        클래스        높이 중앙   상단 접촉    성격
        bollard          75px      0.0%     땅에 붙은 짧은 기둥
        pole            648px     87.3%     화면 위로 뻗는 장대 (8.6배 높다)
        tree_trunk      395px     21.2%     굵고 길다

    `pole`·`tree_trunk` 를 배경으로 가르치는 것은 오히려 **"장대에는 반응하지 마라"**
    는 올바른 지도다. 함정은 *같은 클래스*가 라벨 없이 남을 때 생기는 것이고,
    여기서는 다른 물체다.

    ⚠️ 단 `person` 은 다르다 — AIHub 에도 person 이 525박스(213/299장) 있어 **반드시
    라벨해야 한다.** 빼면 그 순간 진짜 "사람=배경" 이 된다. 프레임을 버리는 선택지는
    213/299 장을 버리는 것이라 성립하지 않는다.

⚠️ `bollard` 는 작다 — 낮은 recall 을 예상하고 시작할 것
    1920×1080 원본을 640 입력으로 줄이면 스케일이 0.333 이다.

        640 환산 폭   p5 2.2px · p25 3.8px · **p50 7.8px** · p95 57.6px
        폭 8px 미만 **50.3%** · 4px 미만 28.0%

    YOLO11 의 최소 stride 가 8 이라 **절반이 특징맵 한 칸도 못 채운다.** 데이터를 더
    넣어 풀리는 문제가 아니라 입력 해상도의 구조적 한계다. 작은 것을 버리지는 않는다 —
    라벨이 있어야 "못 잡는다"를 측정할 수 있고, 없으면 "안 잡힌 게 정상"이 되어
    영영 안 보인다 (→ docs/labeling_stairs.md 3장과 같은 논리).

⚠️ **분할은 파일명 블록 단위로 한다 — 무작위 금지**
    샘플 296장은 `MP_SEL_B027451`~`B027750` 의 **연속 영상 프레임**이다. 인접 프레임은
    사실상 같은 장면이라 무작위로 나누면 train 과 val 에 쌍둥이가 갈려 **val 이 부풀어
    오른다.** NightOwls 를 recording 단위로 나눈 것과 같은 이유다
    (→ detection.md 6-2). 앞쪽 블록을 train, 뒤쪽을 val 로 자른다.

⚠️ **주간 전용이다** — 단 이 단정은 **검증 중이다** (2026-08-05)
    근거는 샘플 297장 육안이었는데, 전량 1/30 에서 저조도 ~600장이 보고됐다.
    `aihub_pack_for_colab.py --dry-run` 의 정찰이 **밝기와 광원을 별개 축으로** 재서
    가른다 — 어둡기만 한 것(LOL·LoLI 부류, 포화 광원 0.00%)은 야간이 아니다
    (→ data.md 3-1-4). 결과가 나오기 전까지는 `C4` 의 "주간 학습 → 실야간 붕괴"
    전례가 그대로 적용된다고 보고, 야간 볼라드 성능은 `C2` 자체 촬영분으로 판정한다.

출력 — 원본 미수정 (CLAUDE.md)
    outputs/datasets/aihub_yolo/
    ├── images/{train,val}/   원본 **하드링크**
    └── labels/{train,val}/

사용:
    uv run python scripts/aihub_to_yolo.py                    # 샘플 변환
    uv run python scripts/aihub_to_yolo.py --val-ratio 0.25
    uv run python scripts/aihub_to_yolo.py --pole-as-bollard  # 2 = 기둥형 전체 (아래 참고)
    uv run python scripts/aihub_to_yolo.py --src <전량 다운로드 경로> --dst ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 클래스 배치의 유일한 정의처. 여기서 다시 정의하면 소스별로 id 가 어긋난다.
from nightowls_yolo import BOLLARD_ID, CLASS_NAMES, PERSON_ID  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/AIHub인도보행영상_sample/bbox"
DST = ROOT / "outputs/datasets/aihub_yolo"

# AIHub 라벨명 → 통합 클래스 id. 여기 없는 29종의 나머지는 **버린다**(=배경).
# `stairs` 는 AIHub 에 없으므로(스키마·박스 모두 0건) 여기 등장하지 않는다.
LABEL_MAP = {"person": PERSON_ID, "bollard": BOLLARD_ID}

# `--pole-as-bollard` 를 켜면 여기까지 클래스 2 로 흡수한다.
#
#   🗣️ 이것은 **제품 정의 문제**이지 기술 문제가 아니다. STATUS 2장은 필요한 것을
#   "일반 장애물(볼라드·**전봇대** 등)" 이라고 적고 있는데 전봇대가 곧 `pole` 이다.
#   확정된 클래스명은 "볼라드" 라서 기본값은 **좁은 정의**(bollard 만)로 두었다.
#   넓히려면 이 플래그를 켜고 재학습하면 된다 — 라벨 원본은 그대로 있다.
POLE_LIKE = {"pole": BOLLARD_ID, "tree_trunk": BOLLARD_ID}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC,
                   help="CVAT XML 과 이미지가 있는 디렉토리")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="뒤쪽 이 비율을 val 로. **무작위가 아니라 블록 분할**")
    p.add_argument("--pole-as-bollard", action="store_true",
                   help="pole·tree_trunk 도 클래스 2 로 흡수 (제품 정의 — 위 참고)")
    p.add_argument("--min-box-px", type=float, default=0.0,
                   help="원본 해상도 기준 이 폭 미만 박스를 버린다. 기본 0(=전부 유지)")
    return p.parse_args()


def _rewritten_by_ultralytics(src: Path) -> bool:
    """ultralytics 가 데이터셋 스캔 중에 이 파일을 **덮어쓸 것인가**.

    `data/utils.check_image` 는 JPEG 의 **마지막 2바이트가 EOI(`FFD9`)가 아니면**
    "corrupt JPEG restored and saved" 를 찍고 `Image.save(im_file, quality=100)` 로
    **그 경로에 다시 쓴다.** 하드링크면 그 쓰기가 **원본을 관통한다**
    (→ STATUS 3장 함정 19 · 실측 3.37MB → 7.27MB). 그런 파일만 복사한다 —
    대부분은 정상이라 하드링크의 이득(AIHub 113,163장)은 그대로 남는다.
    """
    if src.suffix.lower() not in (".jpg", ".jpeg"):
        return False
    try:
        with open(src, "rb") as f:
            f.seek(-2, 2)
            return f.read() != b"\xff\xd9"
    except OSError:
        return True          # 못 읽으면 링크하지 않는다 (안전한 쪽)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if _rewritten_by_ultralytics(src):   # → STATUS 3장 함정 19
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def parse_xml(xml_path: Path, mapping: dict[str, int], min_px: float):
    """CVAT `image/box` → (파일명, 폭, 높이, [(cls, cx, cy, w, h)]) 목록.

    CVAT 좌표는 절대 픽셀(`xtl,ytl,xbr,ybr`)이라 YOLO 정규화 중심좌표로 바꾼다.
    좌표가 프레임 밖으로 조금 나간 라벨이 있어 클램프한다 — 안 하면 ultralytics 가
    학습 중에 조용히 거른다.
    """
    root = ET.parse(xml_path).getroot()
    out, dropped_small, dropped_class = [], 0, Counter()

    for im in root.findall("image"):
        name = im.get("name")
        W, H = float(im.get("width")), float(im.get("height"))
        rows = []
        for b in im.findall("box"):
            lab = b.get("label")
            if lab not in mapping:
                dropped_class[lab] += 1
                continue
            x1 = max(0.0, min(W, float(b.get("xtl"))))
            y1 = max(0.0, min(H, float(b.get("ytl"))))
            x2 = max(0.0, min(W, float(b.get("xbr"))))
            y2 = max(0.0, min(H, float(b.get("ybr"))))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < min_px:
                dropped_small += 1
                continue
            rows.append((mapping[lab],
                         (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                         (x2 - x1) / W, (y2 - y1) / H))
        out.append((name, W, H, rows))
    return out, dropped_small, dropped_class


def main() -> None:
    args = parse_args()
    xmls = sorted(args.src.glob("*.xml"))
    if not xmls:
        raise SystemExit(f"CVAT XML 이 없다: {args.src}")

    mapping = dict(LABEL_MAP)
    if args.pole_as_bollard:
        mapping.update(POLE_LIKE)

    print("=" * 78)
    print("AIHub 인도보행 → YOLO (3클래스: person / stairs / bollard)")
    print(f"   원본 {args.src}")
    print(f"   매핑 {mapping}"
          f"{'   ★ pole·tree_trunk 흡수' if args.pole_as_bollard else ''}")
    print(f"   나머지 클래스는 버린다(=배경). stairs 는 AIHub 에 없다.")
    print("=" * 78)

    records, n_small, dropped = [], 0, Counter()
    for x in xmls:
        r, s, d = parse_xml(x, mapping, args.min_box_px)
        records += r
        n_small += s
        dropped += d

    # ★ 무작위 금지 — 연속 프레임이라 파일명 순으로 앞/뒤를 자른다
    records.sort(key=lambda r: r[0])
    n_val = int(len(records) * args.val_ratio)
    splits = {"train": records[:len(records) - n_val], "val": records[len(records) - n_val:]}
    if n_val:
        print(f"\n블록 분할 — train {splits['train'][0][0]} … {splits['train'][-1][0]}")
        print(f"           val   {splits['val'][0][0]} … {splits['val'][-1][0]}")

    total = Counter()
    for split, recs in splits.items():
        out_img = args.dst / "images" / split
        out_lab = args.dst / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)

        n_img = n_missing = 0
        per_class = Counter()
        for name, W, H, rows in recs:
            src_img = args.src / name
            if not src_img.is_file():
                n_missing += 1
                continue
            stem = Path(name).stem
            link_or_copy(src_img, out_img / f"aihub_{stem}{Path(name).suffix}")
            (out_lab / f"aihub_{stem}.txt").write_text(
                "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                        for c, cx, cy, w, h in rows), encoding="utf-8")
            for c, *_ in rows:
                per_class[c] += 1
            n_img += 1

        total += per_class
        detail = " · ".join(f"{CLASS_NAMES[c]} {per_class[c]:,}"
                            for c in sorted(per_class)) or "박스 없음"
        print(f"\n[{split}] 이미지 {n_img:,}장 · {detail}")
        if n_missing:
            print(f"  ⚠️ XML 에 있으나 이미지가 없는 것 {n_missing}장 — 건너뜀")

    print("\n" + "-" * 78)
    print(f"합계 " + " · ".join(f"{CLASS_NAMES[c]} {total[c]:,}" for c in sorted(total)))
    if args.min_box_px:
        print(f"폭 {args.min_box_px}px 미만으로 버린 박스 {n_small:,}개")
    print(f"버린 클래스 상위: "
          + ", ".join(f"{k} {v}" for k, v in dropped.most_common(6)))
    print(f"\n출력: {args.dst}")
    print("\n⚠️ 주간 전용이다. 야간 볼라드 성능은 이 데이터로 판정되지 않는다 —")
    print("   `C2` 자체 촬영분에 볼라드 장면이 있어야 한다 (→ data.md 3-1).")


if __name__ == "__main__":
    main()
