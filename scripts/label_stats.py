"""YOLO 라벨 디렉토리의 박스 통계 — 라벨 규칙이 실제로 지켜졌는지 재는 자.

두 가지 용도가 하나의 자로 묶인다.

    1. **규칙 역산** — StairNet 처럼 문서화된 라벨 규칙이 없는 외부 데이터에서
       "실제로 적용된 규칙"을 숫자로 뽑는다. `docs/labeling_stairs.md` 의 근거표가
       이 스크립트 출력이다.
    2. **자체 촬영분 검수** (`C3`) — 라벨링이 끝난 뒤 같은 자로 재서 가이드와
       어긋난 곳을 찾는다. 특히 **경계 접촉률**과 **작은 박스 비율**이 StairNet 과
       크게 다르면 라벨러가 규칙을 다르게 이해했다는 신호다.

왜 경계 접촉률을 보나
    "부분 가림을 어떻게 처리하는가" 가 이 한 숫자에 드러난다. 보이는 부분까지만
    박스를 치면 잘린 계단이 화면 끝에 붙어 접촉률이 높고, 가려진 곳을 추정해서
    넓히면 낮아진다. StairNet 은 76.3%(train) 다 — **잘리면 잘린 대로 친다.**

왜 작은 박스 비율을 보나
    `stairs` 의 최대 미검증 항목이 "보행 중 멀리 있는 계단" 이다. StairNet 은
    높이 32px 미만 박스가 **0.0%** 라 이 질문에 답할 표본이 없다. 자체 촬영분에서
    이 비율이 여전히 0 에 가깝다면 촬영이든 라벨링이든 거리 계단을 담지 못한
    것이므로, `C5` 판정 전에 잡아야 한다 (→ docs/labeling_stairs.md).

⚠️ 이미지당 박스 수가 1 로 고정돼 보이더라도 그것이 원본의 성질이라고 읽지 말 것.
    StairNet 은 선분 라벨이고, "계단 전체 1박스" 는 `stairnet_to_bbox.py` 가 만든
    것이다. 즉 **한 프레임에 계단이 둘일 때 어떻게 하는지는 원본에 답이 없다.**

사용:
    uv run python scripts/label_stats.py                         # stairs_yolo train·val
    uv run python scripts/label_stats.py --labels outputs/datasets/own_night/labels/train
    uv run python scripts/label_stats.py --class-id 1 --imgsz 1280 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 콘솔 코드페이지가 cp949 면 ⚠️ 하나에 스크립트가 죽는다. 출력만 utf-8 로 고정한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [ROOT / "outputs/datasets/stairs_yolo/labels/train",
           ROOT / "outputs/datasets/stairs_yolo/labels/val"]

# 경계 접촉 판정 여유. 라벨 툴의 반올림·1px 오차를 흡수한다.
EDGE_TOL_PX = 2.0
# "작은 박스" 기준 — YOLO11 의 최소 stride 가 8 이라 높이 32px 은 특징맵 4칸이다.
# 이보다 작으면 탐지기가 구조적으로 불리하다.
SMALL_H_PX = 32


def load(label_dir: Path, class_id: int | None, imgsz: tuple[int, int] | None):
    """YOLO txt 를 읽어 픽셀 좌표 박스로 편다.

    이미지 크기는 짝이 되는 이미지에서 읽는다. 없으면 `--imgsz` 를 쓰고, 그것도
    없으면 정규화 좌표만으로 낼 수 있는 통계(면적·종횡비·경계)만 낸다.
    """
    import cv2

    img_root = label_dir.parent.parent / "images" / label_dir.name
    rows = []
    for f in sorted(label_dir.glob("*.txt")):
        wh = imgsz
        if wh is None and img_root.is_dir():
            img = next((p for p in img_root.glob(f.stem + ".*")), None)
            if img is not None:
                im = cv2.imread(str(img))
                if im is not None:
                    wh = (im.shape[1], im.shape[0])
        lines = [l.split() for l in f.read_text().strip().splitlines() if l.strip()]
        keep = [L for L in lines if class_id is None or int(L[0]) == class_id]
        for L in keep:
            cx, cy, bw, bh = (float(v) for v in L[1:5])
            rows.append((f.stem, wh, cx, cy, bw, bh, len(keep)))
    return rows


def report(name: str, rows: list) -> None:
    import numpy as np

    if not rows:
        print(f"[{name}] 박스 0개 — 건너뜀\n")
        return

    cx = np.array([r[2] for r in rows])
    cy = np.array([r[3] for r in rows])
    bw = np.array([r[4] for r in rows])
    bh = np.array([r[5] for r in rows])
    nb = np.array([r[6] for r in rows])
    n_img = len(set(r[0] for r in rows))

    have_wh = all(r[1] is not None for r in rows)
    if have_wh:
        W = np.array([r[1][0] for r in rows], float)
        H = np.array([r[1][1] for r in rows], float)
        sizes = sorted(set(zip(W.astype(int), H.astype(int))))
    else:
        W = H = None

    print("=" * 74)
    print(f"[{name}]  이미지 {n_img}장 · 박스 {len(rows)}개 · "
          f"이미지당 {nb.min()}~{nb.max()} (중앙 {int(np.median(nb))})")
    if have_wh:
        shown = ", ".join(f"{w}×{h}" for w, h in sizes[:4])
        print(f"   해상도: {shown}{' …' if len(sizes) > 4 else ''}")
    print("-" * 74)

    def q(v, label, unit=""):
        print(f"  {label:<20}min {v.min():7.1f}{unit} │ p1 {np.percentile(v,1):7.1f}{unit}"
              f" │ p5 {np.percentile(v,5):7.1f}{unit} │ p50 {np.percentile(v,50):7.1f}{unit}"
              f" │ max {v.max():7.1f}{unit}")

    if have_wh:
        q(bw * W, "박스 폭 (px)")
        q(bh * H, "박스 높이 (px)")
    q(bw * bh * 100, "프레임 대비 면적", "%")
    q((bw * W) / (bh * H) if have_wh else bw / bh, "종횡비 (w/h)")

    tol_x = EDGE_TOL_PX / (W if have_wh else 512.0)
    tol_y = EDGE_TOL_PX / (H if have_wh else 512.0)
    left, right = cx - bw / 2 <= tol_x, cx + bw / 2 >= 1 - tol_x
    top, bottom = cy - bh / 2 <= tol_y, cy + bh / 2 >= 1 - tol_y
    any_edge = left | right | top | bottom

    print("-" * 74)
    print(f"  경계 접촉(잘린 채로 라벨): 좌 {left.mean()*100:5.1f}%  우 {right.mean()*100:5.1f}%"
          f"  상 {top.mean()*100:5.1f}%  하 {bottom.mean()*100:5.1f}%"
          f"  → 하나라도 {any_edge.mean()*100:5.1f}%")
    print(f"  좌우 관통 {(left & right).mean()*100:5.1f}%   "
          f"상하 관통 {(top & bottom).mean()*100:5.1f}%")

    area = bw * bh * 100
    print(f"  작은 박스: 면적<10% {(area<10).mean()*100:5.1f}%   "
          f"<5% {(area<5).mean()*100:5.1f}%   <1% {(area<1).mean()*100:5.1f}%", end="")
    if have_wh:
        small = (bh * H) < SMALL_H_PX
        print(f"   │ 높이<{SMALL_H_PX}px {small.mean()*100:5.1f}%")
        if small.mean() < 0.001:
            print(f"  ⚠️ 높이 {SMALL_H_PX}px 미만이 **사실상 0** 이다 — 이 클래스의 "
                  f"'멀리 있는' 사례를 측정할 표본이 없다 (→ docs/labeling_stairs.md 3장)")
    else:
        print()
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", type=Path, nargs="*", default=DEFAULT,
                   help="YOLO 라벨 디렉토리. 여러 개 주면 각각 따로 낸다")
    p.add_argument("--class-id", type=int, default=None,
                   help="이 클래스만 본다 (통합 데이터셋은 stairs=1)")
    p.add_argument("--imgsz", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="이미지를 못 찾을 때 쓸 고정 해상도")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    imgsz = tuple(args.imgsz) if args.imgsz else None
    for d in args.labels:
        d = Path(d)
        if not d.is_dir():
            print(f"⚠️ 없는 디렉토리 — 건너뜀: {d}\n")
            continue
        report(f"{d.parent.parent.name}/{d.name}", load(d, args.class_id, imgsz))


if __name__ == "__main__":
    main()
