"""C1 — StairNet 선분 라벨 → YOLO BBox 변환 (`stairs` 클래스 사전학습용).

원본 라벨 형식 (실측 2026-08-01)
    `class x1 y1 x2 y2` — 계단 모서리 **선분의 두 끝점**. 512×512 픽셀 좌표.
    class 0 = convex(볼록, 디딤판 앞코) 18,478개 · class 1 = concave(오목) 8,055개.
    선분의 99.2% 가 |dy| <= |dx| 인 **수평에 가까운 선**이다 (좌표계 해석 검증용).

왜 BBox 인가
    ③ 은 YOLO 단일 모델로 `person` + `stairs` 를 함께 검출한다 (→ data.md 4-3).
    StairNet 의 선분 표현은 그 모델에 그대로 못 넣으므로 박스로 바꾼다.

⚠️ 박스 정의 — **계단 전체를 하나의 박스**로 잡는다
    선분 단위로 박스를 만들면 높이 0 에 가까운 박스가 이미지당 중앙 9개씩 쌓인다.
    탐지기 학습에도 나쁘고, ④ 강조 렌더에서 저시력 사용자 화면을 가로줄로
    덮어버려 **목적 자체에 반한다**(비채움·최소 개입 설계, → scripts/emphasize.py).
    보행 보조에 필요한 신호는 "여기 계단이 있다" 하나다.

    ⚠️ 이 정의는 **잠정**이다. `stairs` 박스 정의(전체/단위·부분가림·최소크기)는
    자체 촬영분 라벨링 착수 전에 팀 회의로 확정해야 하며(→ TODO.md 회의 안건 4),
    그때 결론이 다르면 **본 스크립트를 다시 돌려 맞추면 된다** — 원본은 손대지
    않으므로 재생성이 자유롭다.

출력 — 원본을 수정하지 않는다 (CLAUDE.md 규칙)
    outputs/datasets/stairs_yolo/
    ├── images/{train,val}/   원본 이미지 **하드링크** (복사 아님 — 디스크 0 추가)
    ├── labels/{train,val}/   변환된 YOLO 라벨
    └── data.yaml             ultralytics 학습 설정

사용:
    uv run python scripts/stairnet_to_bbox.py              # 변환 + 검증 시트
    uv run python scripts/stairnet_to_bbox.py --class-id 1 # 통합 모델 클래스 id
    uv run python scripts/stairnet_to_bbox.py --no-sheet
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/Stair dataset"
DST = ROOT / "outputs/datasets/stairs_yolo"

# 통합 모델의 클래스 배치 — LoLI-Street 동봉 라벨(COCO person=0)과 맞춘다.
# 클래스 배치는 nightowls_yolo 한 곳에만 둔다 — 예전엔 이 dict 가 파일마다 복사돼
# 있어서, 클래스를 늘릴 때 한쪽만 고치면 소스별로 id 가 어긋날 수 있었다.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nightowls_yolo import CLASS_NAMES, STAIRS_ID  # noqa: E402

# 선분 끝점만으로 만든 박스는 계단 '면'보다 얇다. 디딤판 두께만큼 위아래로
# 살짝 넓혀야 실제 계단 영역을 덮는다. 이미지 높이 대비 비율.
PAD_RATIO = 0.01


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class-id", type=int, default=STAIRS_ID,
                   help=f"출력 라벨의 클래스 id (기본 {STAIRS_ID} = stairs)")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--pad", type=float, default=PAD_RATIO, help="박스 상하 여유 비율")
    p.add_argument("--no-sheet", action="store_true", help="검증 시트를 만들지 않는다")
    p.add_argument("--sheet-n", type=int, default=12, help="검증 시트에 실을 장 수")
    return p.parse_args()


def read_segments(path: Path) -> list[tuple[int, int, int, int, int]]:
    out = []
    for row in path.read_text().splitlines():
        parts = row.split()
        if len(parts) < 5:
            continue
        c, x1, y1, x2, y2 = (int(float(v)) for v in parts[:5])
        out.append((c, x1, y1, x2, y2))
    return out


def to_box(segments, w: int, h: int, pad: float) -> tuple[float, float, float, float]:
    """선분 전체를 감싸는 박스 하나. 반환은 YOLO 정규화 (cx, cy, bw, bh)."""
    xs = [v for _, x1, _, x2, _ in segments for v in (x1, x2)]
    ys = [v for _, _, y1, _, y2 in segments for v in (y1, y2)]

    py = pad * h
    # 원본에 512 를 넘는 좌표가 소수 있다(val 최대 513) — 경계로 자른다
    x0, x1 = max(0.0, min(xs)), min(float(w), max(xs))
    y0, y1 = max(0.0, min(ys) - py), min(float(h), max(ys) + py)

    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


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
    """하드링크 우선 — 디스크가 늘지 않는다.

    ⚠️ **"수정 위험이 없다"는 틀렸다** (8/25 실측) — ultralytics 가 EOI 없는 JPEG 을
    다시 쓰면 링크를 타고 원본이 재인코딩된다. 그 경우만 복사한다 (→ 함정 19).
    """
    if dst.exists():
        return
    if _rewritten_by_ultralytics(src):   # → STATUS 3장 함정 19
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:  # 다른 볼륨·파일시스템 미지원 시
        shutil.copy2(src, dst)


def convert(split: str, args) -> tuple[int, list[float], list[np.ndarray]]:
    img_dir = SRC / split / "images"
    lab_dir = SRC / split / "labels"
    out_img = args.dst / "images" / split
    out_lab = args.dst / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    areas: list[float] = []
    n = 0
    for lab in sorted(lab_dir.glob("*.txt")):
        img = next((p for p in img_dir.glob(f"{lab.stem}.*")), None)
        if img is None:
            continue
        segments = read_segments(lab)
        if not segments:
            continue  # 박스 없는 이미지는 배경으로 남길 수도 있으나, 원본에 0건이다

        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        cx, cy, bw, bh = to_box(segments, w, h, args.pad)

        (out_lab / f"{lab.stem}.txt").write_text(
            f"{args.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        link_or_copy(img, out_img / img.name)
        areas.append(bw * bh)
        n += 1

    return n, areas, []


def build_sheet(split: str, args, out_path: Path) -> None:
    """변환 결과를 눈으로 검증한다 — 선분(초록)과 생성된 박스(주황)를 겹쳐 그린다."""
    img_dir = SRC / split / "images"
    lab_dir = SRC / split / "labels"
    labs = sorted(lab_dir.glob("*.txt"))
    step = max(1, len(labs) // args.sheet_n)
    picked = labs[::step][:args.sheet_n]

    cells, cell_w = [], 256
    for lab in picked:
        img = next((p for p in img_dir.glob(f"{lab.stem}.*")), None)
        if img is None:
            continue
        im = cv2.imread(str(img))
        h, w = im.shape[:2]
        segments = read_segments(lab)
        for c, x1, y1, x2, y2 in segments:
            color = (120, 220, 120) if c == 0 else (220, 200, 120)
            cv2.line(im, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        cx, cy, bw, bh = to_box(segments, w, h, args.pad)
        p1 = (int((cx - bw / 2) * w), int((cy - bh / 2) * h))
        p2 = (int((cx + bw / 2) * w), int((cy + bh / 2) * h))
        cv2.rectangle(im, p1, p2, (40, 120, 250), 2)
        cv2.putText(im, f"{lab.stem}  {len(segments)}선", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cv2.resize(im, (cell_w, cell_w)))

    cols = 4
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells) - cols + 1, cols)]
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    print(f"  검증 시트: {out_path}  (초록·노랑 = 원본 선분 / 주황 = 생성된 박스)")


def main() -> None:
    args = parse_args()
    if not SRC.is_dir():
        raise SystemExit(f"StairNet 을 찾을 수 없다: {SRC}")

    print("=" * 78)
    print("C1 — StairNet 선분 → YOLO BBox 변환")
    print(f"   박스 정의: **계단 전체 1박스** (선분 union + 상하 {args.pad:.1%} 여유)")
    print(f"   클래스 id: {args.class_id} ({CLASS_NAMES.get(args.class_id, '?')})")
    print("=" * 78)

    total = {}
    for split in ("train", "val"):
        n, areas, _ = convert(split, args)
        total[split] = n
        a = np.array(areas)
        print(f"\n[{split}] {n:,}장 변환")
        print(f"  박스 면적(프레임 대비): 중앙 {np.median(a):.1%} · "
              f"최소 {a.min():.1%} · 최대 {a.max():.1%}")
        print(f"  프레임의 80% 이상을 덮는 박스: {100 * (a >= 0.8).mean():.1f}% "
              f"· 5% 미만: {100 * (a < 0.05).mean():.1f}%")
        if not args.no_sheet:
            build_sheet(split, args, ROOT / f"outputs/stairs/bbox_check_{split}.png")

    yaml = args.dst / "data.yaml"
    yaml.write_text(
        "# C1 산출물 — scripts/stairnet_to_bbox.py 가 생성한다. 직접 수정하지 말 것.\n"
        f"path: {args.dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items())),
        encoding="utf-8")

    print(f"\n출력: {args.dst}")
    print(f"  data.yaml — 클래스 {len(CLASS_NAMES)}개 {list(CLASS_NAMES.values())}")
    print(f"  train {total['train']:,} / val {total['val']:,}장 (이미지는 하드링크)")
    print("\n⚠️ 박스 정의는 잠정이다 — 회의 안건 4 확정 후 필요하면 다시 돌릴 것")


if __name__ == "__main__":
    main()
