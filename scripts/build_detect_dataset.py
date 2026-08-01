"""C4 준비 — ③ 통합 탐지 데이터셋 구성 (`person` + `stairs` 단일 모델).

두 소스를 하나의 YOLO 데이터셋으로 합친다.

    person  ← LoLI-Street (동봉 YOLO 라벨에서 person 만 추출)
    stairs  ← StairNet (scripts/stairnet_to_bbox.py 산출물)

★ 핵심 설계 — **`low` 이미지에 `high` 라벨을 붙인다** (2026-08-01 실측 근거)

    LoLI-Street 는 low/high 양쪽에 라벨이 따로 들어 있는데, **둘이 일치하지 않는다.**
    실측(train 200장):

        low/high 픽셀 정합    구조 상관 0.991 — 같은 장면·같은 크기 ✅
        person 박스 수        low 294 vs high 367 (**high 가 +24.8%**)
        개수가 같은 경우에도  좌표가 다른 것이 19/45

    → 두 버전에 **탐지기를 각각 돌려 만든 pseudo-label** 이다. 그리고 어두운 쪽에서
      탐지기가 놓친 사람이 `low` 라벨에서 **배경으로 기록**돼 있다.

    `low` 이미지 + `low` 라벨로 학습하면 **"어두우면 놓치는 것"을 그대로 학습한다.**
    이 프로젝트가 원하는 것과 정반대다. 두 버전이 픽셀 정합이므로 **`high` 라벨을
    `low` 이미지의 정답으로 쓰는 것이 기하학적으로 타당하고 더 완전하다.**

    ⚠️ 그래도 `high` 라벨 역시 pseudo-label 이다 — 사람 손 라벨이 아니다.
    **최종 평가셋으로 쓰지 말 것** (→ data.md 2-1). 평가는 자체 촬영 야간분(`C5`)으로.

출력 — 원본 미수정, 이미지는 하드링크
    outputs/datasets/detect_v1/
    ├── images/{train,val}/
    ├── labels/{train,val}/
    └── data.yaml

사용:
    uv run python scripts/build_detect_dataset.py                 # 기본
    uv run python scripts/build_detect_dataset.py --loli-n 12000  # person 표본 늘리기
    uv run python scripts/build_detect_dataset.py --label-src low # 대조 실험용(비권장)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOLI = ROOT / "data/LoLI-Street/LoLI-Street Dataset"
STAIRS = ROOT / "outputs/datasets/stairs_yolo"
DST = ROOT / "outputs/datasets/detect_v1"

CLASS_NAMES = {0: "person", 1: "stairs"}
PERSON_SRC_ID = 0  # LoLI-Street 는 COCO 인덱스 — person 이 0

# 라벨 디렉토리는 split 마다 이름 규칙이 다르다 (원본 배포가 그렇다)
LABEL_DIRS = {
    ("Train", "low"): LOLI / "YOLO Annotations/Train/low/Labels",
    ("Train", "high"): LOLI / "YOLO Annotations/Train/high/Labels",
    ("Val", "low"): LOLI / "YOLO Annotations/Val/YOLO Annotations (low)/Labels",
    ("Val", "high"): LOLI / "YOLO Annotations/Val/YOLO Annotations (high)/Labels",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loli-n", type=int, default=6000,
                   help="train 에 넣을 LoLI 이미지 수 (person 박스 있는 것 우선). "
                        "0 이면 전량. 기본 6000 — stairs 2,670 과의 균형·학습시간 고려")
    p.add_argument("--loli-val-n", type=int, default=800, help="val 에 넣을 LoLI 수")
    p.add_argument("--label-src", choices=["high", "low"], default="high",
                   help="LoLI 라벨을 어느 쪽에서 가져올지. 기본 high (모듈 docstring 참고)")
    p.add_argument("--image-src", choices=["low", "high", "both"], default="low",
                   help="LoLI 이미지를 어느 쪽에서 가져올지. 기본 low (야간 도메인)")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def person_rows(label_path: Path) -> list[str]:
    """person 만 남기고 클래스 id 를 통합 배치(0)로 맞춘다."""
    if not label_path.is_file():
        return []
    out = []
    for row in label_path.read_text().splitlines():
        parts = row.split()
        if len(parts) < 5 or int(float(parts[0])) != PERSON_SRC_ID:
            continue
        out.append("0 " + " ".join(parts[1:5]))
    return out


def add_loli(split: str, out_split: str, n: int, args, rng) -> tuple[int, int]:
    lab_dir = LABEL_DIRS[(split, args.label_src)]
    sides = ["low", "high"] if args.image_src == "both" else [args.image_src]

    # person 박스가 있는 이미지를 우선한다 — 빈 프레임만 잔뜩 넣으면 학습이 흐려진다
    stems = sorted(p.stem for p in (LOLI / split / sides[0]).glob("*"))
    rng.shuffle(stems)

    picked: list[tuple[str, list[str]]] = []
    empties: list[str] = []
    for s in stems:
        rows = person_rows(lab_dir / f"{s}.txt")
        if rows:
            picked.append((s, rows))
        else:
            empties.append(s)
        if n and len(picked) >= n:
            break

    # 배경(음성) 표본을 10% 섞는다 — 전부 양성이면 오탐이 늘어난다
    n_bg = int(len(picked) * 0.1)
    picked += [(s, []) for s in empties[:n_bg]]

    out_img = args.dst / "images" / out_split
    out_lab = args.dst / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    boxes = 0
    for stem, rows in picked:
        for side in sides:
            img = next((LOLI / split / side).glob(f"{stem}.*"), None)
            if img is None:
                continue
            name = f"loli_{side}_{stem}"
            link_or_copy(img, out_img / f"{name}{img.suffix}")
            (out_lab / f"{name}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            boxes += len(rows)

    return len(picked) * len(sides), boxes


def add_stairs(split: str, out_split: str, args) -> tuple[int, int]:
    src_img = STAIRS / "images" / split
    src_lab = STAIRS / "labels" / split
    if not src_img.is_dir():
        raise SystemExit(
            f"StairNet 변환본이 없다: {src_img}\n"
            "먼저 실행할 것: uv run python scripts/stairnet_to_bbox.py")

    out_img = args.dst / "images" / out_split
    out_lab = args.dst / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    n = boxes = 0
    for img in sorted(src_img.glob("*")):
        lab = src_lab / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        name = f"stair_{img.stem}"
        link_or_copy(img, out_img / f"{name}{img.suffix}")
        text = lab.read_text(encoding="utf-8")
        (out_lab / f"{name}.txt").write_text(text, encoding="utf-8")
        boxes += len([r for r in text.splitlines() if r.strip()])
        n += 1
    return n, boxes


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 78)
    print("C4 준비 — ③ 통합 탐지 데이터셋 (person + stairs)")
    print(f"   LoLI 이미지 {args.image_src} · 라벨 {args.label_src}"
          f"{'  ★ 권장 조합' if (args.image_src, args.label_src) == ('low', 'high') else '  ⚠️ 비권장'}")
    print("=" * 78)

    stats = {}
    for split, out_split, n in (("Train", "train", args.loli_n),
                                ("Val", "val", args.loli_val_n)):
        li, lb = add_loli(split, out_split, n, args, rng)
        si, sb = add_stairs(out_split, out_split, args)
        stats[out_split] = (li, lb, si, sb)
        print(f"\n[{out_split}]")
        print(f"  LoLI   이미지 {li:,} · person 박스 {lb:,}")
        print(f"  Stairs 이미지 {si:,} · stairs 박스 {sb:,}")
        print(f"  합계   이미지 {li + si:,} · 박스 {lb + sb:,}")
        if sb:
            print(f"  ⚠️ 클래스 불균형 person:stairs = {lb / sb:.1f} : 1")

    yaml = args.dst / "data.yaml"
    yaml.write_text(
        "# C4 산출물 — scripts/build_detect_dataset.py 가 생성한다. 직접 수정하지 말 것.\n"
        f"# LoLI 이미지={args.image_src} / 라벨={args.label_src}\n"
        f"path: {args.dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items())),
        encoding="utf-8")

    print(f"\n출력: {args.dst}")
    print(f"  {yaml}")
    print("\n⚠️ 이 val 은 **개발용**이다. LoLI 라벨은 pseudo-label 이고 StairNet 은")
    print("   84% 가 주간이라, **야간 실효 성능의 최종 판정에 쓰면 안 된다.**")
    print("   최종 평가셋은 자체 촬영 야간분(`C5`) 이다 (→ TODO.md).")


if __name__ == "__main__":
    main()
