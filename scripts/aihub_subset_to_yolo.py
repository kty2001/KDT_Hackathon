"""`C4e` S2 — AIHub **YOLO 변환본 서브셋** → 학습용 `aihub_yolo` 레이아웃.

왜 이 스크립트가 필요한가 — 다리가 하나 없다
    `D:\\datasets\\bammasil_aihub_subset` 에 **3클래스 YOLO 변환본 113,163장**이 이미 있는데
    학습에 못 들어간다.

      · `build_detect_dataset.py` 의 `AIHUB` 가 보는 `outputs/datasets/aihub_yolo` 에는
        **옛 샘플 238/58장**뿐이다.
      · `aihub_to_yolo.py` 는 `glob("*.xml")` 로 **CVAT XML** 을 기대한다. 서브셋은 이미
        YOLO 라 그 스크립트가 못 받는다 (그쪽은 XML 전용으로 그대로 둔다).

    즉 막고 있는 것은 **데이터가 아니라 배선 한 칸**이다. 이 스크립트가 그 칸이다.

입력 레이아웃 (서브셋 원본 — 수정하지 않는다)
    <src>/bammasil_aihub_part1_images-002/<그룹>/images/*.jpg
    <src>/bammasil_aihub_part1_labels/<그룹>/labels/*.txt
    <src>/bammasil_aihub_part2_images-001/bollard_day/images/*.jpg
    <src>/bammasil_aihub_part2_labels/bollard_day/labels/*.txt

    | 그룹            |    장수 | person박스 | bollard박스 | 세션 |
    |-----------------|--------:|-----------:|------------:|-----:|
    | bollard_night   |  12,806 |     34,019 |      42,397 | 1,305|
    | negative_night  |   3,451 |          0 |           0 |   521|
    | person_night    |  21,704 |     71,625 |           0 | 1,721|
    | bollard_day     |  75,202 |     96,673 |     250,955 | 2,116|

⚠️ **`_night` 접미사를 믿지 말 것** — 무효 필드 `is_night`(파일 mtime 기반)에서 나온
    이름이라 **99.1% 가 실제로는 주간**이다 (→ STATUS ★인수분 R2 · 3장 함정 13).
    데이터는 실재하고 이름만 틀렸다. 여기서도 그룹명은 **내용 구분**으로만 쓴다.

★ 기본값이 `person_night` 를 **제외**하는 이유
    AIHub 를 쓰는 이유는 **`bollard` 가 여기밖에 없어서**다. `person_night` 는 person 만
    있는 (사실상) 주간 프레임이라 **볼라드를 하나도 안 주면서 주간 person 만 늘린다** —
    `C4b` 가 어렵게 얻은 실야간 person 비중을 희석할 뿐이다(`C4` 붕괴가 정확히 그
    조건이었다). person 은 NightOwls 4,819장이 **실야간 손 라벨**로 이미 대고 있다.

    ⚠️ 다만 `bollard_*` 그룹 안의 person 라벨은 **반드시 그대로 실어야 한다.** 빼면
    "사람=배경"을 가르치게 된다 (→ STATUS 3장 함정 4 · aihub_to_yolo.py 머리말).
    필요하면 `--groups` 로 덮어쓸 수 있다.

★ 분할은 **세션(블록) 단위** — 무작위 금지
    파일명이 `<zip>__<블록>__<프레임>` 이라 블록 키를 **파일명에서 그대로 뽑는다**
    (새 파싱 불요). 인접 프레임은 사실상 같은 장면이라 무작위로 나누면 train 과 val 에
    쌍둥이가 갈려 **val 이 부풀어 오른다** (→ detection.md 6-2 의 recording 단위 분할과
    같은 논리). 마지막에 **누수 0 을 검사**하고 어긋나면 종료한다.

    🔴 **블록 키에 그룹을 넣지 말 것** (2026-08-23 실측으로 잡은 결함).
    처음엔 `(그룹, 블록)` 을 키로 썼는데, **전체 블록 2,348개 중 1,808개(77%)가 여러
    그룹에 걸쳐 있다** — 그룹은 무효 필드 `is_night` 기반 분류라 **같은 촬영 세션이
    그룹마다 쪼개져 있기** 때문이다. 그 상태로 나누면 같은 세션이 train 과 val 에
    갈리고도 "누수 0" 으로 보고된다(실측: val 2,205장 중 **568장 오염**).
    → **블록은 그룹을 가로질러 원자 단위**로 다룬다. 한 블록이 여러 그룹에 걸쳐 있으면
    선택된 그룹의 프레임이 **함께 움직인다.**

★ 서브샘플과 배경 비율
    전량(113K)을 그대로 합치면 주간이 압도해 **`C4` 붕괴 조건이 재현된다**
    (→ STATUS 2장 5). `--max-images` 로 c4d 선례와 같은 자릿수(11K)에 맞춰 규모 축을
    고정한다. 배경(빈 라벨) **10%** 는 인수분 R1 2번이 근거다 — `noneg`(배경 0%)는 mAP 가
    좋아 **보이지만** 6런 중 오탐이 최다였다. 배경의 효과는 mAP 가 아니라 **FP 에 나타난다.**
    ★ 이 배경이 `stairs` 실야간 오탐의 **유일한 해법**이기도 하다 (→ detection.md 9-7 1번).

🔴 `--dst` 는 `D:` 에 둘 것 (→ STATUS 3장 함정 14)
    원본이 전부 `D:` 인데 기본 `outputs/` 는 `C:` 라 `os.link` 가 볼륨을 못 넘어
    **통째로 복사**된다(장당 ~1MB). dst 를 `D:` 에 두면 하드링크로 붙는다.

사용:
    uv run python scripts/aihub_subset_to_yolo.py --dry-run
    uv run python scripts/aihub_subset_to_yolo.py `
        --dst "D:\\datasets\\_derived\\aihub_yolo_11k" --max-images 11000 --bg-ratio 0.10
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nightowls_yolo import CLASS_NAMES, write_data_yaml  # noqa: E402

SRC = Path(r"D:\datasets\bammasil_aihub_subset")
DST = ROOT / "outputs/datasets/aihub_yolo"

# 기본 그룹 — 머리말의 "person_night 를 빼는 이유" 참고
POS_GROUPS = "bollard_night,bollard_day"
BG_GROUPS = "negative_night"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC,
                   help="서브셋 루트 (<part>_images*/<그룹>/images 구조)")
    p.add_argument("--dst", type=Path, default=DST,
                   help="출력. 🔴 원본과 **같은 드라이브**에 둘 것 (함정 14)")
    p.add_argument("--groups", default=POS_GROUPS,
                   help=f"양성 그룹 (쉼표). 기본 {POS_GROUPS} — `person_night` 는 볼라드를 "
                        "안 주면서 주간 person 만 늘려 기본 제외한다")
    p.add_argument("--bg-groups", default=BG_GROUPS,
                   help=f"배경(빈 라벨) 그룹 (쉼표). 기본 {BG_GROUPS}")
    p.add_argument("--max-images", type=int, default=11000,
                   help="총 장수 상한. 전량을 넣으면 `C4` 붕괴 조건이 재현된다")
    p.add_argument("--bg-ratio", type=float, default=0.10,
                   help="배경 비율. 인수분 R1 2번이 근거 (0% 는 오탐 최다)")
    p.add_argument("--val-ratio", type=float, default=0.2, help="세션 단위 val 비율")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="분포·선별 결과만 출력하고 파일은 만들지 않는다")
    return p.parse_args()


def block_of(stem: str) -> str:
    """`<zip>__<블록>__<프레임>` → `<zip>__<블록>`. 파일명이 곧 세션 키다."""
    parts = stem.split("__")
    return "__".join(parts[:2]) if len(parts) >= 3 else stem


def scan(src: Path, groups: set[str]) -> dict[str, list[tuple[Path, Path, str]]]:
    """**블록 → [(이미지, 라벨, 그룹)]**. 블록이 유일한 키다 — 머리말의 🔴 참고.

    라벨이 없는 이미지는 버린다. 그룹은 통계·필터용으로만 싣는다.
    """
    blocks: dict[str, list[tuple[Path, Path, str]]] = defaultdict(list)
    for lab_root in sorted(src.glob("*_labels")):
        prefix = lab_root.name[: -len("_labels")]
        img_roots = sorted(src.glob(f"{prefix}_images*"))
        if not img_roots:
            continue
        for group_dir in sorted(lab_root.iterdir()):
            if not group_dir.is_dir() or group_dir.name not in groups:
                continue
            labels = group_dir / "labels"
            images = None
            for ir in img_roots:
                cand = ir / group_dir.name / "images"
                if cand.is_dir():
                    images = cand
                    break
            if images is None or not labels.is_dir():
                continue
            by_img = {p.stem: p for p in images.iterdir() if p.is_file()}
            for lab in sorted(labels.glob("*.txt")):
                img = by_img.get(lab.stem)
                if img is not None:
                    blocks[block_of(lab.stem)].append((img, lab, group_dir.name))
    return dict(blocks)


def pick_blocks(pool: dict[str, list], target: int, rng: random.Random,
                exclude: set[str] | None = None) -> dict[str, list]:
    """목표 장수에 닿을 때까지 **블록 통째로** 고른다 (세션을 쪼개지 않는다)."""
    keys = [b for b in pool if not exclude or b not in exclude]
    rng.shuffle(keys)
    chosen, n = {}, 0
    for b in keys:
        if n >= target:
            break
        chosen[b] = pool[b]
        n += len(pool[b])
    return chosen


def split_blocks(chosen: dict[str, list], val_ratio: float,
                 rng: random.Random) -> dict[str, dict[str, list]]:
    """고른 블록을 train/val 로 가른다 — **블록이 쪼개지지 않으므로 누수가 없다.**"""
    keys = list(chosen)
    rng.shuffle(keys)
    total = sum(len(chosen[b]) for b in keys)
    want_val = total * val_ratio
    val, train, n = {}, {}, 0
    for b in keys:
        if n < want_val:
            val[b] = chosen[b]
            n += len(chosen[b])
        else:
            train[b] = chosen[b]
    return {"train": train, "val": val}


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
    """같은 볼륨이면 하드링크, 아니면 복사 (→ STATUS 3장 함정 14)."""
    if dst.exists():
        return
    if _rewritten_by_ultralytics(src):   # → STATUS 3장 함정 19
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def count_boxes(items: list[tuple[Path, Path, str]]) -> tuple[dict[int, int], int]:
    """클래스별 박스 수와 **빈 라벨(배경) 장수**. 선별된 것만 읽는다."""
    box: dict[int, int] = {c: 0 for c in CLASS_NAMES}
    empty = 0
    for _, lab, _g in items:
        rows = [r for r in lab.read_text(encoding="utf-8").splitlines() if r.strip()]
        if not rows:
            empty += 1
            continue
        for r in rows:
            cid = int(float(r.split()[0]))
            if cid in box:
                box[cid] += 1
    return box, empty


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"서브셋이 없다: {args.src}")
    rng = random.Random(args.seed)

    pos_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    bg_groups = [g.strip() for g in args.bg_groups.split(",") if g.strip()]

    print("=" * 84)
    print(f"AIHub 서브셋 → 학습 레이아웃   {args.src}")
    print(f"   양성 {pos_groups} · 배경 {bg_groups}")
    print(f"   상한 {args.max_images:,}장 · 배경 {args.bg_ratio:.0%} · val {args.val_ratio:.0%}")
    print("=" * 84)

    # 🔴 블록이 유일한 키다 — 그룹을 키에 넣으면 같은 세션이 갈린다 (머리말 참고).
    #    한 번에 훑고, 그룹은 항목 필터로만 쓴다.
    pool = scan(args.src, set(pos_groups) | set(bg_groups))
    if not pool:
        raise SystemExit(f"블록을 못 찾았다: {pos_groups + bg_groups}\n{args.src} 구조 확인")

    def group_counts(blocks: dict[str, list]) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for items in blocks.values():
            for _, _, g in items:
                out[g] += 1
        return dict(out)

    print("\n[보유]")
    for g, n in sorted(group_counts(pool).items()):
        kind = "양성" if g in pos_groups else "배경"
        print(f"  {kind} {g:<16} 장수 {n:>7,}")
    spanning = sum(1 for items in pool.values() if len({g for _, _, g in items}) > 1)
    print(f"  블록 {len(pool):,}개 · 그중 **여러 그룹에 걸친 것 {spanning:,}개** "
          f"({spanning / len(pool):.0%}) — 블록을 원자 단위로 다루는 이유")

    # ★ **배경을 가진 블록부터 채운다.** 배경 프레임의 대부분(실측 91%)이 양성 블록에
    #   섞여 있어, 배경 전용 블록만 긁으면 목표 비율에 영영 못 닿는다.
    #   블록이 원자 단위이므로 배경을 고르면 그 블록의 양성 프레임도 따라온다 — 정상이다.
    #   ⚠️ 배경을 **밀도 순**으로 고른다 — 무작위로 고르면 큰 블록이 걸려 장수 상한을
    #   빨리 먹고 세션 다양성이 3분의 1로 줄어든다(실측 611 → 190). 같은 배경을 더 적은
    #   프레임으로 얻으면 남는 예산이 전부 다른 세션으로 간다.
    n_bg = round(args.max_images * args.bg_ratio)
    has_bg = {b: i for b, i in pool.items() if any(g in bg_groups for _, _, g in i)}

    def bg_density(b: str) -> float:
        items = has_bg[b]
        return sum(1 for _, _, g in items if g in bg_groups) / len(items)

    chosen: dict[str, list] = {}
    got = 0
    for b in sorted(has_bg, key=lambda b: (-bg_density(b), rng.random())):
        if got >= n_bg:
            break
        chosen[b] = has_bg[b]
        got += sum(1 for _, _, g in has_bg[b] if g in bg_groups)
    if got < n_bg:
        print(f"\n⚠️ 배경이 {got:,}장뿐이다 (목표 {n_bg:,}) — 보유량 자체가 모자란다.")
    # 나머지를 양성 블록으로 채워 장수 상한을 맞춘다
    chosen |= pick_blocks(pool, args.max_images - sum(len(i) for i in chosen.values()),
                          rng, exclude=set(chosen))
    if not has_bg:
        print(f"\n⚠️ 배경 그룹 {bg_groups} 을 못 찾았다 — **배경 0% 로 학습하면 오탐이 는다**"
              " (인수분 R1 2번). 그룹명을 확인할 것")

    splits = split_blocks(chosen, args.val_ratio, rng)

    # --- 누수 검사: 한 블록이 train 과 val 에 동시에 있으면 안 된다 ---
    leak = set(splits["train"]) & set(splits["val"])
    if leak:
        raise SystemExit(f"🔴 세션 누수 {len(leak)}건 — 분할 로직이 틀렸다: {sorted(leak)[:3]}")

    print("\n[선별]")
    total_box: dict[int, int] = {c: 0 for c in CLASS_NAMES}
    total_empty = total_n = 0
    for split, blocks in splits.items():
        items = [it for i in blocks.values() for it in i]
        box, empty = count_boxes(items)
        for c in box:
            total_box[c] += box[c]
        total_empty += empty
        total_n += len(items)
        desc = " · ".join(f"{CLASS_NAMES[c]} {box[c]:,}" for c in sorted(box))
        print(f"  {split:<6} 장수 {len(items):>7,} · 세션 {len(blocks):>5,} · {desc}"
              f" · 배경 {empty:,} ({empty / max(1, len(items)):.1%})")

    print(f"\n  합계   장수 {total_n:,} · 세션 {len(chosen):,} · "
          + " · ".join(f"{CLASS_NAMES[c]} {total_box[c]:,}" for c in sorted(total_box)))
    print(f"  배경   {total_empty:,}장 ({total_empty / max(1, total_n):.1%}) — 목표 {args.bg_ratio:.0%}")
    print(f"  ✅ 세션 누수 0 (train {len(splits['train']):,} · val {len(splits['val']):,} 블록, 교집합 없음)")

    if args.dry_run:
        print("\n--dry-run — 파일을 만들지 않았다.")
        return

    print(f"\n[출력] {args.dst}")
    same_volume = args.dst.drive.upper() == args.src.drive.upper()
    if not same_volume:
        print(f"  🔴 dst 가 원본과 **다른 드라이브**다 ({args.src.drive} → {args.dst.drive}).")
        print("     하드링크가 안 걸려 통째로 복사된다 — 장당 ~1MB (→ STATUS 3장 함정 14).")

    for split, blocks in splits.items():
        out_img = args.dst / "images" / split
        out_lab = args.dst / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)
        n = 0
        for items in blocks.values():
            for img, lab, _g in items:
                name = f"aihub_{img.stem}"          # 병합 시 다른 소스와 안 겹치게
                link_or_copy(img, out_img / f"{name}{img.suffix}")
                link_or_copy(lab, out_lab / f"{name}.txt")
                n += 1
        print(f"  {split:<6} {n:,}장")

    yaml = write_data_yaml(args.dst, "scripts/aihub_subset_to_yolo.py",
                           "images/train", "images/val",
                           extra=f"AIHub 서브셋 · 양성={','.join(pos_groups)} "
                                 f"· 배경={total_empty / max(1, total_n):.1%} · 세션 단위 분할")
    print(f"  {yaml}")

    print("\n다음 — 이 산출물을 `--aihub-src` 로 넘긴다")
    print("  uv run python scripts/build_detect_dataset.py --nightowls --aihub --loli-n 0 \\")
    print(f"      --aihub-src \"{args.dst}\" --dst \"<detect_v3 경로>\"")
    print("\n⚠️ 판정 게이트는 `C4c` 것을 그대로 쓴다 — rec34 person recall **0.609 무회귀** ·")
    print("   `stairs` 야간 오탐 **0.2% 이하**. 볼라드를 얻자고 기존 성능을 깎으면 실패다.")


if __name__ == "__main__":
    main()
