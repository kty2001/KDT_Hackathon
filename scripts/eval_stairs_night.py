"""③ `stairs` — **야간과 주간을 갈라서** 잰다.

왜 필요한가 — 프로젝트에 비어 있던 측정이다
    `C4b` 는 `stairs` 의 **오탐**을 0 으로 만들었지만(NightOwls 야간 거리), 그건
    *계단이 없는 곳에서 헛보지 않는가*를 잰 것이다. **정작 어두운 곳에서 계단을
    찾아내는가(재현율)는 한 번도 따로 잰 적이 없다.**

    개발 val 의 `stairs` mAP50 0.988 은 **야간 76장과 주간 348장을 섞은** 값이라,
    야간 성능이 낮아도 주간에 묻혀 보이지 않는다. 이 서비스의 목적이 *야간* 보행
    보조이므로 섞어 재면 안 된다.

    ⚠️ 표본이 **야간 76장뿐**이다. 방향은 읽을 수 있어도 정밀한 판정은 못 한다.
    최종 판정은 자체 촬영 야간 계단(`C2`·`C3`) 몫이다.

야간 식별
    StairNet 파일명에 `_night_` 가 들어 있다 (train 429 / val 76 · 전체의 16.3%).
    밝기 추정이 아니라 **원 데이터셋의 라벨**이므로 신뢰할 수 있다.

사용:
    uv run python scripts/eval_stairs_night.py
    uv run python scripts/eval_stairs_night.py --arm D1A1+bf   # ② 를 앞단에 붙였을 때
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2

from nightowls_yolo import write_data_yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/datasets/detect_v2_loli0"
DST_ROOT = ROOT / "outputs/datasets/stairs_split"
STAIR_PREFIX = "stair_"
NIGHT_TAG = "_night_"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=ROOT / "outputs/detect/c4b_loli0/weights/best.pt")
    p.add_argument("--arm", default="none",
                   help="② 를 앞단에 붙여 재려면 arm 이름 (기본 none = 무처리)")
    p.add_argument("--width", type=int, default=0,
                   help="② 내부 처리 가로. 0 이면 원본(512×512) 그대로")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(split: str, arm_name: str, args) -> tuple[Path, int]:
    """야간/주간 부분집합을 만든다. `--arm` 이 주어지면 ② 를 적용한다."""
    tag = f"{split}_{arm_name.replace('+', '_')}"
    dst = DST_ROOT / tag
    out_img, out_lab = dst / "images/val", dst / "labels/val"
    src_img, src_lab = SRC / "images/val", SRC / "labels/val"
    if not src_img.is_dir():
        raise SystemExit(
            f"평가셋이 없다: {src_img}\n"
            "먼저: uv run python scripts/build_detect_dataset.py --nightowls --loli-n 0 "
            "--dst outputs/datasets/detect_v2_loli0")

    want_night = split == "night"
    picked = [p for p in sorted(src_img.glob(f"{STAIR_PREFIX}*"))
              if (NIGHT_TAG in p.name) == want_night]
    if not picked:
        raise SystemExit(f"{split} 표본이 0장")

    if out_img.is_dir() and not args.rebuild and sum(1 for _ in out_img.glob("*")) == len(picked):
        return dst, len(picked)

    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    arm = None
    if arm_name != "none":
        from lowlight import build as build_arm
        arm = build_arm(arm_name)
        if arm.is_temporal:
            raise SystemExit(f"'{arm_name}' 은 시간축 arm 이라 정지영상에 쓸 수 없다")

    for p in picked:
        if arm is None:
            link_or_copy(p, out_img / p.name)
        else:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            h0, w0 = img.shape[:2]
            work = img
            if args.width and w0 != args.width:
                work = cv2.resize(img, (args.width, round(h0 * args.width / w0)),
                                  interpolation=cv2.INTER_AREA)
            out = arm(work)
            if out.shape[:2] != (h0, w0):
                out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_img / p.name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        link_or_copy(src_lab / f"{p.stem}.txt", out_lab / f"{p.stem}.txt")

    write_data_yaml(dst, "scripts/eval_stairs_night.py", "images/val", "images/val")
    return dst, len(picked)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.weights.is_file():
        raise SystemExit(f"가중치가 없다: {args.weights}")

    print("=" * 84)
    print("③ `stairs` — 야간 / 주간 분리 평가 (StairNet val)")
    print(f"   가중치 {args.weights.parent.parent.name} · ② {args.arm}")
    print("=" * 84)

    model = YOLO(str(args.weights))
    rows = []
    for split in ("night", "day"):
        dst, n = build(split, args.arm, args)
        m = model.val(data=str(dst / "data.yaml"), imgsz=args.imgsz, device=0,
                      project=str(ROOT / "outputs/detect"),
                      name=f"stairs_{split}_{args.arm.replace('+', '_')}",
                      exist_ok=True, plots=False, verbose=False)
        idx = list(m.box.ap_class_index)
        i = idx.index(1) if 1 in idx else None      # stairs = 1
        rows.append((split, n,
                     float(m.box.ap50[i]) if i is not None else float("nan"),
                     float(m.box.ap[i]) if i is not None else float("nan"),
                     float(m.box.p[i]) if i is not None else float("nan"),
                     float(m.box.r[i]) if i is not None else float("nan")))

    print("\n" + "=" * 84)
    print("결과 — `stairs`")
    print("=" * 84)
    print(f"{'구간':<8}{'장수':>7}{'mAP50':>10}{'mAP50-95':>11}{'precision':>12}{'recall':>9}")
    print("-" * 57)
    for split, n, ap50, ap, p_, r_ in rows:
        print(f"{split:<8}{n:>7,}{ap50:>10.3f}{ap:>11.3f}{p_:>12.3f}{r_:>9.3f}")

    night, day = rows[0], rows[1]
    gap = night[2] - day[2]
    print(f"\n야간 − 주간 mAP50 차이: {gap:+.3f}")
    if gap < -0.05:
        print("  → ⚠️ **야간에서 유의하게 나쁘다.** 저조도 계단 탐지는 아직 미완이다.")
    elif gap > 0.05:
        print("  → 야간이 오히려 높다. 표본 편중을 의심할 것(야간 76장뿐).")
    else:
        print("  → 야간·주간 차이가 작다. 단 **StairNet 은 계단 중심 사진**이라")
        print("     '보행 중 마주치는 계단'과 다르다는 한계는 그대로다.")

    print("\n⚠️ 야간 표본 76장뿐이다 — 방향만 읽고, 판정은 자체 촬영분(`C2`·`C3`)에서.")
    print("⚠️ StairNet 은 **계단이 화면 대부분을 차지하는 사진**이다. 실제 보행 시점의")
    print("   '멀리 있는 계단'은 이 데이터에 없다 (→ docs/detection.md 8장).")


if __name__ == "__main__":
    main()
