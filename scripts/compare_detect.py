"""C4b 판정 — 여러 탐지 가중치를 **같은 자로** 재서 한 표로 대조한다.

왜 별도 하네스인가
    `eval_nightowls.py` 는 가중치 하나를 잰다. C4b 는 **before 1개 + after 2개**를
    비교해야 하는데, 세 번 따로 돌리면 옵션이 어긋날 위험이 있고(그러면 비교가
    조용히 깨진다) 결과를 손으로 옮겨 적어야 한다. 여기서 한 번에 돌린다.

무엇을 재는가 — 판정 축은 **개발 val 이 아니다**
    person   NightOwls **rec 34**(held-out) mAP50 · recall — 학습에 쓰지 않은 recording
    stairs   같은 이미지에서의 오탐율 + rec 38(4,605장, 계단 없음) 오탐율

    ⚠️ rec 34·38 은 `build_detect_dataset.py` 가 학습에서 원천 차단한다. 그래서
    before(C4, NightOwls 미사용)와 after(C4b) 를 같은 자로 비교할 수 있다.

사용:
    uv run python scripts/compare_detect.py
    uv run python scripts/compare_detect.py --runs c4_baseline,c4b_loli6000,c4b_loli0
    uv run python scripts/compare_detect.py --skip-fp38        # rec38 오탐 스캔 생략(빠름)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETECT = ROOT / "outputs/detect"
DEFAULT_RUNS = "c4_baseline,c4b_loli6000,c4b_loli0"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=DEFAULT_RUNS,
                   help="outputs/detect 아래 run 이름 (쉼표 구분). 첫 번째가 before 기준")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--skip-fp38", action="store_true",
                   help="rec 38 stairs 오탐 스캔 생략 (4,605장이라 시간이 든다)")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/detect/c4b_compare.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import eval_nightowls as ev            # 변환·스캔 로직을 그대로 재사용한다
    from ultralytics import YOLO

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    weights = {}
    for r in runs:
        w = DETECT / r / "weights/best.pt"
        if not w.is_file():
            raise SystemExit(f"가중치가 없다: {w}\n학습이 끝났는지 확인할 것")
        weights[r] = w

    # 평가셋을 **한 번만** 깔고 모든 가중치가 같은 것을 본다
    class _A:                              # eval_nightowls.build 가 기대하는 인자 묶음
        drop_unlabeled_person = True
        rebuild = False
    a34, a38 = _A(), _A()
    a34.dst = ev.DST_ROOT / "rec34_labeled"
    a38.dst = ev.DST_ROOT / "rec38"
    a38.drop_unlabeled_person = False      # rec 38 은 stairs 오탐 전용 — GT 를 안 쓴다

    n34, b34, _ = ev.build(a34, {"34"})
    print(f"판정셋  rec 34 (held-out): 이미지 {n34:,} · pedestrian {b34:,}")
    if not args.skip_fp38:
        n38, _, _ = ev.build(a38, {"38"})
        print(f"오탐셋  rec 38            : 이미지 {n38:,} · 계단 없음")

    rows = []
    for name, w in weights.items():
        print(f"\n--- {name} ---")
        model = YOLO(str(w))
        m = model.val(data=str(a34.dst / "data.yaml"), imgsz=args.imgsz, device=0,
                      project=str(DETECT), name=f"{name}__rec34_labeled",
                      exist_ok=True, plots=False, verbose=False)
        i = list(m.box.ap_class_index).index(0) if 0 in list(m.box.ap_class_index) else None
        t34, f34, _ = ev.scan_stairs_fp(model, a34.dst / "images/val", args.conf, args.imgsz)
        row = {
            "run": name,
            "map50": float(m.box.ap50[i]) if i is not None else float("nan"),
            "map": float(m.box.ap[i]) if i is not None else float("nan"),
            "precision": float(m.box.p[i]) if i is not None else float("nan"),
            "recall": float(m.box.r[i]) if i is not None else float("nan"),
            "fp34": f34 / t34,
        }
        if not args.skip_fp38:
            t38, f38, _ = ev.scan_stairs_fp(model, a38.dst / "images/val",
                                            args.conf, args.imgsz)
            row["fp38"] = f38 / t38
        rows.append(row)

    base = rows[0]
    print("\n" + "=" * 92)
    print("C4b 판정 — NightOwls rec 34 (학습 미사용) · person")
    print("=" * 92)
    hdr = f"{'run':<16}{'mAP50':>9}{'Δ':>9}{'mAP50-95':>10}{'recall':>9}{'Δ':>9}" \
          f"{'precision':>11}{'stairs오탐(34)':>15}"
    if not args.skip_fp38:
        hdr += f"{'stairs오탐(38)':>15}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['run']:<16}{r['map50']:>9.3f}"
                f"{r['map50'] - base['map50']:>+9.3f}{r['map']:>10.3f}"
                f"{r['recall']:>9.3f}{r['recall'] - base['recall']:>+9.3f}"
                f"{r['precision']:>11.3f}{r['fp34']:>14.1%}")
        if not args.skip_fp38:
            line += f"{r['fp38']:>14.1%}"
        print(line)
    print(f"\nΔ 는 첫 행({base['run']}) 대비다. stairs 오탐은 낮을수록 좋다 "
          "— 두 recording 모두 계단이 없다.")

    args.out.write_text(json.dumps({"conf": args.conf, "imgsz": args.imgsz, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {args.out}")
    print("\n⚠️ 차량 주행 시점이다 — 보행 시점 최종 판정은 자체 촬영분(`C5`).")


if __name__ == "__main__":
    main()
