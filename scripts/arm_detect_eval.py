"""C7 — ② arm 을 앞단에 붙였을 때 ③ 탐지가 좋아지는가 · 나빠지는가.

프로젝트에서 ② 방식 판정의 **주 지표**로 정해 둔 실험이다 (→ docs/data.md 2-3-3).
`무처리` 대조군이 없으면 "②가 도움이 되는가" 자체를 답할 수 없으므로 `none` 은 필수다.

평가셋 — **NightOwls rec 34** (학습에 쓰지 않은 held-out)
    ③ 채점기(`c4b_loli0`)가 rec 36·none·37 로 학습됐으므로 rec 34 는 깨끗하다.
    사람 손 라벨이고 전량 야간이라, 보유분 중 이 질문에 답할 수 있는 유일한 데이터다.

    ⚠️ **차량 대시캠**이라 보행 시점과 다르다. 최종 판정은 자체 촬영분(`C5`) 몫이며,
    여기서 얻는 것은 **방향과 크기**다.

⚠️ 이 실험이 재는 것은 '화질'이 아니라 **도메인 정합**이다
    ③ 는 **전처리하지 않은** 야간 이미지로 학습됐다. 그러니 ② 를 앞단에 붙이면
    학습 때 못 본 톤·색 분포가 들어간다. 여기서 mAP 가 떨어져도 그것이 "② 가
    나쁘다"는 뜻은 아니고, **"현재 ③ 와 짝이 안 맞는다"** 는 뜻이다. 선택지는 셋이다.

        (a) 표시 경로에만 ② 를 쓰고 탐지는 원본으로 돌린다  ← 🗣️ 회의 안건 1
        (b) ② 를 적용한 이미지로 ③ 를 재학습한다
        (c) ② 를 쓰지 않는다

    본 스크립트는 (a)/(b) 중 무엇을 택할지의 **근거를 만든다.**

⚠️ 시간축 arm(`+ts`/`+td`/`+a3`)은 여기서 평가하지 않는다
    rec 34 의 해제분은 **라벨 있는 프레임만** 뽑은 것이라 연속이 아니다. 이전 프레임이
    남의 장면이므로 시간축 arm 의 상태가 무의미해진다. 시간축 arm 판정은 연속 영상
    (`rec 38` 또는 `C5` 자체 촬영)에서만 유효하다 (→ scripts/temporal_eval.py).

사용:
    uv run python scripts/arm_detect_eval.py
    uv run python scripts/arm_detect_eval.py --arms none D1A1+bf A1+bf
    uv run python scripts/arm_detect_eval.py --weights outputs/detect/c4b_loli6000/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

from lowlight import build
from nightowls_yolo import STAIRS_ID, write_data_yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/datasets/nightowls_split/rec34_labeled"   # eval_nightowls.py 산출물
DST_ROOT = ROOT / "outputs/datasets/arm_detect"
DEFAULT_ARMS = ["none", "A1", "A1+bf", "D1A1", "D1A1+bf"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=ROOT / "outputs/detect/c4b_loli0/weights/best.pt")
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="stairs 오탐 집계 임계")
    p.add_argument("--width", type=int, default=640,
                   help="② 내부 처리 해상도(가로). 0 이면 원본 그대로. "
                        "기본 640 — C6 가 검토한 내부 처리 해상도")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/detect/arm_detect.json")
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


def make_set(arm_name: str, args) -> Path:
    """arm 을 적용한 평가셋을 만든다. 라벨은 원본을 그대로 하드링크한다."""
    dst = DST_ROOT / arm_name.replace("+", "_")
    out_img, out_lab = dst / "images/val", dst / "labels/val"
    src_img, src_lab = SRC / "images/val", SRC / "labels/val"
    if not src_img.is_dir():
        raise SystemExit(
            f"평가셋 원본이 없다: {src_img}\n먼저 실행할 것:\n"
            "  uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person")

    srcs = sorted(src_img.glob("*"))
    if out_img.is_dir() and not args.rebuild and sum(1 for _ in out_img.glob("*")) == len(srcs):
        print(f"  [{arm_name}] 기존 변환본 재사용 ({len(srcs):,}장)")
        return dst

    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    arm = build(arm_name)
    if arm.is_temporal:
        raise SystemExit(
            f"'{arm_name}' 은 시간축 arm 이라 이 평가셋에서 쓸 수 없다 "
            "(rec 34 해제분은 연속 프레임이 아니다 — 모듈 docstring 참고)")

    for p in srcs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        work = img
        if args.width and w0 != args.width:      # ② 는 내부 처리 해상도에서 돈다
            work = cv2.resize(img, (args.width, round(h0 * args.width / w0)),
                              interpolation=cv2.INTER_AREA)
        out = arm(work)
        if out.shape[:2] != (h0, w0):            # ③ 입력은 원 해상도로 되돌린다
            out = cv2.resize(out, (w0, h0), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out_img / f"{p.stem}.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        link_or_copy(src_lab / f"{p.stem}.txt", out_lab / f"{p.stem}.txt")

    write_data_yaml(dst, "scripts/arm_detect_eval.py", "images/val", "images/val")
    print(f"  [{arm_name}] 생성 {len(srcs):,}장 → {dst.name}")
    return dst


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.weights.is_file():
        raise SystemExit(f"가중치가 없다: {args.weights}")

    print("=" * 92)
    print("C7 — ② arm × ③ 탐지 (NightOwls rec 34 · held-out)")
    print(f"   채점기 {args.weights.parent.parent.name} · ② 내부 처리 가로 "
          f"{args.width or '원본'}px")
    print("=" * 92)

    sets = {a: make_set(a, args) for a in args.arms}
    model = YOLO(str(args.weights))

    rows = []
    for name, dst in sets.items():
        m = model.val(data=str(dst / "data.yaml"), imgsz=args.imgsz, device=0,
                      project=str(ROOT / "outputs/detect"),
                      name=f"arm_{name.replace('+', '_')}", exist_ok=True,
                      plots=False, verbose=False)
        idx = list(m.box.ap_class_index)
        i = idx.index(0) if 0 in idx else None

        n_img = n_fp = 0
        for r in model.predict(source=str(dst / "images/val"), conf=args.conf,
                               imgsz=args.imgsz, device=0, stream=True, verbose=False):
            n_img += 1
            if r.boxes is not None and int((r.boxes.cls == STAIRS_ID).sum()):
                n_fp += 1

        rows.append({
            "arm": name,
            "map50": float(m.box.ap50[i]) if i is not None else 0.0,
            "map": float(m.box.ap[i]) if i is not None else 0.0,
            "precision": float(m.box.p[i]) if i is not None else 0.0,
            "recall": float(m.box.r[i]) if i is not None else 0.0,
            "stairs_fp": n_fp / max(n_img, 1),
        })
        print(f"  [{name}] mAP50 {rows[-1]['map50']:.3f} · recall {rows[-1]['recall']:.3f}")

    base = next((r for r in rows if r["arm"] == "none"), rows[0])
    print("\n" + "=" * 92)
    print(f"판정 — `{base['arm']}` 대비 (person)")
    print("=" * 92)
    hdr = (f"{'arm':<12}{'mAP50':>9}{'Δ':>9}{'mAP50-95':>10}{'recall':>9}{'Δ':>9}"
           f"{'precision':>11}{'stairs오탐':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<12}{r['map50']:>9.3f}{r['map50'] - base['map50']:>+9.3f}"
              f"{r['map']:>10.3f}{r['recall']:>9.3f}{r['recall'] - base['recall']:>+9.3f}"
              f"{r['precision']:>11.3f}{r['stairs_fp']:>10.1%}")
        r["d_map50"] = r["map50"] - base["map50"]
        r["d_recall"] = r["recall"] - base["recall"]

    args.out.write_text(json.dumps(
        {"weights": str(args.weights), "width": args.width, "conf": args.conf,
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    print("\n읽는 법 — Δ 가 음수면 **그 arm 을 탐지 앞단에 두면 안 된다**는 신호다.")
    print("  단 원인이 '② 가 나쁘다'가 아니라 '③ 가 원본으로 학습됐다'일 수 있다.")
    print("  → 🗣️ 표시/탐지 경로 분리(a) vs ② 적용본으로 ③ 재학습(b) 의 근거로 쓴다.")


if __name__ == "__main__":
    main()
