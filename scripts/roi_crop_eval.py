"""③ **ROI 크롭**이 작은 볼라드를 살리는가 — 스케일 효과와 해상도 효과를 갈라서 잰다.

`review_response_20260825` 3-(e) 가 "학습 비용 0 · 미측정"으로 남긴 카드다. 겨냥하는 곳은
이 프로젝트의 최악 축 — `<4px` 볼라드 recall **0.319**(GT 의 22%) 이고, 로컬 실측으로
`<8px` 가 **GT 의 49.2%** 임을 확인했다.

★ 왜 **3-arm** 인가 — 그냥 크롭하고 재면 오독한다
    YOLO 는 최소 stride 가 8 이라 **박스가 커지기만 해도** 검출이 오를 수 있다. 그래서
    "크롭 후 recall 상승"을 곧바로 *해상도 이득*으로 읽으면 안 된다. 정보 증가가 **0** 인
    arm(=이미 640 인 이미지를 크롭해 되키우기)을 사이에 끼워야 둘이 갈린다.

        A  full frame        → square letterbox 640      기준선(배포와 같은 자)
        B  같은 이미지 크롭  → 640                        정보 증가 0 · **스케일 단독**
        C  원본 고해상 크롭  → 640                        **스케일 + 해상도**

        B − A = 스케일 효과 · C − B = 진짜 해상도 이득

★ 왜 **프레임 전체 기준**으로 세는가
    크롭 밖 GT 를 분모에서 빼면 "recall 이 올랐다"가 자동으로 나온다. 창을 좁힐수록
    남은 것만 쉬워지기 때문이다. 그래서 **창 밖 GT 도 미탐으로 센다** — 이 프로젝트가
    반복해서 밟은 함정(재는 대상이 바뀐다 → STATUS 3장 함정 2)과 같은 부류다.

★ 왜 폭 구간(bucket)을 **arm A 기준으로 고정**하는가
    크롭하면 같은 물체의 화면상 폭이 커진다. 그 커진 폭으로 구간을 나누면 **arm 마다
    구간의 정의가 달라져** 비교가 성립하지 않는다. 구간은 **배포 조건(arm A)에서의 폭**
    으로 한 번 정하고 전 arm 이 그대로 쓴다.

⚠️ **평가셋 오염** — `c4e_s3_11n` 은 이 AIHub 서브셋 8,802장으로 학습됐고, 어느 장이
    train 이었는지 로컬에서 알 수 없다(`detect_v3` 는 학습 PC 에 있다). **절대 recall 은
    부풀려져 있다.** A·B 는 같은 이미지·같은 모델이라 **델타는 유효**하므로 결과는
    **델타로만** 인용할 것.

⚠️ 예측은 **한 장씩** 돌리고 letterbox 는 **square 고정**이다 (→ STATUS 3장 함정 18).

사용:
    uv run python scripts/roi_crop_eval.py                    # 기본 4창 · AIHub 400장 + 실촬영
    uv run python scripts/roi_crop_eval.py --n 150            # 빨리 훑기
    uv run python scripts/roi_crop_eval.py --window-cy 0.45   # 창을 살짝 위로
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")      # → STATUS 3장 함정 9

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")            # → STATUS 3장 함정 12

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_onnx import letterbox                       # noqa: E402
from eval_own_night import (BUCKETS, BUCKET_LABEL, Sample, _iou,      # noqa: E402
                            match_one)
from nightowls_yolo import BOLLARD_ID, CLASS_NAMES      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "outputs/export/bammasil_det_c4e_s3_11n_640/bammasil_det_c4e_s3_11n_640.onnx"
SRC = ROOT / "data/bammasil_aihub_subset"
REAL = ROOT / "data/test_real_data"
OUT = ROOT / "outputs/detect/c4e_roi_crop.json"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, default=ONNX, help="배포 FP32 ONNX")
    p.add_argument("--src", type=Path, default=SRC, help="arm A·B 소재 (AIHub 서브셋)")
    p.add_argument("--real-src", type=Path, default=REAL, help="arm C 소재 (원본 고해상)")
    p.add_argument("--windows", default="1.0x1.0,0.6x0.6,0.5x0.5,0.34x0.34",
                   help="창 비율 (가로x세로). **1.0x1.0 은 자기검증용** — arm A 와 "
                        "완전히 같은 수치가 나와야 한다")
    p.add_argument("--window-cy", type=float, default=0.5,
                   help="창 중심의 세로 위치. 작은 볼라드 cy 중앙이 0.48 이라 "
                        "0.45~0.5 가 후보다")
    p.add_argument("--n", type=int, default=400, help="AIHub 표본 수 (CPU 시간)")
    p.add_argument("--imgsz", type=int, default=640, help="배포 해상도 고정 전제")
    p.add_argument("--conf", type=float, default=0.25, help="운영값 (C4e S1 확정)")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--match-iou", type=float, default=0.5, help="GT 매칭 IoU")
    p.add_argument("--union", default="",
                   help="★ 2패스 — 기준선 예측과 이 창의 예측을 **합집합**으로 본다. "
                        "단일 창이 기각돼도 '추론 2배를 쓰면 얼마를 사는가'는 별개 물음이라 "
                        "`C11b` 예산 판정의 입력이 된다. 예: 0.6x0.6")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args()


# --------------------------------------------------------------------------- 전처리

def lb_params(h: int, w: int, size: int):
    """`export_onnx.letterbox` 와 **같은 산술**로 (scale, top, left) 를 낸다.

    이미지 자체는 그 함수가 만들고, 여기서는 좌표 역변환에 필요한 값만 다시 센다.
    두 곳의 식이 어긋나면 박스가 통째로 밀리므로 아래 `_assert_lb` 가 한 번 확인한다.
    """
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    return s, (size - nh) // 2, (size - nw) // 2


def _assert_lb(size: int) -> None:
    import numpy as np

    for h, w in ((360, 640), (4080, 3060), (720, 1280)):
        canvas = letterbox(np.zeros((h, w, 3), np.uint8), size)
        s, top, left = lb_params(h, w, size)
        filled = (canvas == 0).all(axis=2)
        ys, xs = filled.nonzero()
        assert (ys.min(), xs.min()) == (top, left), (h, w, ys.min(), xs.min(), top, left)


def predict_boxes(model, bgr, imgsz: int, conf: float, iou: float):
    """한 장을 square letterbox 로 밀어 넣고 **원본 픽셀 좌표**로 되돌려 준다."""
    h, w = bgr.shape[:2]
    s, top, left = lb_params(h, w, imgsz)
    square = letterbox(bgr, imgsz)
    r = model.predict(square, imgsz=imgsz, conf=conf, iou=iou,
                      device="cpu", verbose=False)[0].boxes
    out = []
    for c, cf, xy in zip(r.cls.tolist(), r.conf.tolist(), r.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = xy
        out.append((int(c), float(cf),
                    (x1 - left) / s, (y1 - top) / s, (x2 - left) / s, (y2 - top) / s))
    return out


def crop_box(w: int, h: int, fw: float, fh: float, cy: float):
    """중앙(세로는 `cy`) 기준 창 → 원본 픽셀 (x0, y0, x1, y1)."""
    cw, ch = w * fw, h * fh
    x0 = max(0.0, min(w - cw, w * 0.5 - cw / 2))
    y0 = max(0.0, min(h - ch, h * cy - ch / 2))
    return int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch))


# --------------------------------------------------------------------------- 소재

def load_aihub(src: Path, n: int, seed: int):
    """볼라드 GT 가 있는 이미지를 표본으로 뽑는다 → [(img_path, label_path)]."""
    pairs = []
    for lab_dir in sorted(src.glob("*_labels")):
        part = lab_dir.name.replace("_labels", "")
        img_root = next((d for d in src.glob(f"{part}_images*") if d.is_dir()), None)
        if img_root is None:
            continue
        for grp in sorted(p for p in lab_dir.iterdir() if p.is_dir()):
            imgs = img_root / grp.name / "images"
            if not imgs.is_dir():
                continue
            for txt in (grp / "labels").glob("*.txt"):
                pairs.append((imgs, txt))
    random.Random(seed).shuffle(pairs)

    out = []
    for imgs, txt in pairs:
        body = txt.read_text(encoding="utf-8", errors="replace")
        if f"\n{BOLLARD_ID} " not in "\n" + body:
            continue                                     # 볼라드 GT 가 없으면 건너뛴다
        img = next((imgs / (txt.stem + e) for e in (".jpg", ".png", ".jpeg")
                    if (imgs / (txt.stem + e)).is_file()), None)
        if img is None:
            continue
        out.append((img, txt))
        if len(out) >= n:
            break
    return out


def to_sample(img_path: Path, txt: Path, w: int, h: int) -> Sample:
    s = Sample(img_path.stem, img_path, w, h)
    for line in txt.read_text(encoding="utf-8", errors="replace").split("\n"):
        p = line.split()
        if len(p) != 5:
            continue
        cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        if cid not in CLASS_NAMES:
            continue
        s.gt.append((cid, (cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h))
    return s


# --------------------------------------------------------------------------- 집계

def nms(preds, iou_thr: float):
    """클래스별 NMS — 2패스 합집합에서 같은 물체가 두 번 잡히는 것을 접는다."""
    keep = []
    for cid in {p[0] for p in preds}:
        cand = sorted((p for p in preds if p[0] == cid), key=lambda p: -p[1])
        picked = []
        for p in cand:
            if all(_iou(p[2:], q[2:]) < iou_thr for q in picked):
                picked.append(p)
        keep += picked
    return keep


def bucket_of(w640: float) -> str:
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABEL):
        if lo <= w640 < hi:
            return lab
    return BUCKET_LABEL[-1]


def blank_rows():
    rows = {lab: {"gt": 0, "hit": 0, "in_win": 0} for lab in (*BUCKET_LABEL, "전체")}
    rows["_fp"] = 0          # 오탐 박스 수 — recall 만 보면 2패스가 공짜처럼 보인다
    return rows


def fold(rows: dict) -> dict:
    out = {"fp_box": rows["_fp"]}
    tp = rows["전체"]["hit"]
    out["precision"] = tp / (tp + rows["_fp"]) if tp + rows["_fp"] else None
    for lab, v in rows.items():
        if lab == "_fp":
            continue
        out[lab] = {**v,
                    "recall": (v["hit"] / v["gt"]) if v["gt"] else None,
                    "coverage": (v["in_win"] / v["gt"]) if v["gt"] else None}
    return out


# --------------------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    args.onnx = args.onnx.resolve()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX 가 없다: {args.onnx}")

    import cv2
    import numpy as np
    from ultralytics import YOLO

    _assert_lb(args.imgsz)
    wins = []
    for spec in args.windows.split(","):
        fw, fh = (float(x) for x in spec.strip().split("x"))
        wins.append((spec.strip(), fw, fh))

    model = YOLO(str(args.onnx), task="detect")
    thr = {c: args.conf for c in CLASS_NAMES}

    pairs = load_aihub(args.src, args.n, args.seed)
    print(f"모델   {args.onnx.relative_to(ROOT)}")
    print(f"소재   AIHub 볼라드 보유 {len(pairs)}장  (conf {args.conf} · square {args.imgsz})")
    print(f"창     {', '.join(w[0] for w in wins)}  · 세로 중심 {args.window_cy}")
    print("⚠️ 평가셋 오염 — 이 서브셋으로 학습된 모델이다. **델타로만** 읽을 것.\n")

    # arm A/B — 같은 이미지, 창만 다르다
    union_key = f"union(+{args.union})" if args.union else None
    base = wins[0][0]
    per_win = {spec: blank_rows() for spec, _, _ in wins}
    if union_key:
        per_win[union_key] = blank_rows()
        uf, uh = (float(x) for x in args.union.split("x"))
    for i, (img_path, txt) in enumerate(pairs, 1):
        bgr = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = to_sample(img_path, txt, w, h)
        scale_a = min(args.imgsz / h, args.imgsz / w)     # 구간의 자 = arm A 기준

        cache = {}
        for spec, fw, fh in wins:
            x0, y0, x1, y1 = crop_box(w, h, fw, fh, args.window_cy)
            preds = predict_boxes(model, bgr[y0:y1, x0:x1], args.imgsz, args.conf, args.iou)
            preds = [(c, cf, bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                     for c, cf, bx1, by1, bx2, by2 in preds]
            cache[spec] = preds
            res = match_one(s, preds, thr, args.match_iou, scale_a)[BOLLARD_ID]

            rows = per_win[spec]
            seen = [(round((g[3] - g[1]) * scale_a, 2), (g[1] + g[3]) / 2, (g[2] + g[4]) / 2)
                    for g in s.gt if g[0] == BOLLARD_ID]
            for w640, gcx, gcy in seen:
                lab = bucket_of(w640)
                rows[lab]["gt"] += 1
                rows["전체"]["gt"] += 1
                if x0 <= gcx <= x1 and y0 <= gcy <= y1:
                    rows[lab]["in_win"] += 1
                    rows["전체"]["in_win"] += 1
            for w640, _ in res["matched"]:
                lab = bucket_of(w640)
                rows[lab]["hit"] += 1
                rows["전체"]["hit"] += 1
            rows["_fp"] += res["fp"]

        if union_key:
            base_preds = cache.get(base)
            if base_preds is None:
                bx = crop_box(w, h, 1.0, 1.0, 0.5)
                base_preds = predict_boxes(model, bgr, args.imgsz, args.conf, args.iou)
            win_preds = cache.get(args.union)
            if win_preds is None:
                x0, y0, x1, y1 = crop_box(w, h, uf, uh, args.window_cy)
                win_preds = [(c, cf, a + x0, b + y0, cc + x0, d + y0) for c, cf, a, b, cc, d
                             in predict_boxes(model, bgr[y0:y1, x0:x1],
                                              args.imgsz, args.conf, args.iou)]
            merged = nms(base_preds + win_preds, args.iou)
            res = match_one(s, merged, thr, args.match_iou, scale_a)[BOLLARD_ID]
            rows = per_win[union_key]
            for g in s.gt:
                if g[0] != BOLLARD_ID:
                    continue
                lab = bucket_of(round((g[3] - g[1]) * scale_a, 2))
                rows[lab]["gt"] += 1; rows[lab]["in_win"] += 1
                rows["전체"]["gt"] += 1; rows["전체"]["in_win"] += 1
            for w640, _ in res["matched"]:
                lab = bucket_of(w640)
                rows[lab]["hit"] += 1
                rows["전체"]["hit"] += 1
            rows["_fp"] += res["fp"]
        if i % 50 == 0:
            print(f"  … {i}/{len(pairs)}")

    aihub = {spec: fold(rows) for spec, rows in per_win.items()}

    # arm C — 원본 고해상 실촬영. 라벨이 없어 recall 을 못 낸다(방향 확인용)
    real = []
    for img_path in sorted(args.real_src.glob("*.jpg")):
        bgr = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        # 배포 조건 재현본 — 원본을 720p 급으로 먼저 줄인 판(정보 증가 0 대조군)
        s720 = min(1280 / w, 720 / h, 1.0)
        small = cv2.resize(bgr, (round(w * s720), round(h * s720)),
                           interpolation=cv2.INTER_AREA)
        row = {"image": img_path.name, "orig": f"{w}x{h}", "windows": {}}
        for spec, fw, fh in wins:
            cell = {}
            for tag, src_img in (("scale_only(720p)", small), ("full_res(원본)", bgr)):
                hh, ww = src_img.shape[:2]
                x0, y0, x1, y1 = crop_box(ww, hh, fw, fh, args.window_cy)
                preds = predict_boxes(model, src_img[y0:y1, x0:x1],
                                      args.imgsz, args.conf, args.iou)
                cell[tag] = {CLASS_NAMES[c]: round(max(
                    (p[1] for p in preds if p[0] == c), default=0.0), 3)
                    for c in sorted({p[0] for p in preds})} or {}
                cell[tag]["n"] = len(preds)
            row["windows"][spec] = cell
        real.append(row)

    # ------------------------------------------------------------------ 출력
    print("\n" + "=" * 86)
    print("arm A·B — AIHub 볼라드 · GT 폭 구간별 recall (구간의 자는 **arm A 기준 640 환산**)")
    print("           coverage = 창 안에 중심이 들어온 GT 비율 · recall 분모는 **프레임 전체**")
    print("=" * 86)
    head = (f"{'창':>15s} │ " + " │ ".join(f"{lab:>11s}" for lab in (*BUCKET_LABEL, "전체"))
            + " │ FP / 정밀도")
    print(head)
    print("─" * len(head))
    for spec in per_win:
        r = aihub[spec]
        cells = []
        for lab in (*BUCKET_LABEL, "전체"):
            v = r[lab]
            cells.append("         -" if v["recall"] is None
                         else f"{v['recall']:.3f}({v['coverage']:.2f})")
        tag = f"{spec}{'*' if spec == base else ''}"
        pr = r["precision"]
        cells.append(f"{r['fp_box']:>4d} / {pr:.3f}" if pr is not None else "     -")
        print(f"{tag:>15s} │ " + " │ ".join(f"{c:>11s}" for c in cells))
    print(f"\n  * 기준선(arm A). GT 박스 수 — " + " · ".join(
        f"{lab} {aihub[base][lab]['gt']}" for lab in (*BUCKET_LABEL, "전체")))

    print("\nΔ recall (기준선 대비 · **스케일 단독 효과**)")
    print(head)
    print("─" * len(head))
    for spec in per_win:
        if spec == base:
            continue
        cells = []
        for lab in (*BUCKET_LABEL, "전체"):
            a, b = aihub[base][lab]["recall"], aihub[spec][lab]["recall"]
            cells.append("         -" if a is None or b is None else f"{b - a:+.3f}")
        d = aihub[spec]["fp_box"] - aihub[base]["fp_box"]
        cells.append(f"{d:+5d}")
        print(f"{spec:>15s} │ " + " │ ".join(f"{c:>11s}" for c in cells))

    print("\n" + "=" * 86)
    print("arm C — 원본 고해상 실촬영 (라벨 없음 · 방향 확인용)")
    print("        같은 창을 720p 축소본과 원본에 각각 적용 → 차이가 **해상도 이득**")
    print("=" * 86)
    for row in real:
        print(f"\n{row['image']}  ({row['orig']})")
        for spec, cell in row["windows"].items():
            a = cell["scale_only(720p)"]
            b = cell["full_res(원본)"]
            fmt = lambda d: " ".join(f"{k} {v}" for k, v in d.items() if k != "n") or "-"
            print(f"   창 {spec:>9s} │ 720p [{a['n']}] {fmt(a):<34s}"
                  f"│ 원본 [{b['n']}] {fmt(b)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "onnx": str(args.onnx.relative_to(ROOT)),
        "conf": args.conf, "iou": args.iou, "match_iou": args.match_iou,
        "imgsz": args.imgsz, "window_cy": args.window_cy,
        "n_aihub": len(pairs), "baseline_window": base,
        "note": ("구간의 자는 arm A(배포 조건) 기준 640 환산으로 고정 · recall 분모는 "
                 "프레임 전체(창 밖 GT 도 미탐) · 평가셋 오염이 있어 델타로만 인용할 것"),
        "aihub": aihub, "real": real,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n저장: {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
