"""`C5` 판정 — **자체 촬영 야간분(라벨 있음)** 에서 후보 가중치를 한 표로 가른다.

왜 별도 하네스인가
    기존 셋은 전부 다른 것을 잰다.

        compare_detect.py    rec34/38 — NightOwls 전용 · **차량 대시캠 도메인**
        eval_real_night.py   test_real_data — **라벨이 없어** 오탐만 센다
        eval_stairs_night.py StairNet 야간 — `stairs` 단일 클래스 · 외부 데이터셋

    `C2` 촬영 → `C3` 라벨이 오면 **보행 시점 · 라벨 있음 · 3클래스**가 처음으로 갖춰진다.
    그때 후보 3개(`c4b_loli0` 현 배포 · `c4d_11n_640` 인수분 · `c4e_s3_11n` 자체)를
    **같은 자로** 재는 것이 이 스크립트다.

무엇을 재는가 — 5축
    A  클래스별 mAP50 · mAP50-95 (+종합)      임계값 무관 · `model.val()` 1회
    B  운영점 recall·precision·F1·FP·미탐     클래스별 **+ 종합(micro)**
    C  음성 프레임 오탐 박스 수 · 발화 프레임  8/23 게이트와 **같은 자**
    D  GT 폭 구간별 recall (+**전체 열**)      `<4 / 4~8 / 8~16 / >=16px` @640 환산
    E  볼라드 **박스별** confidence            장면 maxconf 로는 안 보인다

    ★ 축 A 만 임계값과 무관하다. 축 B~E 는 **운영 conf 에서** 잰다 — 게이트가 원래
    운영점 기준이고, 클래스별 conf(`bollard` 0.15 채택안)를 걸 수 있는 곳도 여기뿐이다.
    ★ 쪼갠 표(B·D)에는 **종합 행/열을 같이 낸다.** 구간별만 보면 총합 우열이 안 보인다.

★ `ignore` 박스 — 찾아도 감점 없고, 못 찾아도 감점 없다
    라벨 규칙은 `docs/labeling_stairs.md` 4장이 정본이다. 확신이 안 서는 것을 지우면
    "배경"으로 가르치게 되므로 `ignore` 로 남긴다. 여기서는

      · `ignore` GT 는 미탐(FN)에 넣지 않는다
      · `ignore` 영역에 걸린 예측은 오탐(FP)에서 뺀다 (**IoA** 기준 · 클래스 무관)
      · val 라벨 파일에서도 뺀다 — ultralytics 에 ignore 개념이 없어 그냥 두면
        **4번째 클래스가 되어 mAP 가 깨진다**

⚠️ 주의 3가지
    · 배포 640 고정 전제다. `--imgsz` 를 바꾸면 축 D 의 픽셀 구간이 같이 움직인다.
    · **2클래스 가중치(`c4b_loli0`)는 `bollard` 행이 `-` 로 나온다.** 0 으로 찍으면
      "오탐 0" 으로 오독된다.
    · 원본은 건드리지 않는다(→ STATUS 3장 함정 7). 평가셋은 `outputs/` 아래에
      **복사**해서 깐다 — 🔴 하드링크로 깔면 `model.val()` 이 JPEG 을 재인코딩하며
      **원본을 관통해 덮어쓴다**(8/25 실측 · → 함정 18). 촬영 원본은 다시 못 찍는다.

사용:
    uv run python scripts/eval_own_night.py
    uv run python scripts/eval_own_night.py --runs c4b_loli0,c4d_11n_640,c4e_s3_11n
    uv run python scripts/eval_own_night.py --conf 0.25,0.10

    # 클래스별 conf (C4e S1 E2 채택안) — 안 적은 클래스는 --base-conf
    uv run python scripts/eval_own_night.py --class-conf bollard=0.15

    # 학습 레이아웃 밖의 가중치를 복사 없이 태운다 (스모크·인수분)
    uv run python scripts/eval_own_night.py --src <셋> --device cpu `
        --weights outputs/bammasil_results/bammasil_weights/c4d_11n_640/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # → STATUS 3장 함정 9
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nightowls_yolo import CLASS_NAMES, BOLLARD_ID, write_data_yaml  # noqa: E402
from eval_real_night import NAME_TO_ID, parse_class_conf, thr_label  # noqa: E402

SRC_DEFAULT = ROOT / "data/own_night"
DST_DEFAULT = ROOT / "outputs/datasets/own_night"
DETECT = ROOT / "outputs/detect"
DEFAULT_RUNS = "c4b_loli0,c4d_11n_640,c4e_s3_11n"

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# GT 폭 구간 — 640 입력 환산 픽셀. 볼라드 폭 중앙이 7.8px 이고 `<4px`(GT의 22%)는
# 960 에서도 recall 0.49 가 상한이라, 이 경계에서 "데이터 부족"과 "해상도 한계"가 갈린다
# (→ STATUS ★인수분 R1).
BUCKETS = ((0.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, float("inf")))
BUCKET_LABEL = ("<4px", "4~8px", "8~16px", ">=16px")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=SRC_DEFAULT,
                   help="C3 산출물 루트 — images/ + labels/ (기본 data/own_night)")
    p.add_argument("--dst", type=Path, default=DST_DEFAULT,
                   help="평가용 YOLO 레이아웃을 깔 곳 (재생성 가능한 산출물)")
    p.add_argument("--runs", default=DEFAULT_RUNS,
                   help="outputs/detect 아래 run 이름 (쉼표 구분)")
    p.add_argument("--weights", action="append", type=Path, metavar="PATH",
                   help="가중치 직접 경로 (여러 번 가능). 주면 --runs 는 무시된다")
    p.add_argument("--conf", default="0.25",
                   help="운영 conf 후보 (쉼표 구분). 8/23 에 단일 0.25 로 확정됐다")
    p.add_argument("--class-conf", action="append", metavar="SPEC",
                   help="클래스별 임계. 예 'bollard=0.15'. 여러 번 주면 격자가 된다. "
                        "안 적은 클래스는 --base-conf. 주면 --conf 는 무시된다")
    p.add_argument("--base-conf", type=float, default=0.25,
                   help="--class-conf 에서 안 적은 클래스의 임계 (현 운영값 0.25)")
    p.add_argument("--iou", type=float, default=0.5, help="TP 판정 IoU")
    p.add_argument("--imgsz", type=int, default=640, help="배포 해상도 고정 전제")
    p.add_argument("--ignore-class-id", type=int, default=3,
                   help="라벨의 ignore 클래스 id (기본 3 — 0/1/2 다음)")
    p.add_argument("--device", default="0", help="'0' 또는 'cpu'")
    p.add_argument("--skip-map", action="store_true",
                   help="축 A(model.val) 를 건너뛴다 — 배선 확인·CPU 스모크용")
    p.add_argument("--rebuild", action="store_true", help="평가셋을 다시 깐다")
    p.add_argument("--out", type=Path, default=DETECT / "c5_own_night.json")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# 소재 읽기
# ──────────────────────────────────────────────────────────────────────────

class Sample:
    """이미지 한 장 — GT 박스(픽셀) · ignore 박스(픽셀) · 원본 크기."""

    __slots__ = ("stem", "img", "w", "h", "gt", "ign", "keep_lines")

    def __init__(self, stem, img, w, h):
        self.stem, self.img, self.w, self.h = stem, img, w, h
        self.gt: list[tuple[int, float, float, float, float]] = []
        self.ign: list[tuple[float, float, float, float]] = []
        self.keep_lines: list[str] = []      # ignore 를 뺀 라벨 (val 로 나갈 것)


def _xywhn_to_xyxy(cx, cy, bw, bh, w, h):
    return ((cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h)


def load_samples(src: Path, ignore_id: int) -> list[Sample]:
    """`images/` + `labels/` 를 읽어 GT 를 픽셀 좌표로 편다.

    라벨 파일이 없는 이미지는 **전부 음성**으로 본다 (배경 표본). 라벨 디렉토리
    자체가 없으면 그건 `C3` 산출물이 아니므로 즉시 실패한다.
    """
    import cv2
    import numpy as np

    img_dir, lab_dir = src / "images", src / "labels"
    if not img_dir.is_dir():
        raise SystemExit(f"없음: {img_dir}\n`C3` 산출물 루트를 --src 로 줄 것")
    if not lab_dir.is_dir():
        raise SystemExit(
            f"없음: {lab_dir}\n라벨이 없으면 recall 을 못 잰다 — "
            "라벨 없는 소재는 scripts/eval_real_night.py 를 쓸 것")

    out: list[Sample] = []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXT):
        # cv2.imread 는 비ASCII 경로를 못 연다 (→ STATUS 3장 함정 12)
        im = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise SystemExit(f"이미지를 못 읽었다: {img}")
        s = Sample(img.stem, img, im.shape[1], im.shape[0])

        txt = lab_dir / f"{img.stem}.txt"
        if txt.is_file():
            for line in txt.read_text(encoding="utf-8").splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                cid = int(float(f[0]))
                box = _xywhn_to_xyxy(*(float(v) for v in f[1:5]), s.w, s.h)
                if cid == ignore_id:
                    s.ign.append(box)
                elif cid in CLASS_NAMES:
                    s.gt.append((cid, *box))
                    s.keep_lines.append(line.strip())
                else:
                    raise SystemExit(
                        f"모르는 클래스 id {cid} — {txt.name}. "
                        f"클래스는 {CLASS_NAMES} 이고 ignore 는 --ignore-class-id "
                        f"({ignore_id}) 다")
        out.append(s)

    if not out:
        raise SystemExit(f"이미지가 0장이다: {img_dir}")
    return out


def build_val(samples: list[Sample], src: Path, dst: Path, rebuild: bool) -> Path:
    """`model.val()` 이 먹을 YOLO 레이아웃을 깐다 — **ignore 를 뺀 라벨**로.

    🔴 **하드링크를 쓰지 않는다 — 반드시 복사한다** (2026-08-25 실측).
        ultralytics 의 `verify_image` 는 EXIF 회전이 있거나 JPEG 이 조금이라도
        어긋나면 `"corrupt JPEG restored and saved"` 를 찍고 **그 경로에 다시 쓴다.**
        하드링크면 그 쓰기가 **원본을 관통한다** — 스모크에서 원본 3.37MB 가
        7.27MB 로 재인코딩되는 것을 확인했다(inode 동일). `C2` 촬영 원본은 다시
        못 찍으므로 링크로 이득을 볼 자리가 아니다 (→ STATUS 3장 함정 7·18).
        `eval_nightowls.py` 가 `link_or_copy` 를 쓰고도 무사한 것은 원본이 `D:`,
        `outputs/` 가 `C:` 라 **볼륨이 갈려 우연히 복사되기 때문**이다.

    ⚠️ 대가는 용량이다 — `C5` 는 자체 촬영분이라 규모가 작아 감당되지만, 지우고
       필요할 때 되살리는 산출물로 볼 것 (→ STATUS 3장 함정 14).
    """
    import shutil
    out_img, out_lab = dst / "images/val", dst / "labels/val"
    if out_lab.is_dir() and not rebuild and \
            sum(1 for _ in out_img.glob("*")) == len(samples):
        print(f"  기존 변환본 재사용: {len(samples):,}장 (--rebuild 로 다시 만든다)")
        return dst / "data.yaml"

    total_mb = sum(s.img.stat().st_size for s in samples) / 1024 / 1024
    print(f"  평가셋을 깐다 — 이미지 {len(samples):,}장 **복사** ({total_mb:,.0f}MB). "
          "하드링크를 안 쓰는 이유는 build_val docstring 참조 (원본 보호)")

    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    n_ign = 0
    for s in samples:
        shutil.copy2(s.img, out_img / s.img.name)
        (out_lab / f"{s.stem}.txt").write_text(
            "\n".join(s.keep_lines) + ("\n" if s.keep_lines else ""), encoding="utf-8")
        n_ign += len(s.ign)
    if n_ign:
        print(f"  ignore 박스 {n_ign:,}개를 val 라벨에서 뺐다 "
              "(mAP 가 깨지지 않게 — 감점 제외는 축 B·C 에서 처리한다)")

    write_data_yaml(dst, "scripts/eval_own_night.py", "images/val", "images/val",
                    extra="EVAL ONLY — C5 판정 전용. 학습에 넣지 말 것")
    return dst / "data.yaml"


# ──────────────────────────────────────────────────────────────────────────
# 매칭 — 축 B~E 의 공통 기반
# ──────────────────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _ioa(pred, region) -> float:
    """예측 면적 대비 겹침 — `ignore` 영역 판정용.

    IoU 를 쓰면 큰 ignore 영역 안의 작은 예측이 걸러지지 않는다. "이 영역의 판정은
    보류한다"는 뜻이므로 **예측이 그 안에 들어갔는가**를 봐야 한다.
    """
    ix1, iy1 = max(pred[0], region[0]), max(pred[1], region[1])
    ix2, iy2 = min(pred[2], region[2]), min(pred[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    return inter / area if area > 0 else 0.0


def match_one(s: Sample, preds, thr: dict[int, float], iou_thr: float, scale: float):
    """이미지 한 장을 매칭한다.

    preds: [(cls, conf, x1, y1, x2, y2)] — 격자 최저 conf 로 한 번 뽑아 둔 것.
    반환: 클래스별 {tp, fp, fn, matched:[(gt_w640, conf)], missed:[gt_w640],
                   ignored_fp} + 프레임 단위 오탐 박스 수.
    """
    res = {c: {"tp": 0, "fp": 0, "fn": 0, "ignored_fp": 0,
               "matched": [], "missed": []} for c in CLASS_NAMES}
    for cid in CLASS_NAMES:
        gts = [g[1:] for g in s.gt if g[0] == cid]
        cand = sorted((p for p in preds if p[0] == cid and p[1] >= thr[cid]),
                      key=lambda p: -p[1])
        used = [False] * len(gts)
        for _, conf, *box in cand:
            best, best_i = iou_thr, -1
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                v = _iou(box, g)
                if v >= best:
                    best, best_i = v, i
            if best_i >= 0:
                used[best_i] = True
                res[cid]["tp"] += 1
                res[cid]["matched"].append(
                    (round((gts[best_i][2] - gts[best_i][0]) * scale, 2), round(conf, 3)))
            elif any(_ioa(box, r) >= 0.5 for r in s.ign):
                # ignore 영역에 걸린 예측 — 찾아도 감점 없다
                res[cid]["ignored_fp"] += 1
            else:
                res[cid]["fp"] += 1
        for i, g in enumerate(gts):
            if not used[i]:
                res[cid]["fn"] += 1
                res[cid]["missed"].append(round((g[2] - g[0]) * scale, 2))
    return res


def aggregate(samples, per_image, nc: int):
    """축 B·C·D·E 를 한 번에 접는다."""
    cls_ids = [c for c in CLASS_NAMES if c < nc]

    ops = {}
    for c in cls_ids:
        tp = sum(r[c]["tp"] for r in per_image)
        fp = sum(r[c]["fp"] for r in per_image)
        fn = sum(r[c]["fn"] for r in per_image)
        ops[CLASS_NAMES[c]] = _rates(tp, fp, fn,
                                     sum(r[c]["ignored_fp"] for r in per_image))
    ops["종합"] = _rates(*(sum(ops[CLASS_NAMES[c]][k] for c in cls_ids)
                           for k in ("tp", "fp", "fn", "ignored_fp")))
    for c in CLASS_NAMES:
        if c not in cls_ids:
            ops[CLASS_NAMES[c]] = dict(_NA)

    # 축 C — 해당 클래스 GT 가 0인 프레임에서 나온 박스는 정의상 전부 오탐
    neg = {}
    any_box = any_frame = 0
    neg_frames_any = set()
    for c in cls_ids:
        box = frames = n_neg = 0
        for s, r in zip(samples, per_image):
            if any(g[0] == c for g in s.gt):
                continue
            n_neg += 1
            k = r[c]["fp"]
            if k:
                box += k
                frames += 1
                neg_frames_any.add(s.stem)
        neg[CLASS_NAMES[c]] = {"n_frame": n_neg, "fp_box": box, "hit_frame": frames}
        any_box += box
    for c in CLASS_NAMES:
        if c not in cls_ids:
            neg[CLASS_NAMES[c]] = {"n_frame": None, "fp_box": None, "hit_frame": None}
    # 종합은 **순수 음성 프레임**(세 클래스 GT 가 전부 0)이다 — 클래스별 행의
    # 프레임 수를 더하면 같은 프레임을 여러 번 세게 된다.
    pure = [s for s in samples if not s.gt]
    pure_hit = sum(1 for s in pure if s.stem in neg_frames_any)
    pure_box = sum(r[c]["fp"] for s, r in zip(samples, per_image) if not s.gt
                   for c in cls_ids)
    neg["종합(순수음성)"] = {"n_frame": len(pure), "fp_box": pure_box,
                             "hit_frame": pure_hit}
    neg["종합(전 프레임)"] = {"n_frame": len(samples), "fp_box": any_box,
                              "hit_frame": len(neg_frames_any)}

    # 축 D — GT 폭 구간별 recall (+ 전체)
    buckets = {}
    for c in cls_ids:
        row = {}
        for (lo, hi), lab in zip(BUCKETS, BUCKET_LABEL):
            hit = sum(1 for r in per_image for w, _ in r[c]["matched"] if lo <= w < hi)
            miss = sum(1 for r in per_image for w in r[c]["missed"] if lo <= w < hi)
            row[lab] = {"gt": hit + miss, "hit": hit,
                        "recall": (hit / (hit + miss)) if hit + miss else None}
        t_hit = sum(v["hit"] for v in row.values())
        t_gt = sum(v["gt"] for v in row.values())
        row["전체"] = {"gt": t_gt, "hit": t_hit,
                       "recall": (t_hit / t_gt) if t_gt else None}
        buckets[CLASS_NAMES[c]] = row
    for c in CLASS_NAMES:
        if c not in cls_ids:
            buckets[CLASS_NAMES[c]] = {lab: {"gt": None, "hit": None, "recall": None}
                                       for lab in (*BUCKET_LABEL, "전체")}
    total = {}
    for lab in (*BUCKET_LABEL, "전체"):
        hit = sum(buckets[CLASS_NAMES[c]][lab]["hit"] for c in cls_ids)
        gt = sum(buckets[CLASS_NAMES[c]][lab]["gt"] for c in cls_ids)
        total[lab] = {"gt": gt, "hit": hit, "recall": (hit / gt) if gt else None}
    buckets["종합"] = total

    # 축 E — 볼라드는 박스별로 본다 (장면 maxconf 로는 붕괴가 안 보인다)
    bollard = []
    if BOLLARD_ID < nc:
        for s, r in zip(samples, per_image):
            for w, cf in r[BOLLARD_ID]["matched"]:
                bollard.append({"img": s.stem, "gt_w640": w, "conf": cf})
            for w in r[BOLLARD_ID]["missed"]:
                bollard.append({"img": s.stem, "gt_w640": w, "conf": None})
        bollard.sort(key=lambda d: d["gt_w640"])

    order = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    return {"ops": _ordered(ops, order), "negative": _ordered(neg, order),
            "buckets": _ordered(buckets, order), "bollard_boxes": bollard}


def _ordered(d: dict, order: list[str]) -> dict:
    """클래스 행을 `0 person → 1 stairs → 2 bollard` 순으로 두고 종합을 맨 뒤에."""
    return {**{k: d[k] for k in order if k in d},
            **{k: v for k, v in d.items() if k not in order}}


# 가중치가 그 클래스를 아예 모를 때 쓰는 행 — 0 과 구분된다
_NA = {"tp": None, "fp": None, "fn": None, "ignored_fp": None,
       "recall": None, "precision": None, "f1": None}


def _rates(tp, fp, fn, ignored_fp=0):
    rec = tp / (tp + fn) if tp + fn else None
    pre = tp / (tp + fp) if tp + fp else None
    f1 = (2 * pre * rec / (pre + rec)) if pre and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "ignored_fp": ignored_fp,
            "recall": rec, "precision": pre, "f1": f1}


# ──────────────────────────────────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────────────────────────────────

def _f(v, w=8, nd=3):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.{nd}f}"


def _i(v, w=6):
    """정수 칸 — `None`(그 클래스를 모르는 가중치)은 0 이 아니라 '-' 다."""
    return f"{'-':>{w}}" if v is None else f"{v:>{w}}"


def print_map(per_run: dict) -> None:
    print("\n" + "=" * 84)
    print("축 A. mAP — 임계값 무관 (model.val)")
    print("=" * 84)
    hdr = f"{'run':<24}{'class':<10}{'mAP50':>10}{'mAP50-95':>11}"
    print(hdr); print("-" * len(hdr))
    for run, info in per_run.items():
        m = info.get("map")
        if not m:
            print(f"{run:<24}{'(건너뜀 — --skip-map)':<10}")
            continue
        for name, v in m["per_class"].items():
            print(f"{run:<24}{name:<10}{_f(v['ap50'], 10)}{_f(v['ap'], 11)}")
        print(f"{run:<24}{'종합':<10}{_f(m['map50'], 10)}{_f(m['map'], 11)}")


def print_ops(rows: list) -> None:
    print("\n" + "=" * 96)
    print("축 B. 운영점 — recall · precision · F1 (클래스별 + 종합)")
    print("=" * 96)
    hdr = (f"{'run':<24}{'conf':>16}  {'class':<10}{'recall':>9}{'prec':>9}{'F1':>9}"
           f"{'TP':>6}{'FP':>6}{'FN':>6}{'ign':>5}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        for name, v in r["ops"].items():
            print(f"{r['run']:<24}{r['conf_label']:>16}  {name:<10}"
                  f"{_f(v['recall'], 9)}{_f(v['precision'], 9)}{_f(v['f1'], 9)}"
                  f"{_i(v['tp'])}{_i(v['fp'])}{_i(v['fn'])}{_i(v['ignored_fp'], 5)}")


def print_negative(rows: list) -> None:
    print("\n" + "=" * 92)
    print("축 C. 음성 프레임 오탐 — 해당 클래스 GT 가 0인 프레임. 나온 박스는 전부 오탐")
    print("=" * 92)
    hdr = (f"{'run':<24}{'conf':>16}  {'class':<16}"
           f"{'음성프레임':>11}{'오탐박스':>10}{'발화프레임':>12}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        for name, v in r["negative"].items():
            hit = ("-" if v["hit_frame"] is None
                   else f"{v['hit_frame']}/{v['n_frame']}")
            print(f"{r['run']:<24}{r['conf_label']:>16}  {name:<16}"
                  f"{_i(v['n_frame'], 11)}{_i(v['fp_box'], 10)}{hit:>12}")


def print_buckets(rows: list) -> None:
    print("\n" + "=" * 104)
    print("축 D. GT 폭 구간별 recall (입력 640 환산 · 괄호는 GT 박스 수) "
          "— 맨 오른쪽 열과 맨 아래 행이 종합")
    print("=" * 104)
    cols = (*BUCKET_LABEL, "전체")
    hdr = f"{'run':<24}{'conf':>16}  {'class':<10}" + "".join(f"{c:>14}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        for name, row in r["buckets"].items():
            line = f"{r['run']:<24}{r['conf_label']:>16}  {name:<10}"
            for lab in cols:
                v = row[lab]
                rec = "-" if v["recall"] is None else format(v["recall"], ".3f")
                cell = rec if v["gt"] is None else f"{rec}({v['gt']})"
                line += cell.rjust(14)
            print(line)


def print_bollard(rows: list) -> None:
    if not any(r["bollard_boxes"] for r in rows):
        return
    print("\n" + "=" * 84)
    print("축 E. 볼라드 **박스별** confidence — 장면 maxconf 로는 붕괴가 안 보인다")
    print("=" * 84)
    hdr = f"{'run':<24}{'conf':>16}  {'img':<24}{'GT폭@640':>10}{'conf':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        for b in r["bollard_boxes"]:
            cf = "미탐" if b["conf"] is None else f"{b['conf']:.3f}"
            print(f"{r['run']:<24}{r['conf_label']:>16}  {b['img'][-23:]:<24}"
                  f"{b['gt_w640']:>10.1f}{cf:>9}")


# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    # 가중치 — --weights 가 있으면 그쪽이 우선(학습 레이아웃 밖도 태울 수 있게)
    weights: dict[str, Path] = {}
    if args.weights:
        for w in args.weights:
            if not w.is_file():
                raise SystemExit(f"가중치가 없다: {w}")
            # `<런>/weights/best.pt` 레이아웃이면 런 이름은 두 단계 위,
            # 아니면 파일이 든 디렉토리 이름이 곧 런 이름이다
            name = (w.parent.parent.name if w.parent.name == "weights"
                    else w.parent.name) or w.stem
            weights[name] = w
    else:
        for r in (s.strip() for s in args.runs.split(",") if s.strip()):
            w = DETECT / r / "weights/best.pt"
            if not w.is_file():
                raise SystemExit(f"가중치가 없다: {w}\n"
                                 "학습 레이아웃 밖이면 --weights 로 직접 줄 것")
            weights[r] = w

    grids = ([parse_class_conf(s, args.base_conf) for s in args.class_conf]
             if args.class_conf else
             [{cid: float(c) for cid in CLASS_NAMES}
              for c in args.conf.split(",") if c.strip()])

    samples = load_samples(args.src, args.ignore_class_id)
    n_gt = {CLASS_NAMES[c]: sum(1 for s in samples for g in s.gt if g[0] == c)
            for c in CLASS_NAMES}
    n_ign = sum(len(s.ign) for s in samples)

    print("=" * 84)
    print("C5 판정 — 자체 촬영 야간분 (보행 시점 · 라벨 있음)")
    print(f"   소재     {args.src}  ·  이미지 {len(samples):,}장")
    print(f"   GT       " + " · ".join(f"{k} {v:,}" for k, v in n_gt.items())
          + (f"  ·  ignore {n_ign:,}" if n_ign else ""))
    print(f"   가중치   {', '.join(weights)}")
    print(f"   설정     imgsz {args.imgsz} · IoU {args.iou} · device {args.device}")
    print("=" * 84)

    data_yaml = build_val(samples, args.src, args.dst, args.rebuild)

    conf_floor = min(min(t.values()) for t in grids)
    per_run, rows = {}, []
    for run, w in weights.items():
        model = YOLO(str(w))
        nc = len(model.names)
        per_run[run] = {"weights": str(w), "nc": nc}
        if nc < len(CLASS_NAMES):
            print(f"\n⚠️ {run} 은 {nc}클래스다 — "
                  f"{', '.join(CLASS_NAMES[c] for c in CLASS_NAMES if c >= nc)} "
                  "행은 '-' 로 나온다 (0 으로 읽으면 '오탐 0' 오독이다)")

        if not args.skip_map:
            m = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                          project=str(DETECT), name=f"{run}__c5_own_night",
                          exist_ok=True, plots=False, verbose=False)
            per_run[run]["map"] = {
                "per_class": {model.names[int(c)]: {"ap50": float(m.box.ap50[i]),
                                                    "ap": float(m.box.ap[i])}
                              for i, c in enumerate(m.box.ap_class_index)},
                "map50": float(m.box.map50), "map": float(m.box.map)}

        # 예측은 격자 최저 conf 로 **한 번만** 뽑고 뒤에서 클래스별로 거른다.
        # NMS 는 클래스별로 돌고 높은 conf 부터 남기므로 이 둘은 같은 결과다
        # (→ eval_real_night.py 의 --class-conf 와 같은 논리).
        #
        # 🔴 **한 장씩 넘긴다 — 리스트로 묶으면 안 된다** (2026-08-25 실측).
        # ultralytics 는 rect 레터박스를 쓰는데, 세로(3060×4080)와 가로(4080×3060)가
        # 한 predict 호출에 섞이면 공통 입력 shape 로 맞춰지면서 **모든 예측이 바뀐다**
        # — 같은 볼라드가 conf 0.669 → 0.501 이 됐고 세로 이미지들까지 같이 흔들렸다.
        # 자체 촬영분은 방향이 섞일 수 있으므로 여기서 반드시 갈라야 한다
        # (→ STATUS 3장 함정 10 계열).
        cache: dict[str, list] = {}
        for s in samples:
            r = model.predict(source=str(s.img), conf=conf_floor, imgsz=args.imgsz,
                              device=args.device, verbose=False)[0]
            box = []
            if r.boxes is not None and len(r.boxes):
                for c, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                       r.boxes.xyxy.tolist()):
                    box.append((int(c), float(cf), *xyxy))
            cache[s.stem] = box

        for thr in grids:
            per_image = [
                match_one(s, cache[s.stem], thr, args.iou,
                          args.imgsz / max(s.w, s.h))
                for s in samples]
            rec = {"run": run, "nc": nc, "conf_label": thr_label(thr),
                   "class_conf": {CLASS_NAMES[c]: thr[c] for c in sorted(thr)}}
            rec.update(aggregate(samples, per_image, nc))
            rows.append(rec)
            print(f"  {run:<24}conf {thr_label(thr):<18} 완료")

    print_map(per_run)
    print_ops(rows)
    print_negative(rows)
    print_buckets(rows)
    print_bollard(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"src": str(args.src), "imgsz": args.imgsz, "iou": args.iou,
         "n_images": len(samples), "gt": n_gt, "n_ignore": n_ign,
         "runs": per_run, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n저장: {args.out}")
    print("\n라벨 규칙 검수는 별도다 — `C5` 확정 전에 한 번 돌릴 것 "
          "(→ docs/labeling_stairs.md 7장):")
    print(f"  uv run python scripts/label_stats.py "
          f"--labels {(args.dst / 'labels/val').as_posix()}")


if __name__ == "__main__":
    main()
