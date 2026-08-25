"""C4 / C4b 검증 — NightOwls 실제 야간 라벨로 탐지 성능을 잰다.

왜 이게 필요한가
    `detect_v1` 의 val 은 **개발용**이다 — LoLI 는 pseudo-label 이고 StairNet 은 84% 가
    주간이며, 두 소스의 도메인이 분리돼 있다. 반면 **NightOwls 는 사람 손 라벨이고
    51,848장 전량 야간**이라, 현재 보유분 중 **야간 보행자 성능을 정직하게 잴 수 있는
    유일한 데이터**다 (→ data.md 5-2).

    ⚠️ 단 차량 주행 시점(대시캠)이라 보행 시점과는 다르다. 최종 판정은 여전히 자체
    촬영 야간분(`C5`) 몫이다.

무엇을 재는가
    person   NightOwls `pedestrian` 라벨 기준 mAP·recall — **진짜 야간 성능**
    stairs   NightOwls 에 계단은 사실상 없다 → 여기서 나오는 stairs 는 **전부 오탐**이다.
             GT 가 0개라 mAP 행이 안 나오므로 **예측을 직접 세서** 오탐율을 보고한다.

★ C4b 대응 — recording 단위 평가 (2026-08-02 추가)
    NightOwls 를 학습에 투입하면(C4b) **학습에 쓴 recording 으로 평가할 수 없다.**
    프레임이 연속(≈16fps)이라 같은 recording 안에서는 사실상 같은 장면이다.
    `--recordings` 로 평가 대상을 recording 으로 제한한다.

        train  36 · none · 37       (박스 7,974)
        val    35                   (모델 선택용, 박스 348)
        test   34                   ← **평가는 여기서** (박스 1,577, 학습에 미사용)
        제외   38                   (② A3 시간축 전용 + stairs 오탐 전용)

    ⚠️ **before/after 를 같은 자로 재야 한다.** 기존에 보고된 person mAP50 0.205 /
    recall 0.226 은 해제분 13,602장 **전체** 기준이라 rec 34 한정 수치와 직접 비교할
    수 없다. C4b 판정 전에 기존 가중치를 같은 옵션으로 다시 돌릴 것.

★ `--drop-unlabeled-person` — 사람이 있는데 정답을 모르는 프레임
    `ignore`(군중·불명확) 만 있거나 자전거/오토바이 운전자만 있는 프레임은 GT 가
    빈 라벨이 된다. 평가에서 이걸 두면 **맞게 찾은 사람이 오탐으로 집계**돼 precision
    이 부당하게 낮아진다. 이 옵션으로 평가에서 제외한다.
    (학습에서는 **반드시** 제외한다 → `scripts/nightowls_yolo.py` 모듈 docstring)

사용:
    uv run python scripts/eval_nightowls.py                              # 해제분 전체
    uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person
    uv run python scripts/eval_nightowls.py --recordings 38 --fp-only    # stairs 오탐 전용
    uv run python scripts/eval_nightowls.py --weights outputs/detect/c4b_loli0/weights/best.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from nightowls_yolo import NightOwlsIndex, STAIRS_ID, write_data_yaml

ROOT = Path(__file__).resolve().parent.parent
NIGHTOWLS = ROOT / "data/NightOwls"
IMAGES = NIGHTOWLS / "images"
LABEL_JSON = NIGHTOWLS / "nightowls_validation.json"
DST_ROOT = ROOT / "outputs/datasets/nightowls_split"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=ROOT / "outputs/detect/c4_baseline/weights/best.pt")
    p.add_argument("--recordings", default="",
                   help="평가에 쓸 recording (쉼표 구분, 예 '34'). "
                        "빈 값이면 해제분 전체. id 가 없는 그룹은 'none'")
    p.add_argument("--drop-unlabeled-person", action="store_true",
                   help="ignore 만 있거나 자전거/오토바이 운전자만 있는 프레임을 제외")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="오탐 집계용 임계")
    p.add_argument("--fp-only", action="store_true",
                   help="mAP 는 건너뛰고 stairs 오탐만 센다 (GT 없는 recording 용)")
    p.add_argument("--name", default="", help="ultralytics run 이름 (기본: 자동 생성)")
    p.add_argument("--dst", type=Path, default=None)
    p.add_argument("--rebuild", action="store_true", help="변환본을 다시 만든다")
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


def build(args, recs: set[str]) -> tuple[int, int, int]:
    """해제분 중 선택 대상을 YOLO 형식으로 깐다. (이미지 수, 박스 수, 제외 수)"""
    idx = NightOwlsIndex(LABEL_JSON)
    available = {p.name for p in IMAGES.glob("*")}

    picked = idx.select(available, recs, args.drop_unlabeled_person)
    if not picked:
        raise SystemExit("선택된 이미지가 0장이다 — --recordings 값을 확인할 것")
    n_dropped = (len(idx.select(available, recs)) - len(picked)
                 if args.drop_unlabeled_person else 0)
    n_box = sum(len(idx.rows.get(i, [])) for i in picked)

    out_img = args.dst / "images/val"
    out_lab = args.dst / "labels/val"
    if out_lab.is_dir() and not args.rebuild:
        if sum(1 for _ in out_img.glob("*")) == len(picked):
            print(f"  기존 변환본 재사용: {len(picked):,}장 (--rebuild 로 다시 만든다)")
            return len(picked), n_box, n_dropped

    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)
    for iid in picked:
        fname = idx.file_name(iid)
        rows = idx.rows.get(iid, [])
        link_or_copy(IMAGES / fname, out_img / fname)
        (out_lab / f"{Path(fname).stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    write_data_yaml(args.dst, "scripts/eval_nightowls.py", "images/val", "images/val")
    return len(picked), n_box, n_dropped


def scan_stairs_fp(model, img_dir: Path, conf: float, imgsz: int) -> tuple[int, int, int]:
    """stairs 예측을 직접 센다 — NightOwls 에 계단이 없으므로 전부 오탐이다."""
    n_img = n_fp_img = n_fp_box = 0
    for r in model.predict(source=str(img_dir), conf=conf, imgsz=imgsz, device=0,
                           stream=True, verbose=False):
        n_img += 1
        k = int((r.boxes.cls == STAIRS_ID).sum()) if r.boxes is not None else 0
        if k:
            n_fp_img += 1
            n_fp_box += k
    return n_img, n_fp_img, n_fp_box


def main() -> None:
    args = parse_args()
    for p in (IMAGES, LABEL_JSON):
        if not p.exists():
            raise SystemExit(f"없음: {p}\n먼저 scripts/extract_nightowls.py --run 실행")

    recs = {s.strip() for s in args.recordings.split(",") if s.strip()}
    tag = ("rec" + "-".join(sorted(recs)) if recs else "all") + \
          ("_labeled" if args.drop_unlabeled_person else "")
    if args.dst is None:
        args.dst = DST_ROOT / tag
    run_name = args.name or f"{args.weights.parent.parent.name}__{tag}"

    print("=" * 78)
    print("NightOwls 평가 — 사람 손 라벨 · 전량 야간")
    print(f"   가중치   {args.weights}")
    print(f"   대상     {'recording ' + ', '.join(sorted(recs)) if recs else '해제분 전체'}"
          f"{'  · 정답 불명 프레임 제외' if args.drop_unlabeled_person else ''}")
    print("=" * 78)

    n_img, n_box, n_dropped = build(args, recs)
    print(f"  평가 대상: 이미지 {n_img:,} · pedestrian 박스 {n_box:,} (ignore 제외)")
    if n_dropped:
        print(f"  제외한 정답 불명 프레임: {n_dropped:,}장")

    from ultralytics import YOLO  # noqa: E402

    model = YOLO(str(args.weights))

    if not args.fp_only:
        m = model.val(data=str(args.dst / "data.yaml"), imgsz=args.imgsz, device=0,
                      project=str(ROOT / "outputs/detect"), name=run_name,
                      exist_ok=True, plots=True)

        print("\n" + "=" * 78)
        print("결과 — 진짜 야간 보행자 성능")
        print("=" * 78)
        names = model.names
        print(f"{'class':<10}{'mAP50':>10}{'mAP50-95':>12}{'precision':>12}{'recall':>10}")
        print("-" * 54)
        for i, c in enumerate(m.box.ap_class_index):
            print(f"{names[int(c)]:<10}{m.box.ap50[i]:>10.3f}{m.box.ap[i]:>12.3f}"
                  f"{m.box.p[i]:>12.3f}{m.box.r[i]:>10.3f}")

    # stairs 오탐 — GT 가 0개라 mAP 로는 안 잡힌다. 예측을 직접 센다
    print("\n" + "=" * 78)
    print(f"stairs 오탐 (conf {args.conf}) — NightOwls 에 계단은 사실상 없다")
    print("=" * 78)
    t, fi, fb = scan_stairs_fp(model, args.dst / "images/val", args.conf, args.imgsz)
    print(f"  오탐 이미지 {fi:,} / {t:,}  ({fi / t:.1%})  ·  오탐 박스 {fb:,}")

    print("\n⚠️ 차량 주행 시점이라 보행 시점과 다르다 — 최종 판정은 자체 촬영분(C5).")


if __name__ == "__main__":
    main()
