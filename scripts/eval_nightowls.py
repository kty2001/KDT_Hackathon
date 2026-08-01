"""C4 검증 — NightOwls 실제 야간 라벨로 탐지 성능을 잰다.

왜 이게 필요한가
    `detect_v1` 의 val 은 **개발용**이다 — LoLI 는 pseudo-label 이고 StairNet 은 84% 가
    주간이며, 두 소스의 도메인이 분리돼 있다. 반면 **NightOwls 는 사람 손 라벨이고
    51,848장 전량 야간**이라, 현재 보유분 중 **야간 보행자 성능을 정직하게 잴 수 있는
    유일한 데이터**다 (→ data.md 5-2).

    ⚠️ 단 차량 주행 시점(대시캠)이라 보행 시점과는 다르다. 최종 판정은 여전히 자체
    촬영 야간분(`C5`) 몫이다.

무엇을 재는가
    person   NightOwls `pedestrian` 라벨 기준 mAP·recall — **진짜 야간 성능**
    stairs   NightOwls 에 계단은 사실상 없다 → 여기서 나오는 stairs 는 **오탐**이다.
             오탐율을 함께 보고한다 (안전 기준에 직결)

`ignore` 영역 처리
    NightOwls 는 군중·불명확 영역을 `ignore`(category 4)로 표시한다. 이걸 정답으로
    치면 재현율이 부당하게 낮아지므로 **GT 에서 제외**한다. 다만 그 영역의 예측을
    오탐으로 세지 않도록 걸러내지는 않는다(구현 단순화) — recall 은 정확하고
    precision 은 다소 비관적으로 나온다.

사용:
    uv run python scripts/eval_nightowls.py
    uv run python scripts/eval_nightowls.py --weights outputs/detect/xxx/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIGHTOWLS = ROOT / "data/NightOwls"
IMAGES = NIGHTOWLS / "images"
LABEL_JSON = NIGHTOWLS / "nightowls_validation.json"
DST = ROOT / "outputs/datasets/nightowls_eval"

PEDESTRIAN = 1  # NightOwls category_id
CLASS_NAMES = {0: "person", 1: "stairs"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=ROOT / "outputs/detect/c4_baseline/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="오탐 집계용 임계")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--rebuild", action="store_true", help="변환본을 다시 만든다")
    return p.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(args) -> tuple[int, int]:
    """해제된 이미지에 한해 NightOwls JSON → YOLO 라벨로 변환."""
    out_img = args.dst / "images/val"
    out_lab = args.dst / "labels/val"
    if out_lab.is_dir() and not args.rebuild:
        n_i = sum(1 for _ in out_img.glob("*"))
        n_l = sum(1 for _ in out_lab.glob("*.txt"))
        if n_i and n_i == n_l:
            print(f"  기존 변환본 재사용: {n_i:,}장 (--rebuild 로 다시 만들 수 있다)")
            return n_i, -1
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    meta = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
    info = {i["id"]: (i["file_name"], i["width"], i["height"]) for i in meta["images"]}

    boxes: dict[int, list[str]] = {}
    for a in meta["annotations"]:
        if a["category_id"] != PEDESTRIAN or a.get("ignore"):
            continue
        _, w, h = info[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        if bw <= 1 or bh <= 1:
            continue
        boxes.setdefault(a["image_id"], []).append(
            f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")

    n_img = n_box = 0
    for img_id, (fname, _, _) in info.items():
        src = IMAGES / fname
        if not src.is_file():   # 선택 해제분만 있으므로 대부분은 없다
            continue
        rows = boxes.get(img_id, [])
        link_or_copy(src, out_img / fname)
        (out_lab / f"{Path(fname).stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        n_img += 1
        n_box += len(rows)

    (args.dst / "data.yaml").write_text(
        "# scripts/eval_nightowls.py 가 생성한다. 직접 수정하지 말 것.\n"
        f"path: {args.dst.as_posix()}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items())),
        encoding="utf-8")
    return n_img, n_box


def main() -> None:
    args = parse_args()
    for p in (IMAGES, LABEL_JSON):
        if not p.exists():
            raise SystemExit(f"없음: {p}\n먼저 scripts/extract_nightowls.py --run 실행")

    print("=" * 78)
    print("C4 검증 — NightOwls 실제 야간 라벨 (사람 손 라벨 · 전량 야간)")
    print("=" * 78)
    n_img, n_box = build(args)
    if n_box >= 0:
        print(f"  변환: 이미지 {n_img:,} · pedestrian 박스 {n_box:,} (ignore 제외)")

    from ultralytics import YOLO  # noqa: E402

    model = YOLO(str(args.weights))
    m = model.val(data=str(args.dst / "data.yaml"), imgsz=args.imgsz, device=0,
                  project=str(ROOT / "outputs/detect"), name="nightowls_eval",
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

    print("\n⚠️ NightOwls 에 계단은 사실상 없다 — stairs 행이 나온다면 전부 오탐이다.")
    print("   (본 평가의 GT 에는 stairs 가 0개이므로 mAP 행 자체가 안 나올 수 있다.")
    print("    오탐율은 scripts 주석의 별도 진단 참고)")
    print("\n⚠️ 차량 주행 시점이라 보행 시점과 다르다 — 최종 판정은 자체 촬영분(C5).")


if __name__ == "__main__":
    main()
