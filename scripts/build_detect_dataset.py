"""C4 / C4b 준비 — ③ 통합 탐지 데이터셋 구성 (`person` + `stairs` 단일 모델).

세 소스를 하나의 YOLO 데이터셋으로 합친다.

    person  ← LoLI-Street (동봉 YOLO 라벨에서 person 만 추출)
    person  ← NightOwls   (사람 손 라벨 · 전량 야간)          ★ C4b 에서 추가
    stairs  ← StairNet    (scripts/stairnet_to_bbox.py 산출물)

★ 핵심 설계 1 — **`low` 이미지에 `high` 라벨을 붙인다** (2026-08-01 실측 근거)

    LoLI-Street 는 low/high 양쪽에 라벨이 따로 들어 있는데, **둘이 일치하지 않는다.**
    실측(train 200장):

        low/high 픽셀 정합    구조 상관 0.991 — 같은 장면·같은 크기 ✅
        person 박스 수        low 294 vs high 367 (**high 가 +24.8%**)
        개수가 같은 경우에도  좌표가 다른 것이 19/45

    → 두 버전에 **탐지기를 각각 돌려 만든 pseudo-label** 이다. 그리고 어두운 쪽에서
      탐지기가 놓친 사람이 `low` 라벨에서 **배경으로 기록**돼 있다.

    `low` 이미지 + `low` 라벨로 학습하면 **"어두우면 놓치는 것"을 그대로 학습한다.**
    두 버전이 픽셀 정합이므로 **`high` 라벨을 `low` 이미지의 정답으로 쓰는 것이
    기하학적으로 타당하고 더 완전하다.**

★ 핵심 설계 2 — **NightOwls 를 학습에 투입한다 (C4b, 2026-08-02)**

    C4 baseline 은 개발 val 에서 person 0.794 인데 **NightOwls 실야간에서 mAP50
    0.127 · recall 0.195 로 무너졌다** (rec 34 기준). 원인은 모델이 아니라 데이터다 —
    LoLI-Street 는 *주간을 어둡게 만든 합성본*이라 실제 야간 통계로 전이되지 않는다
    (→ docs/detection.md 5장). NightOwls 는 사람 손 라벨이고 전량 야간이라 이 갭을
    직접 메운다.

    동시에 **`stairs` 야간 오탐(7.7%) 도 같이 잡힌다.** NightOwls 는 계단이 없는 야간
    거리라 stairs 라벨이 비어 있고, 오탐 대상으로 지목된 횡단보도·차선·포장 텍스처가
    그 안에 들어 있다 — 별도 음성 표본을 만들 필요가 없다.

    ⚠️ **recording 단위로 나눈다.** 프레임이 연속(≈16fps)이라 랜덤 분할은 누수다.

        train  36 · none · 37   4,819장 / 7,974박스
        val    35                 212장 /   348박스   (모델 선택용)
        제외   34                              ← held-out 판정용. 절대 학습에 넣지 말 것
        제외   38                              ← ② A3 시간축 · stairs 오탐 전용

    ⚠️ **`ignore` 만 있거나 자전거/오토바이 운전자만 있는 프레임은 뺀다.** 사람이
    있는데 GT 가 비어 있어, 넣으면 "사람인데 배경"을 가르친다 — 고치려는 낮은
    재현율을 오히려 강화한다 (→ scripts/nightowls_yolo.py).

출력 — 원본 미수정
    outputs/datasets/<이름>/
    ├── images/{train,val}/     LoLI·StairNet 은 하드링크, NightOwls 는 JPG 변환본 하드링크
    ├── labels/{train,val}/
    └── data.yaml

    NightOwls 는 1024×640 PNG(장당 ~1MB)라 C4 에서 병목이던 디스크 부하를 키운다.
    `outputs/datasets/nightowls_jpg/` 에 JPG 로 한 번 변환해 캐시하고 각 데이터셋에는
    하드링크만 건다 — 두 번째 데이터셋을 만들어도 변환·용량이 늘지 않는다.

사용:
    uv run python scripts/build_detect_dataset.py                        # C4 재현(LoLI만)
    uv run python scripts/build_detect_dataset.py --nightowls \
        --dst outputs/datasets/detect_v2_loli6000                        # C4b 런1
    uv run python scripts/build_detect_dataset.py --nightowls --loli-n 0 \
        --dst outputs/datasets/detect_v2_loli0                           # C4b 런2
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

from nightowls_yolo import CLASS_NAMES, NightOwlsIndex, write_data_yaml  # noqa: E402,F401

ROOT = Path(__file__).resolve().parent.parent
LOLI = ROOT / "data/LoLI-Street/LoLI-Street Dataset"
STAIRS = ROOT / "outputs/datasets/stairs_yolo"
NIGHTOWLS = ROOT / "data/NightOwls"
NO_IMAGES = NIGHTOWLS / "images"
NO_JSON = NIGHTOWLS / "nightowls_validation.json"
NO_JPG = ROOT / "outputs/datasets/nightowls_jpg"
DST = ROOT / "outputs/datasets/detect_v1"

AIHUB = ROOT / "outputs/datasets/aihub_yolo"

# 클래스 배치는 nightowls_yolo.CLASS_NAMES 한 곳에만 있다 (여기서 다시 정의하지 말 것 —
# 예전엔 같은 dict 가 두 파일에 복사돼 있어 클래스를 늘릴 때 한쪽만 고칠 위험이 있었다).
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
                        "**0 이면 LoLI 를 아예 빼고** 전량은 --loli-all. 기본 6000")
    p.add_argument("--loli-all", action="store_true", help="LoLI 전량 사용")
    p.add_argument("--loli-val-n", type=int, default=800, help="val 에 넣을 LoLI 수")
    p.add_argument("--label-src", choices=["high", "low"], default="high",
                   help="LoLI 라벨을 어느 쪽에서 가져올지. 기본 high (모듈 docstring 참고)")
    p.add_argument("--image-src", choices=["low", "high", "both"], default="low",
                   help="LoLI 이미지를 어느 쪽에서 가져올지. 기본 low (야간 도메인)")
    p.add_argument("--nightowls", action="store_true",
                   help="NightOwls 를 학습에 투입한다 (C4b)")
    p.add_argument("--aihub", action="store_true",
                   help="AIHub 인도보행을 투입한다 — **bollard 의 유일한 소스** (C4c). "
                        "선행: scripts/aihub_to_yolo.py(XML) 또는 "
                        "scripts/aihub_subset_to_yolo.py(이미 YOLO 인 서브셋). ⚠️ 주간 전용")
    p.add_argument("--aihub-src", type=Path, default=AIHUB,
                   help="AIHub 변환본 경로 ({images,labels}/{train,val}). "
                        "🔴 원본이 D: 이면 dst 도 D: 에 둘 것 — 볼륨을 넘으면 하드링크가 "
                        "안 걸려 통째로 복사된다 (→ STATUS 3장 함정 14)")
    p.add_argument("--no-train-recs", default="36,none,37",
                   help="NightOwls 학습용 recording (쉼표 구분)")
    p.add_argument("--no-val-recs", default="35",
                   help="NightOwls 검증용 recording. **34·38 은 넣지 말 것**")
    p.add_argument("--jpg-quality", type=int, default=92)
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--seed", type=int, default=0)
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


def add_loli(split: str, out_split: str, n: int, args) -> tuple[int, int]:
    lab_dir = LABEL_DIRS[(split, args.label_src)]
    sides = ["low", "high"] if args.image_src == "both" else [args.image_src]

    # ⚠️ split 마다 **독립된 난수원**을 쓴다. 공용 rng 를 돌려쓰면 train 을 건너뛴
    # 런(--loli-n 0)에서 val 표본까지 달라져, "LoLI 를 뺀 것" 외의 차이가 섞인다.
    rng = random.Random(f"{args.seed}:{split}")

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


def add_aihub(out_split: str, args) -> tuple[int, int, int]:
    """AIHub 인도보행 변환본을 얹는다 → (이미지, person 박스, bollard 박스).

    ⚠️ **주간 전용 소스다.** `C4` 가 "주간 학습 → 실야간 붕괴" 를 이미 보여 줬으므로
    (→ detection.md 4-2), 이걸 넣은 뒤에도 판정은 반드시 held-out(rec 34)에서 한다.
    `C4c` 게이트는 **person recall 0.609 회귀 없음**이다 — 볼라드를 얻자고 사람
    성능을 깎으면 실패다.

    분할은 aihub_to_yolo.py 가 이미 블록 단위로 나눠 뒀다(연속 프레임 누수 방지).
    여기서 다시 섞지 않는다.
    """
    src_img = args.aihub_src / "images" / out_split
    src_lab = args.aihub_src / "labels" / out_split
    if not src_img.is_dir():
        raise SystemExit(
            f"AIHub 변환본이 없다: {src_img}\n"
            "먼저 실행할 것: uv run python scripts/aihub_to_yolo.py (CVAT XML 에서) 또는\n"
            "               uv run python scripts/aihub_subset_to_yolo.py (이미 YOLO 인 서브셋에서)")

    out_img = args.dst / "images" / out_split
    out_lab = args.dst / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    n = n_person = n_bollard = 0
    for img in sorted(src_img.glob("*")):
        lab = src_lab / f"{img.stem}.txt"
        if not lab.is_file():
            continue
        link_or_copy(img, out_img / img.name)
        text = lab.read_text(encoding="utf-8")
        (out_lab / f"{img.stem}.txt").write_text(text, encoding="utf-8")
        for row in text.splitlines():
            if not row.strip():
                continue
            cid = int(float(row.split()[0]))
            n_person += cid == 0
            n_bollard += cid == 2
        n += 1
    return n, n_person, n_bollard


def ensure_jpg(names: list[str], quality: int) -> int:
    """NightOwls PNG → JPG 캐시. 이미 있으면 건너뛴다. (새로 변환한 장수)"""
    from PIL import Image

    NO_JPG.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (NO_JPG / f"{Path(n).stem}.jpg").is_file()]
    if not todo:
        return 0

    def conv(name: str) -> None:
        with Image.open(NO_IMAGES / name) as im:
            im.convert("RGB").save(NO_JPG / f"{Path(name).stem}.jpg",
                                   quality=quality, subsampling=0)

    print(f"  JPG 변환 {len(todo):,}장 …", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(conv, todo))
    return len(todo)


def add_nightowls(out_split: str, recs: set[str], args, idx: NightOwlsIndex,
                  available: set[str]) -> tuple[int, int]:
    picked = idx.select(available, recs, drop_unlabeled_person=True)
    if not picked:
        raise SystemExit(f"NightOwls recording {sorted(recs)} 에서 선택된 이미지가 0장")

    names = [idx.file_name(i) for i in picked]
    ensure_jpg(names, args.jpg_quality)

    out_img = args.dst / "images" / out_split
    out_lab = args.dst / "labels" / out_split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    boxes = 0
    for iid in picked:
        stem = Path(idx.file_name(iid)).stem
        rows = idx.rows.get(iid, [])
        link_or_copy(NO_JPG / f"{stem}.jpg", out_img / f"no_{stem}.jpg")
        (out_lab / f"no_{stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        boxes += len(rows)
    return len(picked), boxes


def main() -> None:
    args = parse_args()
    loli_n = 0 if args.loli_all else args.loli_n
    # LoLI 제외는 **train 에만** 적용한다. val 을 두 런에서 똑같이 유지해야
    # "LoLI 를 뺀 것" 말고는 달라진 게 없는 대조가 된다 (모델 선택 기준 고정).
    use_loli_train = args.loli_all or args.loli_n > 0

    print("=" * 78)
    print("③ 통합 탐지 데이터셋 — 클래스 "
          + " / ".join(f"{i} {n}" for i, n in sorted(CLASS_NAMES.items())))
    if use_loli_train:
        print(f"   LoLI 이미지 {args.image_src} · 라벨 {args.label_src}"
              f"{'  ★ 권장 조합' if (args.image_src, args.label_src) == ('low', 'high') else '  ⚠️ 비권장'}")
    else:
        print("   LoLI **train 에서 제외** — 합성 야간이 실야간 전이를 막는지 보는 대조군")
        print("   (val 은 두 런 비교를 위해 그대로 둔다)")
    if args.nightowls:
        print(f"   NightOwls train rec [{args.no_train_recs}] · val rec [{args.no_val_recs}]")
    print(f"   출력 {args.dst}")
    print("=" * 78)

    idx = available = None
    if args.nightowls:
        for p in (NO_IMAGES, NO_JSON):
            if not p.exists():
                raise SystemExit(f"없음: {p}\n먼저 scripts/extract_nightowls.py --run 실행")
        for bad in ("34", "38"):
            if bad in args.no_train_recs.split(",") or bad in args.no_val_recs.split(","):
                raise SystemExit(
                    f"recording {bad} 은 학습·검증에 쓸 수 없다 — "
                    "34 는 held-out 판정용, 38 은 ② A3 시간축 전용이다")
        idx = NightOwlsIndex(NO_JSON)
        available = {p.name for p in NO_IMAGES.glob("*")}

    total_person = total_stairs = total_bollard = 0
    for split, out_split, n, recs in (
            ("Train", "train", loli_n, args.no_train_recs),
            ("Val", "val", args.loli_val_n, args.no_val_recs)):
        print(f"\n[{out_split}]")
        li = lb = 0
        if use_loli_train or out_split == "val":
            li, lb = add_loli(split, out_split, n, args)
            print(f"  LoLI      이미지 {li:,} · person 박스 {lb:,}")
        ni = nb = 0
        if args.nightowls:
            ni, nb = add_nightowls(out_split, {r.strip() for r in recs.split(",") if r.strip()},
                                   args, idx, available)
            print(f"  NightOwls 이미지 {ni:,} · person 박스 {nb:,}  (실야간·손 라벨)")
        ai = ap = ab = 0
        if args.aihub:
            ai, ap, ab = add_aihub(out_split, args)
            print(f"  AIHub     이미지 {ai:,} · person 박스 {ap:,} · "
                  f"bollard 박스 {ab:,}  (⚠️ 주간)")
        si, sb = add_stairs(out_split, out_split, args)
        print(f"  Stairs    이미지 {si:,} · stairs 박스 {sb:,}")
        print(f"  합계      이미지 {li + ni + ai + si:,} · 박스 {lb + nb + ap + ab + sb:,}")
        person = lb + nb + ap
        if sb:
            bal = f"person:stairs = {person / sb:.1f} : 1"
            if ab:
                bal += f"  ·  person:bollard = {person / ab:.1f} : 1"
            print(f"  클래스 균형 {bal}")
        if person:
            print(f"  person 박스 중 실야간(NightOwls) 비중 {nb / person:.1%}"
                  f"{f' · 주간(AIHub) {ap / person:.1%}' if ap else ''}")
        total_person += person
        total_stairs += sb
        total_bollard += ab

    loli_desc = ("train 제외" if not use_loli_train
                 else "전량" if loli_n == 0 else str(loli_n))
    no_desc = (f"train[{args.no_train_recs}] val[{args.no_val_recs}]"
               if args.nightowls else "미사용")
    yaml = write_data_yaml(
        args.dst, "scripts/build_detect_dataset.py", "images/train", "images/val",
        extra=f"LoLI 이미지={args.image_src}/라벨={args.label_src}/{loli_desc}"
              f" · NightOwls={no_desc}"
              f" · AIHub={'투입(주간·bollard)' if args.aihub else '미사용'}")

    print(f"\n출력: {args.dst}\n  {yaml}")
    if args.aihub:
        print(f"\n★ bollard 박스 총 {total_bollard:,}개 — **주간 전용 소스**다.")
        print("   `C4c` 판정 게이트: held-out(rec 34) person recall 0.609 · stairs 오탐")
        print("   0.2% 에 회귀가 없어야 한다. 볼라드를 얻자고 기존 성능을 깎으면 실패다.")
        if total_bollard < 2000:
            print(f"   ⚠️ {total_bollard:,}개는 **학습에 턱없이 부족하다**(NightOwls person 은")
            print("      7,972개였다). 지금은 배선 검증용이고, 실학습은 전량 부분 다운로드")
            print("      뒤에 한다 (→ data.md 3-1 · aihubshell 부분 다운로드).")
    print("\n⚠️ 이 val 은 **개발용**이다. 최종 판정은 held-out 인 NightOwls rec 34")
    print("   (uv run python scripts/eval_nightowls.py --recordings 34 "
          "--drop-unlabeled-person) 와 자체 촬영 야간분(`C5`) 이다.")


if __name__ == "__main__":
    main()
