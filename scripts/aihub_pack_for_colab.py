"""AIHub 인도보행 다운로드분 → **Colab 으로 옮길 수 있는 크기**의 YOLO 데이터셋.

★ 왜 이 스크립트가 필요한가 — **AI Hub 는 해외 IP 다운로드를 차단한다** (2026-08-04 실측)

    Colab 런타임에서 `aihubshell` 을 돌리면 이렇게 끝난다.

        Download failed with HTTP status 502.
        AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다.

    Colab 은 미국·유럽에 뜨므로 이 경로는 원천적으로 막혀 있다. 그래서 **다운로드는
    국내에서** 하고, 학습만 Colab 에서 한다. 문제는 원천이 태스크당 10GB 씩이라
    그대로는 Drive 로 못 옮긴다는 것인데 — **학습 입력이 어차피 640 이라 미리 줄이면
    정보 손실 없이 10배 이상 작아진다.** 1920×1080 3만 장 30GB → 640×360 2~3GB.

★ 그리고 이 데이터셋은 **라벨과 원천이 안 갈려 있다** (2026-08-04 실측)

    `aihubshell -mode l -datasetkey 189` 의 실제 트리는 이렇다.

        뎁스프리딕션        Depth_001~005.zip     19GB …
        바운딩박스          Bbox_1_new.zip        10GB … (30개)
        서피스마스킹        Surface_1.zip         10GB … (5개)
        폴리곤세그멘테이션  Polygon_1_new.zip     10GB … (14개)

    **태스크별로만 나뉘고 라벨/원천 구분이 없다.** data.md 3-1-3 이 계획한 "라벨 전량을
    먼저 받아 분포를 실측한 뒤 원천을 고른다" 는 순서가 성립하지 않는다. 대신 **태스크별
    1번 zip 만 먼저 받아 정찰**한다 — 목적(원천 600GB 를 헛되이 받지 않기)은 같다.

    받을 것:  바운딩박스(person·bollard) · 서피스마스킹(stairs)
    안 받을 것: 폴리곤세그멘테이션(같은 장애물 29종이라 얻을 게 없다) · 뎁스(무관)

무엇을 하는가
    1. 받은 zip/디렉토리를 훑어 **CVAT XML 을 태스크별로 자동 분류**한다
       (장애물 29종이 보이면 bbox · 노면 20종이 보이면 surface)
    2. 장애물 → `person`(0) · `bollard`(2)   — `aihub_to_yolo.parse_xml` 재사용
    3. 노면 `caution_zone > stairs` 폴리곤 → bbox → `stairs`(1)
    4. ★ **게이트** — 노면 stairs 의 면적·종횡비를 StairNet 과 대조해
       **학습 투입 / 평가 전용**을 가른다 (→ detection.md 8-3)
    5. 이미지를 긴 변 640 으로 줄여 JPEG 재인코딩
    6. **파일명 블록 분할**(무작위 금지 — 연속 프레임이다) · data.yaml · stats.json
    7. `--zip` 이면 Drive 에 올릴 zip 까지 만든다

사용:
    # 1) 국내에서 AIHub 로그인 → 바운딩박스 1번·서피스마스킹 1번 zip 다운로드·압축해제
    # 2) 변환·리사이즈·패키징
    uv run python scripts/aihub_pack_for_colab.py --src D:\\datasets\\AIHub인도보행 --zip
    uv run python scripts/aihub_pack_for_colab.py --src ... --no-surface   # 장애물만
    uv run python scripts/aihub_pack_for_colab.py --src ... --dry-run      # 분포만 센다

    # 3) outputs/datasets/aihub_colab.zip 을 Drive/bammasil/datasets/ 에 올린다
    # 4) notebooks/colab_aihub_train.ipynb 실행
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aihub_to_yolo import LABEL_MAP, POLE_LIKE, parse_xml  # noqa: E402
from nightowls_yolo import CLASS_NAMES, STAIRS_ID, write_data_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "outputs/datasets/aihub_colab"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 노면 20종 중 우리가 찾는 것. `caution_zone` 아래 5종 중 하나라 라벨명이 그대로
# `stairs` 일 수도, `caution_zone` + 속성일 수도 있어 양쪽을 다 본다.
STAIRS_KEYS = ("stairs", "stair")

# 태스크 판별용 지문 — 두 스키마에만 있는 이름들
BBOX_MARKERS = {"bollard", "tree_trunk", "traffic_sign", "pole", "car"}
SURFACE_MARKERS = {"sidewalk", "roadway", "caution_zone", "braille_guide_blocks",
                   "bike_lane", "alley"}

# StairNet 실측 (→ detection.md 8-1) — 게이트의 비교 기준
STAIRNET_AREA_MEDIAN = {"train": 0.475, "val": 0.383}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True,
                   help="압축 해제한 AIHub 다운로드분의 루트 (하위를 전부 훑는다)")
    p.add_argument("--dst", type=Path, default=DST)
    p.add_argument("--imgsz", type=int, default=640,
                   help="긴 변 기준. 학습 입력과 같게 두면 정보 손실이 없다")
    p.add_argument("--quality", type=int, default=90, help="JPEG 품질")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--max-images", type=int, default=0,
                   help="장수 상한. **균등 간격으로 솎아 낸다**(앞부분만 자르지 않는다). "
                        "0=전부. Drive 여유가 빠듯할 때 쓴다 — 장당 ~80KB")
    p.add_argument("--no-surface", action="store_true",
                   help="노면 stairs 를 아예 다루지 않는다")
    p.add_argument("--stairs", choices=["auto", "train", "eval"], default="auto",
                   help="노면 stairs 처리. auto=게이트 판정에 따름 · "
                        "train=학습 투입 · eval=평가 전용 분리")
    p.add_argument("--pole-as-bollard", action="store_true",
                   help="pole·tree_trunk 도 클래스 2 로 (제품 정의 — 회의 안건 7)")
    p.add_argument("--workers", type=int, default=8, help="리사이즈 스레드")
    p.add_argument("--dry-run", action="store_true",
                   help="분포·게이트만 보고 파일은 안 쓴다")
    p.add_argument("--zip", action="store_true", help="Drive 업로드용 zip 까지 만든다")
    return p.parse_args()


# ────────────────────────────────────────────────────────────── 분류·파싱

def _has_stairs(*vals) -> bool:
    return any(v is not None and any(k in str(v).lower() for k in STAIRS_KEYS)
               for v in vals)


def classify_xml(xml_path: Path) -> str:
    """XML 하나를 훑어 'bbox' / 'surface' / 'unknown' 판별.

    파일명·디렉토리명으로 가르지 않는다 — AIHub 는 배포마다 폴더 이름을 바꾼다.
    **안에 들어 있는 라벨명**이 스키마의 지문이라 그것으로 가른다.
    """
    labels = set()
    try:
        for _, el in ET.iterparse(xml_path, events=("start",)):
            if el.tag in ("box", "polygon", "polyline", "mask"):
                if el.get("label"):
                    labels.add(el.get("label").lower())
            if len(labels) >= 12:
                break
    except ET.ParseError:
        return "unknown"
    if labels & SURFACE_MARKERS:
        return "surface"
    if labels & BBOX_MARKERS:
        return "bbox"
    return "unknown"


def parse_surface_stairs(xml_path: Path):
    """노면 CVAT XML → (이미지명, W, H, [(x1,y1,x2,y2)]). stairs 만 담는다.

    폴리곤 → bbox 는 min/max 다. `C1`(StairNet 선분→bbox)에서 이미 한 변환이라
    새로운 위험이 없다.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return
    for im in root.findall("image"):
        name = im.get("name")
        W, H = float(im.get("width") or 0), float(im.get("height") or 0)
        if not (name and W and H):
            continue
        boxes = []
        for shape in ("polygon", "polyline", "box"):
            for el in im.findall(shape):
                attrs = [a.text for a in el.findall("attribute")]
                if not _has_stairs(el.get("label"), *attrs):
                    continue
                if shape == "box":
                    boxes.append((float(el.get("xtl")), float(el.get("ytl")),
                                  float(el.get("xbr")), float(el.get("ybr"))))
                    continue
                pts = [tuple(map(float, q.split(",")))
                       for q in (el.get("points") or "").split(";") if "," in q]
                if len(pts) >= 3:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
        if boxes:
            yield name, W, H, boxes


def to_yolo_rows(cls: int, boxes, W: float, H: float) -> list[str]:
    rows = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                    f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return rows


# ────────────────────────────────────────────────────────────── 게이트

def stairs_gate(records) -> tuple[bool, dict]:
    """노면 stairs 가 우리 '계단 전체 1박스' 정의와 같은 것인지 가른다.

    노면 라벨은 `grating`·`manhole` 과 나란한 **"노면 구역"** 이다. 폴리곤이 디딤면만
    덮었다면 bbox 가 StairNet 보다 작고 납작해진다. 그러면 학습에 섞지 않고 **평가
    전용**으로 쓴다 — 정의가 섞이면 두 소스를 다 망친다 (→ detection.md 8-3).
    """
    areas, ratios = [], []
    for _, W, H, boxes in records:
        for x1, y1, x2, y2 in boxes:
            w, h = (x2 - x1) / W, (y2 - y1) / H
            if w > 0 and h > 0:
                areas.append(w * h)
                ratios.append(w / h)
    if not areas:
        return False, {}

    def q(v, p):
        return st.quantiles(v, n=100)[p - 1] if len(v) > 2 else float("nan")

    stat = {
        "n_box": len(areas),
        "area_p25": q(areas, 25), "area_median": st.median(areas), "area_p75": q(areas, 75),
        "ratio_median": st.median(ratios),
        "stairnet_area_median": STAIRNET_AREA_MEDIAN,
    }
    return st.median(areas) >= 0.15, stat


# ────────────────────────────────────────────────────────────── 출력

def make_portable(yaml_path: Path) -> None:
    """`data.yaml` 에서 `path:` 를 **지운다** — zip 이 다른 머신에서 풀려도 돌게.

    ⚠️ 2026-08-04 실측한 함정. ultralytics 는 `path:` 를 이렇게 푼다.

        절대경로  → 그대로 쓴다
        상대경로  → **`settings['datasets_dir']` 기준** (Colab 은 `/content/datasets`)
        없음      → **yaml 파일이 있는 디렉토리 기준** ← 우리가 원하는 것

    즉 상대경로를 적어 두면 Colab 에서
    `images not found, missing path '/content/datasets/outputs/datasets/...'`
    로 죽고, 절대경로를 적으면 이번엔 **Windows 경로가 박혀** 역시 못 쓴다.
    zip 으로 옮기는 데이터셋은 **`path:` 를 아예 빼는 것**이 유일하게 이식성 있는 답이다.
    """
    lines = [ln for ln in yaml_path.read_text(encoding="utf-8").splitlines()
             if not ln.startswith("path:")]
    lines.insert(1, "# path: 를 일부러 뺐다 — ultralytics 가 **이 파일의 위치**를 기준으로")
    lines.insert(2, "# 잡으므로 어느 머신에서 풀든 동작한다 (→ aihub_pack_for_colab.py)")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resize_one(args_tuple) -> bool:
    from PIL import Image

    src, dst, imgsz, quality = args_tuple
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = imgsz / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return True
    except Exception:
        return False


def emit(records: dict, dst: Path, args, prefix: str, img_index: dict,
         val_ratio: float, max_images: int = 0) -> dict:
    """{이미지명: [YOLO 행]} → images/labels 트리. 반환: split 별 (이미지, 박스) 수."""
    names = sorted(records)                     # ★ 무작위 금지 — 연속 프레임이다

    # 상한이 걸리면 **균등 간격**으로 솎는다. 앞에서 자르면 특정 촬영 구간만 남지만,
    # 균등 추출은 전 구간을 유지하면서 인접 프레임(≈16fps 의 쌍둥이)을 걷어내
    # 오히려 같은 장수당 정보량이 늘어난다. 정렬 순서는 그대로라 블록 분할은 유효하다.
    if 0 < max_images < len(names):
        step = len(names) / max_images
        names = [names[int(i * step)] for i in range(max_images)]
        print(f"  ⚠️ 장수 상한 {max_images:,} — {step:.1f} 프레임마다 하나씩 균등 추출")

    n_val = int(len(names) * val_ratio)
    splits = {"train": names[:len(names) - n_val], "val": names[len(names) - n_val:]}

    out = {}
    for split, keys in splits.items():
        img_dir, lab_dir = dst / "images" / split, dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        jobs, pending = [], []
        for name in keys:
            src = img_index.get(Path(name).name)
            if src is None:
                continue
            stem = f"{prefix}_{Path(name).stem}"
            jobs.append((src, img_dir / f"{stem}.jpg", args.imgsz, args.quality))
            pending.append((stem, records[name]))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            ok = list(ex.map(resize_one, jobs))

        n_img = n_box = 0
        for good, (stem, rows) in zip(ok, pending):
            if not good:
                continue
            (lab_dir / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            n_img += 1
            n_box += len(rows)
        out[split] = (n_img, n_box)
    return out


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"경로가 없다: {args.src}")

    mapping = dict(LABEL_MAP)
    if args.pole_as_bollard:
        mapping.update(POLE_LIKE)

    print("=" * 78)
    print("AIHub 인도보행 → Colab 용 YOLO 데이터셋")
    print(f"   원본 {args.src}")
    print(f"   출력 {args.dst}  ·  긴 변 {args.imgsz}px JPEG q{args.quality}")
    print(f"   매핑 {mapping}")
    print("=" * 78)

    # 1) 이미지 인덱싱 + XML 분류
    img_index: dict[str, Path] = {}
    xmls: list[Path] = []
    for p in args.src.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXT:
            img_index.setdefault(p.name, p)
        elif p.suffix.lower() == ".xml":
            xmls.append(p)

    kinds = Counter()
    by_kind: dict[str, list[Path]] = {"bbox": [], "surface": [], "unknown": []}
    for x in xmls:
        k = classify_xml(x)
        kinds[k] += 1
        by_kind[k].append(x)

    print(f"\n이미지 {len(img_index):,}장 · XML {len(xmls):,}개")
    print(f"  장애물(bbox) {kinds['bbox']:,} · 노면(surface) {kinds['surface']:,} "
          f"· 미상 {kinds['unknown']:,}")
    if not img_index:
        raise SystemExit("이미지를 못 찾았다 — zip 압축을 풀었는지 확인할 것")
    if kinds["unknown"]:
        print("  ⚠️ 미상 XML 은 건너뛴다. 스키마가 바뀌었는지 하나 열어 볼 것")

    # 2) 장애물 → person · bollard
    obstacle: dict[str, list[str]] = {}
    dropped = Counter()
    n_person = n_bollard = 0
    for x in by_kind["bbox"]:
        recs, _, drop = parse_xml(x, mapping, 0.0)
        dropped += drop
        for name, W, H, rows in recs:
            if not rows:
                continue
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
            obstacle.setdefault(name, []).extend(lines)
            for c, *_ in rows:
                n_person += c == 0
                n_bollard += c == 2

    print(f"\n[장애물] 프레임 {len(obstacle):,}장 · person {n_person:,} · bollard {n_bollard:,}")
    print("  버린 클래스 상위: "
          + ", ".join(f"{k} {v:,}" for k, v in dropped.most_common(6)))
    if 0 < n_bollard < 1000:
        print(f"  ⚠️ bollard {n_bollard:,}개는 적다 — zip 을 더 받을지 판단할 것")

    # 3) 노면 → stairs (+ 게이트)
    surface_records, gate_stat, use_train = [], {}, False
    if not args.no_surface:
        for x in by_kind["surface"]:
            surface_records += list(parse_surface_stairs(x))
        n_sbox = sum(len(r[3]) for r in surface_records)
        print(f"\n[노면] stairs 프레임 {len(surface_records):,}장 · 박스 {n_sbox:,}개")

        if surface_records:
            use_train, gate_stat = stairs_gate(surface_records)
            print(f"\n  ── 게이트: StairNet 대조 ({gate_stat['n_box']:,} 박스)")
            print(f"     면적 비율  p25 {gate_stat['area_p25']:.3f} · "
                  f"중앙 {gate_stat['area_median']:.3f} · p75 {gate_stat['area_p75']:.3f}")
            print(f"     종횡비 중앙 {gate_stat['ratio_median']:.2f}")
            print(f"     StairNet 면적 중앙 — train {STAIRNET_AREA_MEDIAN['train']} · "
                  f"val {STAIRNET_AREA_MEDIAN['val']}")
            if args.stairs != "auto":
                use_train = args.stairs == "train"
                print(f"     → --stairs {args.stairs} 로 수동 지정됨")
            elif use_train:
                print("     → 계단 덩어리를 덮는 쪽. **학습 투입**")
            else:
                print("     → StairNet 보다 훨씬 작다(디딤면만?). **평가 전용으로 분리**")
            print("     ⚠️ 자동 판정은 참고값이다. 예측 몇 장을 눈으로 보고 확정할 것")
        elif by_kind["surface"]:
            print("  ⚠️ 노면 XML 은 있는데 stairs 가 0 이다 — 라벨명이 다를 수 있다")
        else:
            png = [p for p in args.src.rglob("*.png") if "mask" in p.name.lower()]
            if png:
                print(f"  ⚠️ 마스크로 보이는 PNG {len(png):,}개 — 노면이 **PNG 마스크 포맷**이면")
                print("     신규 변환 코드가 필요하다 (data.md 3-1-2 미확인 항목의 답)")

    if args.dry_run:
        print("\n--dry-run — 파일은 쓰지 않았다.")
        return

    # 4) 출력
    stairs_in_train = bool(surface_records) and use_train
    if stairs_in_train:
        for name, W, H, boxes in surface_records:
            rows = to_yolo_rows(STAIRS_ID, boxes, W, H)
            if rows:
                obstacle.setdefault(name, []).extend(rows)

    print()
    res = emit(obstacle, args.dst, args, "aihub", img_index, args.val_ratio,
               args.max_images)
    n_out = sum(v[0] for v in res.values())
    for split, (i, b) in res.items():
        print(f"[{split}] 이미지 {i:,}장 · 박스 {b:,}개")
    print(f"  예상 zip 크기 ≈ {n_out * 80 / 2**20:.2f} GiB  (장당 ~80KB 실측)")

    data_yaml = write_data_yaml(
        args.dst, "scripts/aihub_pack_for_colab.py", "images/train", "images/val",
        extra=f"AIHub 인도보행 · 긴 변 {args.imgsz}px · ⚠️ 주간 전용")
    make_portable(data_yaml)

    eval_res = {}
    if surface_records and not stairs_in_train:
        eval_dst = args.dst.parent / f"{args.dst.name}_stairs_eval"
        recs = {name: to_yolo_rows(STAIRS_ID, boxes, W, H)
                for name, W, H, boxes in surface_records}
        recs = {k: v for k, v in recs.items() if v}
        eval_res = emit(recs, eval_dst, args, "aihubsurf", img_index, 1.0)
        make_portable(write_data_yaml(
            eval_dst, "scripts/aihub_pack_for_colab.py", "images/val", "images/val",
            extra="노면 stairs — **평가 전용**. 학습에 넣지 말 것"))
        print(f"\n[평가 전용] {eval_dst} — val {eval_res.get('val', (0, 0))[0]:,}장")

    stats = {
        "생성": "scripts/aihub_pack_for_colab.py",
        "원본": str(args.src),
        "클래스": {str(k): v for k, v in CLASS_NAMES.items()},
        "imgsz": args.imgsz,
        "장애물": {"프레임": len(obstacle), "person": n_person, "bollard": n_bollard,
                   "버린_클래스": dict(dropped.most_common(10))},
        "노면_stairs": {"프레임": len(surface_records),
                        "박스": sum(len(r[3]) for r in surface_records),
                        "학습투입": stairs_in_train, "게이트": gate_stat},
        "분할": {k: {"이미지": v[0], "박스": v[1]} for k, v in res.items()},
    }
    (args.dst / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n산출물: {args.dst}")
    print(f"  data.yaml · stats.json (분포 실측 — **문서에 옮길 것**)")

    if args.zip:
        import shutil
        z = shutil.make_archive(str(args.dst), "zip", root_dir=args.dst)
        print(f"\nzip: {z}  ({Path(z).stat().st_size / 2**30:.2f} GiB)")
        print("  → Drive 의 bammasil/datasets/ 에 올린 뒤")
        print("     notebooks/colab_aihub_train.ipynb 를 실행한다")

    print("\n⚠️ 주간 전용이다. 야간 성능은 이 데이터로 판정되지 않는다 —")
    print("   held-out(NightOwls rec 34) 또는 자체 촬영분(C5)에서 한다.")


if __name__ == "__main__":
    main()
