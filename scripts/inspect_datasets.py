"""데이터셋 무결성 검증 및 인벤토리 생성.

data/ 에 압축 해제된 공개 데이터셋 4종을 스캔해서
 - 이미지/라벨 개수
 - 쌍(pair) 정합성 — 짝 없는 파일 탐지
 - 야간 비율, 클래스 분포
를 확인한다. 학습 착수 전 1회, 이후 데이터 추가 시마다 실행.

사용:
    uv run python scripts/inspect_datasets.py
    uv run python scripts/inspect_datasets.py --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def stems(directory: Path, exts: set[str] | None = None) -> set[str]:
    """디렉토리 내 파일의 확장자를 제거한 이름 집합.

    ExDark 어노테이션은 `2015_00001.png.txt` 처럼 이중 확장자를 쓰므로
    `.txt` 를 벗긴 뒤 남는 `.png` 까지 한 번 더 제거한다.
    """
    if not directory.is_dir():
        return set()
    out = set()
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if Path(stem).suffix.lower() in IMG_EXT:  # 이중 확장자 처리
            stem = Path(stem).stem
        out.add(stem)
    return out


def check_pair(name: str, a_dir: Path, b_dir: Path,
               a_label: str, b_label: str,
               a_exts: set[str] | None = None,
               b_exts: set[str] | None = None) -> dict:
    """두 디렉토리가 1:1로 대응하는지 검사."""
    a, b = stems(a_dir, a_exts), stems(b_dir, b_exts)
    only_a, only_b = a - b, b - a
    result = {
        "name": name,
        f"{a_label}": len(a),
        f"{b_label}": len(b),
        "paired": len(a & b),
        f"only_{a_label}": sorted(only_a)[:5],
        f"only_{b_label}": sorted(only_b)[:5],
        "ok": not only_a and not only_b and bool(a),
    }
    return result


def inspect_lol() -> list[dict]:
    root = DATA_ROOT / "LOLdataset"
    return [
        check_pair(f"LOL / {split}", root / split / "low", root / split / "high",
                   "low", "high", IMG_EXT, IMG_EXT)
        for split in ("our485", "eval15")
    ]


def loli_label_dir(ann_split: Path, exposure: str) -> Path:
    """LoLI-Street 라벨 디렉토리 경로 해결.

    배포본의 명명이 split마다 다르다 — Train은 `low/`, Val은 `YOLO Annotations (low)/`.
    둘 다 지원하고, 없으면 첫 후보를 그대로 돌려줘 검사에서 0건으로 드러나게 한다.
    """
    candidates = [ann_split / exposure, ann_split / f"YOLO Annotations ({exposure})"]
    return next((c for c in candidates if c.is_dir()), candidates[0]) / "Labels"


def inspect_loli() -> list[dict]:
    root = DATA_ROOT / "LoLI-Street" / "LoLI-Street Dataset"
    ann = root / "YOLO Annotations"
    rows = []
    for split in ("Train", "Val"):
        rows.append(check_pair(f"LoLI-Street / {split} (저조도 쌍)",
                               root / split / "low", root / split / "high",
                               "low", "high", IMG_EXT, IMG_EXT))
        # YOLO 라벨은 low/high 각각 별도 제공 — low 기준으로 이미지와 대응 확인
        rows.append(check_pair(f"LoLI-Street / {split} (YOLO 라벨, low)",
                               root / split / "low", loli_label_dir(ann / split, "low"),
                               "image", "label", IMG_EXT, {".txt"}))
    test = root / "Test"
    n_test = len(stems(test, IMG_EXT))
    rows.append({"name": "LoLI-Street / Test (RLLT, 실제 저조도)",
                 "image": n_test, "ok": n_test > 0})
    return rows


def inspect_stair() -> list[dict]:
    root = DATA_ROOT / "Stair dataset"
    rows = []
    for split in ("train", "val"):
        img_dir, lbl_dir = root / split / "images", root / split / "labels"
        row = check_pair(f"StairNet / {split}", img_dir, lbl_dir,
                         "image", "label", IMG_EXT, {".txt"})
        names = stems(img_dir, IMG_EXT)
        night = sum(1 for s in names if "night" in s.lower())
        row["night"] = night
        row["night_ratio"] = f"{night / len(names):.1%}" if names else "-"
        rows.append(row)
    return rows


def inspect_exdark() -> list[dict]:
    img_root = DATA_ROOT / "ExDark" / "ExDark_data"
    ann_root = DATA_ROOT / "ExDark" / "ExDark_Annno"
    rows, totals = [], Counter()
    for cls_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        row = check_pair(f"ExDark / {cls}", cls_dir, ann_root / cls,
                         "image", "label", IMG_EXT, {".txt"})
        totals["image"] += row["image"]
        totals["label"] += row["label"]
        totals["paired"] += row["paired"]
        rows.append(row)
    rows.append({"name": "ExDark / 합계", **dict(totals),
                 "ok": all(r["ok"] for r in rows)})
    return rows


def fmt(rows: list[dict]) -> None:
    for r in rows:
        mark = "OK " if r.get("ok") else "!! "
        counts = " ".join(f"{k}={v}" for k, v in r.items()
                          if k not in {"name", "ok"} and not k.startswith("only_")
                          and not isinstance(v, list))
        print(f"  {mark}{r['name']:<45} {counts}")
        for key in (k for k in r if k.startswith("only_")):
            if r[key]:
                print(f"       ⚠ 짝 없음 {key}: {r[key]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, help="결과를 JSON으로 저장")
    args = ap.parse_args()

    if not DATA_ROOT.is_dir():
        print(f"data/ 없음: {DATA_ROOT}")
        return 1

    sections = {
        "LOL (② 저조도 개선 벤치마크)": inspect_lol(),
        "LoLI-Street (② 주력 + ③ 보조)": inspect_loli(),
        "StairNet (③ 계단)": inspect_stair(),
        "ExDark (③ 저조도 다중객체)": inspect_exdark(),
    }

    failed = 0
    for title, rows in sections.items():
        print(f"\n[{title}]")
        fmt(rows)
        failed += sum(1 for r in rows if not r.get("ok"))

    print(f"\n{'=' * 70}")
    print("전체 정상" if failed == 0 else f"이상 {failed}건 — 위 ⚠ 항목 확인 필요")

    if args.json:
        args.json.write_text(json.dumps(sections, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"JSON 저장: {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
