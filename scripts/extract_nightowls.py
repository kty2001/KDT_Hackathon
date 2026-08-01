"""NightOwls Validation 선택 압축 해제.

`nightowls_validation.zip` 은 51,848장 / 53.53 GiB 이지만 **81.7% 가 빈 프레임**이다.
전량 해제하지 않고 두 용도에 필요한 것만 뽑는다 (→ docs/data.md 5-2).

 - ③ 보행자 학습용      : 박스가 있는 이미지 9,489장 (~10.3 GiB)
 - A3 시간축 검증용     : recording 1개 통째 (기본 38번, 4,605장 ~5.0 GiB)
                          연속 프레임 ≈16fps 이라 플리커·시간축 노이즈를
                          정지영상 벤치마크 없이 볼 수 있다 (→ lowlight_classical.md 3장)

합쳐서 약 15 GiB. 두 집합은 일부 겹치므로 실제 용량은 그보다 작다.

사용:
    uv run python scripts/extract_nightowls.py                 # 계획만 출력 (기본, 쓰기 없음)
    uv run python scripts/extract_nightowls.py --run           # 실제 해제
    uv run python scripts/extract_nightowls.py --run --recording 34
    uv run python scripts/extract_nightowls.py --run --labeled-only
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

# data/NightOwls 는 D:\datasets\NightOwls 로의 Junction 이다 (→ docs/data.md 5-2).
# 경로 분기가 필요 없도록 저장소 기준 경로만 쓴다.
NIGHTOWLS = Path(__file__).resolve().parent.parent / "data" / "NightOwls"
ZIP_PATH = NIGHTOWLS / "nightowls_validation.zip"
JSON_PATH = NIGHTOWLS / "nightowls_validation.json"
ZIP_PREFIX = "nightowls_validation/"  # zip 내부 루트 디렉토리

DEFAULT_RECORDING = 38  # 4,605장 — 5개 recording 중 가장 작아 시간축 검증에 적합


def load_selection(recording: int | None, labeled_only: bool) -> tuple[set[str], dict]:
    """해제할 파일명 집합과 내역을 반환."""
    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    labeled_ids = {a["image_id"] for a in meta["annotations"]}
    labeled = {i["file_name"] for i in meta["images"] if i["id"] in labeled_ids}

    rec: set[str] = set()
    if not labeled_only and recording is not None:
        rec = {i["file_name"] for i in meta["images"]
               if i["recordings_id"] == float(recording)}
        if not rec:
            avail = sorted({i["recordings_id"] for i in meta["images"]
                            if i["recordings_id"] is not None})
            raise SystemExit(f"recording {recording} 없음. 사용 가능: {avail}")

    return labeled | rec, {"labeled": labeled, "recording": rec}


def main() -> None:
    ap = argparse.ArgumentParser(description="NightOwls Validation 선택 해제")
    ap.add_argument("--run", action="store_true",
                    help="실제 해제 (미지정 시 계획만 출력하고 종료)")
    ap.add_argument("--recording", type=int, default=DEFAULT_RECORDING,
                    help=f"통째로 뽑을 recording 번호 (기본 {DEFAULT_RECORDING})")
    ap.add_argument("--labeled-only", action="store_true",
                    help="라벨 있는 이미지만 — recording 통째 해제 생략")
    ap.add_argument("--dest", type=Path, default=NIGHTOWLS / "images",
                    help="해제 위치 (기본 data/NightOwls/images)")
    args = ap.parse_args()

    for p in (ZIP_PATH, JSON_PATH):
        if not p.exists():
            raise SystemExit(f"없음: {p}")

    selected, parts = load_selection(args.recording, args.labeled_only)

    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [m for m in z.infolist()
                   if not m.is_dir() and m.filename[len(ZIP_PREFIX):] in selected]
        total = sum(m.file_size for m in members)

        print(f"zip          : {ZIP_PATH}")
        print(f"해제 위치    : {args.dest}")
        print(f"라벨 있는 것 : {len(parts['labeled']):,}장")
        if parts["recording"]:
            print(f"recording {args.recording:<3}: {len(parts['recording']):,}장 (연속 프레임)")
        print(f"합계(중복 제거): {len(members):,}장 / {total / 1024**3:.2f} GiB")

        missing = len(selected) - len(members)
        if missing:
            print(f"⚠️ zip 에 없는 항목 {missing}건 — 라벨과 이미지가 어긋난다")

        if not args.run:
            print("\n계획만 출력했다. 실제 해제는 --run 을 붙일 것.")
            return

        args.dest.mkdir(parents=True, exist_ok=True)
        done = 0
        for m in members:
            out = args.dest / Path(m.filename).name
            if out.exists() and out.stat().st_size == m.file_size:
                done += 1
                continue
            with z.open(m) as src, open(out, "wb") as dst:
                dst.write(src.read())
            done += 1
            if done % 500 == 0:
                print(f"  {done:,}/{len(members):,}", flush=True)

    print(f"\n완료 — {done:,}장 → {args.dest}")
    print("zip 은 유지된다. 공간이 필요하면 확인 후 직접 삭제할 것 "
          "(재다운로드에 53.5 GiB 가 든다).")


if __name__ == "__main__":
    main()
