"""`data/own_night_KU/` 신규 촬영분을 `data/own_night/` 로 병합 + 파일명 순차 통일.

왜 필요한가
    `data/own_night/`(27장)은 두 raw 소스(`test_real_data` 7 + `test_real_data2` 20)가
    섞여 있고, 파일명도 KakaoTalk 내보내기 이름과 `YYYYMMDD_HHMMSS` 타임스탬프 이름이
    뒤섞여 있다(→ docs/data.md 2-2-1 8/30 정정). 여기에 `data/own_night_KU/`(신규 18장,
    라벨 없음)가 도착해 세 소스를 하나로 합치고 파일명을 `own_night_0001`~`0045`로
    통일한다.

순서 결정 (촬영 시점 순으로 이어 붙임 — 배치 내부는 아래 정렬키)
    1. `test_real_data`  (7장)  — KakaoTalk 내보내기, `_NN` 접미사 오름차순(base=0 먼저)
    2. `test_real_data2` (20장) — `YYYYMMDD_HHMMSS` 파일명 문자열 오름차순
    3. `own_night_KU`    (18장) — KakaoTalk 내보내기, `_NN` 접미사 오름차순(base=0 먼저)

    ⚠️ KakaoTalk 배치 내보내기는 한 번에 고른 사진 전부가 **같은 내보내기 타임스탬프**를
    공유해(파일명만으로는 실제 촬영 순서를 구분 못 함) `_NN` 접미사를 촬영 순서의
    대리 지표로 쓴다 — **검증된 사실이 아니라 가정**이다.

입력 레이아웃 (raw 원본 — 이 스크립트는 아래 두 폴더를 수정하지 않는다)
    data/test_real_data/*.jpg    (+ .mp4, lowlight/, lowlight_x02/ 는 무시)
    data/test_real_data2/*.jpg
    data/own_night_KU/*.jpg      (라벨 없음)

출력 (기존 `data/own_night/` 를 새 이름으로 정리)
    data/own_night/images/own_night_0001.jpg .. own_night_0045.jpg
    data/own_night/labels/own_night_0001.txt .. own_night_0027.txt   (기존 27장분만 — 라벨 그대로 rename)
    data/own_night/manifest.csv  (신규 파일명 ↔ 원래 이름/소스 매핑, gitignore 대상이라 그대로 둠)

    - `test_real_data`/`test_real_data2` 분: 이미 `data/own_night/images|labels/`에
      들어와 있는 파일을 **rename**(그 폴더는 이미 raw가 아니라 병합된 작업 사본이므로
      제자리 rename이 안전하다). raw 폴더 `data/test_real_data*`는 순서 계산에만 읽는다.
    - `own_night_KU` 분: raw이므로 **복사**만 하고 원본은 그대로 둔다. 라벨 파일은
      만들지 않는다(수작업 라벨링은 별도 단계).

안전장치
    - 배치별 예상 개수(7/20/18, 합계 45)가 실제와 다르면 즉시 중단.
    - 대상 파일(새 이름) 중 하나라도 이미 있으면 **아무것도 바꾸지 않고** 중단
      (재실행 시 덮어쓰기 방지 — 부분 실행 후 재실행하면 이 상태에 걸린다).
    - `--dry-run` 으로 실행 전 전체 매핑만 확인 가능.

사용:
    uv run python scripts/merge_own_night.py --dry-run
    uv run python scripts/merge_own_night.py
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OWN_NIGHT = ROOT / "data/own_night"
IMAGES_DIR = OWN_NIGHT / "images"
LABELS_DIR = OWN_NIGHT / "labels"
MANIFEST = OWN_NIGHT / "manifest.csv"

# base 파일명은 `KakaoTalk_<8자리 날짜>_<9자리 시:분:초:밀리초>` 로 고정 길이라,
# 그 뒤에 붙는 `_NN` 접미사만 별도 그룹으로 잡는다 — 그냥 `_(\d+)$` 로 잡으면 base
# 파일명 자체의 밀리초 부분(9자리 숫자)을 접미사로 오인해 base가 맨 뒤로 밀린다.
_KAKAO_SUFFIX_RE = re.compile(r"^KakaoTalk_\d{8}_\d{9}(?:_(\d+))?$")


def kakao_suffix_key(stem: str) -> int:
    """`KakaoTalk_..._NN` → NN, 접미사 없는 base 파일은 0(가장 먼저)."""
    m = _KAKAO_SUFFIX_RE.match(stem)
    if not m:
        raise ValueError(f"KakaoTalk 파일명 형식이 예상과 다르다: {stem}")
    return int(m.group(1)) if m.group(1) else 0


def lexical_key(stem: str) -> str:
    """`YYYYMMDD_HHMMSS` 형태는 문자열 그대로가 시간 순."""
    return stem


@dataclass
class Batch:
    name: str
    src_dir: Path
    mode: str          # "rename_in_place" | "copy_from_raw"
    sort_key: object    # Callable[[str], int | str]
    expected_count: int


BATCHES = [
    Batch("test_real_data", ROOT / "data/test_real_data", "rename_in_place", kakao_suffix_key, 7),
    Batch("test_real_data2", ROOT / "data/test_real_data2", "rename_in_place", lexical_key, 20),
    Batch("own_night_KU", ROOT / "data/own_night_KU", "copy_from_raw", kakao_suffix_key, 18),
]


@dataclass
class PlanItem:
    new_stem: str
    batch: str
    mode: str
    original_name: str
    # rename_in_place 인 경우: 현재 own_night/ 안의 경로. copy_from_raw 인 경우: raw 소스 경로.
    src_image: Path
    src_label: Path | None  # rename_in_place 이면서 라벨이 있을 때만


def build_plan() -> list[PlanItem]:
    plan: list[PlanItem] = []
    counter = 0
    for batch in BATCHES:
        if not batch.src_dir.is_dir():
            raise SystemExit(f"소스 폴더가 없다: {batch.src_dir}")
        files = sorted(batch.src_dir.glob("*.jpg"))
        if len(files) != batch.expected_count:
            raise SystemExit(
                f"{batch.name}: 예상 {batch.expected_count}장인데 실제 {len(files)}장 — "
                f"소스 폴더 내용이 계획과 다르다. {batch.src_dir} 확인 후 재시도할 것.")
        files_sorted = sorted(files, key=lambda p: batch.sort_key(p.stem))
        for f in files_sorted:
            counter += 1
            new_stem = f"own_night_{counter:04d}"
            if batch.mode == "rename_in_place":
                src_image = IMAGES_DIR / f.name
                src_label = LABELS_DIR / f"{f.stem}.txt"
                if not src_image.is_file():
                    raise SystemExit(f"{batch.name}: 이미 own_night/images 에 있어야 할 파일이 없다: {src_image}")
                if not src_label.is_file():
                    raise SystemExit(f"{batch.name}: 라벨이 없어야 할 리 없다: {src_label}")
            else:  # copy_from_raw
                src_image = f
                src_label = None
            plan.append(PlanItem(new_stem, batch.name, batch.mode, f.name, src_image, src_label))
    return plan


def check_conflicts(plan: list[PlanItem]) -> None:
    conflicts = []
    for item in plan:
        target_img = IMAGES_DIR / f"{item.new_stem}.jpg"
        if target_img.exists():
            conflicts.append(str(target_img))
        if item.src_label is not None:
            target_lbl = LABELS_DIR / f"{item.new_stem}.txt"
            if target_lbl.exists():
                conflicts.append(str(target_lbl))
    if conflicts:
        print(f"\n대상 파일 {len(conflicts)}개가 이미 존재한다 — 이전 실행이 부분/전체 완료된 것으로 보인다.")
        for c in conflicts[:20]:
            print(f"  {c}")
        if len(conflicts) > 20:
            print(f"  ... 외 {len(conflicts) - 20}개")
        raise SystemExit("덮어쓰기를 막기 위해 중단한다. data/own_night/ 상태를 확인하고 "
                          "이전 실행 결과를 정리한 뒤 재시도할 것.")


def print_plan(plan: list[PlanItem]) -> None:
    print(f"{'새 이름':<20} {'배치':<16} {'모드':<17} 원래 이름")
    for item in plan:
        print(f"{item.new_stem:<20} {item.batch:<16} {item.mode:<17} {item.original_name}")
    counts = {}
    for item in plan:
        counts[item.batch] = counts.get(item.batch, 0) + 1
    print(f"\n합계 {len(plan)}장 — " + " · ".join(f"{b} {n}" for b, n in counts.items()))


def execute(plan: list[PlanItem]) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    for item in plan:
        target_img = IMAGES_DIR / f"{item.new_stem}.jpg"
        if item.mode == "rename_in_place":
            item.src_image.rename(target_img)
            target_lbl = LABELS_DIR / f"{item.new_stem}.txt"
            item.src_label.rename(target_lbl)
        else:
            shutil.copy2(item.src_image, target_img)


def write_manifest(plan: list[PlanItem]) -> None:
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_stem", "batch", "mode", "original_name"])
        for item in plan:
            w.writerow([item.new_stem, item.batch, item.mode, item.original_name])
    print(f"\nmanifest 기록: {MANIFEST}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="매핑만 출력하고 파일은 바꾸지 않는다")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan()
    print_plan(plan)
    check_conflicts(plan)
    if args.dry_run:
        print("\n--dry-run — 파일을 바꾸지 않았다.")
        return
    execute(plan)
    write_manifest(plan)
    n_labeled = sum(1 for it in plan if it.src_label is not None)
    print(f"\n완료 — images {len(plan)}장, labels {n_labeled}장(기존분만). "
          f"신규 {len(plan) - n_labeled}장은 라벨 없음 — 수작업 라벨링 필요.")


if __name__ == "__main__":
    main()
