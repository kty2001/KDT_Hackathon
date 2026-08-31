"""발표용 시각화 자료를 outputs/presentation/ 한 곳에 모은다.

`outputs/`는 `.gitignore` 대상이지만 `outputs/presentation/`만 예외로 추적한다
(`.gitignore` 참고) — 노트북(`notebooks/dl_presentation_summary.ipynb`)이 참조하는
이미지는 전부 이 스크립트로 이 폴더 안에 모여 있어야 한다. 세 단계로 동작한다:

1. 이미 실행된 리뷰 노트북(`c5_own_night_review.ipynb`, `c4f_distill_test_real_viz.ipynb`)의
   JSON에 임베드된 이미지/표 출력을 재실행 없이 그대로 디코드
2. 증류 teacher/student는 RunPod 학습이라 로컬에 `results.png`가 없으므로
   `results.csv`에서 Ultralytics 표준 함수로 즉석 생성
3. 위에서 생성된 것 포함, `outputs/detect/*/`에 있는 학습곡선류 PNG 8종을
   `outputs/presentation/curve_*.png`로 복사(원본은 그대로 둠 — 복사이지 이동 아님)

사용:
    uv run python scripts/build_presentation_assets.py
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from ultralytics.utils.plotting import plot_results

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
DETECT = ROOT / "outputs" / "detect"
OUT_DIR = ROOT / "outputs" / "presentation"

# (노트북, 셀 인덱스, 셀 안의 이미지 출력 순번, 저장 파일명)
IMAGE_TARGETS = [
    ("c5_own_night_review.ipynb", 11, 0, "01_bollard.png"),
    ("c5_own_night_review.ipynb", 11, 1, "02_person_bollard_crosswalk.png"),
    ("c5_own_night_review.ipynb", 11, 2, "03_stairs.png"),
    ("c5_own_night_review.ipynb", 11, 3, "04_three_person.png"),
    ("c5_own_night_review.ipynb", 11, 4, "05_negative_fp.png"),
    ("c4f_distill_test_real_viz.ipynb", 9, 0, "06_distill_predictions.png"),
    ("c4f_distill_test_real_viz.ipynb", 11, 0, "07_distill_gt.png"),
]

# (노트북, 셀 인덱스, 저장 파일명) — text/html 표 출력
TABLE_TARGETS = [
    # 후보 모델 4종 비교 (c4b_loli0 / c4d_11n_640 / c4e_s3_11n / c4e_s3_11n_INT8)
    ("c5_own_night_review.ipynb", 6, "08_candidates_map50_table.html"),
    ("c5_own_night_review.ipynb", 8, "09_candidates_negative_fp_table.html"),
    # 증류 teacher/student 비교 (c4f_11s_640_teacher / c4f_distill_11n_640 / c4e_s3_11n / INT8)
    ("c4f_distill_test_real_viz.ipynb", 4, "10_distill_map50_table.html"),
    ("c4f_distill_test_real_viz.ipynb", 5, "11_distill_operating_point_table.html"),
    ("c4f_distill_test_real_viz.ipynb", 6, "12_distill_negative_fp_table.html"),
]

# results.csv 만 있고 results.png 가 없는 런 — 복사 전에 CSV에서 재생성
DISTILL_RUNS = ["c4f_11s_640_teacher", "c4f_distill_11n_640"]

# (outputs/detect/ 기준 상대경로, 저장 파일명) — 학습곡선류 PNG 복사
CURVE_TARGETS = [
    ("c4b_loli0/results.png", "curve_c4b_loli0_results.png"),
    ("c4e_s3_11n/results.png", "curve_c4e_s3_11n_results.png"),
    ("c4e_s3_11n/confusion_matrix.png", "curve_c4e_s3_11n_confusion_matrix.png"),
    ("c4e_s3_11n/BoxPR_curve.png", "curve_c4e_s3_11n_prcurve.png"),
    ("c4f_11s_640_teacher/results.png", "curve_teacher_results.png"),
    ("c4f_distill_11n_640/results.png", "curve_student_results.png"),
    ("c4f_11s_640_teacher__rec34_labeled/BoxPR_curve.png", "curve_teacher_rec34_prcurve.png"),
    ("c4f_distill_11n_640__rec34_labeled/BoxPR_curve.png", "curve_student_rec34_prcurve.png"),
]


def load_notebook(name: str) -> dict:
    return json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))


def image_outputs(cell: dict) -> list[str]:
    return [o["data"]["image/png"] for o in cell.get("outputs", [])
            if "data" in o and "image/png" in o["data"]]


def html_outputs(cell: dict) -> list[str]:
    return [o["data"]["text/html"] for o in cell.get("outputs", [])
            if "data" in o and "text/html" in o["data"]]


def extract_review_outputs() -> None:
    nb_cache: dict[str, dict] = {}

    for nb_name, cell_idx, img_idx, out_name in IMAGE_TARGETS:
        nb = nb_cache.setdefault(nb_name, load_notebook(nb_name))
        cell = nb["cells"][cell_idx]
        imgs = image_outputs(cell)
        b64 = imgs[img_idx]
        data = b64 if isinstance(b64, str) else "".join(b64)
        (OUT_DIR / out_name).write_bytes(base64.b64decode(data))
        print(f"저장: {out_name}  <- {nb_name} cell[{cell_idx}] image[{img_idx}]")

    for nb_name, cell_idx, out_name in TABLE_TARGETS:
        nb = nb_cache.setdefault(nb_name, load_notebook(nb_name))
        cell = nb["cells"][cell_idx]
        html = html_outputs(cell)
        content = html[-1] if isinstance(html[-1], str) else "".join(html[-1])
        (OUT_DIR / out_name).write_text(content, encoding="utf-8")
        print(f"저장: {out_name}  <- {nb_name} cell[{cell_idx}] html table")


def generate_distill_curves() -> None:
    for run in DISTILL_RUNS:
        plot_results(file=None, dir=str(DETECT / run))
        print(f"생성: {run}/results.png  <- results.csv")


def copy_curve_images() -> None:
    for rel_src, out_name in CURVE_TARGETS:
        src = DETECT / rel_src
        shutil.copy2(src, OUT_DIR / out_name)
        print(f"복사: {out_name}  <- outputs/detect/{rel_src}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extract_review_outputs()
    generate_distill_curves()
    copy_curve_images()
    n_images = len(IMAGE_TARGETS) + len(CURVE_TARGETS)
    print(f"\n완료: {OUT_DIR} 에 이미지 {n_images}개 + 표 {len(TABLE_TARGETS)}개 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
