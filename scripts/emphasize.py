"""④ 선택적 강조 — 검출 BBox 의 **경계선만** 대비색으로 그린다.

README 2장의 ④ 정의: *"검출된 위험 요소의 경계선만 대비 색으로 표시하고 내부는
원본 유지 — 잔존시력을 가리지 않으면서 '봐야 할 것'만 강조"*.

③ YOLO 와의 인터페이스는 `Detection` 하나다. ③이 아직 학습 전이므로(→ TODO C4)
어댑터 `from_ultralytics()` 로 사전학습 COCO 모델 출력도 그대로 받는다 — 통합(C9)
시점에 ③ 쪽이 무엇이 되든 이 파일은 바뀌지 않는다.

    from emphasize import Detection, draw_emphasis
    out = draw_emphasis(frame, [Detection("person", 0.87, (x1, y1, x2, y2))])

설계 근거 (저시력·야맹 사용자 기준)
    이중 스트로크   검정 밑선 + 대비색 본선. 단색 선은 배경 밝기에 따라 사라진다 —
                    강조선이 가로등 옆에서는 어둡게, 암부에서는 밝게 보여야 하므로
                    양쪽 모두에 대비를 만드는 밑선이 필수다
    비채움          내부를 칠하거나 반투명 오버레이를 얹으면 잔존시력이 보는 원본
                    정보를 가린다. 경계선 밖은 일절 건드리지 않는다
    색 선택         stairs=노랑(위험 관례색·휘도 최상위), person=시안(노랑과 색상·
                    휘도 모두 분리). **빨강은 쓰지 않는다** — 휘도가 낮아 야간
                    배경에 묻히고, 적록색약에서 구분이 무너진다
    깜빡임 금지     대상 사용자가 광과민이다. 점멸·펄스 강조는 금지하고 항상 정적
                    윤곽으로 그린다 (플리커는 안전 이슈 — lowlight_classical.md 3-1)
    두께            프레임 짧은 변에 비례(720p 기준 4px). 저시력 가독을 위해 통상
                    디버그 렌더보다 굵게 잡았다. 실기기·사용자 검증으로 조정할 것

시간축 주의 — ③을 매 2~3프레임만 돌리면(→ hardware_inference.md) 박스가 프레임마다
튀거나 사라져 깜빡임처럼 보인다. 박스 좌표의 IIR 평활·유지(hold) 는 ③ 트래킹 쪽
과제로, 이 모듈은 **단일 프레임 렌더만** 책임진다.

데모 (이 머신에서 실행 가능 — data/sample_image.png 만 필요):
    uv run python scripts/emphasize.py            # ② D1A1+bf → ③ COCO 사전학습 → ④
    uv run python scripts/emphasize.py --no-yolo  # ③ 없이 합성 박스로 렌더만 확인
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

Frame = np.ndarray  # BGR uint8 (H, W, 3)

ROOT = Path(__file__).resolve().parent.parent

# 클래스 → 강조색 (BGR). 위 docstring 의 "색 선택" 참고.
PALETTE: dict[str, tuple[int, int, int]] = {
    "stairs": (0, 215, 255),  # 노랑
    "person": (255, 230, 0),  # 시안
}
FALLBACK_COLOR = (255, 255, 255)  # 미지정 클래스 — 흰색
UNDERLAY_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class Detection:
    """③ → ④ 인터페이스. ③ 구현이 무엇이든 이 형태로 변환해 넘긴다."""

    name: str                        # 클래스명 ("person", "stairs", ...)
    conf: float                      # 신뢰도 0~1
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 (픽셀, 프레임 좌표)


def from_ultralytics(result, min_conf: float = 0.0) -> list[Detection]:
    """ultralytics `Results` 1건 → Detection 목록."""
    dets = []
    for b in result.boxes:
        conf = float(b.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        dets.append(Detection(result.names[int(b.cls[0])], conf, (x1, y1, x2, y2)))
    return dets


def stroke_width(frame: Frame, base: int = 4) -> int:
    """프레임 짧은 변에 비례한 본선 두께 (720p 기준 base px)."""
    return max(2, round(base * min(frame.shape[:2]) / 720))


def draw_emphasis(frame: Frame, detections: Iterable[Detection], *,
                  min_conf: float = 0.35, labels: bool = False,
                  inplace: bool = False) -> Frame:
    """경계선만 이중 스트로크로 그린다. 내부·외부 픽셀은 건드리지 않는다.

    min_conf: 렌더 단계의 최종 필터. ③이 이미 걸렀다면 0 으로 두면 된다.
    labels:   클래스명 텍스트. 사용자용 화면에는 끄고(작은 글씨는 저시력에
              무의미) 팀 디버그용으로만 켠다.
    """
    out = frame if inplace else frame.copy()
    h, w = out.shape[:2]
    t = stroke_width(out)
    rim = max(1, t // 2)  # 밑선이 본선 밖으로 드러나는 폭

    for d in detections:
        if d.conf < min_conf:
            continue
        x1 = int(np.clip(d.box[0], 0, w - 1))
        y1 = int(np.clip(d.box[1], 0, h - 1))
        x2 = int(np.clip(d.box[2], 0, w - 1))
        y2 = int(np.clip(d.box[3], 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = PALETTE.get(d.name, FALLBACK_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), UNDERLAY_COLOR, t + 2 * rim, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)
        if labels:
            text = f"{d.name} {d.conf:.2f}"
            org = (x1, max(y1 - t - 4, 12))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        UNDERLAY_COLOR, 4, cv2.LINE_AA)
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# 데모 — ②(D1A1+bf) → ③(COCO 사전학습) → ④ 를 한 장에서 잇는다
# --------------------------------------------------------------------------


def _demo():
    import time

    from lowlight import build

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=ROOT / "data/sample_image.png")
    p.add_argument("--arm", default="D1A1+bf", help="② 전처리 arm (none 이면 원본)")
    p.add_argument("--no-yolo", action="store_true",
                   help="③ 을 생략하고 합성 박스로 렌더만 확인")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/emphasize_demo.png")
    args = p.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"이미지를 읽지 못했다: {args.image}")
    if frame.shape[1] > 1280:
        h = int(1280 * frame.shape[0] / frame.shape[1])
        frame = cv2.resize(frame, (1280, h), interpolation=cv2.INTER_AREA)

    enhanced = build(args.arm)(frame)

    dets: list[Detection] = []
    used = "합성 박스"
    if not args.no_yolo:
        # 사전학습 COCO 모델 — ③ baseline(C4) 전의 임시 탐지기. stairs 클래스는
        # 없으므로 이 데모에서 계단은 잡히지 않는다 (person 등 COCO 클래스만).
        try:
            from ultralytics import YOLO
            weights = ROOT / "outputs/yolo11n.pt"  # outputs/ 는 git 비추적
            model = YOLO(str(weights) if weights.exists() else "yolo11n.pt")
            r_raw = model.predict(frame, conf=0.25, verbose=False)[0]
            r_enh = model.predict(enhanced, conf=0.25, verbose=False)[0]
            print(f"③ COCO 사전학습(yolo11n) — 원본 {len(r_raw.boxes)}건 / "
                  f"② {args.arm} 후 {len(r_enh.boxes)}건 검출 (n=1 참고용)")
            dets = from_ultralytics(r_enh)
            used = (f"yolo11n ({len(dets)}건)" if dets
                    else "yolo11n 0건 → 합성 박스")
        except Exception as e:  # 오프라인 등 — 렌더 확인은 계속한다
            print(f"⚠️ YOLO 사용 불가({e}) → 합성 박스로 대체")

    if not dets:
        h, w = enhanced.shape[:2]
        dets = [
            Detection("person", 0.91, (int(w * .12), int(h * .35), int(w * .30), int(h * .78))),
            Detection("stairs", 0.84, (int(w * .48), int(h * .55), int(w * .86), int(h * .92))),
            Detection("pole", 0.51, (int(w * .68), int(h * .18), int(w * .74), int(h * .55))),
        ]

    emphasized = draw_emphasis(enhanced, dets, labels=True)

    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        draw_emphasis(enhanced, dets)
    ms = (time.perf_counter() - t0) * 1000 / n
    print(f"④ 렌더 비용: {ms:.2f}ms/frame ({len(dets)}박스, {enhanced.shape[1]}x{enhanced.shape[0]})")

    cell_w = 420
    hh = int(cell_w * frame.shape[0] / frame.shape[1])
    row = np.hstack([cv2.resize(im, (cell_w, hh))
                     for im in (frame, enhanced, emphasized)])
    header = np.full((30, row.shape[1], 3), 32, np.uint8)
    for i, text in enumerate(["INPUT", f"(2) {args.arm}", f"(2)+(4) {used}"]):
        cv2.putText(header, text, (i * cell_w + 8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack([header, row]))
    print(f"데모 저장: {args.out}")


if __name__ == "__main__":
    _demo()
