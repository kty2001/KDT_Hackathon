"""①②③④ 파이프라인 end-to-end 데모 — 연속 프레임 → 동영상.

`C9` 통합의 **사전 검증**이다. 앱이 없어도 데스크톱에서 파이프라인 전체가 도는지,
그리고 정지영상으로는 안 보이던 것(박스 깜빡임·플리커)이 실제로 어떤지 눈으로 본다.

★ 기본 구조는 **표시/탐지 경로 분리**다 (2026-08-02 `C7` 판정, → docs/detection.md 7장)

    프레임 ─┬─→ [탐지 경로] 원본 그대로 → ③ YOLO ──────────┐
            │                                              ↓
            └─→ [표시 경로] ② arm(글레어·대비) ────→ ④ 경계선 강조 → 화면

    ② 를 탐지 앞단에 두면 mAP 가 내려가고 무엇보다 **`stairs` 야간 오탐이
    0.1% → 5.7% 로 되살아난다.** 그래서 탐지는 원본을 먹는다. 분리는 비용이 아니라
    이득이다 — 두 경로가 독립이라 병렬화할 수 있고, ② 의 지연이 ③ 에 더해지지 않는다.
    `--fused` 로 옛 직렬 구조(②→③)를 재현해 비교할 수 있다.

★ 탐지 프레임 스킵 (`--detect-every`)
    실시간 예산상 ③ 를 매 프레임 돌릴 수 없다 (→ README 2장: 2~3프레임 주기).
    스킵 구간은 직전 박스를 **그대로 유지(hold)** 한다. 이때 박스가 튀거나 깜빡이는
    것이 보이면 트래킹 보간(`W6`)이 필요하다는 신호다 — 이 데모의 관찰 목적 중 하나.

사용:
    uv run python scripts/pipeline_demo.py                       # rec38 600프레임
    uv run python scripts/pipeline_demo.py --frames 1200 --detect-every 1
    uv run python scripts/pipeline_demo.py --arm D1A1+bf --layout single
    uv run python scripts/pipeline_demo.py --fused               # 옛 직렬 구조 비교
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from emphasize import draw_emphasis, from_ultralytics
from lowlight import build
from temporal_eval import frame_paths, load_sequence

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs/demo"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path,
                   default=ROOT / "outputs/detect/c4b_loli0/weights/best.pt")
    p.add_argument("--recording", type=int, default=38)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--arm", default="D1A1+bf+ts", help="표시 경로 ② arm")
    p.add_argument("--width", type=int, default=640, help="② 내부 처리 가로")
    p.add_argument("--detect-every", type=int, default=2,
                   help="③ 실행 주기(프레임). 사이 구간은 직전 박스 유지")
    p.add_argument("--conf", type=float, default=0.25,
                   help="확정 운영값 0.25 (`C4e` S1 — 일괄 하향은 기각됐다)")
    p.add_argument("--fps", type=float, default=16.0, help="NightOwls 촬영 fps")
    p.add_argument("--layout", choices=["side", "single"], default="side",
                   help="side = 원본 | 처리본 나란히 (차이를 보여주는 용도)")
    p.add_argument("--fused", action="store_true",
                   help="옛 직렬 구조(②→③) 재현 — 탐지도 처리본을 먹는다")
    p.add_argument("--labels", action="store_true", help="클래스명 표시(디버그용)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def label_bar(frame: np.ndarray, text: str) -> np.ndarray:
    """상단에 얇은 캡션 — 시연에서 무엇을 보는지 알려준다."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"가중치가 없다: {args.weights}")
    from ultralytics import YOLO

    paths = frame_paths(args.recording, args.start, args.frames)
    seq = load_sequence(paths, 0)          # 원 해상도 유지 — 탐지 경로가 원본을 쓴다
    if not seq:
        raise SystemExit("프레임을 읽지 못했다")
    h, w = seq[0].shape[:2]

    arm = build(args.arm)
    arm.reset()
    model = YOLO(str(args.weights))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"rec{args.recording}_{args.arm.replace('+', '_')}" + ("_fused" if args.fused else "")
    out_path = args.out or OUT_DIR / f"pipeline_{tag}.mp4"
    ow = w * 2 if args.layout == "side" else w
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (ow, h))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter 를 열 수 없다: {out_path}")

    print("=" * 84)
    print(f"파이프라인 데모 — recording {args.recording} · {len(seq)}프레임 · {w}×{h}")
    print(f"   구조   {'②→③ 직렬 (fused, 옛 구조)' if args.fused else '표시/탐지 분리 (C7 판정 구조)'}")
    print(f"   ② arm {args.arm} @{args.width}px · ③ {args.weights.parent.parent.name}"
          f" · 탐지 주기 {args.detect_every}프레임")
    print("=" * 84)

    t_arm = t_det = t_draw = 0.0
    dets: list = []
    n_det_frames = 0
    for i, frame in enumerate(seq):
        # --- 표시 경로: ② ---
        t0 = time.perf_counter()
        work = frame
        if args.width and w != args.width:
            work = cv2.resize(frame, (args.width, round(h * args.width / w)),
                              interpolation=cv2.INTER_AREA)
        shown = arm(work)
        if shown.shape[:2] != (h, w):
            shown = cv2.resize(shown, (w, h), interpolation=cv2.INTER_LINEAR)
        t_arm += time.perf_counter() - t0

        # --- 탐지 경로: ③ (원본 입력이 기본, --fused 면 처리본) ---
        if i % args.detect_every == 0:
            t0 = time.perf_counter()
            src = shown if args.fused else frame
            r = model.predict(src, conf=args.conf, imgsz=640, device=0, verbose=False)[0]
            dets = from_ultralytics(r, min_conf=args.conf)
            t_det += time.perf_counter() - t0
            n_det_frames += 1
        # else: 직전 dets 를 그대로 유지(hold) — 깜빡임 관찰 지점

        # --- ④ 강조 ---
        t0 = time.perf_counter()
        rendered = draw_emphasis(shown, dets, min_conf=args.conf, labels=args.labels)
        t_draw += time.perf_counter() - t0

        if args.layout == "side":
            writer.write(np.hstack([label_bar(frame, "原本 / original"),
                                    label_bar(rendered, f"② {args.arm} + ④ 강조")]))
        else:
            writer.write(rendered)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(seq)}", flush=True)

    writer.release()
    n = len(seq)
    total = (t_arm + t_det + t_draw) / n * 1000

    print(f"\n저장: {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MB)")
    print("\n" + "=" * 84)
    print("단계별 비용 (프레임당 평균, 개발 PC)")
    print("=" * 84)
    print(f"  ② {args.arm:<14} {t_arm / n * 1000:6.1f} ms   (표시 경로 · 매 프레임)")
    print(f"  ③ YOLO11n            {t_det / n * 1000:6.1f} ms   "
          f"(탐지 {n_det_frames}회 / {n}프레임 = 주기 {args.detect_every})")
    print(f"  ④ 강조 렌더          {t_draw / n * 1000:6.1f} ms")
    print(f"  {'합계':<21} {total:6.1f} ms  →  {1000 / total:5.1f} FPS")
    print("\n⚠️ 개발 PC(RTX 4060 Ti) 기준이다. 목표는 **폰에서** 720p 15FPS 이므로")
    print("   이 수치로 게이트를 판정하지 말 것 — 절대 판정은 `C11` 실기기다.")
    print("\n👁️ 영상에서 확인할 것")
    print("   1) 가로등 통과 구간에서 화면 전체가 출렁이는가 (플리커 — `+ts` 유무 비교)")
    print("   2) 탐지 스킵 구간에서 박스가 튀거나 깜빡이는가 (→ 트래킹 보간 `W6` 필요 신호)")
    print("   3) 횡단보도를 `stairs` 로 잡는가 (→ C7 이 지적한 ② 의 오탐 되살아남)")


if __name__ == "__main__":
    main()
