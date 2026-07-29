"""② 저조도 개선 — arm 비교 하네스 (A1 vs A2, 무처리 대조군 포함).

docs/data.md 2-3-3 의 4-arm 실험과 **동일한 골격**이다. 여기서는 아직 ③ 탐지 mAP 를
붙이지 않고, 그 전에 확인할 수 있는 것만 잰다.

    속도    720p 단독 ms/frame — 1차 게이트(<=20ms) 판정 입력
    화질    PSNR / SSIM  (GT 재현도 — 가설 H1 의 반증 재료)
    노이즈  밝기이득 정규화 증폭률 (가설 H2 의 측정량)
    육안    arm 나란히 붙인 대조표 이미지

⚠️ **PSNR 우위를 채택 근거로 쓰지 말 것.** H1 이 정확히 "PSNR 순위가 실효 순위와
   다르다"는 가설이다. 최종 판정은 탐지 mAP 로 하며 본 스크립트는 그 준비 단계다.

사용법:
    uv run python scripts/compare_lowlight.py
    uv run python scripts/compare_lowlight.py --dataset loli --n 12
    uv run python scripts/compare_lowlight.py --arms none A1 A2 --size 854x480
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from lowlight import ARM_NAMES, build, noise_amplification

ROOT = Path(__file__).resolve().parent.parent
GATE_MS = 20.0  # data.md 2-3-3 1차 게이트

DATASETS = {
    "lol": (ROOT / "data/LOLdataset/eval15/low", ROOT / "data/LOLdataset/eval15/high"),
    "loli": (
        ROOT / "data/LoLI-Street/LoLI-Street Dataset/Val/low",
        ROOT / "data/LoLI-Street/LoLI-Street Dataset/Val/high",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASETS, default="lol")
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    p.add_argument("--n", type=int, default=8, help="평가에 쓸 쌍 개수")
    p.add_argument("--size", default="1280x720", help="속도 측정 해상도 WxH")
    p.add_argument("--runs", type=int, default=10, help="속도 측정 반복 횟수")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/lowlight")
    p.add_argument("--no-sheet", action="store_true", help="대조표 이미지 생성 생략")
    return p.parse_args()


def load_pairs(dataset: str, n: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """low/high 같은 파일명끼리 짝지어 읽는다."""
    low_dir, high_dir = DATASETS[dataset]
    if not low_dir.is_dir():
        raise SystemExit(f"데이터가 없다: {low_dir}\n  README '데이터 배치' 절 참고")

    pairs = []
    for low_path in sorted(low_dir.iterdir()):
        if low_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        high_path = high_dir / low_path.name
        if not high_path.exists():  # inspect_datasets.py 가 잡아야 할 상황
            continue
        low, high = cv2.imread(str(low_path)), cv2.imread(str(high_path))
        if low is None or high is None or low.shape != high.shape:
            continue
        pairs.append((low_path.name, low, high))
        if len(pairs) >= n:
            break

    if not pairs:
        raise SystemExit(f"짝이 맞는 쌍을 찾지 못했다: {low_dir}")
    return pairs


def measure_speed(arm, frame: np.ndarray, runs: int) -> float:
    for _ in range(2):  # warm-up
        arm(frame)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        arm(frame)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def measure_quality(arm, pairs) -> dict[str, float]:
    psnr, ssim, amp, bright = [], [], [], []
    for _, low, high in pairs:
        out = arm(low)
        psnr.append(peak_signal_noise_ratio(high, out, data_range=255))
        ssim.append(structural_similarity(high, out, channel_axis=2, data_range=255))
        amp.append(noise_amplification(low, out))
        bright.append(float(out.mean()) / (float(low.mean()) + 1e-6))
    return {
        "psnr": float(np.mean(psnr)),
        "ssim": float(np.mean(ssim)),
        "noise_amp": float(np.mean(amp)),
        "gain": float(np.mean(bright)),
    }


def build_sheet(arms, pairs, out_path: Path, rows: int = 4, cell_w: int = 320):
    """행=샘플, 열=[입력 | arm들 | GT] 대조표를 만든다."""
    used = pairs[:rows]
    labels = ["입력(low)"] + [a.name for a in arms] + ["GT(high)"]
    ascii_labels = ["INPUT(low)"] + [a.name for a in arms] + ["GT(high)"]  # OpenCV 는 한글 못 그림

    grid_rows = []
    for _, low, high in used:
        h = int(cell_w * low.shape[0] / low.shape[1])
        cells = [low] + [arm(low) for arm in arms] + [high]
        grid_rows.append(np.hstack([cv2.resize(c, (cell_w, h)) for c in cells]))

    body = np.vstack(grid_rows)
    header = np.full((34, body.shape[1], 3), 32, np.uint8)
    for i, text in enumerate(ascii_labels):
        cv2.putText(header, text, (i * cell_w + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([header, body]))
    return labels


def main():
    args = parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))

    pairs = load_pairs(args.dataset, args.n)
    arms = [build(name) for name in args.arms]
    speed_frame = cv2.resize(pairs[0][1], (w, h), interpolation=cv2.INTER_CUBIC)

    print(f"데이터셋 {args.dataset} · 쌍 {len(pairs)}개 · 속도 {w}x{h} median of {args.runs}")
    print(f"게이트: ② 단독 <= {GATE_MS:.0f}ms/frame (data.md 2-3-3)\n")

    print(f"{'arm':<8}{'ms':>8}{'FPS':>8}{'게이트':>8}{'PSNR':>8}{'SSIM':>8}{'노이즈증폭':>12}{'밝기이득':>10}")
    print("-" * 74)
    results = []
    for arm in arms:
        ms = measure_speed(arm, speed_frame, args.runs)
        q = measure_quality(arm, pairs)
        results.append((arm, ms, q))
        fps = "—" if ms < 0.05 else f"{1000 / ms:.1f}"  # 무처리 arm 은 측정 의미 없음
        print(f"{arm.name:<8}{ms:>8.2f}{fps:>8}"
              f"{('OK' if ms <= GATE_MS else '탈락'):>8}"
              f"{q['psnr']:>8.2f}{q['ssim']:>8.3f}{q['noise_amp']:>12.3f}{q['gain']:>10.2f}x")

    print("\narm 구성")
    for arm in arms:
        print(f"  {arm.name:<8}{arm.describe()}")

    print("\n읽는 법")
    print("  노이즈증폭  밝기이득으로 정규화한 σ 비. >1.0 이면 신호보다 노이즈를 더 키운 것 (H2)")
    print("  PSNR/SSIM   GT 재현도일 뿐 탐지 성능이 아니다 (H1). 채택 근거로 쓰지 말 것")

    if not args.no_sheet:
        sheet = args.out / f"compare_{args.dataset}.png"
        labels = build_sheet(arms, pairs, sheet)
        print(f"\n대조표 저장: {sheet}")
        print(f"  열 순서: {' | '.join(labels)}")


if __name__ == "__main__":
    main()
