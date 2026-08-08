"""실촬영 테스트 소재를 **더 저조도로** 합성한다 (이미지 + 동영상).

`scripts/lowlight.py` 가 어두운 입력을 밝히는 쪽이라면, 이쪽은 그 반대 —
이미 야간에 찍은 `data/test_real_data/` 를 한 단계 더 어두운 조건으로 낮춰
② 저조도 개선 arm 의 하한을 눈으로 보기 위한 용도다.

퇴화 모델 (센서가 노출을 줄였을 때 실제로 일어나는 순서 그대로)

    sRGB → 선형화 → 노출 이득 k(<1) → 광자 산탄노이즈 + 리드노이즈 → sRGB

- **선형 도메인에서 곱한다.** 픽셀값에 그냥 0.4 를 곱하면 감마가 걸린 값을
  줄이는 것이라 카메라의 노출 감소와 다른 톤이 나온다.
- **노이즈를 반드시 넣는다.** 단순 스케일링은 곱셈 하나로 되돌아가므로
  ②의 난이도가 전혀 오르지 않는다. 저조도가 어려운 진짜 이유는 어두움이
  아니라 어두운 곳의 **SNR** 이다. 산탄노이즈는 √신호에 비례하므로
  암부일수록 상대 노이즈가 커진다 — 이 성질이 모델의 핵심이다.
- 동영상은 프레임마다 독립 노이즈를 넣는다(센서 노이즈는 시간축 무상관).
  A3 시간축 스테이지를 시험할 소재가 되려면 이래야 한다.

⚠️ 한계 — `night_eval.py`·`build_detect_dataset.py` 에 적어둔 것과 같은 주의:
**주간을 어둡게 만든 합성본은 실제 야간 통계가 아니다.** 여기 입력은 이미
야간 실촬영이라 그 함정이 덜하지만, 그래도 강광원의 블루밍·플레어나
카메라 ISO 상승 시의 크로마 노이즈·NR 뭉갬은 재현되지 않는다. 정량 평가의
근거로 쓰지 말고 **정성 확인용**으로만 쓸 것.

사용
    uv run python scripts/darken.py                      # 기본 gain 0.35
    uv run python scripts/darken.py --gain 0.2 --dst data/test_real_data/lowlight_x02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

_SRGB_TO_LINEAR = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2)


class Darkener:
    """노출을 gain 배로 줄이고 그만큼의 센서 노이즈를 얹는다.

    gain        선형 도메인 노출 배율. 0.35 면 약 -1.5 EV.
    full_well   화이트(1.0)에 해당하는 광자 수. **작을수록 노이즈가 크다.**
    read_noise  전자 단위 리드 노이즈 — 신호와 무관한 바닥 노이즈.
    """

    def __init__(self, gain: float = 0.35, full_well: float = 3000.0,
                 read_noise: float = 4.0, seed: int = 0):
        self.gain = float(gain)
        self.full_well = float(full_well)
        self.read_noise = float(read_noise)
        self.rng = np.random.default_rng(seed)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        # 선형화는 uint8 입력이라 256엔트리 LUT 로 정확히 대체된다
        lin = _SRGB_TO_LINEAR[bgr] * self.gain

        # 산탄노이즈: 광자 수는 포아송. full_well 이 커서 정규근사로 충분하고
        # (μ≈100 부터 오차 무시 가능) 4000만 화소에 poisson 을 돌리는 것보다 훨씬 싸다.
        electrons = lin * self.full_well
        sigma = np.sqrt(electrons + self.read_noise ** 2, dtype=np.float32)
        electrons += self.rng.standard_normal(lin.shape, dtype=np.float32) * sigma

        lin = np.clip(electrons / self.full_well, 0.0, 1.0)
        return np.clip(np.power(lin, 1 / 2.2) * 255.0, 0, 255).astype(np.uint8)


def process_image(src: Path, dst: Path, dk: Darkener) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했다: {src}")
    # 원본이 JPEG 여도 품질 95 로 다시 쓴다 — 압축 아티팩트를 새로 얹지 않기 위해
    cv2.imwrite(str(dst), dk(img), [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_video(src: Path, dst: Path, dk: Darkener) -> int:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"동영상을 열지 못했다: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"인코더를 열지 못했다 (mp4v): {dst}")

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(dk(frame))   # 프레임마다 독립 노이즈 — 시간축 무상관
            n += 1
    finally:
        cap.release()
        writer.release()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/test_real_data"))
    ap.add_argument("--dst", type=Path, default=None,
                    help="기본값: <src>/lowlight")
    ap.add_argument("--gain", type=float, default=0.35, help="선형 노출 배율 (<1)")
    ap.add_argument("--full-well", type=float, default=3000.0,
                    help="화이트에 해당하는 광자 수. 작을수록 노이즈가 크다")
    ap.add_argument("--read-noise", type=float, default=4.0, help="리드 노이즈 (전자)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dst_dir = args.dst or args.src / "lowlight"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dk = Darkener(args.gain, args.full_well, args.read_noise, args.seed)
    print(f"gain={args.gain}  full_well={args.full_well}  read_noise={args.read_noise}")
    print(f"{args.src} → {dst_dir}\n")

    # 하위 디렉토리는 훑지 않는다 — 출력 디렉토리를 다시 먹는 사고를 막는다
    for src in sorted(p for p in args.src.iterdir() if p.is_file()):
        ext = src.suffix.lower()
        dst = dst_dir / src.name
        if ext in IMAGE_EXT:
            process_image(src, dst, dk)
            print(f"  [img] {src.name}")
        elif ext in VIDEO_EXT:
            n = process_video(src, dst, dk)
            print(f"  [vid] {src.name}  ({n} frames, 오디오 없음)")


if __name__ == "__main__":
    main()
