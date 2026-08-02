"""W1 — ② A3 시간축 arm 실측 (NightOwls recording 38 연속 프레임).

무엇을 재는가 — **정지영상 벤치마크로는 원리적으로 잡히지 않는 축**이다
(→ docs/lowlight_classical.md 3장).

    1) 플리커 증폭률   프레임 간 전역 휘도의 흔들림이 **입력 대비 몇 배**가 되는가
    2) 시간축 σ        정지 화소에서 프레임 간 요동 — 시간축 노이즈의 직접 측정
    3) 공간 σ (색/휘도) 기존 목적 축과의 연속성 확인 (metrics.noise_absolute)
    4) 속도            `bf` 를 대체할 수 있는지의 전제

★ 왜 '증폭률'인가 — 절대값이 아니라 비(比)로 봐야 한다
    rec 38 은 **차량 주행** 영상이라 장면 자체가 계속 변한다. 출력의 흔들림에는
    (a) 실제 장면 변화와 (b) arm 이 만든 플리커가 섞여 있다. 같은 프레임 열에서
    입력의 흔들림으로 나누면 (a) 가 상쇄돼 **arm 이 더한 몫**만 남는다.
    1.0 이면 arm 이 플리커를 만들지 않은 것이고, >1 이면 만든 것이다.

★ 시간축 σ 는 **인접 프레임 차분**으로 잰다 (1차 시도의 실패를 고친 것)
    처음에는 '정지 화소'를 골라 시퀀스 전체의 표준편차를 쟀는데, 값이 20 이상으로
    나왔다 — 센서 노이즈가 아니라 **장면 변화**를 잰 것이다. 대시캠은 60프레임
    동안에도 장면이 통째로 바뀌므로, 한 화소의 '전 구간 σ' 에는 느린 변화가 전부
    들어온다.

    시간축 노이즈는 정의상 **프레임 간 무상관**이므로 인접 프레임 차분에 전부
    실린다. 느린 장면 변화는 차분에서 거의 상쇄된다. 그래서

        σ_t = std( Δ출력 |  입력 기준 정지 화소 ) / √2

    로 잰다 (√2 는 두 프레임의 독립 노이즈가 차분에서 더해지는 몫). 마스크는
    **쌍마다 입력에서** 고른다 — 출력에서 고르면 노이즈를 키운 arm 일수록 조용한
    곳만 남아 **키울수록 좋게 나오는** 역전이 생긴다 (metrics.flat_dark_mask 와
    같은 설계 원칙).

⚠️ 이 데이터의 한계 — 대시캠이라 카메라가 늘 움직인다. 정지 화소가 적어 시간축 σ 는
   보수적으로(과대) 나온다. 보행 시점 자체 촬영 영상이 오면 다시 잴 것 (`C2`).

사용:
    uv run python scripts/temporal_eval.py                       # 기본 300프레임
    uv run python scripts/temporal_eval.py --frames 600 --width 640
    uv run python scripts/temporal_eval.py --arms none,D1A1,D1A1+bf,D1A1+a3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lowlight import build
from metrics import cell, flat_dark_mask, luma, noise_absolute

ROOT = Path(__file__).resolve().parent.parent
NIGHTOWLS = ROOT / "data/NightOwls"
LABEL_JSON = NIGHTOWLS / "nightowls_validation.json"
IMAGES = NIGHTOWLS / "images"

DEFAULT_ARMS = "none,A1,A1+bf,A1+a3,D1A1,D1A1+bf,D1A1+ts,D1A1+td,D1A1+a3"
STATIC_MAX_DELTA = 6.0    # 이 미만으로 움직인 화소를 '정지'로 본다 (0~255)
MOVING_MIN_DELTA = 20.0   # 이 이상 움직인 화소 — 고스팅이 나타나는 곳


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", type=int, default=38,
                   help="연속 프레임을 가진 recording. 38 만 전량 해제돼 있다")
    p.add_argument("--frames", type=int, default=300, help="앞에서부터 몇 프레임")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--width", type=int, default=640,
                   help="내부 처리 해상도(가로). C6 에서 640×360 이 후보였다")
    p.add_argument("--arms", default=DEFAULT_ARMS)
    # A3 파라미터 스윕용 — 값을 주면 모든 시간축 arm 의 기본값을 덮어쓴다.
    # ⚠️ motion_thresh 는 **키울수록 억제가 강해진다** (같은 움직임에서 a 가 작아진다).
    p.add_argument("--blend-min", type=float, default=None)
    p.add_argument("--motion-thresh", type=float, default=None)
    p.add_argument("--cutoff-hz", type=float, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/lowlight/temporal_rec38.json")
    return p.parse_args()


def frame_paths(recording: int, start: int, n: int) -> list[Path]:
    """recording 의 프레임을 **촬영 순서대로**. 파일명(ObjectId)이 곧 시간순이다."""
    meta = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
    names = sorted(i["file_name"] for i in meta["images"]
                   if i.get("recordings_id") == float(recording))
    paths = [IMAGES / n_ for n_ in names]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(
            f"recording {recording} 이미지가 없다. "
            "먼저: uv run python scripts/extract_nightowls.py --run")
    return paths[start:start + n]


def load_sequence(paths: list[Path], width: int) -> list[np.ndarray]:
    seq = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if width and img.shape[1] != width:
            h = round(img.shape[0] * width / img.shape[1])
            img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
        seq.append(img)
    return seq


def pair_masks(seq: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """쌍마다 (정지 화소, 움직인 화소). **입력에서** 골라 모든 arm 이 같은 화소를 본다.

    움직인 화소는 고스팅 측정용이다 — 시간축 억제의 대가가 나타나는 곳이 정확히
    여기다.
    """
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        d = cv2.absdiff(luma(a), luma(b))
        stat = d < STATIC_MAX_DELTA
        if stat.sum() < 500:   # 장면이 통째로 바뀐 구간 — 가장 조용한 5% 로 후퇴
            stat = d <= np.percentile(d, 5.0)
        move = d >= MOVING_MIN_DELTA
        if move.sum() < 500:
            move = d >= np.percentile(d, 95.0)
        out.append((stat, move))
    return out


def override_params(arm, args) -> None:
    """CLI 로 준 A3 파라미터를 arm 에 반영한다 (스윕용)."""
    d = getattr(arm, "denoise", None)
    if d is not None:
        if args.blend_min is not None:
            d.blend_min = args.blend_min
        if args.motion_thresh is not None:
            d.motion_thresh = args.motion_thresh
        d.params.update(blend_min=d.blend_min, motion_thresh=d.motion_thresh)
        d._lut = None                      # LUT 는 파라미터에서 파생 — 무효화
    t = getattr(arm, "tone", None)
    if t is not None and args.cutoff_hz is not None:
        t.alpha = float(1.0 - np.exp(-2.0 * np.pi * args.cutoff_hz / t.params["fps"]))
        t.params.update(cutoff_hz=args.cutoff_hz, alpha=round(t.alpha, 4))


def run_arm(name: str, seq: list[np.ndarray], masks: list[np.ndarray],
            noise_mask: np.ndarray, args) -> dict:
    arm = build(name)
    override_params(arm, args)
    arm.reset()

    means: list[float] = []
    lumas: list[np.ndarray] = []
    sig_l = sig_c = 0.0
    t0 = time.perf_counter()
    for f in seq:
        out = arm(f)
        L = luma(out).astype(np.float32)
        means.append(float(L.mean()))
        lumas.append(L)
        a, b = noise_absolute(out, noise_mask)
        sig_l += a
        sig_c += b
    elapsed = (time.perf_counter() - t0) / len(seq) * 1000.0

    # 시간축 σ — 인접 프레임 차분의 σ / √2 (정지 화소에 국한)
    diffs = [(lumas[i + 1] - lumas[i])[stat] for i, (stat, _) in enumerate(masks)]
    temporal = float(np.concatenate(diffs).std() / np.sqrt(2.0))

    # ★ 고스팅 — 움직인 화소의 고주파 디테일. 시간축 억제의 **대가**가 여기 나온다.
    #   이 축이 없으면 "화면을 뭉갤수록 모든 지표가 좋아지는" 착시를 못 잡는다.
    detail = float(np.mean([
        np.abs(cv2.Laplacian(lumas[i + 1], cv2.CV_32F))[move].mean()
        for i, (_, move) in enumerate(masks)]))

    d = np.diff(np.array(means))
    return {
        "arm": name,
        "ms": elapsed,
        "flicker": float(d.std()),                       # 전역 휘도 1차차분의 σ
        "temporal_sigma": temporal,
        "motion_detail": detail,
        "sigma_l": sig_l / len(seq),
        "sigma_c": sig_c / len(seq),
        "mean_luma": float(np.mean(means)),
    }


def main() -> None:
    args = parse_args()
    paths = frame_paths(args.recording, args.start, args.frames)
    print("=" * 96)
    print(f"W1 — A3 시간축 실측 · recording {args.recording} · "
          f"{len(paths)}프레임 · 가로 {args.width}px")
    print("=" * 96)

    seq = load_sequence(paths, args.width)
    if len(seq) < 3:
        raise SystemExit("프레임이 너무 적다")
    print(f"  로드 {len(seq)}프레임 {seq[0].shape[1]}×{seq[0].shape[0]}")

    masks = pair_masks(seq)
    noise_m = flat_dark_mask(luma(seq[0]))   # 입력 기준 고정 — 모든 arm 이 같은 화소를 본다
    print(f"  정지 화소 {np.mean([s.mean() for s, _ in masks]):.1%}"
          f" · 움직인 화소 {np.mean([m.mean() for _, m in masks]):.1%}"
          f" · 노이즈 측정 화소 {noise_m.mean():.1%}")

    rows = [run_arm(a.strip(), seq, masks, noise_m, args)
            for a in args.arms.split(",") if a.strip()]
    base = next((r for r in rows if r["arm"] == "none"), rows[0])
    print(f"  증폭률·디테일 유지율의 기준 arm: {base['arm']}")

    print()
    head = ["arm", "ms/frame", "플리커σ", "증폭률", "시간축σ", "동적 디테일",
            "유지율", "공간 휘도σ", "공간 색σ"]
    w = [12, 10, 9, 8, 9, 12, 8, 11, 9]
    print("".join(cell(h, x) for h, x in zip(head, w)))
    print("-" * sum(w))
    for r in rows:
        amp = r["flicker"] / base["flicker"] if base["flicker"] > 1e-9 else float("nan")
        keep = r["motion_detail"] / base["motion_detail"] if base["motion_detail"] else float("nan")
        vals = [r["arm"], f"{r['ms']:.1f}", f"{r['flicker']:.3f}", f"{amp:.2f}×",
                f"{r['temporal_sigma']:.2f}", f"{r['motion_detail']:.2f}",
                f"{keep:.2f}×", f"{r['sigma_l']:.2f}", f"{r['sigma_c']:.2f}"]
        print("".join(cell(v, x, "<" if i == 0 else ">")
                      for i, (v, x) in enumerate(zip(vals, w))))
        r["flicker_amp"] = amp
        r["detail_keep"] = keep

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"recording": args.recording, "frames": len(seq), "width": args.width,
         "static_px_ratio": float(np.mean([s.mean() for s, _ in masks])),
         "blend_min": args.blend_min, "motion_thresh": args.motion_thresh,
         "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")

    print("\n읽는 법")
    print("  증폭률 1.0 = arm 이 플리커를 만들지 않음. >1 이면 만든 것이다.")
    print("  ★ **유지율과 함께 볼 것.** 시간축 억제를 세게 걸면 시간축σ·플리커가")
    print("     같이 내려가지만, 그건 화면을 뭉갠 결과일 수 있다. 유지율이 떨어지면")
    print("     고스팅이 생긴 것이고 보행 보조에서는 **위험 요소를 지우는** 방향이다.")
    print("  `+a3` 가 `+bf` 보다 빠르고, 색σ 를 더 낮추고, 유지율을 덜 깎아야")
    print("  비로소 **bf 대체 근거**가 된다.")
    print("\n⚠️ 대시캠이라 카메라가 늘 움직인다 — 시간축σ 는 보수적(과대)이다.")
    print("   보행 시점 자체 촬영 영상(C2)이 오면 다시 잰다.")


if __name__ == "__main__":
    main()
