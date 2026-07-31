"""② 저조도 개선 — **실제 야간** 표본으로 arm 을 목적 축에서 평가한다.

왜 필요한가 (docs/lowlight_classical.md 6-2)
    LOL·LoLI-Street Val 의 GT 는 전부 주간 장면이고, low 쪽 노이즈가 GT 보다 오히려
    적다. 즉 '주간을 어둡게 만든 것'에 가까워 **실제 야간 성능을 예측하지 못한다.**
    여기서는 보유 데이터셋에서 진짜 어두운 장면만 골라 다시 잰다.

paired GT 가 없으므로 PSNR/SSIM 은 재지 않는다. 대신 **GT 없이 잴 수 있고
프로젝트 목적에 직접 대응하는 축**만 본다. 지표 정의는 전부 `metrics.py` 에 있다.

    글레어      광원(휘도>=235) 코어 밝기의 전후 변화 + 포화 면적 — 목적은 '억제'(①)
    대비        같은 밝기의 감마 영상 대비 국소대비 비 — 목적은 '밝기 아닌 대비'
    노이즈      평활 암부 국한 절대 σ (휘도·색 분리) — 가설 H2

세 축 모두 **절대 기준**이다. 예전 정의(상위 1% 백분위 · 밝기이득 정규화)는 영상마다
재는 대상이 달라져 육안과 어긋났다 — 표의 ║ 오른쪽에 대조용으로만 남아 있다.
경위는 `metrics.py` 모듈 docstring 과 `docs/lowlight_classical.md` 6-4 참고.

사용법:
    uv run python scripts/night_eval.py                    # 프로파일 + 평가
    uv run python scripts/night_eval.py --max-luma 45      # 더 어두운 표본만
    uv run python scripts/night_eval.py --profile-only
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from lowlight import ARM_NAMES, build
# 지표 정의는 metrics.py 한 곳에 있다 (구·신 지표와 그 결함 설명 포함).
# 기존 import 경로(`from night_eval import dark_mask, local_rms, luma, measure`)를
# 노트북이 쓰고 있어 그대로 재수출한다.
from metrics import (  # noqa: F401
    LAMP_MIN, LAMP_MIN_PX, cell, dark_mask, estimate_noise, local_rms, luma, measure,
)

ROOT = Path(__file__).resolve().parent.parent

# 후보 풀 — 실제 촬영본만. LoLI-Street Val(low) 은 비교용 기준선으로만 넣는다.
POOLS = {
    "loli_test": ROOT / "data/LoLI-Street/LoLI-Street Dataset/Test",
    "exdark": ROOT / "data/ExDark/ExDark_data",
    "stair": ROOT / "data/Stair dataset/train/images",
    "loli_val_low": ROOT / "data/LoLI-Street/LoLI-Street Dataset/Val/low",  # 합성 대조군
}
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", nargs="+", default=["loli_test", "exdark", "stair"], choices=POOLS)
    p.add_argument("--max-luma", type=float, default=60.0, help="야간 판정 평균밝기 상한")
    p.add_argument("--profile-n", type=int, default=400, help="풀별 프로파일 표본 수")
    p.add_argument("--n", type=int, default=120, help="평가에 쓸 야간 영상 수")
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    p.add_argument("--speed-size", default="1280x720,854x480",
                   help="속도 측정 해상도 목록 (쉼표 구분). 1차 게이트는 720p 기준")
    p.add_argument("--profile-only", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/lowlight")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


# --------------------------------------------------------------------------
# 1. 풀 프로파일 — 어디에 진짜 야간이 있는가
# --------------------------------------------------------------------------


def profile(pools: list[str], n: int, rng: random.Random) -> dict[str, list[Path]]:
    print("=" * 88)
    print("1. 풀 프로파일 — 실제 야간 장면이 어디에 있는가")
    print("=" * 88)
    print(f"{'pool':<14}{'전체':>8}{'표본':>7}{'평균밝기':>10}{'중앙':>8}"
          f"{'어두운<60':>11}{'매우어두움<40':>14}{'포화(>=250)':>13}")
    print("-" * 88)

    dark_by_pool: dict[str, list[Path]] = {}
    for name in pools:
        paths = list_images(POOLS[name])
        if not paths:
            print(f"{name:<14}{'없음':>8}   (경로 확인: {POOLS[name]})")
            continue

        sample = rng.sample(paths, min(n, len(paths)))
        stats, dark = [], []
        for p in sample:
            # 밝기 판정에는 축소 로드로 충분하다 (밝기는 스케일 불변)
            im = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
            if im is None:
                continue
            L = luma(im)
            mean = float(L.mean())
            stats.append((mean, 100.0 * (L >= 250).mean()))
            if mean <= 60.0:
                dark.append(p)

        arr = np.array([s[0] for s in stats])
        sat = np.mean([s[1] for s in stats])
        print(f"{name:<14}{len(paths):>8}{len(arr):>7}{arr.mean():>10.1f}{np.median(arr):>8.1f}"
              f"{100 * (arr < 60).mean():>10.1f}%{100 * (arr < 40).mean():>13.1f}%{sat:>12.2f}%")
        dark_by_pool[name] = dark

    return dark_by_pool


def collect_night(pools, dark_by_pool, max_luma, n, rng) -> list[tuple[str, Path, np.ndarray]]:
    """풀별로 고르게 섞어 야간 표본을 만든다 (한 풀이 결과를 지배하지 않도록)."""
    per_pool = max(1, n // max(1, len([p for p in pools if dark_by_pool.get(p)])))
    picked: list[tuple[str, Path, np.ndarray]] = []
    for name in pools:
        cands = dark_by_pool.get(name, [])
        rng.shuffle(cands)
        taken = 0
        for p in cands:
            if taken >= per_pool:
                break
            im = cv2.imread(str(p))  # 노이즈 측정 때문에 원해상도로 읽는다
            if im is None:
                continue
            if float(luma(im).mean()) > max_luma:
                continue
            picked.append((name, p, im))
            taken += 1
    return picked


# --------------------------------------------------------------------------
# 2. 야간 표본에서 목적 축 평가
# --------------------------------------------------------------------------


def _print_table(arms, samples, title: str) -> None:
    """목적 축 3개를 **신 지표 우선**으로 표시한다.

    구 지표(강광원 백분위·정규화대비·노이즈증폭)는 문서 6-2·6-3 수치와 대조하기
    위해 오른쪽 블록에 남겨 두되, **판정은 왼쪽 신 지표로 한다.**
    세 지표의 결함과 교체 근거는 `metrics.py` 모듈 docstring 참고.
    """
    print(f"\n[{title}]  n={len(samples)}")
    print(cell("arm", 8, "<") + "│" + cell("[신] 노이즈 절대σ", 22, "^")
          + "│" + cell("[신] 대비", 10) + "│" + cell(f"광원코어(>={LAMP_MIN:.0f})", 17)
          + cell("표본", 8) + "│" + cell("포화율 전", 11) + cell("후", 8)
          + "║" + cell("[구] 강광원Δ", 14) + cell("[구] 정규화대비", 16)
          + cell("[구] 노이즈증폭", 16))
    print(cell("", 8) + "│" + cell("휘도", 11) + cell("색", 11)
          + "│" + cell("감마대비", 10) + "│" + cell("전→후", 17) + cell("", 8)
          + "│" + cell("", 11) + cell("", 8) + "║" + cell("(결함 — 대조용)", 46, "^"))
    print("-" * 134)
    for arm in arms:
        r = measure(arm, samples)
        core_n = r["glare_core_n"]
        core = ("—" if core_n == 0 else
                f"{r['glare_core_before']:.0f}→{r['glare_core_after']:.0f}")
        print(cell(arm.name, 8, "<") + "│"
              + cell(f"{r['noise_luma_after']:.2f}", 11)
              + cell(f"{r['noise_chroma_after']:.2f}", 11) + "│"
              + cell(f"{r['contrast_gain']:.2f}", 10) + "│"
              + cell(core, 17) + cell(f"{core_n}/{len(samples)}", 8) + "│"
              + cell(f"{r['sat_before']:.2f}%", 11) + cell(f"{r['sat_after']:.2f}%", 8)
              + "║" + cell(f"{r['glare_after'] - r['glare_before']:+.1f}", 14)
              + cell(f"{r['norm_contrast']:.2f}", 16)
              + cell(f"{r['noise_amp']:.3f}", 16))

    # 입력 자체의 절대 노이즈 — 신 지표는 정규화하지 않으므로 기준선이 필요하다
    base = measure(build("none"), samples)
    print(cell("(입력)", 8, "<") + "│"
          + cell(f"{base['noise_luma_before']:.2f}", 11)
          + cell(f"{base['noise_chroma_before']:.2f}", 11) + "│"
          + cell("1.00", 10) + "│" + cell("기준선", 17))


GATE_MS = 20.0  # data.md 2-3-3 1차 게이트 — ② 단독 720p 기준


def speed_table(arms, frame, sizes: list[str], runs: int = 7) -> None:
    """해상도별 ms/frame. 하이라이트 압축 계열은 여기서 걸릴 가능성이 크다."""
    import time

    print()
    print("=" * 88)
    print(f"0. 속도 — 1차 게이트 ② 단독 <= {GATE_MS:.0f}ms/frame @720p (median of {runs})")
    print("=" * 88)
    header = "".join(f"{s:>16}" for s in sizes)
    print(f"{'arm':<10}{header}{'720p 게이트':>14}")
    print("-" * 88)

    for arm in arms:
        cells, gate_ms = [], None
        for size in sizes:
            w, h = (int(v) for v in size.lower().split("x"))
            probe = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)
            for _ in range(2):
                arm(probe)
            samples = []
            for _ in range(runs):
                t0 = time.perf_counter()
                arm(probe)
                samples.append((time.perf_counter() - t0) * 1000)
            ms = float(np.median(samples))
            if size.lower().startswith("1280x720"):
                gate_ms = ms
            cells.append("—" if ms < 0.05 else f"{ms:.1f}ms")
        verdict = "—" if gate_ms is None or gate_ms < 0.05 else ("OK" if gate_ms <= GATE_MS else "탈락")
        print(f"{arm.name:<10}{''.join(f'{c:>16}' for c in cells)}{verdict:>14}")


def evaluate(arms, night, by_pool: bool = True) -> None:
    print()
    print("=" * 88)
    print(f"2. 야간 표본 {len(night)}장에서 목적 축 평가 (GT 없음 — PSNR/SSIM 미측정)")
    print("=" * 88)

    _print_table(arms, [s for _, _, s in night], "전체 야간 표본")

    if by_pool:
        pools: dict[str, list[np.ndarray]] = {}
        for pool, _, src in night:
            pools.setdefault(pool, []).append(src)
        # 합성(loli_val_low) 과 실제 촬영본의 결론이 갈리는지가 핵심이다
        for pool in sorted(pools, key=lambda k: k == "loli_val_low"):
            tag = "합성 감광 — 대조군" if pool == "loli_val_low" else "실제 촬영"
            _print_table(arms, pools[pool], f"{pool} ({tag})")

    print("\n  ── 신 지표 (절대 기준 — 이 열로 판정한다) ──")
    print("  노이즈 절대σ  평활 암부에 국한한 절대 σ. **(입력) 행보다 크면 노이즈를 드러낸 것.**")
    print("                정규화하지 않으므로 밝기이득에 상쇄되지 않는다. 마스크는 입력에서")
    print("                고정해 모든 arm 이 같은 화소를 본다. '색' 열이 육안의 '컬러 노이즈'")
    print("                축에 직접 대응한다 (H2 판정은 이 두 열로 한다)")
    print("  대비(감마대비) 같은 평균밝기의 감마 영상 대비 국소대비 비. >1.0 이어야 '균일하게")
    print("                밝히기'를 넘어선 진짜 대비 강조다")
    print(f"  광원코어      휘도>={LAMP_MIN:.0f} 화소를 입력에서 고정해 전→후를 본다. 낮아지면 실제 억제.")
    print("                '—' 는 그 풀에 광원 자체가 없다는 뜻(예: 합성 저조도 데이터).")
    print("                표본 수가 작으면(예: 2/40) 이 열의 신뢰도도 낮다")
    print("  포화율        광원의 '번짐'. 코어 밝기와 별개 피해다 — 코어를 유지하면서")
    print("                면적만 키우는 arm 이 있다 (A2·L1). 둘 다 봐야 눈부심 거동이 보인다")
    print("\n  ── 구 지표 (║ 오른쪽 — 문서 6-2·6-3 대조용. 서열화에 쓰지 말 것) ──")
    print("  세 열 모두 측정 대상을 영상 통계에 상대적으로 정의해 무너진다. 강광원Δ 는 어두운")
    print("  영상에서 광원을 표본에 넣지 못하고, 정규화대비는 저조도에서 1.0 아래로 눌리며,")
    print("  노이즈증폭은 육안과 반대 결론을 낸다. 상세는 metrics.py docstring.")


def build_sheet(arms, night, out_path: Path, rows: int = 4, cell_w: int = 300):
    used = night[:rows]
    labels = ["INPUT(night)"] + [a.name for a in arms]
    grid = []
    for _, _, src in used:
        h = int(cell_w * src.shape[0] / src.shape[1])
        cells = [src] + [arm(src) for arm in arms]
        grid.append(np.hstack([cv2.resize(c, (cell_w, h)) for c in cells]))

    body = np.vstack(grid)
    header = np.full((34, body.shape[1], 3), 32, np.uint8)
    for i, text in enumerate(labels):
        cv2.putText(header, text, (i * cell_w + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([header, body]))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    dark_by_pool = profile(args.pools, args.profile_n, rng)
    if args.profile_only:
        return

    night = collect_night(args.pools, dark_by_pool, args.max_luma, args.n, rng)
    if not night:
        raise SystemExit(f"평균밝기 <= {args.max_luma} 인 영상을 찾지 못했다. --max-luma 를 올려볼 것")

    counts: dict[str, int] = {}
    for pool, _, _ in night:
        counts[pool] = counts.get(pool, 0) + 1
    print(f"\n야간 표본 구성 (평균밝기 <= {args.max_luma}): "
          + ", ".join(f"{k} {v}장" for k, v in counts.items()))

    arms = [build(n) for n in args.arms]
    speed_table(arms, night[0][2], [s.strip() for s in args.speed_size.split(",")])
    evaluate(arms, night)

    sheet = args.out / "night_arms.png"
    build_sheet(arms, night, sheet)
    print(f"\n대조표 저장: {sheet}")


if __name__ == "__main__":
    main()
