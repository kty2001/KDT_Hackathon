"""② 저조도 개선 — **실제 야간** 표본으로 arm 을 목적 축에서 평가한다.

왜 필요한가 (docs/lowlight_classical.md 6-2)
    LOL·LoLI-Street Val 의 GT 는 전부 주간 장면이고, low 쪽 노이즈가 GT 보다 오히려
    적다. 즉 '주간을 어둡게 만든 것'에 가까워 **실제 야간 성능을 예측하지 못한다.**
    여기서는 보유 데이터셋에서 진짜 어두운 장면만 골라 다시 잰다.

paired GT 가 없으므로 PSNR/SSIM 은 재지 않는다. 대신 **GT 없이 잴 수 있고
프로젝트 목적에 직접 대응하는 축**만 본다.

    글레어      상위 1% 밝기·포화율의 전후 변화  — 목적은 '억제'(①)
    대비        밝기이득으로 정규화한 암부 국소대비 이득 — 목적은 '밝기 아닌 대비'
    노이즈      밝기이득으로 정규화한 σ 증폭률 — 가설 H2

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

from lowlight import ARM_NAMES, build, estimate_noise

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


def luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def local_rms(L: np.ndarray, mask: np.ndarray) -> float:
    """국소 대비 — 9x9 이웃 표준편차의 평균."""
    m = cv2.blur(L, (9, 9))
    var = cv2.blur(L * L, (9, 9)) - m * m
    return float(np.sqrt(np.maximum(var, 0))[mask].mean())


def dark_mask(L: np.ndarray) -> np.ndarray:
    """암부 마스크 — **순수 검정(크러시)은 제외한다.**

    실제 야간 영상은 암부의 상당 부분이 0 으로 뭉개져 있다. 그 영역은 복원할
    정보가 자체가 없어서 포함하면 통계를 지배해버린다(밝기비가 0/0 으로 발산).
    신호가 조금이라도 남아 있는 어두운 영역만 본다.
    """
    m = (L >= 3) & (L <= np.percentile(L, 60))
    return m if m.sum() >= 100 else (L <= np.percentile(L, 60))


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


LAMP_MIN = 235.0  # 광원 판정 절대 임계 (문서 lowlight_classical.md 7장)
LAMP_MIN_PX = 30  # 이보다 광원 화소가 적으면 그 표본은 코어 집계에서 제외


def measure(arm, samples, lamp_min: float = LAMP_MIN, lamp_min_px: int = LAMP_MIN_PX) -> dict[str, float]:
    """arm 을 목적 축(글레어·대비·노이즈)에서 평가한다.

    글레어는 **두 가지 방식**으로 잰다 (`glare_*` 와 `glare_core_*`).
    `glare_*`(상위 1% 백분위)는 영상이 매우 어두우면(평균밝기가 한 자릿수인 경우)
    임계 자체가 휘도 40~50대까지 내려가, 실제 광원이 아니라 '덜 어두운 배경'을
    표본에 넣는다 — 자체 촬영 야간 사진(평균밝기 7.3)에서 이 문제로 D1 의
    하이라이트 억제가 '증폭'으로 오보고됐다 (→ `lowlight_classical.md` 7장).

    `glare_core_*`는 **절대 휘도(`lamp_min`)로 광원을 고정**해 이 문제를 피한다.
    영상 밝기와 무관하게 항상 같은 대상(실제 광원)을 재므로, 어두운 영상에서도
    유효하다. `sat_*`(포화 면적, 이미 절대 임계 250 기반)와 함께 보면
    **코어 밝기**(광원 자체가 얼마나 밝아지는가)와 **번짐**(포화로 뭉개지는 면적)을
    분리해서 읽을 수 있다 — 둘은 서로 다른 종류의 피해다.

    `glare_before`/`glare_after` 등 기존 키는 과거 문서 수치(6-2·6-3)와의 대조를
    위해 그대로 남긴다. **새 판정은 `glare_core_*` 를 우선할 것.**
    """
    glare_b, glare_a, sat_b, sat_a = [], [], [], []
    core_b, core_a = [], []
    c_ratio, gains, n_amp = [], [], []
    for src in samples:
        L = luma(src)
        out_img = arm(src)
        O = luma(out_img)

        hi = L >= np.percentile(L, 99)  # 강광원 후보 = 상위 1% — 어두운 영상에서 취약 (위 docstring)
        glare_b.append(L[hi].mean())
        glare_a.append(O[hi].mean())
        sat_b.append(100.0 * (L >= 250).mean())
        sat_a.append(100.0 * (O >= 250).mean())

        # ★ 절대 임계 기반 코어 — 입력에서 고른 광원 화소 위치를 출력에도 그대로
        # 적용한다(재선택하지 않는다). 같은 화소가 어디로 가는지를 봐야 "억제/증폭"
        # 판정이 의미가 있다. 표본에 광원 자체가 없으면(개수 미달) 이 표본은 뺀다.
        lamp = L >= lamp_min
        if int(lamp.sum()) >= lamp_min_px:
            core_b.append(float(np.median(L[lamp])))
            core_a.append(float(np.median(O[lamp])))

        dark = dark_mask(L)
        c_ratio.append(local_rms(O, dark) / (local_rms(L, dark) + 1e-6))
        # 정규화는 전역 밝기이득으로 한다 — 암부 평균은 야간에서 0 에 붙어 발산한다
        gain = (float(O.mean()) + 1e-6) / (float(L.mean()) + 1e-6)
        gains.append(gain)
        n_amp.append((estimate_noise(out_img) + 1e-6) / (estimate_noise(src) + 1e-6) / gain)

    cr, br = float(np.mean(c_ratio)), float(np.mean(gains))
    core_n = len(core_b)
    return {
        "glare_before": float(np.mean(glare_b)),
        "glare_after": float(np.mean(glare_a)),
        "sat_before": float(np.mean(sat_b)),
        "sat_after": float(np.mean(sat_a)),
        # core_n == 0 이면 이 표본 풀 전체에 광원(>=lamp_min)이 없다는 뜻 —
        # 두 값 다 nan 이며, 호출부가 "—" 같은 표시로 걸러야 한다.
        "glare_core_before": float(np.mean(core_b)) if core_b else float("nan"),
        "glare_core_after": float(np.mean(core_a)) if core_a else float("nan"),
        "glare_core_n": core_n,
        "contrast": cr,
        "gain": br,
        "norm_contrast": cr / br,
        "noise_amp": float(np.mean(n_amp)),
    }


def _print_table(arms, samples, title: str) -> None:
    print(f"\n[{title}]  n={len(samples)}")
    print(f"{'arm':<8}{'강광원(백분위) 전':>17}{'후':>8}{'변화':>8}"
          f"{'포화율 전':>11}{'후':>8}{'대비배율':>10}{'밝기배율':>10}"
          f"{'정규화대비':>12}{'노이즈증폭':>12}"
          f"{'코어(절대 >=' + f'{LAMP_MIN:.0f})':>16}{'표본':>7}")
    print("-" * 118)
    for arm in arms:
        r = measure(arm, samples)
        core_n = r["glare_core_n"]
        core = ("—" if core_n == 0 else
                f"{r['glare_core_before']:.0f}→{r['glare_core_after']:.0f}")
        print(f"{arm.name:<8}{r['glare_before']:>17.1f}{r['glare_after']:>8.1f}"
              f"{r['glare_after'] - r['glare_before']:>+8.1f}"
              f"{r['sat_before']:>10.2f}%{r['sat_after']:>7.2f}%"
              f"{r['contrast']:>9.2f}x{r['gain']:>9.2f}x"
              f"{r['norm_contrast']:>12.2f}{r['noise_amp']:>12.3f}"
              f"{core:>16}{core_n:>4}/{len(samples)}")


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

    print("\n  강광원(백분위) 변화  목적은 '억제'다. 양수면 눈부심을 키운 것 (→ ① 과 상충)")
    print(f"                     ⚠️ 평균밝기가 낮은 풀(한 자릿수)에서는 상위 1% 임계 자체가")
    print(f"                     휘도 40~50대로 내려가 광원(>={LAMP_MIN:.0f})을 못 잡을 수 있다.")
    print(f"                     그럴 때는 '코어(절대)' 열을 볼 것 — 이 열이 우선한다")
    print(f"  코어(절대 >={LAMP_MIN:.0f})  광원 화소만 고정해 전→후를 본다. 낮아지면 실제 억제.")
    print("                     '—' 는 그 풀에 광원 자체가 없다는 뜻(예: 합성 저조도 데이터)")
    print("                     표본 수가 작으면(예: 2/40) 이 열의 신뢰도도 낮다")
    print("  정규화대비   >1.0 이어야 '밝히기'가 아닌 '대비 강조'")
    print("  노이즈증폭   >1.0 이면 신호보다 노이즈를 더 키운 것 (H2)")
    print("\n  ⚠️ 정규화대비·노이즈증폭 지표에도 알려진 결함이 있다 — 판정 전 반드시")
    print("     lowlight_classical.md 7장(지표 재설계 항목)을 확인할 것.")


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
