"""C6 — ② 단독 **해상도 × arm** 스윕: 속도 게이트 + 목적 축을 함께 잰다.

왜 속도만 재지 않는가 (docs/lowlight_classical.md 7장, 2026-08-01 발견)
    글레어 코어 지표가 **처리 해상도에 크게 의존한다.** 같은 사진에서 `D1` 의 광원
    코어가 원본 10 · 짧은변720 151 · 640 158 · 480 161 로 갈렸다. Drago 의 전역 로그
    매핑이 입력 전체 통계로 정해지기 때문이다.

    즉 "내부 처리 해상도를 낮춘다"는 결정은 **속도만의 문제가 아니라 표시 품질
    자체를 바꾼다.** 해상도를 속도로만 고르고 품질은 720p 수치로 판정하면, 실제
    배포 해상도에서 성립하지 않는 결론을 얻는다. 그래서 두 축을 같은 프레임에서
    함께 잰다.

무엇을 판정하는가
    1차 게이트  ② 단독 <= 20ms/frame @720p (data.md 2-3-3 의 **설계 배분값**이지
                실측 근거가 아니다 — 절대 판정은 C11 실기기에서 다시 한다)
    목적 축     글레어(광원 코어) · 대비(감마 대조군 대비) · 노이즈(평활 암부 절대 σ)
                정의는 전부 metrics.py

⚠️ 해상도 간 비교의 함정 — 노이즈는 **배율로 읽어야 한다**
    다운샘플은 그 자체로 노이즈를 평균해 없앤다. 그래서 480p 의 절대 σ 가 720p 보다
    낮은 것은 arm 의 성능이 아니라 리샘플의 부수효과다. 본 스크립트는 **각 해상도의
    입력 σ 를 그 해상도의 기준선**으로 삼아 배율(`×기준선`)로 출력한다.

사용법:
    uv run python scripts/resolution_sweep.py                    # 기본 3해상도 × 전 arm
    uv run python scripts/resolution_sweep.py --n 60             # 표본 늘리기
    uv run python scripts/resolution_sweep.py --arms none A1+bf D1A1+bf
    uv run python scripts/resolution_sweep.py --sizes 1280x720,640x360
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from lowlight import ARM_NAMES, build
from metrics import LAMP_MIN, cell, luma, measure
from night_eval import GATE_MS, POOLS, list_images

ROOT = Path(__file__).resolve().parent.parent

# NightOwls 는 야외 야간 **보행/주행 시점**이라 이 프로젝트 도메인에 가장 가깝다.
# ExDark 는 야간이되 실내가 많아 장면 통계가 다르다 (→ lowlight_classical.md 6-3-5).
POOLS = dict(POOLS, nightowls=ROOT / "data/NightOwls/images")

SELF_SHOT = ROOT / "data/sample_image.png"  # 자체 촬영 야간본 (git 비추적)

# 후보 내부 처리 해상도. 16:9 로 강제한다 — 입력이 카메라 스트림이라 종횡비가
# 고정이고, 속도와 품질을 **같은 프레임**에서 재야 비교가 성립한다.
DEFAULT_SIZES = "1280x720,854x480,640x360"

SUPPRESS_RATIO = 0.95  # 광원 코어가 5% 이상 내려가야 '억제'

INPUT_KEY = "(입력)"  # 기준선 행 — --arms 에 'none' 이 없어도 항상 잰다

# 게이트 판정 불가 밴드. 개발 PC 실측은 **프로세스 간에** 10~20% 씩 흔들려서
# (2026-08-01 실측: A1+bf @720p 18.3 / 18.8 / 19.5 / 22.4ms), 이 폭 안에 든 arm 은
# median 이 선을 넘었는지로 채택·기각을 가를 수 없다. 경계로 표시하고 보류한다.
GATE_BAND = 0.25  # ±25%


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", nargs="+", default=["nightowls", "exdark", "stair"],
                   choices=list(POOLS))
    p.add_argument("--sizes", default=DEFAULT_SIZES, help="해상도 목록 (쉼표 구분)")
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    p.add_argument("--n", type=int, default=36, help="목적 축 평가에 쓸 야간 표본 수")
    p.add_argument("--max-luma", type=float, default=60.0, help="야간 판정 평균밝기 상한")
    p.add_argument("--runs", type=int, default=7, help="속도 median 표본 수")
    p.add_argument("--scan", type=int, default=600, help="풀당 야간 후보 탐색 상한")
    p.add_argument("--no-self-shot", action="store_true",
                   help="자체 촬영본을 표본에서 뺀다 (글레어 표본이 크게 준다)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------
# 표본 — 실제 야간만
# --------------------------------------------------------------------------


def collect(pools: list[str], n: int, max_luma: float, scan: int,
            rng: random.Random) -> list[tuple[str, np.ndarray]]:
    """풀별로 고르게 섞어 야간 표본을 만든다 (한 풀이 결과를 지배하지 않도록)."""
    per_pool = max(1, n // max(1, len(pools)))
    picked: list[tuple[str, np.ndarray]] = []

    for name in pools:
        paths = list_images(POOLS[name])
        if not paths:
            print(f"  ⚠️ {name}: 이미지 없음 — 건너뜀 ({POOLS[name]})")
            continue
        rng.shuffle(paths)
        taken = 0
        for path in paths[:scan]:
            if taken >= per_pool:
                break
            # 밝기 판정은 축소 로드로 충분하다 (평균밝기는 스케일에 거의 불변)
            small = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
            if small is None or float(luma(small).mean()) > max_luma:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            picked.append((name, img))
            taken += 1
        print(f"  {name:<11} {taken}장 (평균밝기 <= {max_luma:.0f})")

    return picked


def resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """목표 해상도로 맞춘다.

    확대는 INTER_CUBIC, 축소는 INTER_AREA — 축소에 CUBIC 을 쓰면 에일리어싱이
    노이즈로 잡혀 노이즈 지표를 오염시킨다.
    """
    interp = cv2.INTER_AREA if (w * h) < (img.shape[1] * img.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


# --------------------------------------------------------------------------
# 1. 속도
# --------------------------------------------------------------------------


def gate_verdict(ms: float) -> str:
    """게이트 판정. 측정 흔들림(GATE_BAND) 안이면 판정을 보류한다."""
    if ms < 0.05:
        return "—"
    if ms <= GATE_MS * (1 - GATE_BAND):
        return "OK"
    if ms >= GATE_MS * (1 + GATE_BAND):
        return "탈락"
    return "경계"


def speed_sweep(arms, frame, sizes, runs: int) -> dict[tuple[str, str], float]:
    print()
    print("=" * 96)
    print(f"1. 속도 — 해상도 × arm (ms/frame, median of {runs})")
    print(f"   1차 게이트: ② 단독 <= {GATE_MS:.0f}ms @720p"
          f"  ·  판정 불가 밴드 {GATE_MS * (1 - GATE_BAND):.0f}~"
          f"{GATE_MS * (1 + GATE_BAND):.0f}ms → '경계'")
    print("=" * 96)

    print(cell("arm", 10, "<") + "".join(cell(s, 13) for s in sizes)
          + cell("720p 게이트", 14) + cell("확실히 통과하는 최대 해상도", 30))
    print("-" * 96)

    out: dict[tuple[str, str], float] = {}
    for arm in arms:
        row, passing = [], []
        for size in sizes:
            w, h = (int(v) for v in size.lower().split("x"))
            probe = resize_to(frame, w, h)
            for _ in range(2):  # 워밍업 — 첫 호출은 LUT·버퍼 할당이 섞인다
                arm(probe)
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter()
                arm(probe)
                ts.append((time.perf_counter() - t0) * 1000)
            ms = float(np.median(ts))
            out[(arm.name, size)] = ms
            mark = {"OK": "", "경계": "~", "탈락": ""}.get(gate_verdict(ms), "")
            row.append("—" if ms < 0.05 else f"{mark}{ms:.1f}ms")
            if gate_verdict(ms) in ("OK", "—"):
                passing.append((w * h, size))

        gate = out.get((arm.name, "1280x720"))
        verdict = "—" if gate is None else gate_verdict(gate)
        best = max(passing)[1] if passing else "없음"
        print(cell(arm.name, 10, "<") + "".join(cell(c, 13) for c in row)
              + cell(verdict, 14) + cell(best, 30))

    print(f"\n  '~' 는 판정 불가 밴드({GATE_MS * (1 - GATE_BAND):.0f}~"
          f"{GATE_MS * (1 + GATE_BAND):.0f}ms) 안에 든 값이다.")
    print("  ⚠️ 개발 PC CPU 실측이며 **프로세스 간 10~20% 흔들린다** (2026-08-01 실측:")
    print("     A1+bf @720p 를 네 번 재서 18.3 / 18.8 / 19.5 / 22.4ms). 이 밴드 안의")
    print("     arm 은 median 이 선을 넘었는지로 채택·기각을 가를 수 없다.")
    print("     목표 실행 환경(모바일 NPU/GPU)도 아니므로 **상대 순위와 명백한 탈락자**만")
    print("     읽고, 절대 판정은 C11 실기기에서 한다 (→ TODO.md C6 의 측정 기준 주의).")
    return out


# --------------------------------------------------------------------------
# 2. 목적 축 — 해상도별
# --------------------------------------------------------------------------


def axes_sweep(arms, samples, sizes) -> dict[tuple[str, str], dict]:
    print()
    print("=" * 96)
    print(f"2. 목적 축 — 해상도 × arm  (야간 표본 {len(samples)}장)")
    print("=" * 96)

    results: dict[tuple[str, str], dict] = {}
    for size in sizes:
        w, h = (int(v) for v in size.lower().split("x"))
        frames = [resize_to(img, w, h) for _, img in samples]

        # 기준선은 --arms 에 'none' 이 있든 없든 항상 잰다 — 노이즈 배율의 분모다.
        base = measure(build("none"), frames)
        results[(INPUT_KEY, size)] = base
        bl, bc = base["noise_luma_before"], base["noise_chroma_before"]
        lamp_n = base["glare_core_n"]

        print(f"\n[{size}]  광원 있는 표본 {lamp_n}/{len(frames)}장"
              f" · 입력 노이즈 기준선 휘도 {bl:.2f} / 색 {bc:.2f}")
        print(cell("arm", 10, "<") + cell("광원코어 전→후", 18) + cell("판정", 7)
              + cell("대비", 8) + cell("노이즈 휘도", 13) + cell("색", 11)
              + cell("3축", 7))
        print("-" * 74)

        for arm in arms:
            r = measure(arm, frames)
            results[(arm.name, size)] = r

            if r["glare_core_n"] == 0:
                core, verdict, ok_glare = "—", "—", False
            else:
                b, a = r["glare_core_before"], r["glare_core_after"]
                core = f"{b:.0f}→{a:.0f}"
                ok_glare = a < b * SUPPRESS_RATIO
                verdict = "억제" if ok_glare else ("증폭" if a > b * 1.01 else "유지")

            # 노이즈는 **그 해상도의 입력 대비 배율**로 읽는다 (모듈 docstring 참고)
            rl = r["noise_luma_after"] / max(bl, 1e-6)
            rc = r["noise_chroma_after"] / max(bc, 1e-6)
            ok_contrast = r["contrast_gain"] > 1.0
            ok_noise = rc <= 1.0
            marks = ("G" if ok_glare else "·") + ("C" if ok_contrast else "·") \
                    + ("N" if ok_noise else "·")

            print(cell(arm.name, 10, "<") + cell(core, 18) + cell(verdict, 7)
                  + cell(f"{r['contrast_gain']:.2f}", 8)
                  + cell(f"{r['noise_luma_after']:.2f} ({rl:.1f}x)", 13)
                  + cell(f"{rc:.1f}x", 11) + cell(marks, 7))

    print("\n  대비    > 1.0 이어야 '균일하게 밝히기'를 넘어선 진짜 대비 강조")
    print("  노이즈  괄호는 **그 해상도의 입력 σ 대비 배율.** 다운샘플 자체가 노이즈를")
    print("          평균해 없애므로 해상도 간 비교는 절대 σ 가 아니라 이 배율로 한다")
    print(f"  광원코어 휘도>={LAMP_MIN:.0f} 화소를 입력에서 고정해 전→후. 5% 이상 내려가야 '억제'")
    print("  3축     G=글레어 억제 · C=대비>1.0 · N=색노이즈 기준선 이하 (모두 만족이 목표)")
    return results


# --------------------------------------------------------------------------
# 3. 해상도 민감도 — 이 스크립트의 존재 이유
# --------------------------------------------------------------------------


def sensitivity(arms, sizes, speed, axes) -> None:
    print()
    print("=" * 96)
    print("3. 해상도 민감도 — 해상도를 바꾸면 결론이 바뀌는가")
    print("=" * 96)

    print(cell("arm", 10, "<") + cell("광원코어(해상도별)", 30)
          + cell("코어 변동폭", 13) + cell("대비(해상도별)", 24) + cell("판정 뒤집힘", 14))
    print("-" * 92)

    flipped_glare, flipped_gate = [], []
    for arm in arms:
        cores, verds, contrasts = [], [], []
        for size in sizes:
            r = axes[(arm.name, size)]
            if r["glare_core_n"] == 0:
                cores.append(None)
                verds.append(None)
            else:
                cores.append(r["glare_core_after"])
                verds.append(r["glare_core_after"]
                             < r["glare_core_before"] * SUPPRESS_RATIO)
            contrasts.append(r["contrast_gain"])

        valid = [c for c in cores if c is not None]
        spread = f"{max(valid) - min(valid):.0f}" if len(valid) >= 2 else "—"
        vset = {v for v in verds if v is not None}
        flip = "★ 억제↔유지" if len(vset) > 1 else ""
        if len(vset) > 1:
            flipped_glare.append(arm.name)

        gates = {gate_verdict(speed.get((arm.name, s), float("inf"))) for s in sizes}
        if len(gates - {"경계"}) > 1:
            flipped_gate.append(arm.name)

        print(cell(arm.name, 10, "<")
              + cell(" / ".join("—" if c is None else f"{c:.0f}" for c in cores), 30)
              + cell(spread, 13)
              + cell(" / ".join(f"{c:.2f}" for c in contrasts), 24)
              + cell(flip, 14))

    print(f"\n  열 순서: {' / '.join(sizes)}")

    print("\n" + "-" * 96)
    print("읽어낸 것")
    print("-" * 96)
    if len(sizes) < 2:
        print("· 해상도가 1개뿐이라 민감도는 판정할 수 없다 (--sizes 에 2개 이상 줄 것).")
    elif flipped_glare:
        print(f"★ **해상도만 바꿔도 글레어 억제 판정이 뒤집히는 arm**: {', '.join(flipped_glare)}")
        print("  → 이 arm 들은 '어느 해상도에서 쓸 것인가'를 정하지 않으면 채택 판정 자체가")
        print("    성립하지 않는다. 속도만으로 해상도를 고르면 안 되는 직접 증거다.")
    else:
        print("· 글레어 억제 판정은 해상도에 걸쳐 유지됐다 — 순위는 안정적이다.")
        print("  (절대값은 여전히 변하므로 문서 간에 옮겨 적을 때는 해상도를 함께 적을 것)")

    if flipped_gate:
        print(f"\n★ **해상도에 따라 게이트 통과 여부가 갈리는 arm**: {', '.join(flipped_gate)}")
        print("  → 내부 처리 해상도 하향으로 되살릴 수 있는 후보군이다.")

    # 종합 — 어느 (arm, 해상도) 조합이 속도와 3축을 동시에 만족하는가
    print("\n" + "-" * 96)
    print("종합 — 속도 게이트 + 3축을 **동시에** 만족하는 조합")
    print("-" * 96)
    winners, borderline = [], []
    for arm in arms:
        if arm.name == "none":
            continue
        for size in sizes:
            r = axes[(arm.name, size)]
            ms = speed.get((arm.name, size), float("inf"))
            v = gate_verdict(ms)
            if v == "탈락" or r["glare_core_n"] == 0:
                continue
            base = axes[(INPUT_KEY, size)]
            ok_g = r["glare_core_after"] < r["glare_core_before"] * SUPPRESS_RATIO
            ok_c = r["contrast_gain"] > 1.0
            ok_n = r["noise_chroma_after"] <= base["noise_chroma_before"]
            if ok_g and ok_c and ok_n:
                (winners if v == "OK" else borderline).append(
                    (arm.name, size, ms, r["contrast_gain"]))

    for name, size, ms, cg in winners:
        print(f"  ✅ {name:<9} @ {size:<10} {ms:>6.1f}ms · 대비 {cg:.2f}")
    for name, size, ms, cg in borderline:
        print(f"  ⚠️ {name:<9} @ {size:<10} {ms:>6.1f}ms · 대비 {cg:.2f}"
              f"   ← 3축은 만족하나 **속도가 판정 불가 밴드**")

    if not winners:
        print("  ❌ **속도까지 확실히 만족하는 조합은 없다.**")
        if borderline:
            print("     위 ⚠️ 는 3축을 채웠으므로 **측정 정밀화가 곧 판정**이다 — 고정 프레임·")
            print("     반복 실행으로 밴드를 좁히거나 C11 실기기 실측으로 넘길 것.")
        print("     남은 카드: (a) bf 를 시간축 억제(A3)로 대체, (b) 셰이더 이식,")
        print("     (c) 게이트 완화(탐지 주기 조정으로 예산 재배분) — 회의 안건 1번.")

    print("\n※ 노이즈 3축 중 N 은 **색** σ 기준이다. 휘도 σ 는 2절 표를 따로 볼 것 —")
    print("  D1A1+bf 의 유일한 미달 축이라 A3 시간축 억제의 대상이다 (→ 6-5-2의 3).")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    sizes = [s.strip() for s in args.sizes.split(",")]
    arms = [build(n) for n in args.arms]

    print("=" * 96)
    print("0. 표본 구성 — 실제 야간만")
    print("=" * 96)
    samples = collect(args.pools, args.n, args.max_luma, args.scan, rng)

    if not args.no_self_shot:
        if SELF_SHOT.is_file():
            img = cv2.imread(str(SELF_SHOT))
            if img is not None:
                samples.append(("self", img))
                print(f"  {'self':<11} 1장 (자체 촬영 — 강광원 표본)")
        else:
            print(f"  ⚠️ 자체 촬영본 없음: {SELF_SHOT} (git 비추적 — 별도 공유 필요)")

    if not samples:
        raise SystemExit("야간 표본을 찾지 못했다. --max-luma 를 올리거나 --pools 확인")

    speed = speed_sweep(arms, samples[0][1], sizes, args.runs)
    axes = axes_sweep(arms, samples, sizes)
    sensitivity(arms, sizes, speed, axes)


if __name__ == "__main__":
    main()
