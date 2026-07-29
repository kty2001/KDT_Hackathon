"""② 저조도 개선 — 고전 기법 속도 프로브.

목적: 어떤 고전 기법이 1차 게이트(② 단독 ≤20ms/frame)를 넘을 수 있는지 가늠한다.
결과 해석·주의사항은 docs/lowlight_classical.md 2장 참고.

⚠️ 이것은 속도 프로브지 화질 평가가 아니다. 또한 CPU·NumPy 구현이라
   GPU 셰이더 이식 시 순위가 바뀔 수 있다. 기각 근거로 쓰지 말 것.

사용법:
    uv run python scripts/probe_classical.py            # 720p
    uv run python scripts/probe_classical.py --size 854x480
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMG = Path(__file__).resolve().parent.parent / "data/LOLdataset/eval15/low/1.png"
GATE_MS = 20.0  # data.md 2-3-3 의 1차 게이트


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=DEFAULT_IMG)
    p.add_argument("--size", default="1280x720", help="WxH (기본 720p)")
    p.add_argument("--runs", type=int, default=10)
    return p.parse_args()


def bench(fn, n=10, warm=2):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def guided_filter(I, p, r=16, eps=1e-3):
    """He 2013. box filter만 사용 — opencv contrib 불요."""
    k = (r, r)
    mI, mp = cv2.boxFilter(I, -1, k), cv2.boxFilter(p, -1, k)
    cov = cv2.boxFilter(I * p, -1, k) - mI * mp
    var = cv2.boxFilter(I * I, -1, k) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


def build_cases(src, srcf):
    """(이름, 계열, 함수) 목록. 계열 번호는 lowlight_classical.md 1장과 대응."""
    gamma_lut = np.clip(((np.arange(256) / 255.0) ** 0.45) * 255, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merge = cv2.createMergeMertens()

    def gamma():
        return cv2.LUT(src, gamma_lut)

    def clahe_lab():
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def agcwd():
        """Huang 2013 — 프레임당 히스토그램 1회 + 1D LUT."""
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        pdf = cv2.calcHist([v], [0], None, [256], [0, 256]).ravel()
        pdf /= pdf.sum()
        pmin, pmax = pdf.min(), pdf.max()
        pw = pmax * ((pdf - pmin) / (pmax - pmin + 1e-8)) ** 0.5
        cdfw = np.cumsum(pw) / (pw.sum() + 1e-8)
        g = np.clip(255.0 * (np.arange(256) / 255.0) ** (1 - cdfw), 0, 255).astype(np.uint8)
        hsv[:, :, 2] = cv2.LUT(v, g)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def retinex(scales):
        log_i = np.log1p(srcf * 255.0)
        r = np.zeros_like(srcf)
        for s in scales:
            r += (log_i - np.log1p(cv2.GaussianBlur(srcf * 255.0, (0, 0), s))) / len(scales)
        return cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def lime_approx():
        """LIME 근사 — max-RGB 조도맵을 guided filter로 정련 (원논문의 최적화 대체)."""
        t = srcf.max(axis=2)
        t = np.clip(guided_filter(t, t, r=16, eps=1e-3), 0.15, 1.0) ** 0.8
        return np.clip(srcf / t[:, :, None], 0, 1)

    def dong_dehaze():
        """Dong 2011 — 반전 후 dark channel dehaze, 재반전."""
        inv = 1.0 - srcf
        dark = cv2.erode(inv.min(axis=2), np.ones((15, 15), np.uint8))
        t = np.maximum(guided_filter(inv.mean(axis=2), 1.0 - 0.9 * dark, r=20, eps=1e-3), 0.1)
        return 1.0 - np.clip((inv - 1.0) / t[:, :, None] + 1.0, 0, 1)

    def mertens_fusion():
        """단일 프레임에서 가상 노출 스택 생성 후 융합.

        ⚠️ MergeMertens 는 **8bit 입력이어야 한다.** float32 [0,1] 을 넣으면
        출력이 0 근처로 뭉개져 새까맣게 나온다(2026-07-29 확인). 속도만 재던
        시절에는 드러나지 않던 버그다.
        """
        stack = [(np.clip(srcf * k, 0, 1) * 255).astype(np.uint8) for k in (1.0, 2.5, 5.0)]
        return np.clip(merge.process(stack), 0, 1)

    tm = {n: getattr(cv2, f"createTonemap{n}")(1.5) for n in ("Reinhard", "Drago", "Mantiuk")}

    return [
        ("감마 LUT (하한선)", "①", gamma),
        ("AGCWD", "①", agcwd),
        ("CLAHE (LAB L채널)", "①", clahe_lab),
        ("SSR", "②", lambda: retinex((80,))),
        ("MSR (3-scale)", "②", lambda: retinex((15, 80, 250))),
        ("LIME 근사 (guided)", "②", lime_approx),
        ("Mertens 가상 3노출 융합", "③", mertens_fusion),
        ("Dong 반전-dehaze", "④", dong_dehaze),
        ("Tonemap Reinhard", "⑤", lambda: tm["Reinhard"].process(srcf)),
        ("Tonemap Drago", "⑤", lambda: tm["Drago"].process(srcf)),
        ("Tonemap Mantiuk", "⑤", lambda: tm["Mantiuk"].process(srcf)),
        ("[노이즈] bilateral d=7", "—", lambda: cv2.bilateralFilter(src, 7, 50, 50)),
        ("[노이즈] guided filter r=8", "—", lambda: guided_filter(srcf, srcf, r=8, eps=4e-3)),
        ("[노이즈] detailEnhance", "—", lambda: cv2.detailEnhance(src, sigma_s=10, sigma_r=0.15)),
        ("[노이즈] edgePreservingFilter", "—", lambda: cv2.edgePreservingFilter(src, flags=1, sigma_s=30, sigma_r=0.3)),
        ("[노이즈] fastNlMeans", "—", lambda: cv2.fastNlMeansDenoisingColored(src, None, 5, 5, 7, 15)),
    ]


def main():
    args = parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))

    src = cv2.imread(str(args.image))
    if src is None:
        raise SystemExit(f"영상을 읽을 수 없다: {args.image}")
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)
    srcf = src.astype(np.float32) / 255.0

    print(f"{w}x{h} CPU 단일 프레임, median of {args.runs}  |  cv2 {cv2.__version__}")
    print(f"{'기법':<30}{'계열':>5}{'ms':>10}{'FPS':>9}   게이트({GATE_MS:.0f}ms)")
    print("-" * 72)
    for name, family, fn in build_cases(src, srcf):
        try:
            ms = bench(fn, n=args.runs)
            verdict = "OK" if ms <= GATE_MS else ("경계" if ms <= 2 * GATE_MS else "탈락")
            print(f"{name:<30}{family:>5}{ms:>10.2f}{1000 / ms:>9.1f}   {verdict}")
        except Exception as e:  # 기법 하나가 죽어도 나머지는 측정한다
            print(f"{name:<30}{family:>5}{'ERR':>10}   {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
