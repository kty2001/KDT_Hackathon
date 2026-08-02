"""② 저조도 개선 — 고전 arm 구현 (A1 / A2) + arm 레지스트리.

docs/data.md 2-3-3 의 4-arm 비교 실험에 그대로 꽂을 수 있는 형태로 만든다.
모든 arm 은 `bgr uint8 -> bgr uint8` 단일 인터페이스라 ③ YOLO 앞단에 바로 붙는다.

    from lowlight import build, ARM_NAMES
    arm = build("A1")
    out = arm(frame)

arm 구성 (docs/lowlight_classical.md 6장)
    none    무처리 — ②가 ③에 도움이 되는지 자체를 검증하는 필수 대조군
    A1      CLAHE(타일 적응) + 감마
    A2      AGCWD(전역 적응 감마, Huang 2013)
    +bf     위에 bilateral 노이즈 억제를 얹은 변형
    +ts/+td/+a3   **A3 시간축** — 톤커브 평활 / 모션적응 노이즈 억제 / 둘 다.
            연속 프레임 전용이며 `arm.is_temporal` 로 구분한다 (→ 아래 A3 절)

설계 메모
- **두 톤커브 모두 LAB L 채널에서 동작한다.** AGCWD 원논문은 HSV V 를 쓰지만,
  여기서 비교하려는 것은 색공간이 아니라 *톤커브 전략(타일 적응 vs 전역 적응)*이다.
  색공간을 맞춰 교란변수를 하나 없앤다.
- **노이즈 억제는 직교 스테이지다.** data.md 의 A안은 "CLAHE + 감마 + 노이즈 억제"로
  묶여 있었으나, 그러면 A1·A2 비교에 노이즈 처리가 섞여 들어간다. 분리해야
  가설 H2(고전의 노이즈 증폭이 탐지를 해치는가)를 arm 차이로 직접 읽을 수 있다.
- 공간축 스테이지는 상태를 갖지 않는다. 시간축 처리(A3, → lowlight_classical.md 3장)만
  상태를 가지며 `TemporalArm` 으로 분리했다 — 정지영상 하네스에 섞이지 않게 하기 위해서다.
"""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np

# 지표는 `metrics.py` 로 옮겼다 — 글레어·대비·노이즈가 같은 결함을 공유해 한 곳에서
# 함께 고쳐야 했다. 여기서는 기존 import 경로(`from lowlight import estimate_noise`)를
# 살리기 위해 재수출만 한다. **새 판정에는 `metrics.noise_absolute` 를 쓸 것** —
# 아래 둘은 밝기이득 정규화 결함이 있는 구 지표다.
from metrics import estimate_noise, noise_amplification  # noqa: F401

Frame = np.ndarray  # BGR uint8 (H, W, 3)
Stage = Callable[[Frame], Frame]


# --------------------------------------------------------------------------
# 톤커브 스테이지
# --------------------------------------------------------------------------


class CLAHE:
    """A1 — 타일 단위 적응 히스토그램 평활 + 감마 (Zuiderveld 1994).

    국소적으로 대비를 올려 암부 디테일을 살린다. clip_limit 로 노이즈 증폭을 제한한다.
    """

    def __init__(self, clip_limit: float = 2.0, tile: int = 8, gamma: float = 0.75):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        self._lut = _gamma_lut(gamma)
        self.params = {"clip_limit": clip_limit, "tile": tile, "gamma": gamma}

    def __call__(self, bgr: Frame) -> Frame:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.LUT(self._clahe.apply(lab[:, :, 0]), self._lut)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class AGCWD:
    """A2 — Adaptive Gamma Correction with Weighting Distribution (Huang 2013).

    프레임 히스토그램에서 픽셀값마다 다른 감마를 유도해 1D LUT 하나로 적용한다.
    CLAHE 와 달리 **전역** 이라 공간 구조를 보지 않지만, 그만큼 국소 노이즈를
    끌어올리지 않는다. 비용은 히스토그램 1회 + LUT 1회.

    alpha: 가중 분포의 지수. 작을수록 강하게 밝힌다 (원논문 기본 0.5).
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.params = {"alpha": alpha}

    def __call__(self, bgr: Frame) -> Frame:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.LUT(lab[:, :, 0], self._curve(lab[:, :, 0]))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _curve(self, luma: np.ndarray) -> np.ndarray:
        """가중 히스토그램 → 픽셀값별 감마 → 8bit LUT."""
        pdf = cv2.calcHist([luma], [0], None, [256], [0, 256]).ravel()
        total = pdf.sum()
        if total <= 0:  # 전부 같은 값인 프레임
            return np.arange(256, dtype=np.uint8)
        pdf /= total

        pmin, pmax = pdf.min(), pdf.max()
        if pmax - pmin < 1e-8:  # 완전 평탄 — 보정할 게 없다
            return np.arange(256, dtype=np.uint8)

        # 빈도가 높은 구간이 톤 범위를 독점하지 않도록 pdf 를 눌러준다
        weighted = pmax * ((pdf - pmin) / (pmax - pmin)) ** self.alpha
        cdf_w = np.cumsum(weighted) / weighted.sum()

        levels = np.arange(256) / 255.0
        return np.clip(255.0 * levels ** (1.0 - cdf_w), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# 하이라이트 압축 계열 — A1/A2 가 원리적으로 못 하는 일을 한다
#
# A1(CLAHE)·A2(AGCWD) 의 톤커브는 단조 증가라 **어떤 입력값도 낮출 수 없다.**
# 그래서 강광원을 억제하기는커녕 밝기를 더 올린다(→ lowlight_classical.md 6-2-3).
# 아래 세 스테이지는 구조적으로 하이라이트를 누르거나 최소한 올리지 않는다.
# --------------------------------------------------------------------------


_SRGB_TO_LINEAR = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2)


class Tonemap:
    """HDR 톤매핑 오퍼레이터(Drago / Reinhard) 래퍼.

    톤매핑은 원래 HDR 라디언스를 LDR 로 **압축**하는 연산이라, 정의상 밝은 쪽을
    누른다. 저조도 영상에 전용하면 암부를 올리면서 강광원을 함께 눌러준다.

    입력을 **선형화(sRGB 감마 제거)한 뒤** 통과시키고, 오퍼레이터의 gamma 로
    표시 감마를 되씌운다. 이 과정을 빼면 오퍼레이터가 가정하는 신호와 어긋나
    결과가 뭉개진다.
    """

    _FACTORIES = {
        "drago": lambda g, kw: cv2.createTonemapDrago(
            g, kw.get("saturation", 1.0), kw.get("bias", 0.85)),
        "reinhard": lambda g, kw: cv2.createTonemapReinhard(
            g, kw.get("intensity", 0.0), kw.get("light_adapt", 0.8),
            kw.get("color_adapt", 0.0)),
    }

    def __init__(self, kind: str = "drago", gamma: float = 2.2, **kw):
        if kind not in self._FACTORIES:
            raise ValueError(f"지원하지 않는 오퍼레이터: {kind}")
        self.kind = kind
        self._tm = self._FACTORIES[kind](gamma, kw)
        self.params = {"kind": kind, "gamma": gamma, **kw}

    def __call__(self, bgr: Frame) -> Frame:
        # 입력이 uint8 이므로 선형화는 256엔트리 LUT 로 **정확히** 대체된다.
        # np.power 를 2.7M 픽셀에 직접 돌리면 그것만으로 60ms 가 나간다.
        linear = _SRGB_TO_LINEAR[bgr]
        out = self._tm.process(linear)
        # 전부 0 인 프레임 등에서 오퍼레이터가 NaN 을 낼 수 있다
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class LIME:
    """Retinex 계열 — 조도맵을 추정해 나눈다 (Guo 2017 근사).

    조도맵 T 를 채널별 최대값으로 초기화한 뒤 구조 보존 평활을 하고 `I / T` 를
    취한다. 원논문은 정련에 가중 최소제곱 최적화를 쓰지만, 실시간이 목표라
    **guided filter 로 대체**했다(구조 보존 평활이라는 목적이 같고 훨씬 싸다).

    A1/A2 와 결정적으로 다른 점: 강광원은 T≈1 이라 `I/T ≈ I` 로 **그대로 남고**,
    암부만 나눠서 올라간다. 즉 하이라이트를 **증폭하지 않는다.**
    """

    def __init__(self, radius: int = 16, eps: float = 1e-3,
                 floor: float = 0.15, gamma: float = 0.8, scale: int = 4):
        self.radius, self.eps, self.floor, self.gamma = radius, eps, floor, gamma
        self.scale = scale
        self.params = {"radius": radius, "eps": eps, "floor": floor,
                       "gamma": gamma, "scale": scale}

    def __call__(self, bgr: Frame) -> Frame:
        src = bgr.astype(np.float32) / 255.0
        t = src.max(axis=2)
        t = guided_filter(t, t, self.radius, self.eps, self.scale)
        # floor 가 없으면 완전 암부에서 0 으로 나눠 노이즈가 폭발한다
        t = np.clip(t, self.floor, 1.0) ** self.gamma
        return np.clip(src / t[:, :, None] * 255.0, 0, 255).astype(np.uint8)


def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 16,
                  eps: float = 1e-3, scale: int = 1) -> np.ndarray:
    """He 2013 guided filter. box filter 만 써서 구현 — opencv contrib 불요.

    scale > 1 이면 **fast guided filter**(He 2015) — 선형계수 a,b 를 축소본에서
    구해 업샘플한 뒤 원해상도 가이드에 적용한다. a,b 는 원래 공간적으로 매끄러워
    축소해도 손실이 거의 없고, 비용은 대략 scale^2 만큼 준다.
    """
    if scale > 1:
        small = (max(1, I.shape[1] // scale), max(1, I.shape[0] // scale))
        Is = cv2.resize(I, small, interpolation=cv2.INTER_AREA)
        ps = cv2.resize(p, small, interpolation=cv2.INTER_AREA)
        rs = max(1, r // scale)
    else:
        Is, ps, rs = I, p, r

    k = (rs, rs)
    mean_I, mean_p = cv2.boxFilter(Is, -1, k), cv2.boxFilter(ps, -1, k)
    cov = cv2.boxFilter(Is * ps, -1, k) - mean_I * mean_p
    var = cv2.boxFilter(Is * Is, -1, k) - mean_I * mean_I
    a = cov / (var + eps)
    b = mean_p - a * mean_I

    if scale > 1:  # 계수만 원해상도로 되돌린다
        full = (I.shape[1], I.shape[0])
        a = cv2.resize(a, full, interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, full, interpolation=cv2.INTER_LINEAR)
        return a * I + b

    return cv2.boxFilter(a, -1, k) * I + cv2.boxFilter(b, -1, k)


# --------------------------------------------------------------------------
# 노이즈 억제 스테이지 (직교 — 어느 톤커브에도 붙일 수 있다)
# --------------------------------------------------------------------------


class Bilateral:
    """엣지 보존 평활. 톤커브가 증폭한 센서 노이즈를 누른다.

    d 를 키우면 급격히 비싸진다. 720p 실시간을 노리면 d<=7 을 유지할 것.
    """

    def __init__(self, d: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0):
        self.args = (d, sigma_color, sigma_space)
        self.params = {"d": d, "sigma_color": sigma_color, "sigma_space": sigma_space}

    def __call__(self, bgr: Frame) -> Frame:
        return cv2.bilateralFilter(bgr, *self.args)


# --------------------------------------------------------------------------
# 시간축 스테이지 (A3) — 상태를 갖는다. **연속 프레임에만 유효하다**
#
# 정지영상 벤치마크로는 원리적으로 검출되지 않는 두 결함을 다룬다
# (→ lowlight_classical.md 3장):
#
#   3-1 플리커      A1·A2·D1 은 프레임마다 톤커브를 다시 계산한다. 가로등이 들고
#                   나며 밝기 분포가 급변하면 화면 전체가 출렁인다. 대상이
#                   광과민 저시력자라 이건 화질이 아니라 **안전 문제**다.
#   3-2 시간축 노이즈  센서 노이즈는 프레임 간 무상관, 신호는 상관 → 누적하면
#                   SNR 이 √N 개선된다. 공간축 필터(`bf`)와 달리 **해상도를
#                   희생하지 않는다.** 미달 축이던 색 σ 가 정확히 이것이 다루는 것.
#
# ⚠️ 두 스테이지 모두 이전 프레임에 의존한다. 서로 무관한 정지영상 목록에
#    돌리면 **의미 없는 값**이 나온다. `Arm.is_temporal` 로 구분하고,
#    시퀀스가 바뀔 때마다 `arm.reset()` 을 부를 것.
# --------------------------------------------------------------------------


def _luma(bgr: Frame) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


class TemporalToneSmooth:
    """3-1 대응 — arm 의 **실효 톤커브를 시간축 IIR 로 평활**한다.

    왜 파라미터가 아니라 실효 커브인가
        "톤커브 파라미터를 평활하라"가 정석이지만, CLAHE 는 타일마다 커브가 다르고
        Drago 는 OpenCV 내부에서 계산돼 **꺼낼 파라미터가 없다.** 대신 프레임마다
        입력 휘도 → 출력 휘도의 **경험적 전달곡선**(256엔트리)을 재고 그것을 평활한다.
        오퍼레이터 종류와 무관하게 동작하고, 조합 arm(`D1A1`)에도 그대로 붙는다.

    보정은 **이득(곱)** 으로 넣는다 — 평활 커브 / 현재 커브 비율만큼 출력을 스케일한다.
    LAB 왕복(≈4ms)을 피하려고 곱으로 택했고, 국소 구조는 비율이 화소마다 같은 방향
    으로 작용하므로 CLAHE 가 만든 대비는 보존된다.

    한 극(one-pole) IIR 의 계수는 컷오프에서 유도한다:  a = 1 - exp(-2π·fc/fs).
    업계 관행 컷오프는 0.5Hz 수준 (→ lowlight_classical.md 3-1).
    """

    #: 전달곡선 통계용 화소 솎음 간격. 256엔트리 평균을 구하는 데 전 화소는 불필요하고
    #: bincount 가 이 스테이지 비용의 대부분이었다 (4 → 화소 1/16).
    SUBSAMPLE = 4
    GAIN_LIMIT = (0.5, 2.0)   # 커브가 0 근처일 때 이득이 발산하는 것을 막는다

    def __init__(self, fps: float = 16.0, cutoff_hz: float = 0.5):
        self.alpha = float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz / max(fps, 1e-6)))
        self.params = {"fps": fps, "cutoff_hz": cutoff_hz, "alpha": round(self.alpha, 4)}
        self._curve: np.ndarray | None = None

    def reset(self) -> None:
        self._curve = None

    @classmethod
    def _transfer(cls, l_in: np.ndarray, l_out: np.ndarray) -> np.ndarray:
        """입력 휘도값별 출력 휘도 평균. 빈 구간은 이웃에서 보간한다."""
        s = cls.SUBSAMPLE
        flat_in = l_in[::s, ::s].ravel()
        flat_out = l_out[::s, ::s].ravel().astype(np.float64)
        cnt = np.bincount(flat_in, minlength=256).astype(np.float64)
        acc = np.bincount(flat_in, weights=flat_out, minlength=256)
        seen = cnt > 0
        if not seen.any():
            return np.arange(256, dtype=np.float64)
        curve = np.empty(256, dtype=np.float64)
        curve[seen] = acc[seen] / cnt[seen]
        idx = np.arange(256)
        curve[~seen] = np.interp(idx[~seen], idx[seen], curve[seen])
        return curve

    def __call__(self, src: Frame, out: Frame) -> Frame:
        l_in = _luma(src)
        curve = self._transfer(l_in, _luma(out))

        if self._curve is None:          # 첫 프레임 — 평활할 과거가 없다
            self._curve = curve
            return out
        self._curve = self.alpha * curve + (1.0 - self.alpha) * self._curve

        lo, hi = self.GAIN_LIMIT
        gain = np.clip(self._curve / np.maximum(curve, 1.0), lo, hi).astype(np.float32)
        g = cv2.LUT(l_in, gain)                       # 화소별 이득 (1ch float32)
        # cv2.multiply 가 포화 캐스팅까지 한 번에 한다 — numpy 로 하면
        # float 변환·곱·clip·uint8 캐스팅이 각각 전체 프레임을 훑어 3배 비싸다.
        return cv2.multiply(out, cv2.merge((g, g, g)), dtype=cv2.CV_8U)


class MotionAdaptiveDenoise:
    """3-2 대응 — 이전 출력과 IIR 블렌딩하되 **움직인 화소는 섞지 않는다**.

        out = a·cur + (1-a)·prev,   a = clip(a_min + (1-a_min)·d/thresh, a_min, 1)

    `d` 는 프레임 차분(평활)이다. 정지 영역은 a→a_min 이라 강하게 누적돼 노이즈가
    √ 로 줄고, 움직이는 영역은 a→1 이라 **고스팅이 생기지 않는다.**

    a_min=0.35 이면 정지 영역의 실효 누적은 약 2.9프레임이라 σ 가 이론상 46% 수준
    으로 내려간다. `bf` 와 달리 공간 해상도를 전혀 깎지 않는다.

    구현 메모 — 비용이 곧 채택 가능성이다
        `bf` 를 대체하려는 스테이지라 `bf` 보다 비싸면 의미가 없다. 그래서
        (1) 이전 프레임을 uint8 로 들고 있어 매 프레임 형변환을 피하고,
        (2) 움직임은 **그레이 1채널**에서만 재며,
        (3) 블렌딩을 numpy 가 아니라 cv2(SIMD·멀티스레드) 연산으로 돌린다.
    """

    #: 기본값은 rec 38 스윕(2026-08-02)에서 고른 중간값이다. 0.35/12 는 억제가
    #: 거의 없었고(시간축σ 6.05→5.45), 0.15/40 은 억제는 크지만 동적부 디테일
    #: 유지율이 0.89→0.83 으로 떨어져 **움직이는 보행자를 지우는** 쪽이었다.
    def __init__(self, blend_min: float = 0.25, motion_thresh: float = 24.0):
        self.blend_min = float(blend_min)
        self.motion_thresh = float(motion_thresh)
        self.params = {"blend_min": blend_min, "motion_thresh": motion_thresh}
        self._prev: np.ndarray | None = None       # uint8 BGR

    def reset(self) -> None:
        self._prev = None

    def __call__(self, bgr: Frame) -> Frame:
        if self._prev is None or self._prev.shape != bgr.shape:
            self._prev = bgr.copy()
            return bgr

        # 움직임 세기 — 그레이 차분을 살짝 평활해 노이즈가 움직임으로 오인되지 않게
        d = cv2.blur(cv2.absdiff(_luma(bgr), _luma(self._prev)), (3, 3))
        a = cv2.LUT(d, self._alpha_lut())                      # 1ch float32

        prev_f = self._prev.astype(np.float32)
        delta = cv2.multiply(cv2.subtract(bgr.astype(np.float32), prev_f),
                             cv2.merge([a, a, a]))
        out = cv2.add(prev_f, delta).clip(0, 255).astype(np.uint8)
        self._prev = out
        return out

    def _alpha_lut(self) -> np.ndarray:
        """차분값(0~255) → 블렌딩 계수. 매 프레임 화소 연산 대신 256엔트리 LUT."""
        if getattr(self, "_lut", None) is None:
            d = np.arange(256, dtype=np.float32)
            self._lut = np.clip(
                self.blend_min + (1.0 - self.blend_min) * d / self.motion_thresh,
                self.blend_min, 1.0).astype(np.float32)
        return self._lut


# --------------------------------------------------------------------------
# arm 조립
# --------------------------------------------------------------------------


class Arm:
    """스테이지를 순서대로 적용하는 arm. 무처리 arm 은 stages 가 빈 리스트다."""

    is_temporal = False

    def __init__(self, name: str, stages: Sequence[Stage] = ()):
        self.name = name
        self.stages = list(stages)

    def reset(self) -> None:
        """시퀀스 경계에서 호출. 상태 없는 arm 은 할 일이 없다."""

    def __call__(self, bgr: Frame) -> Frame:
        out = bgr
        for stage in self.stages:
            out = stage(out)
        return out

    def describe(self) -> str:
        if not self.stages:
            return "무처리 (대조군)"
        return " → ".join(
            f"{s.__class__.__name__}({', '.join(f'{k}={v}' for k, v in s.params.items())})"
            for s in self.stages
        )

    def __repr__(self) -> str:
        return f"<Arm {self.name}: {self.describe()}>"


class TemporalArm(Arm):
    """A3 — 공간축 arm 위에 시간축 스테이지를 얹은 arm.

    ⚠️ **연속 프레임에만 유효하다.** 서로 무관한 정지영상에 돌리면 이전 프레임이
    남의 장면이라 값이 무의미하다. 시퀀스가 바뀌면 `reset()` 을 부를 것.
    인터페이스는 `bgr -> bgr` 그대로라 ③ 앞단 결선은 달라지지 않는다.
    """

    is_temporal = True

    def __init__(self, name: str, stages: Sequence[Stage] = (), *,
                 tone: bool = True, denoise: bool = True,
                 fps: float = 16.0, cutoff_hz: float = 0.5,
                 blend_min: float = 0.35, motion_thresh: float = 12.0):
        super().__init__(name, stages)
        self.tone = TemporalToneSmooth(fps, cutoff_hz) if tone else None
        self.denoise = MotionAdaptiveDenoise(blend_min, motion_thresh) if denoise else None

    def reset(self) -> None:
        for s in (self.tone, self.denoise):
            if s is not None:
                s.reset()

    def __call__(self, bgr: Frame) -> Frame:
        out = super().__call__(bgr)
        if self.tone is not None:
            out = self.tone(bgr, out)      # 실효 톤커브 평활은 원본이 필요하다
        if self.denoise is not None:
            out = self.denoise(out)
        return out

    def describe(self) -> str:
        extra = [s.__class__.__name__ + f"({', '.join(f'{k}={v}' for k, v in s.params.items())})"
                 for s in (self.tone, self.denoise) if s is not None]
        return " → ".join([super().describe()] + extra)


_BUILDERS: dict[str, Callable[[], Arm]] = {
    "none": lambda: Arm("none"),
    # 단조 증가 톤커브 — 하이라이트를 누르지 못한다
    "A1": lambda: Arm("A1", [CLAHE()]),
    "A1+bf": lambda: Arm("A1+bf", [CLAHE(), Bilateral()]),
    "A2": lambda: Arm("A2", [AGCWD()]),
    "A2+bf": lambda: Arm("A2+bf", [AGCWD(), Bilateral()]),
    # 하이라이트 압축 계열
    "D1": lambda: Arm("D1", [Tonemap("drago")]),
    "D1+bf": lambda: Arm("D1+bf", [Tonemap("drago"), Bilateral()]),
    "R1": lambda: Arm("R1", [Tonemap("reinhard")]),
    "R1+bf": lambda: Arm("R1+bf", [Tonemap("reinhard"), Bilateral()]),
    "L1": lambda: Arm("L1", [LIME()]),
    "L1+bf": lambda: Arm("L1+bf", [LIME(), Bilateral()]),
    # 조합 arm — 단일 arm 이 목적 축을 동시 충족하지 못해(→ lowlight_classical.md
    # 6-3-6) 역할을 분담한다: D1(하이라이트 압축) → A1(국소 대비 복원) → bf(노이즈).
    # D1 이 전체를 어둡게 눌러 대비를 잃는 결함(6-3-4의 3)을 A1 의 타일 적응이
    # 보상하는 것이 노림수다.
    #
    # CLAHE 의 감마는 1.0 으로 뺀다(단독 A1 은 0.75). D1 이 이미 표시 감마(2.2)를
    # 씌워 내보내므로 추가 감마는 중복이고, 실측에서도 대비 이득이 거의 없이
    # (1.73→1.67) D1 이 눌러둔 하이라이트만 되올렸다(코어 −58 → γ1.0 에서 −73).
    #
    # 비용은 스테이지 합산이라 720p 게이트는 원리적으로 못 넘는다. 내부 처리
    # 해상도 하향(640×360) 또는 셰이더 이식이 전제다 (→ 6-5).
    "D1A1": lambda: Arm("D1A1", [Tonemap("drago"), CLAHE(gamma=1.0)]),
    "D1A1+bf": lambda: Arm("D1A1+bf", [Tonemap("drago"), CLAHE(gamma=1.0), Bilateral()]),
    # ---- A3 시간축 (연속 프레임 전용) ------------------------------------
    # 접미사 의미 — `bf` 를 직교 스테이지로 분리했던 것과 같은 이유로 둘을 쪼갠다.
    #   +ts  톤커브 시간축 평활만 (플리커)
    #   +td  모션적응 시간축 노이즈 억제만
    #   +a3  둘 다 = **`bf` 를 대체하려는 후보**
    "A1+ts": lambda: TemporalArm("A1+ts", [CLAHE()], tone=True, denoise=False),
    "A1+td": lambda: TemporalArm("A1+td", [CLAHE()], tone=False, denoise=True),
    "A1+a3": lambda: TemporalArm("A1+a3", [CLAHE()]),
    "D1A1+ts": lambda: TemporalArm(
        "D1A1+ts", [Tonemap("drago"), CLAHE(gamma=1.0)], tone=True, denoise=False),
    "D1A1+td": lambda: TemporalArm(
        "D1A1+td", [Tonemap("drago"), CLAHE(gamma=1.0)], tone=False, denoise=True),
    "D1A1+a3": lambda: TemporalArm(
        "D1A1+a3", [Tonemap("drago"), CLAHE(gamma=1.0)]),
    # rec 38 실측 결론(2026-08-02): `+td` 는 `bf` 를 **대체하지 못한다** — 공간
    # 노이즈 두 축에서 모두 지고 더 비싸다. 반면 `+ts` 는 플리커를 낮추면서
    # 동적부 디테일을 거의 안 깎는다(0.99×). 그래서 실사용 후보는 둘의 합이다.
    "D1A1+bf+ts": lambda: TemporalArm(
        "D1A1+bf+ts", [Tonemap("drago"), CLAHE(gamma=1.0), Bilateral()],
        tone=True, denoise=False),
}

ARM_NAMES = tuple(_BUILDERS)


def build(name: str) -> Arm:
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise KeyError(f"알 수 없는 arm '{name}'. 사용 가능: {', '.join(ARM_NAMES)}") from None


# --------------------------------------------------------------------------
# 유틸
# --------------------------------------------------------------------------


def _gamma_lut(gamma: float) -> np.ndarray:
    """gamma < 1 이면 밝아진다."""
    return np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
