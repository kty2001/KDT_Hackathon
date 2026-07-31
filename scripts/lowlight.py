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

설계 메모
- **두 톤커브 모두 LAB L 채널에서 동작한다.** AGCWD 원논문은 HSV V 를 쓰지만,
  여기서 비교하려는 것은 색공간이 아니라 *톤커브 전략(타일 적응 vs 전역 적응)*이다.
  색공간을 맞춰 교란변수를 하나 없앤다.
- **노이즈 억제는 직교 스테이지다.** data.md 의 A안은 "CLAHE + 감마 + 노이즈 억제"로
  묶여 있었으나, 그러면 A1·A2 비교에 노이즈 처리가 섞여 들어간다. 분리해야
  가설 H2(고전의 노이즈 증폭이 탐지를 해치는가)를 arm 차이로 직접 읽을 수 있다.
- 상태를 갖지 않는다. 시간축 처리(A3, → lowlight_classical.md 3장)는 상태가 필요하므로
  별도 스테이지로 추가한다.
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
# arm 조립
# --------------------------------------------------------------------------


class Arm:
    """스테이지를 순서대로 적용하는 arm. 무처리 arm 은 stages 가 빈 리스트다."""

    def __init__(self, name: str, stages: Sequence[Stage] = ()):
        self.name = name
        self.stages = list(stages)

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
