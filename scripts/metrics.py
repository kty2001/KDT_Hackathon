"""② 저조도 개선 — 목적 축 지표 정의 (글레어 · 대비 · 노이즈).

지표가 `lowlight.py`(노이즈)와 `night_eval.py`(글레어·대비)에 흩어져 있었다.
셋 다 같은 결함을 공유하므로 — 아래 참고 — 한 곳에 모아 함께 고친다.
기존 이름은 두 모듈에서 그대로 재수출하므로 노트북·스크립트는 깨지지 않는다.

★ 재설계의 뿌리 (docs/lowlight_classical.md 7장)
    목적 축 지표 3개가 전부 **측정 대상을 영상 통계에 상대적으로 정의**했다.
    영상이 바뀌면 재는 대상 자체가 달라져, 육안과 반대 결론을 낸다.

        글레어  강광원을 '상위 1% 백분위'로 정의 → 매우 어두운 영상에서 임계가
                휘도 47까지 내려가 실제 광원(>=235)이 표본에서 빠진다
        노이즈  σ 비를 '전역 밝기이득'으로 나눔 → 암부를 크게 올리는 arm 은
                이득도 커서, 새로 **드러난** 노이즈가 분모에 상쇄된다
        대비    국소 RMS 비를 '전역 밝기이득'으로 나눔 → 극단적 저조도에서는
                어떤 arm 이든 밝기를 3~9배 올리므로 대비가 늘어도 1.0 아래로 눌린다

    셋 다 **절대 기준**으로 바꾼다. 공통 처방은 두 가지다.

        (a) 측정 대상을 **입력에서 고정**하고 출력에서 같은 화소를 본다
            (재선택하면 '무엇이 어디로 갔는가'를 못 읽는다)
        (b) 정규화가 필요하면 영상 통계로 나누지 말고 **명시적 대조군**과 비교한다

    글레어는 (a)를 절대 휘도 임계로 이미 적용했다(`glare_core_*`). 노이즈는 (a)를
    평활·암부 마스크로, 대비는 (b)를 밝기매칭 감마 대조군으로 적용한다.

자기검증:
    uv run python scripts/metrics.py              # LOL + 자체 촬영본에서 구·신 지표 대조
    uv run python scripts/metrics.py --n 15
"""

from __future__ import annotations

import unicodedata

import numpy as np

import cv2

Frame = np.ndarray  # BGR uint8 (H, W, 3)

LAMP_MIN = 235.0  # 광원 판정 절대 임계 — 백분위와 달리 영상 밝기에 흔들리지 않는다
LAMP_MIN_PX = 30  # 이보다 광원 화소가 적으면 그 표본은 코어 집계에서 제외

# Immerkaer(1996) 노이즈 추정 커널. 라플라시안 유사 응답이라 완만한 밝기 변화에는
# 거의 반응하지 않고 화소 단위 요동만 남긴다.
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
_IMMERKAER_SCALE = float(np.sqrt(np.pi / 2) / 6.0)


def luma(bgr: Frame) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


# --------------------------------------------------------------------------
# 표 출력 유틸 — 한글 헤더가 열을 밀지 않도록
# --------------------------------------------------------------------------


def cell(text: str, width: int, align: str = ">") -> str:
    """터미널 표시폭 기준으로 칸을 채운다.

    한글·기호는 터미널에서 두 칸을 차지하는데 `len()` 은 1로 세므로, f-string 의
    `:>12` 같은 폭 지정이 한글 헤더에서 어긋난다. 지표 표는 열이 맞아야 읽히므로
    동아시아 문자폭(W/F)을 2로 세어 직접 채운다.
    """
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    pad = " " * max(0, width - w)
    if align == ">":
        return pad + text
    if align == "^":
        half = len(pad) // 2
        return pad[:half] + text + pad[half:]
    return text + pad


# --------------------------------------------------------------------------
# 마스크 — 무엇을 재는가
# --------------------------------------------------------------------------


def local_std(L: np.ndarray, k: int = 9) -> np.ndarray:
    """k×k 이웃 표준편차 맵. 국소 대비이자 '구조가 있는가'의 척도다."""
    m = cv2.blur(L, (k, k))
    var = cv2.blur(L * L, (k, k)) - m * m
    return np.sqrt(np.maximum(var, 0.0))


def local_rms(L: np.ndarray, mask: np.ndarray) -> float:
    """마스크 영역의 평균 국소 대비."""
    return float(local_std(L)[mask].mean())


SHADOW_LO = 3.0    # 이 아래는 크러시 — 복원할 신호가 남아 있지 않다
SHADOW_HI = 80.0   # 중간회색(128)의 아래쪽. 통상적인 '그림자' 대역
FLAT_MAX_SD = 2.0  # 국소 표준편차가 이보다 작으면 구조가 없는 것으로 본다
MIN_PX = 200       # 마스크가 이보다 작으면 통계가 성립하지 않는다


def dark_mask(L: np.ndarray, lo: float = SHADOW_LO, hi: float = SHADOW_HI,
              min_px: int = MIN_PX) -> np.ndarray:
    """암부 마스크 — **절대 휘도 대역**. 크러시(순수 검정)는 제외한다.

    ⚠️ 예전 정의는 상한을 **백분위(p60)** 로 잡았고, 그것이 7장이 지적한 상대 정의
    결함 그 자체였다. 자체 촬영 야간 사진(평균밝기 7.3)에서 p60 은 **휘도 5** 였다.
    즉 마스크가 '암부'가 아니라 **크러시된 순수 검정만** 잡았고, 그 영역은 8bit
    양자화로 색차(a,b)의 고유값이 **1개뿐**이라 색 노이즈가 구조적으로 0 이 된다.
    영상마다 재는 대상이 달라지는 것도 문제지만, 이 경우는 아예 측정이 불가능해진다.

    절대 대역으로 고정하면 어떤 영상에서든 같은 '그림자'를 본다. 하한(3)은 신호가
    남아 있는 최소선이고, 상한(80)은 중간회색 아래의 통상적 그림자 대역이다.
    밝은 영상이라 이 대역이 거의 비면 백분위로 되돌아간다(그 경우엔 절대 기준을
    고집할 실익이 없다 — 재려는 암부 자체가 없는 영상이다).
    """
    m = (L >= lo) & (L <= hi)
    return m if m.sum() >= min_px else (L <= np.percentile(L, 60.0))


def flat_dark_mask(L: np.ndarray, flat_max_sd: float = FLAT_MAX_SD,
                   min_px: int = MIN_PX) -> np.ndarray:
    """★ 노이즈 측정용 — **암부 ∩ 평활** 영역. 여기 남는 요동은 노이즈뿐이다.

    왜 평활 영역에 국한하는가: Immerkaer 추정을 영상 전체에 돌리면 질감·엣지가
    응답에 섞인다. 저조도 개선 arm 은 암부 디테일을 드러내는 것이 목적이므로
    "디테일이 늘어난 것"과 "노이즈가 늘어난 것"이 한 숫자에 뭉쳐, 무엇이 나빠졌는지
    구분되지 않는다. 구조가 없는 곳만 보면 남는 것은 노이즈뿐이다.

    왜 **입력 기준**으로 고정하는가: 출력에서 평활 영역을 다시 고르면 arm 마다
    다른 화소를 재게 된다. 노이즈를 키운 arm 일수록 "평활" 판정 영역이 좁아지고
    더 조용한 곳만 남아, **노이즈를 키울수록 좋게 나오는** 역전이 생긴다.
    입력에서 한 번 고르고 모든 arm 이 같은 화소를 보게 해야 비교가 성립한다.

    평활 판정도 **절대 임계**(국소 σ <= 2)다. 백분위로 '가장 평평한 절반'을 고르면
    영상마다 다른 수준의 평활도를 평활이라 부르게 되고, 극단적 저조도에서는
    크러시 영역으로 마스크가 쏠린다.
    """
    dark = dark_mask(L)
    sd = local_std(L)
    flat = dark & (sd <= flat_max_sd)
    if flat.sum() < min_px:  # 입력 자체가 시끄러워 절대 임계로는 안 걸리는 경우
        flat = dark & (sd <= np.percentile(sd[dark], 50.0))
    # 3×3 커널이 마스크 바깥의 구조를 끌어오지 않도록 1픽셀 침식
    flat = cv2.erode(flat.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return flat if flat.sum() >= min_px else dark


# --------------------------------------------------------------------------
# 노이즈 — 절대 σ (구 지표: 밝기이득 정규화 비)
# --------------------------------------------------------------------------


def masked_sigma(chan: np.ndarray, mask: np.ndarray) -> float:
    """마스크 영역에 국한한 Immerkaer σ 추정. **절대값**(0~255 스케일)이다."""
    if not mask.any():
        return float("nan")
    response = np.abs(cv2.filter2D(chan.astype(np.float32), -1, _IMMERKAER))
    return float(response[mask].mean() * _IMMERKAER_SCALE)


def noise_absolute(bgr: Frame, mask: np.ndarray) -> tuple[float, float]:
    """★ 드러난 노이즈를 휘도·색으로 **나눠서** 절대 σ 로 잰다.

    색을 따로 보는 것이 핵심이다. 육안 결함 보고(문서 6-3-4의 1)는 "암부에
    **컬러** 노이즈가 화면 전체로 퍼진다"였는데, 그레이스케일 하나로는 이 축이
    아예 측정되지 않았다. 두 계열이 색 채널을 다루는 방식이 정반대라 판별력도 크다 —
    A1·A2 는 LAB 의 `L` 만 건드려 `a,b` 노이즈를 그대로 두는 반면,
    D1·R1·L1 은 RGB 세 채널을 모두 변환해 색 노이즈를 함께 증폭한다.

    반환: (휘도 σ, 색 σ). 색은 LAB `a`,`b` 의 이차평균이다.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sig_l = masked_sigma(lab[:, :, 0], mask)
    sig_a = masked_sigma(lab[:, :, 1], mask)
    sig_b = masked_sigma(lab[:, :, 2], mask)
    return sig_l, float(np.hypot(sig_a, sig_b) / np.sqrt(2.0))


def estimate_noise(bgr: Frame) -> float:
    """Immerkaer(1996) 전역 σ 추정 — **구 지표**. 문서 대조용으로만 남긴다.

    영상 전체를 평균하므로 질감과 노이즈가 섞인다. 새 판정은 `noise_absolute` 를 쓸 것.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    response = np.abs(cv2.filter2D(gray, -1, _IMMERKAER))
    return float(response.sum() * np.sqrt(np.pi / 2) / (6.0 * (w - 2) * (h - 2)))


def noise_amplification(src: Frame, dst: Frame) -> float:
    """밝기이득 정규화 σ 비 — **구 지표(결함 있음)**. 문서 6-2·6-3 대조용.

    ⚠️ 육안과 반대 결론을 낸다. 암부를 크게 끌어올리는 arm 은 원래 보이지 않던
    노이즈를 드러내는데, 전역 밝기이득도 함께 커서 나누면 1.0 아래로 떨어진다
    (D1 0.834 · R1 0.766 — 육안으로는 이 둘이 가장 지저분하다).
    **arm 서열화에 쓰지 말 것.** → `noise_absolute`
    """
    gain = (float(dst.mean()) + 1e-6) / (float(src.mean()) + 1e-6)
    return (estimate_noise(dst) + 1e-6) / (estimate_noise(src) + 1e-6) / gain


# --------------------------------------------------------------------------
# 대비 — 밝기매칭 감마 대조군 대비 (구 지표: 밝기이득 정규화)
# --------------------------------------------------------------------------


def gamma_matched_luma(L: np.ndarray, target_mean: float,
                       lo: float = 0.05, hi: float = 6.0,
                       iters: int = 30) -> tuple[np.ndarray, float]:
    """L 을 **감마만으로** target_mean 밝기에 맞춘 영상 — '균일하게 밝히기' 대조군.

    감마는 단조 톤커브라 공간 정보를 전혀 쓰지 않는다. 즉 이 영상은 **정확히
    '그냥 밝힌 것'** 이고, 여기서 늘어난 국소 대비는 밝히기의 부수효과일 뿐이다.
    arm 이 이보다 더 대비를 냈다면 그것이 진짜 '대비 강조'다.

    평균은 감마에 대해 단조 감소하므로 이분탐색이 성립한다. 탐색은 256빈
    히스토그램 위에서 하므로 영상 크기와 무관하게 싸다.
    """
    idx = np.clip(L, 0, 255).astype(np.uint8)
    hist = np.bincount(idx.ravel(), minlength=256).astype(np.float64)
    n = max(hist.sum(), 1.0)
    levels = np.arange(256, dtype=np.float64) / 255.0

    for _ in range(iters):
        g = 0.5 * (lo + hi)
        if float((hist * (levels ** g) * 255.0).sum() / n) < target_mean:
            hi = g  # 아직 어둡다 → 감마를 더 내려 밝힌다
        else:
            lo = g

    g = 0.5 * (lo + hi)
    lut = ((levels ** g) * 255.0).astype(np.float32)
    return lut[idx], g


def contrast_gain(L_in: np.ndarray, L_out: np.ndarray,
                  mask: np.ndarray) -> tuple[float, float]:
    """★ '밝기가 아닌 대비' 축의 절대 판정. >1.0 이면 균일 밝히기를 넘어섰다.

    구 지표는 국소 대비 비를 **전역 밝기이득으로 나눴다.** 그 전제는 "밝기를 k배
    올리면 대비도 k배 커진다"는 선형 가정인데, 톤커브는 비선형이고 출력은 8bit 로
    클립되므로 성립하지 않는다. LOL(평균밝기 15)처럼 극단적으로 어두우면 어떤
    arm 이든 밝기를 3~9배 올리기 때문에, 대비가 실제로 늘어도 1.0 아래로 눌렸다.

    여기서는 나누는 대신 **같은 평균밝기를 내는 감마 영상과 직접 비교**한다.
    밝기 효과가 분자·분모 양쪽에 똑같이 들어가 상쇄되므로, 남는 것은 공간 적응
    (타일·조도맵·국소 압축)이 만들어낸 순수 대비 이득뿐이다.

    반환: (대비 이득, 대조군에 쓰인 감마).
    """
    ref, g = gamma_matched_luma(L_in, float(L_out.mean()))
    return local_rms(L_out, mask) / (local_rms(ref, mask) + 1e-6), g


# --------------------------------------------------------------------------
# 통합 측정
# --------------------------------------------------------------------------


def measure(arm, samples, lamp_min: float = LAMP_MIN,
            lamp_min_px: int = LAMP_MIN_PX) -> dict[str, float]:
    """arm 을 목적 축(글레어·대비·노이즈)에서 평가한다.

    각 축마다 **신 지표와 구 지표를 함께** 낸다. 구 지표는 문서 6-2·6-3 의 기존
    수치와 대조하기 위한 것이고, **판정은 신 지표로 한다.**

        축      신 지표(절대 기준)                  구 지표(결함)
        글레어  glare_core_*  (휘도>=235 고정)      glare_*      (상위 1% 백분위)
        대비    contrast_gain (밝기매칭 감마 대비)  norm_contrast(밝기이득 정규화)
        노이즈  noise_luma/chroma_* (평활암부 절대) noise_amp    (밝기이득 정규화)

    `sat_*`(포화 면적)는 원래 절대 임계 250 기반이라 결함이 없다. `glare_core_*`와
    함께 보면 **코어 밝기**(광원이 얼마나 밝아지는가)와 **번짐**(포화로 뭉개지는
    면적)을 분리해 읽을 수 있다 — 둘은 서로 다른 종류의 피해다.
    """
    glare_b, glare_a, sat_b, sat_a = [], [], [], []
    core_b, core_a = [], []
    c_ratio, gains, n_amp = [], [], []
    c_gain, ref_gamma = [], []
    nl_b, nl_a, nc_b, nc_a = [], [], [], []

    for src in samples:
        L = luma(src)
        out_img = arm(src)
        O = luma(out_img)

        # --- 글레어 -------------------------------------------------------
        hi = L >= np.percentile(L, 99)  # 구 지표 — 어두운 영상에서 광원을 놓친다
        glare_b.append(L[hi].mean())
        glare_a.append(O[hi].mean())
        sat_b.append(100.0 * (L >= 250).mean())
        sat_a.append(100.0 * (O >= 250).mean())

        # 입력에서 고른 광원 화소 위치를 출력에도 그대로 적용한다(재선택하지 않는다).
        # 같은 화소가 어디로 가는지를 봐야 "억제/증폭" 판정이 의미가 있다.
        lamp = L >= lamp_min
        if int(lamp.sum()) >= lamp_min_px:
            core_b.append(float(np.median(L[lamp])))
            core_a.append(float(np.median(O[lamp])))

        # --- 대비 ---------------------------------------------------------
        dark = dark_mask(L)
        c_ratio.append(local_rms(O, dark) / (local_rms(L, dark) + 1e-6))
        gain = (float(O.mean()) + 1e-6) / (float(L.mean()) + 1e-6)
        gains.append(gain)
        cg, g_ref = contrast_gain(L, O, dark)
        c_gain.append(cg)
        ref_gamma.append(g_ref)

        # --- 노이즈 -------------------------------------------------------
        n_amp.append((estimate_noise(out_img) + 1e-6) / (estimate_noise(src) + 1e-6) / gain)
        flat = flat_dark_mask(L)  # 입력 기준 고정 — 모든 arm 이 같은 화소를 본다
        sl_b, sc_b = noise_absolute(src, flat)
        sl_a, sc_a = noise_absolute(out_img, flat)
        nl_b.append(sl_b), nl_a.append(sl_a)
        nc_b.append(sc_b), nc_a.append(sc_a)

    cr, br = float(np.mean(c_ratio)), float(np.mean(gains))
    return {
        # 글레어
        "glare_before": float(np.mean(glare_b)),
        "glare_after": float(np.mean(glare_a)),
        "sat_before": float(np.mean(sat_b)),
        "sat_after": float(np.mean(sat_a)),
        # core_n == 0 이면 이 표본 풀 전체에 광원(>=lamp_min)이 없다는 뜻 —
        # 두 값 다 nan 이며, 호출부가 "—" 같은 표시로 걸러야 한다.
        "glare_core_before": float(np.mean(core_b)) if core_b else float("nan"),
        "glare_core_after": float(np.mean(core_a)) if core_a else float("nan"),
        "glare_core_n": len(core_b),
        # 대비
        "contrast": cr,
        "gain": br,
        "norm_contrast": cr / br,           # 구 지표 (결함)
        "contrast_gain": float(np.mean(c_gain)),  # ★ 신 지표
        "ref_gamma": float(np.mean(ref_gamma)),
        # 노이즈
        "noise_amp": float(np.mean(n_amp)),  # 구 지표 (결함)
        "noise_luma_before": float(np.mean(nl_b)),   # ★ 신 지표 (절대 σ)
        "noise_luma_after": float(np.mean(nl_a)),
        "noise_chroma_before": float(np.mean(nc_b)),
        "noise_chroma_after": float(np.mean(nc_a)),
    }


# --------------------------------------------------------------------------
# 자기검증 — 신 지표가 육안 결론을 재현하는가
# --------------------------------------------------------------------------


def _self_check():
    """구·신 지표를 나란히 출력해, 신 지표가 문서 6-3-4 의 육안 결론과 맞는지 본다.

    육안 결론(문서 6-3-4의 1): **D1·R1·L1 은 암부 컬러 노이즈가 심하고,
    A1+bf·A2 가 훨씬 깨끗하다.** 구 노이즈 지표는 이와 반대 순위를 냈다.
    신 지표가 육안 순위를 재현하면 재설계가 성공한 것이다.
    """
    import argparse
    from pathlib import Path

    from lowlight import ARM_NAMES, build

    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=_self_check.__doc__)
    p.add_argument("--n", type=int, default=8, help="LOL 표본 수")
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    args = p.parse_args()

    arms = [build(n) for n in args.arms]

    pools: list[tuple[str, list[np.ndarray]]] = []
    low_dir = root / "data/LOLdataset/eval15/low"
    if low_dir.is_dir():
        imgs = []
        for path in sorted(low_dir.iterdir())[: args.n]:
            im = cv2.imread(str(path))
            if im is not None:
                imgs.append(im)
        if imgs:
            pools.append((f"LOL eval15 (n={len(imgs)}, 합성 감광)", imgs))

    sample_path = root / "data/sample_image.png"
    if sample_path.exists():
        im = cv2.imread(str(sample_path))
        if im is not None:
            # 원본이 크면 지표가 아니라 인내심을 시험한다. 노이즈는 스케일에
            # 민감하므로 축소는 필요한 만큼만(장변 1280) 한다.
            if im.shape[1] > 1280:
                h = int(1280 * im.shape[0] / im.shape[1])
                im = cv2.resize(im, (1280, h), interpolation=cv2.INTER_AREA)
            pools.append((f"자체 촬영 야간 (n=1, 평균밝기 {luma(im).mean():.1f})", [im]))

    if not pools:
        raise SystemExit("LOL 도 sample_image.png 도 없다. README '데이터 배치' 참고")

    RULE = 92
    for title, imgs in pools:
        print("=" * RULE)
        print(title)
        print("=" * RULE)
        print(cell("arm", 8, "<") + "│" + cell("[신] 노이즈 절대σ", 22, "^")
              + "│" + cell("[신] 대비", 10) + "│" + cell("[신] 광원코어", 14)
              + "║" + cell("[구] 노이즈", 12) + cell("[구] 대비", 11))
        print(cell("", 8) + "│" + cell("휘도", 11) + cell("색", 11)
              + "│" + cell("감마대비", 10) + "│" + cell("전→후", 14)
              + "║" + cell("증폭률", 12) + cell("정규화", 11))
        print("-" * RULE)

        base = None
        for arm in arms:
            r = measure(arm, imgs)
            if base is None:
                base = r
            core = ("—" if r["glare_core_n"] == 0 else
                    f"{r['glare_core_before']:.0f}→{r['glare_core_after']:.0f}")
            print(cell(arm.name, 8, "<") + "│"
                  + cell(f"{r['noise_luma_after']:.2f}", 11)
                  + cell(f"{r['noise_chroma_after']:.2f}", 11) + "│"
                  + cell(f"{r['contrast_gain']:.2f}", 10) + "│"
                  + cell(core, 14) + "║"
                  + cell(f"{r['noise_amp']:.3f}", 12)
                  + cell(f"{r['norm_contrast']:.2f}", 11))

        # 신 노이즈는 정규화하지 않으므로 입력 자체의 σ 가 기준선이다
        print(cell("(입력)", 8, "<") + "│"
              + cell(f"{base['noise_luma_before']:.2f}", 11)
              + cell(f"{base['noise_chroma_before']:.2f}", 11) + "│"
              + cell("1.00", 10) + "│" + cell("기준선", 14) + "║"
              + cell("1.000", 12) + cell("1.00", 11))
        print()

    print("읽는 법 —  ║ 왼쪽이 신 지표(절대 기준, 이걸로 판정) · 오른쪽이 구 지표(대조용)")
    print()
    print("  [신] 노이즈   평활 암부에 국한한 **절대** σ. (입력) 행보다 크면 노이즈를 드러낸 것.")
    print("                정규화하지 않아 밝기이득에 상쇄되지 않는다. 마스크는 입력에서 고정해")
    print("                모든 arm 이 같은 화소를 본다. 휘도와 색을 나눈 것이 핵심이다 —")
    print("                A1·A2 는 LAB `L` 만 건드려 색 σ 가 그대로인 반면, D1·R1·L1 은 RGB 를")
    print("                전부 변환해 색 노이즈를 함께 키운다 (육안 결함 보고 6-3-4의 1)")
    print("  [신] 대비     같은 평균밝기의 감마 영상 대비 국소대비 비. >1.0 이면 '균일하게")
    print("                밝히기'를 넘어선 진짜 대비 강조")
    print("  [신] 광원코어 휘도>=235 화소를 입력에서 고정해 전→후. 낮아지면 실제 억제.")
    print("                '—' 는 그 표본에 광원이 없다는 뜻 — LOL 이 그렇다(합성 감광이라")
    print("                가로등·전조등이 아예 없다). 글레어 판정은 자체 촬영본에서만 가능")
    print()
    print("  [구] 노이즈   밝기이득 정규화 σ 비 — **육안과 반대 결론을 내는 결함 지표**")
    print("  [구] 대비     밝기이득 정규화 — 극단적 저조도에서 1.0 아래로 눌린다")


if __name__ == "__main__":
    _self_check()
