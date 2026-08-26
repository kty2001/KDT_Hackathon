"""③ 배포 ONNX → **INT8 양자화** (PTQ · static · QDQ).

`export_onnx.py` 가 구운 FP32 배포판을 받아 8비트 판을 만든다. 산출물은
`outputs/quantization/` 에 모으고, 받는 쪽이 되묻지 않도록 metadata(기계용)·
README(사람용)·calib manifest(재현용)를 같이 굽는다.

    outputs/quantization/<pkg>/          ← 모델마다 한 칸. 여러 모델이 공존한다
        <pkg>_generic.onnx          ★ generic 프리셋 (per-channel · W int8 / A uint8)
        <pkg>_qnn_qdq.onnx          ★ qnn 프리셋 (per-tensor · uint8)
        <pkg>_opset13_fp32.onnx     중간물 — opset 12→13 변환 + quant_pre_process (FP32)
        <pkg>_metadata.json · <pkg>_README.md · <pkg>_calib_manifest.txt

이름은 **`<pkg>_<판>`** 하나로 통일한다 — 평가 산출물도 같은 줄기를 쓰면
(`<pkg>_generic_realnight.json`) 어느 모델의 수치인지 파일명만 보고 안다.

`<pkg>` 는 **원본 ONNX 파일명 + `-INT8`** 이다 (예: `bammasil_det_c4e_s3_11n_640-INT8`).
디렉토리도 파일도 같은 이름으로 시작하므로 **한 파일만 따로 옮겨도 8비트 판인 줄 안다.**
모델·해상도가 다르면 저절로 갈리고, `--name` 으로 덮어쓸 수 있다.
⚠️ `_opset13_fp32.onnx` 만은 이름 그대로 **FP32** 다 — 양자화의 입력이지 산출이 아니다.

왜 **static** PTQ 인가 (`quantize_dynamic` 을 쓰지 않는 이유)
    이 그래프는 Conv 88개짜리 CNN 이다. dynamic 양자화는 활성값 범위를 매 추론마다
    재는 방식이라 RNN·Transformer 용이고, CNN 은 캘리브레이션으로 범위를 미리 고정하는
    static 이 정석이다. 모바일 NPU 는 애초에 고정 스케일을 요구한다.

왜 **QDQ** 형식인가
    QNN·NNAPI·Core ML 계열이 QuantizeLinear/DequantizeLinear 쌍을 보고 자기 커널로
    융합한다. QOperator 형식은 ORT CPU 전용에 가깝다.

왜 **opset 을 13 으로 올리는가**
    per-channel QDQ 는 `QuantizeLinear` 의 `axis` 속성을 쓰는데 그것이 opset 13 에서
    들어왔다. Conv 는 채널별 스케일이 정확도에 크게 유리하다. 배포판이 opset 12 인 것은
    모바일 변환기 호환 폭 때문인데(→ `export_onnx.py`), 양자화 산출물은 어차피 별도
    파일이므로 여기서만 올린다. `onnx.version_converter` 로 되고 재내보내기가 필요 없다.

왜 **프리셋이 둘**인가 — CPU 판·GPU 판이 아니다
    QDQ ONNX 는 런타임 중립이라 같은 파일을 CPU EP·NPU EP 가 모두 읽는다. 두 판의 차이는
    **EP 계열이 요구하는 양자화 형식**이다 — `generic` 은 per-channel int8(NNAPI·Core ML·
    CPU EP 에서 정확도가 낫다), `qnn` 은 per-tensor uint8(스냅드래곤 QNN EP 가 요구).
    GPU 는 INT8 의 짝이 아니다 — ORT CUDA EP 는 QDQ 를 제대로 쓰지 못하고 INT8 GPU 경로는
    사실상 TensorRT 뿐인데 **폰에는 TensorRT 가 없다** (→ hardware_inference.md 1장).
    폰 GPU delegate 를 쓸 거면 INT8 이 아니라 FP16(`export_onnx.py --half`)이다.

🔴 왜 **`Conv` 만** 양자화하는가 (2026-08-26 실측 — 이것을 빼면 모델이 죽는다)
    ORT 기본값으로 전부 양자화하면 **검출이 0 이 된다.** 7장 전부에서 FP32 가 찾던 박스
    (`_04` 야간 볼라드 0.84·0.77 등)가 통째로 사라졌다. 원인은 머리(head)의 마지막
    `Concat`(`/model.23/Concat_3`)이 **박스 좌표(0~640 픽셀)와 클래스 점수(0~1)를 한
    텐서로 합치는** 데 있다. 이 출력을 per-tensor uint8 로 양자화하면 스케일이
    640/255 ≈ 2.5 로 잡혀 **모든 클래스 점수가 0 으로 반올림**된다.
    `op_types_to_quantize=["Conv"]` 로 두면 연산량의 대부분(Conv 88개)은 INT8 로 가면서
    머리의 산술(Concat·Sigmoid·Softmax·Mul)은 float 로 남아 이 붕괴가 없다.
    실측으로 볼라드 2/2 (0.81·0.78) 와 계단이 그대로 살아났다.

⚠️ **캘리브레이션 표본은 배포와 같은 자로 전처리해야 한다** — square letterbox 640 ·
    RGB · /255 · NCHW. `export_onnx.letterbox` 를 그대로 가져다 쓴다 (→ STATUS 3장 함정 18).

⚠️ **속도는 이 PC 에서 판정하지 않는다** (→ STATUS 3장 함정 3). 아래 CPU EP 배율은
    **참고치**이고 실판정은 `C11` 실기기 계측이다.

사용:
    uv run python scripts/quantize_onnx.py                        # generic + qnn 둘 다
    uv run python scripts/quantize_onnx.py --preset generic
    uv run python scripts/quantize_onnx.py --calibrate-method percentile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# ultralytics 가 `onnxruntime`(CPU 판)을 제멋대로 설치하면 `onnxruntime-gpu` 와 디렉토리를
# 공유해 나중에 통째로 망가진다 (→ STATUS 3장 함정 9). 임포트 전에 꺼야 한다.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_onnx import letterbox, parity_check      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC_ONNX = ROOT / "outputs/export/bammasil_det_c4e_s3_11n_640/bammasil_det_c4e_s3_11n_640.onnx"
OUT = ROOT / "outputs/quantization"
CALIB_SRC = ROOT / "data/test_real_data"

PRESETS = ("generic", "qnn")

# ⚠️ 파일 이름이 `_qnn.onnx` 로 끝나면 ultralytics 가 이것을 **QNN 컨텍스트 바이너리**로
#    보고 `onnxruntime_providers_qnn.dll` 을 찾다가 죽는다 (`autobackend.py:348`
#    `name.endswith("_qnn.onnx")`). 실체는 평범한 QDQ ONNX 이므로 접미사를 피한다.
# `<pkg>_<판>` 으로 통일한다. ⚠️ qnn 판을 `_qnn` 으로 끝내지 말 것 — ultralytics 가
# **QNN 컨텍스트 바이너리**로 착각해 없는 DLL 을 찾다 죽는다 (`autobackend.py:348`).
FILE_SUFFIX = {"generic": "_generic", "qnn": "_qnn_qdq"}

# 🔴 Conv 만 양자화한다 — 전부 양자화하면 검출이 0 이 된다 (위 docstring 참고).
OP_TYPES = ["Conv"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, default=SRC_ONNX, help="원본 FP32 ONNX")
    p.add_argument("--out", type=Path, default=OUT,
                   help="산출 루트. 실제 산출은 이 아래 <name>/ 에 들어간다")
    p.add_argument("--name", default=None,
                   help="산출 하위 디렉토리 이름. 기본은 원본 ONNX 의 파일명 — "
                        "모델마다 갈려서 한 루트에 여러 모델이 공존한다")
    p.add_argument("--preset", default=",".join(PRESETS),
                   help=f"쉼표 구분 — {' · '.join(PRESETS)}")
    p.add_argument("--calib-src", type=Path, default=CALIB_SRC,
                   help="캘리브 소재 루트 (야간 실촬영)")
    p.add_argument("--video-stride", type=int, default=4,
                   help="동영상 N 프레임마다 1장")
    p.add_argument("--include-synthetic", action="store_true",
                   help="lowlight/·lowlight_x02/ 합성 어둡게 판도 캘리브에 넣는다. "
                        "기본은 끈다 — 실기기에 없는 분포라 활성값 범위만 넓힌다")
    p.add_argument("--bits", type=int, default=8, choices=(8, 4),
                   help="가중치 비트. 8 = opset 13 · 4 = **opset 21 필수** — 표준 Q/DQ 가 "
                        "4비트를 opset 21 부터 받는다. 그 아래로 구우면 com.microsoft "
                        "도메인으로 떨어져 ORT 밖(QNN·NNAPI·Core ML)에서 못 읽는다")
    p.add_argument("--block-size", type=int, default=64,
                   help="4비트 전용 — 가중치 블록 단위 스케일. 0 이면 per-channel. "
                        "4비트는 per-channel 도 거칠어 블록이 실질 관건이다")
    p.add_argument("--calibrate-method", default="minmax",
                   choices=("minmax", "percentile", "entropy"))
    p.add_argument("--imgsz", type=int, default=640, help="배포 해상도 고정 전제")
    p.add_argument("--check-n", type=int, default=8,
                   help="FP32↔INT8 박스 비교 표본 수. 0 이면 건너뜀")
    p.add_argument("--conf", type=float, default=0.25,
                   help="비교에 쓸 신뢰도 임계 (운영값 0.25 — C4e S1)")
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--bench-runs", type=int, default=20,
                   help="CPU EP 지연 참고치 반복 수. 0 이면 건너뜀")
    return p.parse_args()


# --------------------------------------------------------------------------- 캘리브 표본

def collect_calib_frames(src: Path, imgsz: int, stride: int,
                         include_synthetic: bool) -> tuple[list, list[str]]:
    """야간 실촬영 소재를 square letterbox 640 uint8 로 모은다.

    라벨이 필요 없으므로 스틸뿐 아니라 **동영상 프레임**도 쓴다 — 스틸 7장만으로는
    활성값 범위를 잡기에 표본이 너무 얇다.
    """
    import cv2
    import numpy as np

    dirs = [src]
    if include_synthetic:
        dirs += [src / "lowlight", src / "lowlight_x02"]

    frames, manifest = [], []
    for d in dirs:
        if not d.is_dir():
            continue
        for img in sorted(d.glob("*.jpg")):
            # 한글 경로 대비 — cv2.imread 는 비ASCII 경로를 조용히 None 으로 돌려준다
            bgr = cv2.imdecode(np.fromfile(str(img), np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            frames.append(letterbox(bgr, imgsz))
            manifest.append(str(img.relative_to(ROOT)))
        for vid in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(vid))
            idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    frames.append(letterbox(bgr, imgsz))
                    manifest.append(f"{vid.relative_to(ROOT)}#{idx}")
                idx += 1
            cap.release()
    return frames, manifest


def make_reader(frames, input_name: str):
    """ORT CalibrationDataReader — 전처리는 배포 계약과 같다 (RGB · /255 · NCHW)."""
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader

    class NightCalibReader(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= len(frames):
                return None
            bgr = frames[self.i]
            self.i += 1
            rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
            return {input_name: rgb.transpose(2, 0, 1)[None]}

        def rewind(self):
            self.i = 0

    return NightCalibReader()


# --------------------------------------------------------------------------- 양자화

def to_opset(src: Path, dst: Path, target: int) -> dict:
    """opset 올리기 + shape inference·정리.

    8비트는 **13** (per-channel QDQ 가 쓰는 `QuantizeLinear.axis` 가 여기서 들어왔다),
    4비트는 **21** (표준 Q/DQ 가 4비트 타입을 opset 21 부터 받는다)이 전제다.
    """
    import onnx
    from onnx import version_converter
    from onnxruntime.quantization.shape_inference import quant_pre_process

    m = onnx.load(str(src))
    before = len(m.graph.node)
    m13 = version_converter.convert_version(m, target)
    onnx.checker.check_model(m13)
    tmp = dst.with_suffix(".tmp.onnx")
    onnx.save(m13, str(tmp))
    quant_pre_process(str(tmp), str(dst), skip_symbolic_shape=False)
    tmp.unlink(missing_ok=True)
    after = len(onnx.load(str(dst)).graph.node)
    return {"nodes_before": before, "nodes_after_convert": len(m13.graph.node),
            "nodes_after_preprocess": after}


def copy_metadata(src: Path, dst: Path) -> list[str]:
    """원본의 `metadata_props`(names·stride·task…)를 양자화판에 되박는다.

    ORT 양자화·전처리 과정에서 떨어지면 `YOLO(<int8>.onnx)` 가 클래스 이름을 잃거나
    로드에 실패해 아래 검증이 통째로 막힌다.
    """
    import onnx

    props = {p.key: p.value for p in onnx.load(str(src)).metadata_props}
    m = onnx.load(str(dst))
    have = {p.key for p in m.metadata_props}
    missing = sorted(k for k in props if k not in have)
    if missing:
        onnx.helper.set_model_props(m, props)
        onnx.save(m, str(dst))
    return missing


def quantize(preset: str, src13: Path, dst: Path, frames, input_name: str,
             method: str, bits: int = 8, block_size: int = 0) -> dict:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    cal = {"minmax": CalibrationMethod.MinMax,
           "percentile": CalibrationMethod.Percentile,
           "entropy": CalibrationMethod.Entropy}[method]

    # 활성은 어느 쪽이든 8비트다 (W4A8) — 검출 모델에서 활성까지 4비트는 무모하다.
    signed = QuantType.QInt4 if bits == 4 else QuantType.QInt8
    unsigned = QuantType.QUInt4 if bits == 4 else QuantType.QUInt8
    extra = {"BlockSize": block_size} if (bits == 4 and block_size) else {}
    common = {"bits": bits, "block_size": block_size if bits == 4 else None,
              "calibrate_method": method, "op_types_to_quantize": OP_TYPES}

    if preset == "generic":
        quantize_static(
            str(src13), str(dst), make_reader(frames, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8, weight_type=signed,
            per_channel=True, calibrate_method=cal,
            op_types_to_quantize=OP_TYPES, extra_options=extra,
            calibration_providers=["CPUExecutionProvider"])
        return {"preset": preset, "quant_format": "QDQ", "per_channel": True,
                "activation_type": "QUInt8", "weight_type": signed.name, **common}

    # qnn — 스냅드래곤 HTP 가 요구하는 형태를 ORT 가 직접 짜 준다
    from onnxruntime.quantization import quantize as quantize_with_config
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    cfg = get_qnn_qdq_config(str(src13), make_reader(frames, input_name),
                             calibrate_method=cal,
                             activation_type=QuantType.QUInt8,
                             weight_type=unsigned,
                             per_channel=False,
                             op_types_to_quantize=OP_TYPES,
                             calibration_providers=["CPUExecutionProvider"])
    # 블록 크기는 인자로 안 받으므로 돌려받은 config 에 주입한다
    if extra:
        cfg.extra_options = {**(cfg.extra_options or {}), **extra}
    quantize_with_config(str(src13), str(dst), cfg)
    return {"preset": preset, "quant_format": "QDQ", "per_channel": False,
            "activation_type": "QUInt8", "weight_type": unsigned.name, **common,
            "note": "get_qnn_qdq_config 기본값 — QNN EP(스냅드래곤 HTP) 대상"}


# --------------------------------------------------------------------------- 검증

def graph_io(path: Path) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {"inputs": [{"name": i.name, "shape": i.shape, "dtype": i.type}
                       for i in sess.get_inputs()],
            "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.type}
                        for o in sess.get_outputs()]}


def bench_cpu(path: Path, imgsz: int, runs: int) -> float | None:
    """⚠️ 참고치다 — 이 PC 는 15~25ms 를 판정할 수 없다 (STATUS 3장 함정 3)."""
    if runs <= 0:
        return None
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.zeros((1, 3, imgsz, imgsz), np.float32)
    for _ in range(3):
        sess.run(None, {name: x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {name: x})
    return round((time.perf_counter() - t0) / runs * 1000, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mb(path: Path) -> float:
    return round(path.stat().st_size / 1e6, 2)


# --------------------------------------------------------------------------- README

def file_rows(meta: dict, outdir: Path, pkg: str) -> str:
    """디렉토리에 **실제로 있는 파일 전부**에 설명을 붙인다.

    이 스크립트가 만들지 않은 것(예: `eval_real_night.py` 의 json)도 같은 칸에 쌓이므로
    표를 손으로 적지 않고 디렉토리를 훑어서 만든다 — 안 그러면 표가 곧 거짓말이 된다.
    """
    known = {
        f"{pkg}_opset{meta['opset_conversion']['to']}_fp32.onnx": (
            "중간물 — 원본을 opset 12→13 변환 + `quant_pre_process` 한 것. **FP32 그대로**이고 "
            "양자화의 **입력**이다. 배포물이 아니며 지워도 재실행하면 다시 생긴다"),
        f"{pkg}_metadata.json": (
            "기계 판독용 — 설정 · SHA256 · 그래프 입출력 · 캘리브 표본 · FP32 대비 비교 결과"),
        f"{pkg}_README.md": "이 문서",
        f"{pkg}_calib_manifest.txt": (
            f"캘리브레이션에 **실제로 쓴** 파일·프레임 번호 {meta['calibration']['frames']}줄 (재현용)"),
    }
    for v in meta["variants"]:
        ep = ("NNAPI · Core ML · ORT CPU EP" if v["per_channel"]
              else "스냅드래곤 **QNN EP(HTP)**")
        known[v["file"]] = (
            f"★ **배포 후보 — `{v['preset']}` 프리셋** · QDQ · "
            f"{'per-channel' if v['per_channel'] else 'per-tensor'} · "
            f"가중치 {v['weight_type']} / 활성 {v['activation_type']} · 대상 {ep}")

    def size(f: Path) -> str:
        n = f.stat().st_size
        return f"{n / 1e6:.2f} MB" if n >= 1e5 else f"{n / 1e3:.1f} KB"

    rows = []
    for f in sorted(outdir.iterdir()):
        if f.is_dir():
            continue
        note = known.get(f.name)
        if note is None:
            if "realnight" in f.name:
                # `<pkg>_<판>_realnight.json` — 어느 판의 수치인지 이름에서 되짚는다
                variant = f.stem.removeprefix(f"{pkg}_").removesuffix("_realnight")
                note = (f"`eval_real_night.py` 산출 — **{variant}** 판의 "
                        "야간 음성·양성 장면 대조 원자료")
            else:
                note = "부수 산출물 — 이 스크립트가 만든 것이 아니다"
        rows.append(f"| `{f.name}` | {note} | {size(f)} |")
    readme = f"{pkg}_README.md"
    if readme not in {f.name for f in outdir.iterdir() if f.is_file()}:
        rows.append(f"| `{readme}` | {known[readme]} | — |")
    return chr(10).join(rows)


def collapse_section(meta: dict) -> str:
    """`collapse_check` 가 실려 있으면 '이 판을 쓸 수 있는가' 절을 낸다.

    양자화가 **파일로는 성공하고 예측으로만 죽는** 경우가 있어서(→ 함정 20) 크기·정합
    수치만으로는 판단이 안 된다. 같은 7장에 실제로 태워 본 표가 판정의 근거다.
    """
    c = meta.get("collapse_check")
    if not c:
        return ""

    head = "| 이미지 | " + " | ".join(c["columns"]) + " |"
    sep = "|---|" + "---|" * len(c["columns"])
    body = chr(10).join("| `" + r[0] + "` | " + " | ".join(r[1:]) + " |" for r in c["rows"])
    notes = chr(10).join("- " + n for n in c.get("notes", []))
    probe = c.get("probe")
    probe_block = ""
    if probe:
        probe_block = (chr(10) + "**임계를 내려 보면** — 못 보는 것(`blind`)과 임계 아래"
                       "(`lowconf`)는 다른 고장이다." + chr(10) * 2
                       + chr(10).join("    " + line for line in probe) + chr(10))

    mark = {"passed": "✅ **통과**", "collapsed": "🔴 **붕괴 — 채택하지 않는다**",
            "partial": "🟡 **부분 붕괴 — 채택하지 않는다**"}[c["verdict"]]

    return f"""
### 붕괴 검사 — 이 판을 쓸 수 있는가

{c['source']} · conf **{c['conf']}** · letterbox **{c['letterbox']}** ({c['when']} 측정).
칸은 **검출 개수 · 클래스와 confidence** 다.

{head}
{sep}
{body}

**판정** — {mark}

{notes}
{probe_block}"""


def write_readme(path: Path, meta: dict) -> None:
    src = meta["source"]
    pkg = meta["package"]
    rows = []
    for v in meta["variants"]:
        cmp_ = v.get("compare_with_fp32") or {}
        rows.append(
            f"| `{v['file']}` | {v['preset']} | "
            f"{'per-channel' if v['per_channel'] else 'per-tensor'} | "
            f"{v['size_mb']} MB | "
            f"{cmp_.get('count_mismatch', '-')} | "
            f"{cmp_.get('max_xy_diff_px', '-')} | {cmp_.get('max_conf_diff', '-')} |")

    bench = "\n".join(
        f"| `{v['file']}` | {v.get('cpu_ms_ref', '-')} |" for v in meta["variants"])

    path.write_text(f"""# `{src['run_name']}` — INT8 양자화 산출물

> 만든 날 **{meta['created_on']}** · 원본 `{src['file']}` (FP32 · {src['size_mb']} MB)
> 🔴 **이것은 배포 확정판이 아니다** — 캘리브 표본이 {meta['calibration']['frames']}프레임
> (한 장소·한 밤)이고 mAP·recall 정식 판정을 하지 못했다. 아래 5장.

## 1. 무엇이 들어 있나

이 디렉토리의 **모든 파일**이다. ★ 표시만 앱에 넘길 배포 후보이고 나머지는 중간물·기록이다.

| 파일 | 무엇 | 크기 |
|---|---|---|
{file_rows(meta, path.parent, pkg)}

### FP32 대비 정합 — 같은 이미지에 두 모델을 먹여 박스를 짝지은 결과

| 파일 | 프리셋 | 가중치 스케일 | 크기 | 검출수 불일치 | 좌표 최대오차(px) | conf 최대오차 |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

- **`generic`** — per-channel 가중치 / uint8 활성. NNAPI · Core ML · ORT CPU EP 대상
- **`qnn`** — per-tensor uint8. 스냅드래곤 **QNN EP(HTP)** 가 요구하는 형태
{collapse_section(meta)}

⚠️ **CPU 판·GPU 판이 아니다.** QDQ ONNX 는 런타임 중립이라 같은 파일을 여러 EP 가 읽는다.
두 판의 차이는 EP 계열이 요구하는 **양자화 형식**이다. GPU 는 INT8 의 짝이 아니며
(폰에 TensorRT 없음 → hardware_inference.md 1장), GPU delegate 를 쓸 거면 FP16 이다.

## 2. 입출력 계약 — 원본과 같다

```
images  : float32 [1, 3, {meta['imgsz']}, {meta['imgsz']}]   (NCHW · RGB · /255)
output0 : float32 [1, 7, 8400]
```

전처리·후처리·NMS·좌표 역변환은 **FP32 판과 완전히 동일**하다
(→ `outputs/export/{src['run_name']}/README.md`). 앱이 고칠 것은 없다.

## 3. 어떻게 만들었나

```
opset 12 → 13 (version_converter)   per-channel QDQ 는 QuantizeLinear 의 axis 가 필요
  → quant_pre_process               shape inference + 그래프 정리
    → quantize_static (QDQ)         캘리브 {meta['calibration']['frames']}프레임 · **Conv 만**
      → metadata_props 복사          names·stride·task 유지
```

🔴 **`Conv` 만 양자화한 것은 취향이 아니라 필수다.** 전부 양자화하면 **검출이 0 이 된다** —
머리의 마지막 `Concat`(`/model.23/Concat_3`)이 박스 좌표(0~640)와 클래스 점수(0~1)를 한
텐서로 합치는데, 이걸 per-tensor uint8 로 재면 스케일이 640/255 ≈ 2.5 라 **점수가 전부
0 으로 반올림**된다. 7장 전부에서 박스가 사라지는 것을 실측했다. Conv 만 두면 연산량의
대부분은 INT8 로 가면서 머리의 산술은 float 로 남아 이 붕괴가 없다.

캘리브 소재는 **야간 실촬영** `{meta['calibration']['source']}` 이고 전처리는 배포와 같은 자
(**square letterbox {meta['imgsz']}** · RGB · /255 · NCHW → STATUS 3장 함정 18)다.
실제로 쓴 파일·프레임 목록은 `{meta['calibration']['manifest']}` 에 있다.

재현:

```powershell
uv run python scripts/quantize_onnx.py --calibrate-method {meta['calibration']['method']}
```

## 4. CPU EP 지연 — ⚠️ 참고치일 뿐이다

| 파일 | 1회 추론(ms) |
|---|---|
| `{src['file']}` (FP32) | {meta.get('fp32_cpu_ms_ref', '-')} |
{bench}

🔴 **이 PC 는 15~25ms 를 판정할 수 없다** (프로세스 간 10~20% 변동 → STATUS 3장 함정 3).
게다가 데스크톱 x86 CPU EP 의 INT8 배율은 **폰 NPU 의 배율과 다르다.**
속도 판정은 `C11` 실기기 계측이다.

## 5. 🔴 아직 검증되지 않은 것

- **mAP·recall 을 재지 못했다** — held-out(NightOwls rec34)·`detect_v3` val 이 이 PC 에 없다.
  학습 PC 또는 `C5` 하네스(`scripts/eval_own_night.py`)로 판정할 것.
- **캘리브 표본이 얇다** — {meta['calibration']['frames']}프레임이고 전부 **한 장소·한 밤·세로
  촬영**이다. square letterbox 의 회색 패딩이 좌우에만 생기는 쪽으로 치우쳐 있다.
  정식 캘리브는 `C2` 촬영분이나 학습 PC 의 `detect_v3` 로 다시 구울 것.
- **QNN 판은 실기기에서 그래프가 통째로 올라가는지 확인되지 않았다** — op 하나가 안 올라가면
  CPU fallback 이라 이득이 사라진다. `C10` 에서 확인할 것.
""", encoding="utf-8")


# --------------------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    # `relative_to(ROOT)` 로 출력하므로 상대경로 인자를 먼저 절대경로로 만든다
    args.onnx = args.onnx.resolve()
    args.out = args.out.resolve()
    args.calib_src = args.calib_src.resolve()
    if not args.onnx.is_file():
        raise SystemExit(f"원본 ONNX 가 없다: {args.onnx}")
    presets = [p.strip() for p in args.preset.split(",") if p.strip()]
    for p in presets:
        if p not in PRESETS:
            raise SystemExit(f"모르는 프리셋: {p} (가능: {', '.join(PRESETS)})")

    run_name = args.onnx.parent.name
    # 모델마다 한 칸을 쓴다 — 같은 루트에 다른 모델의 양자화 결과가 같이 살 수 있게.
    # 디렉토리·파일이 전부 `<원본>-INT8` 로 시작해서 한 파일만 옮겨도 8비트 판인 줄 안다.
    opset_target = 21 if args.bits == 4 else 13
    pkg = args.name or f"{args.onnx.stem}-INT{args.bits}"
    outdir = args.out / pkg
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"원본  {args.onnx.relative_to(ROOT)}  ({mb(args.onnx)} MB)")

    # 1) 캘리브 표본
    frames, manifest = collect_calib_frames(
        args.calib_src, args.imgsz, args.video_stride, args.include_synthetic)
    if not frames:
        raise SystemExit(f"캘리브 표본을 한 장도 못 읽었다: {args.calib_src}")
    print(f"캘리브 {len(frames)}프레임  ({args.calib_src.relative_to(ROOT)}"
          f" · 영상 stride {args.video_stride}"
          f"{' · 합성 포함' if args.include_synthetic else ''})")
    (outdir / f"{pkg}_calib_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8")

    # 2) opset 12 → 13 + 전처리
    src13 = outdir / f"{pkg}_opset{opset_target}_fp32.onnx"
    conv = to_opset(args.onnx, src13, opset_target)
    print(f"opset 12→{opset_target}  노드 {conv['nodes_before']} → {conv['nodes_after_convert']}"
          f" → {conv['nodes_after_preprocess']} (전처리 후)")

    import onnxruntime as ort
    input_name = ort.InferenceSession(
        str(src13), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    # 3) 프리셋별 양자화
    variants = []
    for preset in presets:
        dst = outdir / f"{pkg}{FILE_SUFFIX[preset]}.onnx"
        print(f"\n[{preset}] {args.bits}비트 양자화 중… ({args.calibrate_method}"
              + (f" · block {args.block_size}" if args.bits == 4 and args.block_size
                 else "") + ")")
        info = quantize(preset, src13, dst, frames, input_name,
                        args.calibrate_method, args.bits, args.block_size)
        restored = copy_metadata(args.onnx, dst)
        info.update({
            "file": dst.name, "size_mb": mb(dst), "sha256": sha256(dst),
            "graph": graph_io(dst),
            "metadata_props_restored": restored,
        })
        print(f"        → {dst.name}  {info['size_mb']} MB"
              f"  (원본 대비 {mb(args.onnx) / info['size_mb']:.2f}배 작음)")
        if restored:
            print(f"        metadata 복원: {', '.join(restored)}")
        variants.append(info)

    # 4) FP32 ↔ INT8 박스 비교
    images = sorted(args.calib_src.glob("*.jpg"))[:args.check_n] if args.check_n else []
    if images:
        print(f"\nFP32 ↔ INT8 박스 비교  ({len(images)}장 · conf {args.conf})")
        for v in variants:
            v["compare_with_fp32"] = parity_check(
                args.onnx, outdir / v["file"], images,
                args.imgsz, args.conf, args.iou)
            c = v["compare_with_fp32"]
            print(f"  {v['preset']:8s} 검출수 불일치 {c['count_mismatch']}/{c['images']}"
                  f" · 박스 {c['boxes_compared']}개"
                  f" · 좌표 최대 {c['max_xy_diff_px']}px"
                  f" · conf 최대 {c['max_conf_diff']}")

    # 5) CPU EP 지연 — 참고치
    fp32_ms = bench_cpu(args.onnx, args.imgsz, args.bench_runs)
    if fp32_ms is not None:
        print(f"\nCPU EP 지연 ⚠️ 참고치 (함정 3 — 이 PC 는 판정 불가)")
        print(f"  FP32     {fp32_ms} ms")
        for v in variants:
            v["cpu_ms_ref"] = bench_cpu(outdir / v["file"], args.imgsz, args.bench_runs)
            print(f"  {v['preset']:8s} {v['cpu_ms_ref']} ms"
                  f"  ({fp32_ms / v['cpu_ms_ref']:.2f}배)")

    # 6) 산출물 문서
    meta = {
        "project": "밤마실 (bammasil) — 야간 보행 AI 시각보조",
        "stage": "③ 위험요소 탐지 · INT8 양자화 (C10a)",
        "created_on": date.today().isoformat(),
        "package": pkg,
        "imgsz": args.imgsz,
        "precision": f"INT{args.bits}",
        "bits": args.bits,
        "source": {
            "file": args.onnx.name, "run_name": run_name,
            "path": str(args.onnx.relative_to(ROOT)),
            "size_mb": mb(args.onnx), "sha256": sha256(args.onnx),
            "graph": graph_io(args.onnx),
        },
        "opset_conversion": {"from": 12, "to": opset_target, **conv},
        "calibration": {
            "source": str(args.calib_src.relative_to(ROOT)),
            "frames": len(frames),
            "video_stride": args.video_stride,
            "include_synthetic": args.include_synthetic,
            "method": args.calibrate_method,
            "preprocess": "square letterbox 640 · RGB · /255 · NCHW (배포와 동일)",
            "manifest": f"{pkg}_calib_manifest.txt",
        },
        "fp32_cpu_ms_ref": fp32_ms,
        "variants": variants,
        "not_verified": [
            "mAP·recall 정식 판정 — held-out(rec34)·detect_v3 val 이 로컬에 없다",
            "캘리브 표본이 한 장소·한 밤·세로 촬영으로 치우쳐 있다",
            "QNN 판이 실기기에서 CPU fallback 없이 올라가는지 미확인",
            "속도는 실기기(C11)에서만 판정한다 — 위 CPU 배율은 참고치",
        ],
        "environment": {
            "onnxruntime": ort.__version__,
            "python": sys.version.split()[0],
        },
    }
    (outdir / f"{pkg}_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(outdir / f"{pkg}_README.md", meta)

    print(f"\n산출  {outdir.relative_to(ROOT)}/")
    for f in sorted(outdir.iterdir()):
        print(f"        {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
