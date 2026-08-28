"""③ 배포 ONNX → **FP8 양자화 실증 실험** (일회성 · 배포 후보가 아니다).

`docs/hardware_inference.md`가 모바일 경로의 정밀도를 이미 "NPU(QNN/NNAPI) → INT8 ·
폰 GPU delegate → FP16"으로 정했고, FP8 실행 경로는 어느 모바일 런타임 문서에도 없다.
이 스크립트는 "그래도 실제로 돌려보면 뭐가 나오는가"를 코드로 확인해 **기각 판단의
실증 근거**를 남기는 용도다 — `scripts/roi_crop_eval.py`(`C4e` E3 기각)와 같은 성격.

`scripts/quantize_onnx.py`·`scripts/export_onnx.py`의 기존 헬퍼를 그대로 재사용한다.
INT8/INT4 배포 패키지용 본선 스크립트는 건드리지 않는다 — metadata.json/README.md 같은
패키징도 만들지 않고, 콘솔 출력 그대로 옮긴 `result.md` 하나만 남긴다.

세 가지를 확인한다:
    A. generic(CPU) 경로 · Conv 만 양자화 — INT8 채택판과 같은 범위로 FP8 이 변환되는가
    B. generic(CPU) 경로 · 전체 그래프 양자화 — FP8 은 동적 범위가 넓어 함정 20의
       Concat 붕괴(박스 0~640 vs 점수 0~1 스케일 충돌)가 없을 수도 있다 — 그 여부 자체가 정보
    C. QNN(스냅드래곤 Hexagon HTP) 경로 · `get_qnn_qdq_config` — 정적 분석으로는 FP8
       지원이 없어 보이는데, 실제로 config 빌드 단계에서 무엇이 나는지 실행으로 확인

float8 QuantizeLinear/DequantizeLinear 는 ONNX **opset 19** 부터 받는다
(INT8 은 13, INT4 는 21 — 양자화 비트마다 필요한 opset 이 다르다).

사용:
    uv run python scripts/fp8_quant_experiment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_onnx import parity_check                       # noqa: E402
from quantize_onnx import (CALIB_SRC, SRC_ONNX, collect_calib_frames,   # noqa: E402
                           make_reader, mb, to_opset)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/quantization"
OPSET_TARGET = 19


def try_generic(src19: Path, frames, input_name: str, dst: Path, op_types) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    try:
        quantize_static(
            str(src19), str(dst), make_reader(frames, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            per_channel=False,
            # float 양자화는 MinMax/Percentile/Entropy 를 안 받는다 — ORT 가 이렇게
            # 못박아 뒀다(실측: "Only Distribution calibration method is supported
            # for float quantization."). INT8/INT4 캘리브 방식과 다른 지점이다.
            calibrate_method=CalibrationMethod.Distribution,
            op_types_to_quantize=op_types,
            calibration_providers=["CPUExecutionProvider"])
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def try_qnn(src19: Path, frames, input_name: str, dst: Path) -> dict:
    from onnxruntime.quantization import (CalibrationMethod, QuantType,
                                          quantize as quantize_with_config)
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

    try:
        cfg = get_qnn_qdq_config(
            str(src19), make_reader(frames, input_name),
            calibrate_method=CalibrationMethod.Distribution,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            per_channel=False, op_types_to_quantize=["Conv"],
            calibration_providers=["CPUExecutionProvider"])
    except Exception as e:
        return {"ok": False, "stage": "config", "error": f"{type(e).__name__}: {e}"}

    # config 빌드는 그래프 구조만 본다 — 실제 캘리브 실행·QDQ 삽입은 quantize() 단계에서
    # 일어나므로, QNN 전용 전처리가 FP8 을 실제로 다루는지는 여기까지 가 봐야 드러난다.
    try:
        quantize_with_config(str(src19), str(dst), cfg)
        return {"ok": True, "stage": "quantize", "error": None}
    except Exception as e:
        return {"ok": False, "stage": "quantize", "error": f"{type(e).__name__}: {e}"}


def write_result(outdir: Path, result: dict) -> None:
    (outdir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(att: dict) -> str:
        if not att.get("ok"):
            return f"🔴 실패 — `{att.get('error')}`"
        line = f"✅ 성공 — `{att.get('file')}` ({att.get('size_mb')} MB)"
        pc = att.get("parity_check")
        if pc and "error" not in pc:
            line += (f" · parity 검출수 불일치 {pc['count_mismatch']}/{pc['images']}"
                     f" · conf 최대오차 {pc['max_conf_diff']} · 통과 {pc['passed']}")
        elif pc:
            line += f" · parity 비교 중 예외: {pc['error']}"
        return line

    a = result["attempts"].get("A_generic_conv_only", {})
    b = result["attempts"].get("B_generic_full_graph", {})
    c = result["attempts"].get("C_qnn_config", {})
    conv = result["opset_conversion"]
    conv_line = ("✅ 성공" if conv["ok"]
                 else f"🔴 실패 — `{conv.get('error')}`")

    (outdir / "result.md").write_text(f"""# FP8 양자화 — 최소 실증 실험 결과

원본 `{result['source']['file']}` (FP32 · {result['source']['size_mb']} MB) ·
opset 12 → {result['opset_target']} 변환: {conv_line}

## 시도 A — generic(CPU) 경로 · Conv 만 양자화 (INT8 채택판과 같은 범위)

{fmt(a)}

## 시도 B — generic(CPU) 경로 · 전체 그래프 양자화

{fmt(b)}

## 시도 C — QNN(스냅드래곤 Hexagon HTP) 경로 · `get_qnn_qdq_config` → `quantize()`

{fmt(c) if c.get("ok") else f"🔴 {c.get('stage')} 단계에서 실패 — `{c.get('error')}`"}

## 판단

이 결과는 배포 후보를 만들기 위한 것이 아니다 — `docs/hardware_inference.md`가 이미
모바일 경로의 정밀도를 NPU(QNN/NNAPI) → INT8, 폰 GPU delegate → FP16으로 정해 두었고
FP8 실행 경로는 어느 모바일 런타임에도 없다. 이 실험은 그 판단을 실행으로 확인한
기록이다.

시도 C(QNN)가 **파일 생성에는 성공**해 "재검토 필요"로 남겼었지만, 그 파일을 실제로
돌려 보면(위 parity 비교) **A와 같은 이유로 실행이 막힌다** — `float8e4m3fn` 을 받는
`QLinearConv` 자체가 이 PC의 실행 공급자(CPU/CUDA)에는 없다. QNN(Hexagon NPU) 실기기가
있어야만 실행 가능 여부를 알 수 있고, 그 실기기가 없는 한 **파일이 만들어졌다는 것과
동작한다는 것은 다른 이야기**다.

**8비트를 제외한 나머지 압축안은 전부 기각됐다** — `INT4`는 정확도가 붕괴해서
(야간 볼라드 `_04` 2/2 → 0/2, `outputs/quantization/<pkg>-INT4/<pkg>-INT4_README.md`),
`FP8`은 이 PC에서 실행 자체가 안 돼서(위) 기각. 배포 후보는 `INT8`
(`outputs/quantization/<pkg>-INT8/`) 하나뿐이다. 최종 결론은 `docs/STATUS.md` 6장을 본다.
""", encoding="utf-8")


def main() -> int:
    onnx_path = SRC_ONNX
    outdir = OUT / f"{onnx_path.stem}-FP8_experiment"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"원본  {onnx_path.relative_to(ROOT)}  ({mb(onnx_path)} MB)")

    frames, _manifest = collect_calib_frames(CALIB_SRC, 640, stride=4, include_synthetic=False)
    if not frames:
        raise SystemExit(f"캘리브 표본을 한 장도 못 읽었다: {CALIB_SRC}")
    print(f"캘리브 {len(frames)}프레임")

    result: dict = {
        "source": {"file": onnx_path.name, "size_mb": mb(onnx_path)},
        "opset_target": OPSET_TARGET,
        "attempts": {},
    }

    # opset 12 → 19. float8 QuantizeLinear/DequantizeLinear 는 여기부터 들어왔다.
    # 이 변환 자체가 실패하면 이후 시도는 전부 의미가 없으므로 바로 끝낸다.
    src19 = outdir / f"{onnx_path.stem}_opset{OPSET_TARGET}_fp32.onnx"
    try:
        conv = to_opset(onnx_path, src19, OPSET_TARGET)
        print(f"opset 12→{OPSET_TARGET} 변환 성공 "
              f"(노드 {conv['nodes_before']} → {conv['nodes_after_preprocess']})")
        result["opset_conversion"] = {"ok": True, **conv}
    except Exception as e:
        result["opset_conversion"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"opset 변환 실패 — 이후 시도를 모두 건너뜀: {result['opset_conversion']['error']}")
        write_result(outdir, result)
        return 1

    import onnxruntime as ort
    input_name = ort.InferenceSession(
        str(src19), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    # 시도 A — generic 경로 · Conv 만
    dst_a = outdir / f"{onnx_path.stem}_fp8_generic_conv.onnx"
    print("\n[A] generic 경로 · Conv 만 양자화…")
    a = try_generic(src19, frames, input_name, dst_a, ["Conv"])
    if a["ok"]:
        a.update({"file": dst_a.name, "size_mb": mb(dst_a)})
        print(f"  성공 → {dst_a.name}  ({a['size_mb']} MB)")
    else:
        print(f"  실패: {a['error']}")
    result["attempts"]["A_generic_conv_only"] = a

    # 시도 B — generic 경로 · 전체 그래프
    dst_b = outdir / f"{onnx_path.stem}_fp8_generic_full.onnx"
    print("\n[B] generic 경로 · 전체 그래프 양자화…")
    b = try_generic(src19, frames, input_name, dst_b, None)
    if b["ok"]:
        b.update({"file": dst_b.name, "size_mb": mb(dst_b)})
        print(f"  성공 → {dst_b.name}  ({b['size_mb']} MB)")
    else:
        print(f"  실패: {b['error']}")
    result["attempts"]["B_generic_full_graph"] = b

    # 성공한 시도에 대해서만 FP32 대비 parity 비교 — 함정 20의 "야간 볼라드 2/2 유지" 신호
    images = sorted(CALIB_SRC.glob("*.jpg"))[:8]

    def check_parity(key: str, att: dict, dst: Path) -> None:
        if not (att["ok"] and images):
            return
        print(f"\n[{key}] FP32 대비 parity 비교 ({len(images)}장 · conf 0.25)…")
        try:
            cmp_ = parity_check(onnx_path, dst, images, 640, 0.25, 0.7)
            att["parity_check"] = cmp_
            print(f"  검출수 불일치 {cmp_['count_mismatch']}/{cmp_['images']}"
                  f" · 좌표 최대 {cmp_['max_xy_diff_px']}px · conf 최대 {cmp_['max_conf_diff']}"
                  f" · 통과 {cmp_['passed']}")
        except Exception as e:
            att["parity_check"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  parity 비교 중 예외: {att['parity_check']['error']}")

    for key, dst in (("A_generic_conv_only", dst_a), ("B_generic_full_graph", dst_b)):
        check_parity(key, result["attempts"][key], dst)

    # 시도 C — QNN 경로. config 빌드뿐 아니라 실제 quantize() 실행까지 밀어붙인다 —
    # QNN 전용 전처리가 FP8 을 실제로 다루는지는 그 단계에서 드러난다.
    dst_c = outdir / f"{onnx_path.stem}_fp8_qnn_qdq.onnx"
    print("\n[C] QNN(Hexagon HTP) 경로 · get_qnn_qdq_config → quantize() 시도…")
    c = try_qnn(src19, frames, input_name, dst_c)
    if c["ok"]:
        c.update({"file": dst_c.name, "size_mb": mb(dst_c)})
        print(f"  성공 → {dst_c.name}  ({c['size_mb']} MB) — QNN 경로가 FP8 을 받아들였다"
              " (예상과 다름, 재검토 필요)")
    else:
        print(f"  {c['stage']} 단계에서 실패: {c['error']}")
    result["attempts"]["C_qnn_config"] = c

    # config·quantize() 는 그래프를 만들 뿐 실행 가능성은 안 보여준다 — 실제로 돌려서 확인
    # (2026-08-28 실측: QNN 전용 그래프라 이 PC 의 CPU/CUDA EP 로는 실행 자체가 막힌다).
    check_parity("C_qnn_config", c, dst_c)

    write_result(outdir, result)
    print(f"\n산출  {outdir.relative_to(ROOT)}/")
    for f in sorted(outdir.iterdir()):
        print(f"        {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
