"""③ 탐지 가중치 → ONNX 내보내기 + 팀 공유 패키지 생성.

앱(`P3`)·모바일 런타임 담당자에게 넘길 **자기설명적인 한 덩어리**를 만든다.
`.onnx` 파일만 던지면 받는 쪽이 전처리·출력 레이아웃·임계값을 되물어야 하므로,
`metadata.json`(기계용) 과 `README.md`(사람용) 를 같이 굽고 zip 으로 묶는다.

    outputs/export/<name>/
        bammasil_det_<name>_<imgsz>.onnx
        metadata.json      ← 입출력 스펙·클래스·성능·체크섬
        README.md          ← 통합 담당자가 읽을 문서 (전처리/후처리 절차 포함)
    outputs/export/<name>.zip

왜 NMS 를 넣지 않는가 (`--nms` 로 켤 수는 있다)
    NMS 를 그래프에 심으면 `NonMaxSuppression` 연산자 때문에 **QNN/NNAPI/Core ML 에서
    그래프가 통째로 CPU 로 떨어지거나 변환 자체가 거부**된다 (→ hardware_inference.md 1장).
    모바일은 "순수 conv 그래프는 NPU, NMS 는 앱 코드" 가 정석이라 기본값은 NMS 제외다.
    데스크톱에서 onnxruntime 으로 바로 굴려볼 용도라면 `--nms` 를 켜면 된다.

왜 고정 입력 크기인가 (`--dynamic` 으로 풀 수는 있다)
    NPU 컴파일러는 shape 이 고정이어야 그래프를 미리 배치한다. 동적 축을 남기면
    대부분 CPU fallback 이다. 카메라 프레임은 어차피 letterbox 로 정사각에 맞춰
    넣으므로 고정이 손해가 아니다.

왜 FP32 인가 (`--half` 로 바꿀 수는 있다)
    양자화는 **내보낸 뒤 타깃 런타임에서** 하는 것이 정석이다 (③ YOLO 는 INT8 PTQ 권장
    → hardware_inference.md 3장). 여기서 FP16 으로 구우면 INT8 캘리브레이션의 입력이
    이미 손실된 상태가 되고, CPU EP 에서는 FP16 이 오히려 느리다.

⚠️ **이 모델의 입력은 ②를 거치지 않은 원본 프레임이다.**
    ② 저조도 개선을 탐지 앞단에 붙이면 `stairs` 야간 오탐이 0.1% → 5.7% 로 되살아난다
    (`C7`, 8/2 → detection.md 7장). 표시 경로와 탐지 경로는 분리돼 있다.
    이 경고는 `metadata.json`·`README.md` 에도 그대로 실린다 — 받는 쪽이 문서를
    안 읽고 파이프라인을 이어붙이는 것이 이 프로젝트에서 가장 비싼 실수다.

정합성 검증 (`--check-n`, 기본 8장)
    내보내기가 "성공했다" 는 파일이 생겼다는 뜻일 뿐이다. 같은 이미지를 `.pt` 와
    `.onnx` 에 각각 넣어 **검출 개수·박스 좌표·신뢰도**를 비교하고, 어긋나면 종료
    코드를 0 이 아닌 값으로 낸다. 표본은 dataset val 에서 가져온다.

사용:
    uv run python scripts/export_onnx.py                      # 채택 가중치 · 640 · FP32
    uv run python scripts/export_onnx.py --imgsz 320          # 저해상 변형
    uv run python scripts/export_onnx.py --nms --no-zip       # 데스크톱 즉시 테스트용
    uv run python scripts/export_onnx.py --weights outputs/detect/c4b_loli6000/weights/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import zipfile
from datetime import date
from pathlib import Path

# ultralytics 는 ONNX 추론 시 `onnxruntime`(CPU 패키지)이 없으면 **제멋대로 설치한다**.
# 이 프로젝트는 `onnxruntime-gpu` 를 uv.lock 으로 고정해 두었고 둘은 같은 `onnxruntime`
# 모듈을 제공하므로, 자동 설치는 lock 밖 패키지를 끼워 넣어 환경 재현만 깨뜨린다.
# 임포트 전에 꺼야 한다 (ultralytics 가 import 시점에 이 값을 읽는다).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "outputs/detect/c4b_loli0/weights/best.pt"
VAL_IMAGES = ROOT / "outputs/datasets/detect_v2_loli0/images/val"
OUT = ROOT / "outputs/export"

# 채택 가중치 `c4b_loli0` 의 held-out 실측치 (NightOwls rec 34 · --drop-unlabeled-person).
# 개발 val 이 아니라 **판정 표준 조합**의 숫자다 → STATUS.md 1장 / detection.md 6-7.
#
# ⚠️ `c4b_loli6000`(0.691 / 0.625 / 오탐 0.0%) 의 수치를 여기 적지 말 것 — 두 런은
#    사실상 동점이지만 수치를 섞으면 "오탐 0.0%" 라는 과대 표기가 된다.
#    다른 가중치를 내보낼 때는 이 표가 맞지 않으므로 `--weights` 를 바꾸면 실측치도
#    같이 갈아야 한다 (`RUN_METRICS` 에 런 이름으로 등록).
RUN_METRICS = {
    "c4b_loli0": {
        "mAP50": 0.684, "mAP50_95": 0.331, "recall": 0.609, "precision": 0.787,
        "stairs_fp_rec34": 0.2, "stairs_fp_rec38": 0.0,
    },
    "c4b_loli6000": {
        "mAP50": 0.691, "mAP50_95": 0.358, "recall": 0.625, "precision": 0.777,
        "stairs_fp_rec34": 0.0, "stairs_fp_rec38": 0.0,
    },
    # ★ 첫 자체 3클래스 런 (2026-08-23 · C4e S3 → detection.md 11-3c).
    #   rec38 은 돌리지 않았다(`--skip-fp38`) — **0.0 으로 적지 말 것.** 미측정이다.
    "c4e_s3_11n": {
        "mAP50": 0.710, "mAP50_95": 0.354, "recall": 0.635, "precision": 0.787,
        "stairs_fp_rec34": 0.0, "stairs_fp_rec38": None,
    },
}
HELDOUT_COMMON = {
    "source": "NightOwls recording 34 (held-out, 학습 미사용) · person 클래스",
    "eval_cmd": "uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person",
    "note": "개발 val(mAP50 0.892)은 야간 실효 성능을 보지 못한다. 이 표를 쓸 것.",
}

WARN_NO_LOWLIGHT = (
    "이 모델의 입력은 ② 저조도 개선을 거치지 않은 **원본 프레임**이다. "
    "②(D1A1+bf)를 탐지 앞단에 붙이면 stairs 야간 오탐이 0.1% → 5.7% 로 57배가 된다 "
    "(C7, 2026-08-02 · rec 34 held-out). 횡단보도·차선 텍스처를 계단으로 오인하는 것이라 "
    "저시력 보행자가 반드시 지나는 곳에서 헛보게 된다. "
    "표시 경로(②→④)와 탐지 경로(원본→③)는 분리해서 구현할 것."
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--imgsz", type=int, default=640,
                   help="정사각 입력 한 변. 320 은 속도, 640 은 정확도")
    p.add_argument("--opset", type=int, default=12,
                   help="12 가 모바일 변환기 호환 폭이 가장 넓다")
    p.add_argument("--name", default=None, help="기본값은 가중치의 런 이름")
    p.add_argument("--nms", action="store_true",
                   help="그래프에 NMS 를 심는다 (모바일 비권장 — 위 docstring)")
    p.add_argument("--dynamic", action="store_true", help="배치·해상도 축을 동적으로")
    p.add_argument("--half", action="store_true", help="FP16 (모바일 배포에는 비권장)")
    p.add_argument("--no-simplify", dest="simplify", action="store_false",
                   help="onnxslim 그래프 정리를 건너뛴다")
    p.add_argument("--check-n", type=int, default=8,
                   help="PT↔ONNX 정합성 검증 표본 수. 0 이면 건너뜀")
    p.add_argument("--val-images", type=Path, default=VAL_IMAGES,
                   help="정합성 검증 표본 디렉토리. **모델의 학습 도메인과 맞출 것** — "
                        "도메인이 어긋나면 양쪽 다 검출 0개가 되어 검증이 무의미하게 통과한다")
    p.add_argument("--conf", type=float, default=0.35,
                   help="검증·권장 신뢰도 임계값 (pipeline_demo.py 와 동일)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 임계값")
    p.add_argument("--no-zip", dest="zip", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def arch_name(model) -> str:
    """가중치에서 아키텍처를 읽는다.

    하드코딩하면 다른 런을 내보낼 때 **문서가 조용히 거짓말을 한다** — 받는 쪽은
    metadata.json 을 보고 런타임을 고르므로 (YOLO26 은 NMS-free 라 후처리가 다르다)
    틀린 이름 하나가 통합 실패로 이어진다.
    """
    stem = Path(model.model.yaml.get("yaml_file", "")).stem or "unknown"
    n = sum(p.numel() for p in model.model.parameters())
    return f"{stem} (ultralytics · {n / 1e6:.2f}M params)"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def graph_io(onnx_path: Path) -> dict:
    """onnxruntime 세션에서 실제 입출력 스펙을 읽는다.

    문서에 손으로 적은 shape 은 옵션을 바꾸는 순간 거짓말이 된다. 받는 쪽이
    믿을 수 있도록 **내보낸 파일 자체에서** 뽑아 metadata 에 싣는다.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    return {
        "inputs": [{"name": i.name, "shape": i.shape, "dtype": i.type}
                   for i in sess.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.type}
                    for o in sess.get_outputs()],
        "embedded_metadata": {k: meta[k] for k in sorted(meta)},
    }


def letterbox(bgr, size: int, pad: int = 114):
    """종횡비 유지 축소 + 회색 패딩으로 정사각을 만든다.

    앱이 구현해야 할 전처리 1번과 **같은 것**이다 (README 로 같이 나간다).
    """
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    s = min(size / h, size / w)
    nh, nw = round(h * s), round(w * s)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    canvas = np.full((size, size, 3), pad, dtype=bgr.dtype)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=interp)
    return canvas


def parity_check(pt_path: Path, onnx_path: Path, images: list[Path],
                 imgsz: int, conf: float, iou: float) -> dict:
    """같은 이미지를 두 백엔드에 넣어 검출 결과를 비교한다.

    ⚠️ **파일 경로를 그대로 넘기면 안 된다** — ultralytics 는 `.pt` 에는 가변 rect
    레터박스(예: 16:9 → 640×384)를, 고정 shape ONNX 에는 640×640 을 적용한다.
    입력 해상도 자체가 달라 비정사각 이미지에서 신뢰도가 0.1 씩 벌어지는데, 이는
    변환 손실이 아니라 **전처리 차이**다. 그래프 변환만 재려면 미리 정사각으로
    레터박스한 배열을 양쪽에 똑같이 먹여야 한다.

    박스 순서로 짝짓지 않는다 — 신뢰도가 미세하게 흔들리면 NMS 정렬 순서가 뒤집혀
    같은 박스끼리 비교되지 않는다. 좌표가 가장 가까운 것끼리 짝짓는다.
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    pt, ox = YOLO(str(pt_path)), YOLO(str(onnx_path), task="detect")
    n_mismatch, max_xy, max_conf, n_boxes = 0, 0.0, 0.0, 0

    for img in images:
        src = cv2.imread(str(img))
        if src is None:
            continue
        square = letterbox(src, imgsz)
        kw = dict(imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)
        a = pt.predict(square, **kw)[0].boxes
        b = ox.predict(square, **kw)[0].boxes
        if len(a) != len(b):
            n_mismatch += 1
            continue
        if len(a) == 0:
            continue
        n_boxes += len(a)
        ax, bx = a.xyxy.cpu().numpy(), b.xyxy.cpu().numpy()
        ac, bc = a.conf.cpu().numpy(), b.conf.cpu().numpy()
        # 각 pt 박스에 대해 좌표가 가장 가까운 onnx 박스를 짝으로 본다
        for i in range(len(ax)):
            j = int(np.abs(bx - ax[i]).sum(axis=1).argmin())
            max_xy = max(max_xy, float(np.abs(ax[i] - bx[j]).max()))
            max_conf = max(max_conf, float(abs(ac[i] - bc[j])))

    return {
        "images": len(images),
        "count_mismatch": n_mismatch,
        "boxes_compared": n_boxes,
        "max_xy_diff_px": round(max_xy, 4),
        "max_conf_diff": round(max_conf, 5),
        # 좌표 1px·신뢰도 0.01 은 FP32 누적오차 수준. 그 이상은 변환이 뭔가 바꿨다는 뜻.
        "passed": n_mismatch == 0 and max_xy < 1.0 and max_conf < 0.01,
    }


def write_readme(path: Path, meta: dict) -> None:
    """받는 사람이 이 파일만 읽고 통합할 수 있게 쓴다."""
    io_ = meta["graph"]
    inp, outs = io_["inputs"][0], io_["outputs"]
    names = meta["classes"]
    cls_lines = "\n".join(f"| {i} | `{n}` |" for i, n in names.items())
    out_lines = "\n".join(
        f"| `{o['name']}` | `{o['shape']}` | `{o['dtype']}` |" for o in outs)

    hd = meta["heldout"]
    if hd.get("metrics"):
        m = hd["metrics"]
        perf_table = "\n".join([
            f"{hd['source']} 기준 (`{meta['model']['run_name']}`):", "",
            "| 지표 | 값 |", "|---|---|",
            f"| mAP50 | {m['mAP50']} |",
            f"| mAP50-95 | {m['mAP50_95']} |",
            f"| recall | {m['recall']} |",
            f"| precision | {m['precision']} |",
            f"| `stairs` 야간 오탐률 (rec 34) | {m['stairs_fp_rec34']}% |",
            # 미측정을 0.0% 로 보이게 하지 않는다 — 안 잰 것과 0 인 것은 다르다
            "| `stairs` 야간 오탐률 (rec 38) | "
            + (f"{m['stairs_fp_rec38']}% |" if m.get("stairs_fp_rec38") is not None
               else "**미측정** |"),
        ])
    else:
        perf_table = (
            f"⚠️ **이 런(`{meta['model']['run_name']}`)의 held-out 실측치가 등록돼 있지 않다.**\n"
            f"채택 런이 아닌 가중치를 내보냈다는 뜻이다. 성능을 인용하기 전에 아래로 직접 잴 것:\n\n"
            f"```powershell\n{hd['eval_cmd']}\n```")

    # ★ 출력 계약은 세 가지다. **문서에 하나를 하드코딩하면 반드시 거짓말이 된다.**
    #   1) --nms         : 그래프에 NMS 를 심은 것
    #   2) end-to-end    : YOLO26 계열. NMS-free 라 [1, max_det, 6] 으로 **이미 디코딩돼 나온다**
    #   3) raw           : YOLO11 계열 기본. [1, 4+nc, anchors] 원시 헤드 출력
    # 2와 3은 앱이 할 일이 정반대다 — 2에 NMS 를 또 돌리면 박스가 사라진다.
    shape0 = outs[0]["shape"]
    is_end2end = (not meta["export"]["nms"] and len(shape0) == 3
                  and shape0[-1] == 6 and shape0[1] != 4 + len(names))

    if meta["export"]["nms"]:
        nms_section = (
            "그래프에 **NMS 가 포함**돼 있다. 출력은 이미 정리된 박스 목록이므로\n"
            "앱에서 추가 NMS 를 돌리지 말 것.\n")
    elif is_end2end:
        nms_section = f"""### ⚠️ 이 모델은 **NMS-free (end-to-end)** 다 — YOLO11 계열과 계약이 다르다

`{meta['model']['arch']}` 는 NMS 없이 학습된 구조라, 그래프에 NMS 를 심지 않았는데도
출력이 **이미 디코딩·정리된 박스 목록**이다.

출력 `{outs[0]['name']}` 은 `{shape0}` 이며 축 구성은

    [batch, max_det={shape0[1]}, 6]
     └ 채널 0..3 : x1, y1, x2, y2  (letterbox 된 {meta['export']['imgsz']}×{meta['export']['imgsz']} 좌표계, 픽셀 단위)
     └ 채널 4    : 신뢰도 — **conf 내림차순으로 정렬돼 있다**
     └ 채널 5    : 클래스 id (float 로 들어온다. 반올림해서 정수로 쓸 것)

`cxcywh` 가 아니라 **`xyxy`** 다. 그리고 `max_det={shape0[1]}` 행은 **항상 채워져 나온다** —
검출이 적으면 남는 행이 conf 0 으로 패딩된다.

앱에서 할 일 (**3단계뿐이다**):
1. 채널 4 가 `conf`({meta['inference']['conf']}) 미만인 행을 버린다.
   정렬돼 있으므로 **처음으로 임계 미만인 행에서 끊으면 된다** — 전수 순회 불필요.
2. 채널 5 를 반올림해 클래스 id 로 쓴다.
3. letterbox 역변환 — 패딩 offset 을 빼고 scale 로 나눠 **원본 프레임 좌표**로 되돌린다.

> 🔴 **NMS 를 돌리지 말 것.** 이미 적용돼 있다. 한 번 더 돌리면 겹치는 정탐이 지워진다.
> YOLO11 계열용 후처리 코드를 그대로 재사용하면 이 사고가 난다 —
> 두 계열을 같이 비교할 때는 **모델별로 후처리 분기를 둘 것.**
> 대신 NNAPI 가 `NonMaxSuppression` 을 지원하지 않는 문제에서 자유롭다(그래프 분할 없음).
"""
    else:
        nms_section = f"""그래프에 **NMS 가 없다** (의도된 것 — 모바일 NPU 호환).
출력 `{outs[0]['name']}` 은 `{outs[0]['shape']}` 이며 축 구성은

    [batch, 4 + num_classes, num_anchors]
     └ 채널 0..3 : cx, cy, w, h  (letterbox 된 {meta['export']['imgsz']}×{meta['export']['imgsz']} 좌표계, 픽셀 단위)
     └ 채널 4..  : 클래스별 점수 (이미 sigmoid 적용 — 다시 씌우지 말 것)

앱에서 할 일:
1. 채널 4.. 의 최대값이 `conf` 미만인 앵커를 버린다
2. `cx,cy,w,h` → `x1,y1,x2,y2` 변환
3. 클래스별 NMS (IoU {meta['inference']['iou']})
4. letterbox 역변환 — 패딩 offset 을 빼고 scale 로 나눠 **원본 프레임 좌표**로 되돌린다
"""

    path.write_text(f"""# 밤마실 ③ 위험요소 탐지 — ONNX 배포 패키지

> 생성 {meta['exported_on']} · 모델 `{meta['model']['file']}`
> 문의 전에 이 문서를 끝까지 읽을 것. 특히 아래 ⚠️ 두 줄.

## ⚠️ 먼저 읽을 것

1. **입력은 원본 카메라 프레임이다.** {WARN_NO_LOWLIGHT}
2. **좌표는 letterbox 좌표계다.** 원본 프레임에 그대로 그리면 박스가 어긋난다.
   아래 후처리 4번을 반드시 구현할 것.

## 무엇인가

| 항목 | 값 |
|---|---|
| 과제 | 야간 보행 위험요소 탐지 ({len(names)} 클래스) |
| 아키텍처 | {meta['model']['arch']} |
| 입력 | `{inp['name']}` · `{inp['shape']}` · `{inp['dtype']}` |
| 파라미터 정밀도 | {meta['export']['precision']} |
| opset | {meta['export']['opset']} |
| 파일 크기 | {meta['model']['size_mb']} MB |
| SHA-256 | `{meta['model']['sha256']}` |

### 클래스

| id | 이름 |
|---|---|
{cls_lines}

## 전처리 (앱이 구현할 것)

카메라 프레임 → 모델 입력까지, **순서대로**:

1. **letterbox**: 종횡비를 유지한 채 긴 변을 `{meta['export']['imgsz']}` 에 맞춰 축소하고,
   남는 영역을 회색 `(114,114,114)` 으로 채워 `{meta['export']['imgsz']}×{meta['export']['imgsz']}` 정사각을 만든다.
   *늘려서 채우지 말 것* — 종횡비가 깨지면 사람 박스가 무너진다.
2. **BGR → RGB** (OpenCV 로 읽었다면 필요, 카메라 API 가 RGB 면 생략)
3. **0..255 → 0..1** (`/255.0`). 평균·표준편차 정규화는 **하지 않는다.**
4. **HWC → CHW**, 배치 축 추가 → `{inp['shape']}`, `{inp['dtype']}`

> 참고: 원본 PyTorch 가중치는 추론 시 **가변 rect 레터박스**(16:9 프레임이면 640×384)를
> 쓰지만 이 ONNX 는 정사각 고정이다. 세로 패딩만큼 연산이 늘고, 같은 프레임에 대한
> 신뢰도가 PyTorch 쪽과 소수점 둘째 자리에서 달라질 수 있다 — 정상이며, 고정 shape 을
> 택한 대가다(NPU 배치를 위해 필요). 정사각으로 통일해 재면 두 백엔드는 일치한다.

## 출력 · 후처리

| 이름 | shape | dtype |
|---|---|---|
{out_lines}

{nms_section}
### 권장 임계값

| 파라미터 | 값 | 근거 |
|---|---|---|
| `conf` | {meta['inference']['conf']} | 데스크톱 파이프라인 데모와 동일값 |
| `iou` | {meta['inference']['iou']} | 학습·평가 기본값 |

임계값을 바꾸면 성능표(아래)와 비교가 깨진다. 조정은 실기기 검증(`C11`) 뒤에.

## 성능 — 어떤 숫자를 믿을 것인가

{perf_table}

> {meta['heldout']['note']}

**아직 검증되지 않은 것**: 자체 촬영 야간분(`C2`/`C5`)이 없어 *보행 시점에서 멀리 있는
계단*은 측정된 적이 없다. 학습 데이터의 계단은 화면을 가득 채운 사진이라, 낮은 야간
오탐률이 "헛보지 않는다" 인지 **"거리 장면에선 아예 발화하지 않는다"** 인지 구분되지
않는다. `stairs` **재현율은 야간 거리 장면에서 한 번도 측정된 적이 없다.**
데모 시나리오를 짤 때 이 한계를 전제할 것.

또한 NightOwls 는 **차량 대시캠**이라 카메라 높이·속도·모션블러가 보행 시점과 다르다.
위 recall 이 폰을 들고 걷는 상황에서 그대로 재현된다는 보장은 없다.

## 런타임 배포 경로

| 플랫폼 | 실행 경로 |
|---|---|
| 안드로이드(스냅드래곤) | ONNX Runtime + QNN EP / NNAPI EP |
| 안드로이드(Exynos) | ONNX Runtime + NNAPI EP |
| iOS | ONNX Runtime + Core ML EP (또는 Core ML 재변환) |
| 개발 PC | ONNX Runtime CPU/CUDA EP |

**TensorRT 는 폰에서 동작하지 않는다.** Jetson 확장 단계 전용이다.
INT8 양자화는 이 FP32 파일을 입력으로 **타깃 런타임에서** 수행할 것.

## 무결성 확인

```powershell
Get-FileHash {meta['model']['file']} -Algorithm SHA256
# → {meta['model']['sha256']}
```

## 재현

```powershell
uv run python scripts/export_onnx.py {meta['export']['argv_hint']}
```

원본 가중치: `{meta['model']['source_weights']}` (SHA-256 `{meta['model']['source_sha256'][:16]}…`)
""", encoding="utf-8")


def main() -> None:
    args = parse_args()
    # 상대경로로 넘어오면 뒤의 relative_to(ROOT) 가 터진다. 입구에서 절대경로로 고정한다.
    args.weights = args.weights.resolve()
    args.val_images = args.val_images.resolve()
    if not args.weights.is_file():
        raise SystemExit(f"가중치가 없다: {args.weights}")

    name = args.name or args.weights.parent.parent.name
    dst_dir = OUT / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bammasil_det_{name}_{args.imgsz}"
    onnx_path = dst_dir / f"{stem}.onnx"

    print("=" * 78)
    print("③ 탐지 가중치 → ONNX 배포 패키지")
    print(f"   가중치 {args.weights}")
    print(f"   {args.imgsz}×{args.imgsz} · opset {args.opset} · "
          f"{'FP16' if args.half else 'FP32'} · NMS {'포함' if args.nms else '제외'} · "
          f"{'동적' if args.dynamic else '고정'} shape")
    print("=" * 78)

    import torch  # noqa: E402 — 무거워서 인자 검증 뒤에 임포트
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    produced = Path(model.export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=args.simplify,
        dynamic=args.dynamic, half=args.half, nms=args.nms, device="cpu"))

    # ultralytics 는 가중치 옆에 굽는다. outputs/export/ 로 옮겨 이름을 붙인다.
    if produced.resolve() != onnx_path.resolve():
        shutil.move(str(produced), onnx_path)
    print(f"\n내보냄: {onnx_path}")

    graph = graph_io(onnx_path)
    print(f"   입력  {graph['inputs'][0]['name']} {graph['inputs'][0]['shape']}")
    for o in graph["outputs"]:
        print(f"   출력  {o['name']} {o['shape']}")

    check = None
    if args.check_n > 0:
        if not args.val_images.is_dir():
            print(f"\n⚠️ 검증 표본 없음({args.val_images}) — 정합성 검증을 건너뛴다")
        else:
            pool = sorted(args.val_images.glob("*.jpg"))
            random.Random(args.seed).shuffle(pool)
            sample = pool[:args.check_n]
            print(f"\nPT↔ONNX 정합성 검증 — val 표본 {len(sample)}장 "
                  f"(CPU · {args.imgsz}×{args.imgsz} 레터박스 고정)")
            check = parity_check(args.weights, onnx_path, sample,
                                 args.imgsz, args.conf, args.iou)
            mark = "✅ 일치" if check["passed"] else "❌ 불일치"
            print(f"   {mark} · 개수 불일치 {check['count_mismatch']}/{check['images']}"
                  f" · 박스 {check['boxes_compared']}개"
                  f" · 좌표 최대 {check['max_xy_diff_px']}px"
                  f" · conf 최대 {check['max_conf_diff']}")

    argv_hint = f"--imgsz {args.imgsz} --opset {args.opset}"
    if args.nms:
        argv_hint += " --nms"
    if args.dynamic:
        argv_hint += " --dynamic"
    if args.half:
        argv_hint += " --half"
    def _rel(p: Path) -> str:
        """저장소 밖 경로(파생 데이터셋이 D: 에 있다)면 절대경로 그대로 적는다."""
        try:
            return p.relative_to(ROOT).as_posix()
        except ValueError:
            return p.as_posix()

    if args.weights != WEIGHTS.resolve():
        argv_hint += f" --weights {_rel(args.weights)}"
    if args.val_images != VAL_IMAGES.resolve():
        argv_hint += f" --val-images {_rel(args.val_images)}"
    if args.name:
        argv_hint += f" --name {args.name}"

    meta = {
        "project": "밤마실 (bammasil) — 야간 보행 AI 시각보조",
        "stage": "③ 위험요소 탐지",
        "exported_on": date.today().isoformat(),
        "model": {
            "file": onnx_path.name,
            "arch": arch_name(model),
            "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
            "sha256": sha256(onnx_path),
            "source_weights": args.weights.relative_to(ROOT).as_posix(),
            "source_sha256": sha256(args.weights),
            "run_name": name,
        },
        "classes": dict(enumerate(model.names[i] for i in sorted(model.names))),
        "export": {
            "imgsz": args.imgsz, "opset": args.opset, "nms": args.nms,
            "dynamic": args.dynamic, "simplify": args.simplify,
            "precision": "FP16" if args.half else "FP32",
            "argv_hint": argv_hint,
        },
        "preprocess": {
            "resize": "letterbox (종횡비 유지 + 회색 114 패딩)",
            "color": "RGB",
            "scale": "0..255 → 0..1 (/255)",
            "mean_std_normalization": None,
            "layout": "NCHW float32",
        },
        "inference": {"conf": args.conf, "iou": args.iou},
        # 런 이름에 등록된 실측치만 싣는다 — 다른 런의 숫자를 빌려 쓰면 과대 표기가 된다
        "heldout": {**HELDOUT_COMMON, "metrics": RUN_METRICS.get(name)},
        "graph": graph,
        "parity_check": check,
        "warning": WARN_NO_LOWLIGHT,
        "environment": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
        },
    }
    (dst_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(dst_dir / "README.md", meta)
    print(f"   metadata.json · README.md 생성")

    if args.zip:
        zip_path = OUT / f"{stem}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(dst_dir.iterdir()):
                if f.is_file():
                    z.write(f, f.name)
        print(f"\n공유 패키지: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")

    print("\n" + "=" * 78)
    print(f"⚠️ {WARN_NO_LOWLIGHT}")
    print("=" * 78)

    if check is not None and not check["passed"]:
        raise SystemExit("❌ PT↔ONNX 정합성 검증 실패 — 이 파일을 공유하지 말 것")


if __name__ == "__main__":
    main()
