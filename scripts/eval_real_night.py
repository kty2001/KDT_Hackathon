"""`C4e` S0b — 자체 실촬영 야간 소재에서 **오탐을 센다** (라벨 없이).

왜 이것이 필요한가
    `C4e` S0(→ detection.md 9장)에서 c4d 6런을 NightOwls rec 34 에 태워 3런이 게이트를
    통과했지만, **후보를 좁히지 못했다.** rec 34 는 1,001장에서 볼라드·계단 오탐이
    **0~2박스**뿐이라 *recall 은 가르지만 FP 는 못 가르기* 때문이다. 인수분이 `noneg`
    (배경 이미지 0%)를 "6런 중 오탐 최다"로 경계한 것도 **주간 6,988 볼라드 박스**에서
    나온 것이라 여기서는 검증되지 않는다.

    그런데 **라벨이 없어도 오탐은 셀 수 있다.** `data/test_real_data/` 7장 중 5장은
    계단도 볼라드도 없는 야간 노면이고(→ data.md 2-2-1), **사람은 7장 전부 0명**이다.
    즉 그 장면에서 나오는 예측은 **정의상 전부 오탐**이다. 이 저장소에서 유일한
    **보행 시점** 야간 소재라, 대시캠(rec 34)이 못 재는 축을 여기서 잰다.

무엇을 재는가
    후보 가중치 × 밝기 3단 × conf 격자로

      · 음성 5장의 `stairs`/`bollard` 오탐 박스 수 (프레임당)
      · 7장 전부의 `person` 오탐 (사람이 0명이므로 전부 오탐)
      · `_04`(야간 볼라드 2개) · `_05`(계단)에서의 탐지 수와 confidence

    밝기 3단은 `scripts/darken.py` 로 만든 `lowlight/`(gain 0.35) ·
    `lowlight_x02/`(gain 0.2) 다. ⚠️ **합성본이다** — 밝기 축 자체를 정량 근거로 쓰지
    말 것. 여기서 보는 것은 "어두워질수록 무너지는가"라는 **방향**이다.

★ `C4e` S1 (E2) — **클래스별 conf** (2026-08-23 추가)
    S0b 가 보인 것은 **두 클래스가 conf 를 정반대로 요구한다**는 것이다 (→ detection.md
    9-7 3-b). `bollard` 는 가까운 쪽이 0.321 까지 내려가 0.25 를 겨우 넘는데, `stairs` 는
    0.25 → 0.10 에서 오탐이 2~3.7배로 뛴다. **단일 임계로는 둘을 동시에 못 만족시킨다.**

    `--class-conf` 는 클래스마다 다른 임계를 건다. 구현은 **낮은 쪽으로 한 번 뽑고
    클래스별로 다시 거르는** 방식이다 — NMS 는 클래스별로 돌고 높은 conf 부터 남기므로,
    낮게 뽑아 뒤에서 거르는 것과 각 클래스를 제 임계로 돌리는 것이 **같은 결과**다.
    ⚠️ 단 `max_det`(기본 300)에 걸릴 만큼 후보가 쏟아지면 그 등가성이 깨진다.

    ✅ **`--class-conf` 를 안 주면 동작이 예전 그대로다** — 스칼라 격자에서는 최저값이
    곧 그 conf 라 필터가 무의미해진다. `c4e_s0b_realnight.json` 재현이 A/B 의 전제다.

⚠️ 이것은 판정이 아니라 단서다
    음성 표본이 **5장**(밝기 3단 합쳐 15프레임)뿐이다. 최종 판정은 자체 촬영 야간
    평가셋(`C5`)에서 한다. 다만 지금 가진 것 중 **도메인이 맞는 유일한 자료**다.

장면 구성의 출처
    라벨이 없으므로 아래 `SCENE` 은 **문서(data.md 2-2-1 · 8/13 육안 확인)에서 온 것**
    이지 파일에서 읽은 것이 아니다. 파일 구성이 바뀌면 즉시 실패하도록 장수를 검사한다.

사용:
    uv run python scripts/eval_real_night.py
    uv run python scripts/eval_real_night.py --runs c4d_11n_640,c4d_11n_640_noneg
    uv run python scripts/eval_real_night.py --conf 0.25,0.10,0.05
    uv run python scripts/eval_real_night.py --videos --video-stride 30

    # E2 — bollard 만 낮추고 stairs 는 0.25 로 묶는다 (여러 번 주면 격자가 된다)
    uv run python scripts/eval_real_night.py `
        --class-conf bollard=0.25 --class-conf bollard=0.15 --class-conf bollard=0.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # → STATUS 3장 함정 9
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nightowls_yolo import BOLLARD_ID, CLASS_NAMES, PERSON_ID, STAIRS_ID  # noqa: E402

SRC = ROOT / "data/test_real_data"
DETECT = ROOT / "outputs/detect"
BRIGHTNESS = {"원본": ".", "gain0.35": "lowlight", "gain0.20": "lowlight_x02"}

# 라벨이 없다 — 장면 구성은 data.md 2-2-1 에서 왔다. 접미사로 가른다.
SCENE = {"_04": "bollard", "_05": "stairs"}          # 나머지는 전부 음성
N_IMAGES = 7                                          # 바뀌면 SCENE 을 다시 확인할 것

DEFAULT_RUNS = "c4b_loli0,c4d_11n_640,c4d_11s_640,c4d_11n_640_noneg"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=DEFAULT_RUNS,
                   help="outputs/detect 아래 run 이름 (쉼표 구분)")
    p.add_argument("--conf", default="0.25,0.10,0.05",
                   help="운영 conf 후보 (쉼표 구분). 인수분 R1 의 무릎은 0.10")
    p.add_argument("--class-conf", action="append", metavar="SPEC",
                   help="★ E2 — 클래스별 임계. 예: 'bollard=0.10' 또는 "
                        "'person=0.25,stairs=0.25,bollard=0.10'. 여러 번 주면 격자가 된다. "
                        "안 적은 클래스는 --base-conf 를 쓴다. 주면 --conf 는 무시된다")
    p.add_argument("--base-conf", type=float, default=0.25,
                   help="--class-conf 에서 안 적은 클래스의 임계 (기본 0.25 = 현 운영값)")
    p.add_argument("--imgsz", type=int, default=640, help="배포 해상도 고정 전제")
    p.add_argument("--videos", action="store_true",
                   help="동영상 5개도 같이 센다 (전부 음성 — 계단·볼라드·사람 없음)")
    p.add_argument("--video-stride", type=int, default=30, help="동영상 N 프레임마다 1장")
    p.add_argument("--video-max-frames", type=int, default=20, help="동영상당 상한")
    p.add_argument("--out", type=Path, default=DETECT / "c4e_s0b_realnight.json")
    return p.parse_args()


NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def parse_class_conf(spec: str, base: float) -> dict[int, float]:
    """'bollard=0.10' · 'person=0.25,bollard=0.10' → {클래스ID: 임계}.

    안 적은 클래스는 `base` 를 쓴다 — 기본 0.25 는 **현 운영값**이라, 한 클래스만
    적으면 "그 클래스만 건드렸다"가 그대로 성립한다.
    """
    thr = {cid: base for cid in CLASS_NAMES}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--class-conf 형식이 아니다: {part!r} (예: bollard=0.10)")
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in NAME_TO_ID:
            raise SystemExit(
                f"모르는 클래스: {name!r}. 가능한 것 — {', '.join(NAME_TO_ID)}")
        thr[NAME_TO_ID[name]] = float(value)
    return thr


def thr_label(thr: dict[int, float]) -> str:
    """표·JSON 에 쓸 짧은 표기. 전부 같으면 스칼라처럼 보인다."""
    values = set(thr.values())
    if len(values) == 1:
        return f"{values.pop():g}"
    return "/".join(f"{CLASS_NAMES[c][:2]}{thr[c]:g}" for c in sorted(thr))


def scene_of(stem: str) -> str:
    for suffix, name in SCENE.items():
        if stem.endswith(suffix):
            return name
    return "negative"


def collect_images(sub: str) -> dict[str, list[Path]]:
    """밝기 하위 폴더의 이미지를 장면별로 묶는다."""
    d = SRC if sub == "." else SRC / sub
    if not d.is_dir():
        raise SystemExit(f"없음: {d}\ndarken.py 로 저조도 합성본을 먼저 만들 것")
    imgs = sorted(p for p in d.glob("*.jpg"))
    if len(imgs) != N_IMAGES:
        raise SystemExit(
            f"{d} 이미지가 {len(imgs)}장이다 (기대 {N_IMAGES}장). "
            "구성이 바뀌었으면 SCENE 매핑을 다시 확인할 것 — 라벨이 없어 자동 판별이 안 된다")
    out: dict[str, list[Path]] = {"negative": [], "bollard": [], "stairs": []}
    for p in imgs:
        out[scene_of(p.stem)].append(p)
    return out


def count(model, paths: list[Path], thr: dict[int, float], imgsz: int) -> dict:
    """클래스별 박스 수와 최대 confidence. 예측이 없는 클래스는 0 / None.

    `thr` 이 전부 같은 값이면 `min()` 이 곧 그 값이라 **예전 스칼라 동작과 동일**하다.
    """
    box = {PERSON_ID: 0, STAIRS_ID: 0, BOLLARD_ID: 0}
    best = {PERSON_ID: 0.0, STAIRS_ID: 0.0, BOLLARD_ID: 0.0}
    hit_img = 0
    for r in model.predict(source=[str(p) for p in paths], conf=min(thr.values()),
                           imgsz=imgsz, device=0, stream=True, verbose=False):
        if r.boxes is None or not len(r.boxes):
            continue
        cls = r.boxes.cls.tolist()
        cf = r.boxes.conf.tolist()
        touched = False
        for c, v in zip(cls, cf):
            c = int(c)
            if c not in box or float(v) < thr[c]:
                continue
            box[c] += 1
            best[c] = max(best[c], float(v))
            touched = True
        hit_img += int(touched)
    return {"n_img": len(paths), "hit_img": hit_img,
            "box": {CLASS_NAMES[k]: v for k, v in box.items()},
            "maxconf": {CLASS_NAMES[k]: (round(v, 3) if v else None)
                        for k, v in best.items()}}


def count_videos(model, sub: str, thr: dict[int, float], args) -> dict:
    """동영상은 전부 음성이다 — 나오는 것은 전부 오탐."""
    d = SRC if sub == "." else SRC / sub
    vids = sorted(d.glob("*.mp4"))
    box = {PERSON_ID: 0, STAIRS_ID: 0, BOLLARD_ID: 0}
    n_frame = hit = 0
    for v in vids:
        used = 0
        for i, r in enumerate(model.predict(source=str(v), conf=min(thr.values()),
                                            imgsz=args.imgsz, device=0, stream=True,
                                            verbose=False)):
            if i % args.video_stride:
                continue
            used += 1
            n_frame += 1
            if r.boxes is not None and len(r.boxes):
                touched = False
                for c, cv in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    c = int(c)
                    if c in box and float(cv) >= thr[c]:
                        box[c] += 1
                        touched = True
                hit += int(touched)
            if used >= args.video_max_frames:
                break
    return {"n_video": len(vids), "n_frame": n_frame, "hit_frame": hit,
            "box": {CLASS_NAMES[k]: v for k, v in box.items()}}


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    # conf 축 — 스칼라 격자(기존) 또는 클래스별 격자(E2). 섞지 않는다.
    if args.class_conf:
        grids = [parse_class_conf(s, args.base_conf) for s in args.class_conf]
    else:
        grids = [{cid: float(c) for cid in CLASS_NAMES}
                 for c in args.conf.split(",") if c.strip()]

    weights = {}
    for r in runs:
        w = DETECT / r / "weights/best.pt"
        if not w.is_file():
            raise SystemExit(f"가중치가 없다: {w}")
        weights[r] = w

    scenes = {name: collect_images(sub) for name, sub in BRIGHTNESS.items()}
    n_neg = len(next(iter(scenes.values()))["negative"])
    print(f"소재  {SRC}")
    print(f"      음성 {n_neg}장 · `_04` 볼라드 · `_05` 계단 × 밝기 {len(BRIGHTNESS)}단")
    print("      ⚠️ 라벨 없음 — 음성 장면의 예측은 전부 오탐. 사람은 7장 전부 0명이다.")
    if args.class_conf:
        print("★ 클래스별 conf (E2) — 안 적은 클래스는 --base-conf "
              f"{args.base_conf:g}")
        for thr in grids:
            print("      " + " · ".join(f"{CLASS_NAMES[c]} {thr[c]:g}"
                                        for c in sorted(thr)))

    rows = []
    for run, w in weights.items():
        model = YOLO(str(w))
        n_cls = len(model.names)
        for thr in grids:
            for bright in BRIGHTNESS:
                s = scenes[bright]
                rec = {"run": run, "nc": n_cls,
                       # `conf` 는 실제로 predict 에 들어간 값(=최저) — 기존 리더 호환
                       "conf": min(thr.values()), "conf_label": thr_label(thr),
                       "class_conf": {CLASS_NAMES[c]: thr[c] for c in sorted(thr)},
                       "brightness": bright,
                       "negative": count(model, s["negative"], thr, args.imgsz),
                       "bollard_scene": count(model, s["bollard"], thr, args.imgsz),
                       "stairs_scene": count(model, s["stairs"], thr, args.imgsz)}
                if args.videos:
                    rec["video"] = count_videos(model, BRIGHTNESS[bright], thr, args)
                rows.append(rec)
                print(f"  {run:<22}{bright:>9} conf {thr_label(thr):<18} 완료")

    # --- 표 1. 음성 장면 오탐 (라벨 불요 · 여기가 본론) ---
    print("\n" + "=" * 86)
    print(f"음성 {n_neg}장 — 계단·볼라드·사람이 **없는** 야간 노면. 나온 박스는 전부 오탐")
    print("=" * 86)
    hdr = f"{'run':<22}{'conf':>20}{'밝기':>10}{'stairs':>9}{'bollard':>9}{'person':>8}{'오탐장수':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        n = r["negative"]
        print(f"{r['run']:<22}{r['conf_label']:>20}{r['brightness']:>10}"
              f"{n['box']['stairs']:>9}{n['box']['bollard']:>9}{n['box']['person']:>8}"
              f"{n['hit_img']:>7}/{n['n_img']}")

    # --- 표 2. 양성 장면 — 잡히는가, 어두워지면 무너지는가 ---
    print("\n" + "=" * 86)
    print("양성 장면 — `_04` 야간 볼라드 2개 · `_05` 계단 (프로젝트 유일)")
    print("=" * 86)
    hdr2 = (f"{'run':<22}{'conf':>20}{'밝기':>10}"
            f"{'_04 bollard':>13}{'maxconf':>9}{'_05 stairs':>12}{'maxconf':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        b, st = r["bollard_scene"], r["stairs_scene"]
        print(f"{r['run']:<22}{r['conf_label']:>20}{r['brightness']:>10}"
              f"{b['box']['bollard']:>13}{str(b['maxconf']['bollard'] or '-'):>9}"
              f"{st['box']['stairs']:>12}{str(st['maxconf']['stairs'] or '-'):>9}")

    if args.videos:
        print("\n" + "=" * 86)
        print(f"동영상 (전부 음성 · {args.video_stride}프레임마다 1장 · 영상당 {args.video_max_frames}장 상한)")
        print("=" * 86)
        hdr3 = f"{'run':<22}{'conf':>20}{'밝기':>10}{'stairs':>9}{'bollard':>9}{'person':>8}{'오탐프레임':>12}"
        print(hdr3)
        print("-" * len(hdr3))
        for r in rows:
            v = r["video"]
            print(f"{r['run']:<22}{r['conf_label']:>20}{r['brightness']:>10}"
                  f"{v['box']['stairs']:>9}{v['box']['bollard']:>9}{v['box']['person']:>8}"
                  f"{v['hit_frame']:>9}/{v['n_frame']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"imgsz": args.imgsz, "n_negative": n_neg, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    # 상대경로로 --out 을 받으면 relative_to 가 터진다 — 표시용이라 실패해도 넘긴다
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\n저장: {shown}")
    print("\n⚠️ 음성 표본이 5장이다 — **판정이 아니라 단서**다. 최종 판정은 `C5`.")


if __name__ == "__main__":
    main()
