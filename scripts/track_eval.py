"""`C4e` S1 (E1) — **탐지 프레임 스킵의 대가**를 잰다 (`W6` 보간 · `P1` 평활).

왜 이것이 필요한가
    S0b 가 *일괄* conf 하향(0.25→0.10)을 기각했다 — 보행 시점 실야간 음성에서 `stairs`
    오탐이 2~3.7배, 발화 프레임이 6/15 → 14/15 가 된다(→ detection.md 9-6). 그 대가를
    흡수할 유일한 수단이 시간축이라, **일괄 하향을 되살릴 수 있는가가 여기 걸려 있다.**
    그리고 `C9` 통합은 `--detect-every` 로 프레임을 건너뛸 텐데 **그 대가를 잰 적이
    한 번도 없다.**

★ 없는 것은 "구현"이 아니라 **"재는 자"** 다
    hold(직전 박스 유지)는 `pipeline_demo.py:126-134` 에 **이미 있다**(`--detect-every`
    기본 2). `temporal_eval.py` 는 ② **화소** 지표만 내고 YOLO 를 호출조차 하지 않는다.
    그래서 이 스크립트는 **하네스 신설 + hold 대비 A/B** 다.

★ 설계의 핵심 — 탐지는 **오라클 한 번만** 돈다
    `--detect-every k` 는 "k 의 배수 프레임에서만 탐지 결과를 쓴다"는 뜻이다. 매 프레임
    추론을 한 번 돌려 두고(오라클) 조건마다 거기서 골라 쓰면 **재추론이 필요 없다.**
    런당 N회 추론으로 격자 전체가 끝난다 (YOLO predict 는 결정론적이라 성립한다).

⚠️ 지표는 전부 **오라클 대비**다 — GT 를 쓰지 않는다
    여기서 재는 것은 "매 프레임 탐지 대비 스킵·보간이 얼마나 손해인가"이지 "모델이
    얼마나 좋은가"가 아니다. GT 를 끌어들이면 **모델 성능과 시간축 손해가 섞인다** —
    STATUS 3장 함정 2("재는 대상이 딴것이 된다")와 같은 논리다.

⚠️ 소재는 rec 38 하나뿐이다
    rec 34 는 라벨 7,361 중 **해제가 1,191장**뿐이라 프레임이 끊겨 있어 시간축에 못 쓴다.
    rec 38 은 4,605 전량 해제돼 있다(16fps → 600프레임 ≈ 37초). rec 38 은 **학습 금지**
    지정이고 여기서는 측정에만 쓴다.

⚠️ 트래커는 **순수 파이썬 IoU** 다 — `model.track()` 을 쓰지 않는다
    ① 배포 ONNX 는 **NMS 조차 그래프 밖**이라 트래킹은 어차피 **앱 코드 레벨**이어야
      한다. 데스크톱에서만 되는 결과를 만들지 않는다.
    ② 이 환경에서 `model.track()` 은 애초에 못 돈다 — `ultralytics.trackers.utils.matching`
      이 `lap` 을 요구하는데 없고, `YOLO_AUTOINSTALL=false` 가 자동설치를 막는다
      (→ STATUS 3장 함정 9). `lapx` 추가는 의존성 변경이라 하지 않는다.

    제약과 올바른 설계가 일치한다.

사용:
    uv run python scripts/track_eval.py --frames 120 --detect-every 1 --interp none
    uv run python scripts/track_eval.py --frames 600 --detect-every 1,2,3 `
        --interp none,hold,track --videos
    uv run python scripts/track_eval.py --runs c4d_11n_640 --interp track `
        --detect-every 2 --smooth-alpha 0.0,0.3,0.6
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # → STATUS 3장 함정 9
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")         # → STATUS 3장 함정 12

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from emphasize import Detection, from_ultralytics  # noqa: E402
from temporal_eval import frame_paths, load_sequence  # noqa: E402

DETECT = ROOT / "outputs/detect"
VIDEO_SRC = ROOT / "data/test_real_data"

DEFAULT_RUNS = "c4b_loli0,c4d_11n_640"
MODES = ("none", "hold", "track")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=DEFAULT_RUNS,
                   help="outputs/detect 아래 run 이름 (쉼표 구분). 현 배포 · 채택 후보")
    p.add_argument("--recording", type=int, default=38,
                   help="연속 프레임 recording. **38 만 전량 해제돼 있다**")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--detect-every", default="1,2,3", help="탐지 주기 격자 (쉼표)")
    p.add_argument("--interp", default="none,hold,track",
                   help=f"보간 방식 격자 (쉼표). {'|'.join(MODES)}")
    p.add_argument("--smooth-alpha", default="0.0",
                   help="`P1` 박스 좌표 EMA 격자 (쉼표). 0 이면 평활 없음. track 에만 걸린다")
    p.add_argument("--conf", type=float, default=0.25, help="운영 conf (S0b 판정 유지값)")
    p.add_argument("--imgsz", type=int, default=640, help="배포 해상도 고정 전제")
    p.add_argument("--iou-match", type=float, default=0.5,
                   help="오라클 대비 매칭 임계")
    p.add_argument("--max-age", type=int, default=0,
                   help="track — 탐지 프레임에서 **못 붙은** 트랙을 살려 두는 횟수. "
                        "0 이면 즉시 폐기(= k=1 에서 오라클과 정확히 같아진다)")
    p.add_argument("--videos", action="store_true",
                   help="test_real_data mp4 5개에서 **오탐 지속 프레임 수**를 같이 잰다")
    p.add_argument("--video-max-frames", type=int, default=150, help="영상당 상한")
    p.add_argument("--out", type=Path, default=DETECT / "c4e_s1_e1_temporal.json")
    return p.parse_args()


# ---------------------------------------------------------------- 기하

def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match(pred: list[Detection], ref: list[Detection], thresh: float
          ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """클래스별 greedy IoU 매칭 → (쌍, 안 붙은 pred, 안 붙은 ref).

    greedy 로 두는 이유 — 헝가리안을 쓰려면 `lap`/`scipy` 의존이 들어오는데, 한 프레임의
    박스가 한 자릿수라 최적해와 갈릴 여지가 거의 없다. 의존성을 늘리지 않는 쪽을 택한다.
    """
    pairs: list[tuple[int, int, float]] = []
    used_p: set[int] = set()
    used_r: set[int] = set()
    cand = [(iou(p.box, r.box), i, j)
            for i, p in enumerate(pred)
            for j, r in enumerate(ref) if p.name == r.name]
    for v, i, j in sorted(cand, reverse=True):
        if v < thresh or i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        pairs.append((i, j, v))
    return (pairs,
            [i for i in range(len(pred)) if i not in used_p],
            [j for j in range(len(ref)) if j not in used_r])


# ---------------------------------------------------------------- 보간

class Track:
    """IoU 트랙 하나. 중심 속도로 스킵 구간을 외삽한다.

    ⚠️ **`misses` 는 '탐지 프레임에서 못 붙은 횟수'이지 경과 프레임 수가 아니다.**
    둘을 섞으면 스킵 구간의 외삽(정상 동작)과 탐지 실패(트랙 폐기 사유)가 구분되지
    않는다 — 그러면 `k=1` 에서도 스테일 박스가 남아 오라클을 재현하지 못한다.
    """

    __slots__ = ("name", "conf", "box", "vel", "misses")

    def __init__(self, d: Detection) -> None:
        self.name = d.name
        self.conf = d.conf
        self.box = tuple(float(v) for v in d.box)
        self.vel = (0.0, 0.0)
        self.misses = 0

    def observe(self, d: Detection, k: int) -> None:
        """탐지 프레임 — 중심 속도를 갱신하고 박스를 관측으로 맞춘다."""
        cx0, cy0 = center(self.box)          # type: ignore[arg-type]
        cx1, cy1 = center(d.box)
        self.vel = ((cx1 - cx0) / max(1, k), (cy1 - cy0) / max(1, k))
        self.box = tuple(float(v) for v in d.box)
        self.conf = d.conf
        self.misses = 0

    def coast(self) -> None:
        """스킵 프레임 — 속도로 박스를 통째로 민다 (크기는 유지)."""
        dx, dy = self.vel
        x1, y1, x2, y2 = self.box
        self.box = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def det(self) -> Detection:
        return Detection(self.name, self.conf,
                         tuple(int(round(v)) for v in self.box))  # type: ignore[arg-type]


def simulate(oracle: list[list[Detection]], k: int, mode: str,
             *, iou_thresh: float, max_age: int, alpha: float) -> list[list[Detection]]:
    """오라클 결과에서 (주기 k, 보간 mode) 조건의 산출 시퀀스를 만든다. 재추론 없음."""
    if mode == "none":
        return [oracle[i] if i % k == 0 else [] for i in range(len(oracle))]

    if mode == "hold":
        out: list[list[Detection]] = []
        last: list[Detection] = []
        for i in range(len(oracle)):
            if i % k == 0:
                last = oracle[i]
            out.append(last)
        return out

    if mode != "track":
        raise SystemExit(f"모르는 --interp: {mode} (가능: {'|'.join(MODES)})")

    # max_age 0 = 탐지 프레임에서 못 붙은 트랙은 **즉시 폐기**. 그래야 k=1 이 오라클과
    # 정확히 같아져 자기검증이 성립한다. 값을 키우면 스테일 박스를 얼마나 오래 살려
    # 두는지가 `ghost_rate`·오탐 지속에 그대로 나타난다.
    tracks: list[Track] = []
    out = []
    for i in range(len(oracle)):
        if i % k == 0:
            dets = oracle[i]
            pairs, new_p, lost_r = match(
                dets, [t.det() for t in tracks], iou_thresh)
            for di, ti, _ in pairs:
                tracks[ti].observe(dets[di], k)
            for ti in sorted(lost_r, reverse=True):
                tracks[ti].misses += 1
                if tracks[ti].misses > max_age:
                    tracks.pop(ti)
            for di in new_p:
                tracks.append(Track(dets[di]))
        else:
            for t in tracks:
                prev = t.box
                t.coast()
                if alpha > 0.0:                      # `P1` — 좌표 EMA
                    t.box = tuple(alpha * p + (1 - alpha) * c
                                  for p, c in zip(prev, t.box))  # type: ignore[assignment]
        out.append([t.det() for t in tracks])
    return out


# ---------------------------------------------------------------- 지표

def evaluate(produced: list[list[Detection]], oracle: list[list[Detection]],
             thresh: float, k: int) -> dict:
    """전부 **오라클 대비**. GT 를 쓰지 않는다 (모듈 docstring 참고)."""
    ious: list[float] = []
    n_pred = n_ref = n_miss = n_ghost = 0
    for pred, ref in zip(produced, oracle):
        pairs, extra, missed = match(pred, ref, thresh)
        ious += [v for _, _, v in pairs]
        n_pred += len(pred)
        n_ref += len(ref)
        n_miss += len(missed)
        n_ghost += len(extra)

    # 지터 — 연속 프레임에서 같은 물체로 이어지는 박스의 중심 이동량 σ
    def step_sigma(seq: list[list[Detection]]) -> float:
        steps: list[float] = []
        for a, b in zip(seq[:-1], seq[1:]):
            for i, j, _ in match(b, a, thresh)[0]:
                (cx1, cy1), (cx0, cy0) = center(b[i].box), center(a[j].box)
                steps.append(((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5)
        return statistics.pstdev(steps) if len(steps) > 1 else 0.0

    # 깜빡임 — 박스가 있다/없다가 뒤집힌 횟수
    on = [bool(f) for f in produced]
    flip = sum(1 for a, b in zip(on[:-1], on[1:]) if a != b)
    on_ref = [bool(f) for f in oracle]
    flip_ref = sum(1 for a, b in zip(on_ref[:-1], on_ref[1:]) if a != b)

    # 반응 지연 — 오라클에 물체가 **새로** 나타난 프레임부터 산출에 뜰 때까지.
    #
    # ⚠️ **탐색을 주기 k 로 잘라야 한다.** 자르지 않으면 오라클 자신의 깜빡임(다음 탐지
    # 프레임에서 물체가 사라졌다가 수십 프레임 뒤 재등장)이 그대로 지연으로 잡혀,
    # "보간이 느리다"와 "탐지기가 불안정하다"가 섞인다 — 함정 2 와 같은 오염이다.
    # 주기 k 안에 못 뜨면 보간의 문제가 아니므로 `unresolved` 로 따로 센다.
    horizon = max(1, k)
    lat: list[int] = []
    unresolved = 0
    prev_ref: list[Detection] = []
    for i, ref in enumerate(oracle):
        _, appeared, _ = match(ref, prev_ref, thresh)
        for j in appeared:
            target = ref[j]
            for d in range(0, min(horizon, len(oracle) - i - 1) + 1):
                if any(p.name == target.name and iou(p.box, target.box) >= thresh
                       for p in produced[i + d]):
                    lat.append(d)
                    break
            else:
                unresolved += 1
        prev_ref = ref

    sig, sig_ref = step_sigma(produced), step_sigma(oracle)
    return {
        "iou_mean": round(statistics.fmean(ious), 4) if ious else 0.0,
        "miss_rate": round(n_miss / n_ref, 4) if n_ref else 0.0,
        "ghost_rate": round(n_ghost / n_pred, 4) if n_pred else 0.0,
        "jitter_sigma": round(sig, 2),
        "jitter_amp": round(sig / sig_ref, 3) if sig_ref > 0 else None,
        "flip": flip,
        "flip_ref": flip_ref,
        "latency": round(statistics.fmean(lat), 3) if lat else 0.0,
        "latency_unresolved": unresolved,
        "n_pred": n_pred, "n_ref": n_ref,
    }


def self_check(m: dict) -> list[str]:
    """`--detect-every 1` 은 **세 모드 전부** 오라클 자신이어야 한다 — 1차 게이트.

    `none` 은 매 프레임이 탐지 프레임이라 자명하고, `hold` 는 유지할 구간이 없으며,
    `track` 은 `--max-age 0` 이면 못 붙은 트랙을 즉시 버리므로 셋 다 오라클과 같다.
    셋을 한꺼번에 거는 것이 `none` 하나만 거는 것보다 훨씬 강한 검증이다 —
    보간 로직의 버그가 여기서 잡힌다.
    """
    bad = []
    if m["iou_mean"] != 1.0:
        bad.append(f"iou_mean={m['iou_mean']} (기대 1.0)")
    for key in ("miss_rate", "ghost_rate", "latency"):
        if m[key] != 0.0:
            bad.append(f"{key}={m[key]} (기대 0)")
    if m["jitter_amp"] not in (None, 1.0):
        bad.append(f"jitter_amp={m['jitter_amp']} (기대 1.0)")
    return bad


# ---------------------------------------------------------------- 음성 영상

def video_oracles(model, args) -> list[list[list[Detection]]]:
    """영상별 **매 프레임 오라클**. 본 경로와 같은 이유로 런당 한 번만 돈다."""
    out = []
    for v in sorted(VIDEO_SRC.glob("*.mp4")):
        cap = cv2.VideoCapture(str(v))
        frames = []
        while len(frames) < args.video_max_frames:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            continue
        out.append([from_ultralytics(r, min_conf=args.conf) for r in
                    model.predict(source=frames, conf=args.conf, imgsz=args.imgsz,
                                  device=0, stream=True, verbose=False)])
    return out


def video_false_runs(oracles, k: int, mode: str, args) -> dict:
    """`test_real_data` mp4 — 전부 음성이라 나온 박스는 정의상 오탐.

    ⚠️ 트래킹이 **오탐을 오래 살려두는 역효과**를 볼 수 있는 유일한 소재다.
    지연·recall 은 못 잰다 (양성이 없다).
    """
    runs: list[int] = []
    n_frame = n_hit = 0
    for oracle in oracles:
        produced = simulate(oracle, k, mode, iou_thresh=args.iou_match,
                            max_age=args.max_age, alpha=0.0)
        cur = 0
        for f in produced:
            n_frame += 1
            if f:
                n_hit += 1
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
    return {"n_frame": n_frame, "hit_frame": n_hit,
            "fp_runs": len(runs),
            "fp_run_mean": round(statistics.fmean(runs), 2) if runs else 0.0,
            "fp_run_max": max(runs) if runs else 0}


# ---------------------------------------------------------------- main

def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    ks = [int(v) for v in args.detect_every.split(",") if v.strip()]
    modes = [m.strip() for m in args.interp.split(",") if m.strip()]
    alphas = [float(a) for a in args.smooth_alpha.split(",") if a.strip()]

    weights = {}
    for r in runs:
        w = DETECT / r / "weights/best.pt"
        if not w.is_file():
            raise SystemExit(f"가중치가 없다: {w}")
        weights[r] = w

    paths = frame_paths(args.recording, args.start, args.frames)
    seq = load_sequence(paths, 0)        # 원 해상도 유지 — 탐지 경로는 원본을 먹는다
    if len(seq) < 3:
        raise SystemExit(f"프레임이 {len(seq)}장뿐이다 — recording {args.recording} 확인")

    print("=" * 96)
    print(f"E1 박스 시간축 — rec {args.recording} · {len(seq)}프레임 · "
          f"{seq[0].shape[1]}×{seq[0].shape[0]} · conf {args.conf}")
    print(f"   주기 {ks} × 보간 {modes} × EMA {alphas} · 매칭 IoU {args.iou_match}")
    print("   ⚠️ 지표는 전부 **오라클(매 프레임 탐지) 대비**다. GT 를 쓰지 않는다.")
    print("=" * 96)

    rows = []
    for run, w in weights.items():
        model = YOLO(str(w))
        print(f"\n[{run}] 오라클 추론 {len(seq)}프레임 …", flush=True)
        oracle = [from_ultralytics(r, min_conf=args.conf) for r in
                  model.predict(source=seq, conf=args.conf, imgsz=args.imgsz,
                                device=0, stream=True, verbose=False)]
        print(f"   오라클 박스 {sum(len(f) for f in oracle)}개 "
              f"/ 발화 {sum(1 for f in oracle if f)}프레임")
        vid_oracles = video_oracles(model, args) if args.videos else None

        for k in ks:
            for mode in modes:
                # EMA 는 track 에만 의미가 있다 — 나머지는 0 한 번만 돈다
                for alpha in (alphas if mode == "track" else [0.0]):
                    produced = simulate(oracle, k, mode, iou_thresh=args.iou_match,
                                        max_age=args.max_age, alpha=alpha)
                    m = evaluate(produced, oracle, args.iou_match, k)
                    rec = {"run": run, "detect_every": k, "interp": mode,
                           "smooth_alpha": alpha, **m}
                    if k == 1 and alpha == 0.0 and args.max_age == 0:
                        bad = self_check(m)
                        rec["self_check"] = "OK" if not bad else "; ".join(bad)
                        if bad:
                            raise SystemExit(
                                f"🔴 자기검증 실패 (interp={mode}) — 하네스가 틀렸다.\n"
                                "`--detect-every 1` 은 세 모드 전부 오라클 자신이어야 한다:\n  "
                                + "\n  ".join(bad))
                    if args.videos:
                        rec["video"] = video_false_runs(vid_oracles, k, mode, args)
                    rows.append(rec)

    # --- 표 1. 오라클 대비 ---
    print("\n" + "=" * 96)
    print("오라클(매 프레임 탐지) 대비 — 스킵·보간의 대가")
    print("=" * 96)
    hdr = (f"{'run':<20}{'주기':>5}{'보간':>7}{'EMA':>6}"
           f"{'IoU':>8}{'miss':>8}{'ghost':>8}{'지터σ':>9}{'증폭':>7}{'깜빡':>7}{'지연':>7}{'미해결':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<20}{r['detect_every']:>5}{r['interp']:>7}{r['smooth_alpha']:>6}"
              f"{r['iou_mean']:>8}{r['miss_rate']:>8}{r['ghost_rate']:>8}"
              f"{r['jitter_sigma']:>9}{str(r['jitter_amp']):>7}"
              f"{r['flip']:>5}/{r['flip_ref']:<2}{r['latency']:>7}"
              f"{r['latency_unresolved']:>8}")

    if args.videos:
        print("\n" + "=" * 96)
        print("음성 영상 (test_real_data mp4 5개 — 계단·볼라드·사람 없음) — 오탐 지속")
        print("=" * 96)
        hdr2 = (f"{'run':<20}{'주기':>5}{'보간':>7}"
                f"{'오탐프레임':>12}{'구간수':>8}{'평균지속':>10}{'최대':>7}")
        print(hdr2)
        print("-" * len(hdr2))
        seen = set()
        for r in rows:
            key = (r["run"], r["detect_every"], r["interp"])
            if key in seen:
                continue
            seen.add(key)
            v = r["video"]
            print(f"{r['run']:<20}{r['detect_every']:>5}{r['interp']:>7}"
                  f"{v['hit_frame']:>8}/{v['n_frame']:<4}{v['fp_runs']:>8}"
                  f"{v['fp_run_mean']:>10}{v['fp_run_max']:>7}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"recording": args.recording, "frames": len(seq), "conf": args.conf,
         "imgsz": args.imgsz, "iou_match": args.iou_match, "rows": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\n저장: {shown}")
    print("\n👁️ 읽는 법 — **`IoU` 와 `지터σ` 를 반드시 같이 볼 것.**")
    print("   hold 는 박스를 얼려서 지터가 *내려가는* 대신 IoU 가 무너진다.")
    print("   억제의 대가는 `지연` 에 나타난다 (→ temporal_eval.py 의 설계와 같은 꼴).")
    print("⚠️ 속도는 여기서 판정하지 않는다 — 이 PC 는 15~25ms 를 못 가른다(함정 3).")


if __name__ == "__main__":
    main()
