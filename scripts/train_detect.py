"""C4 — ③ 통합 탐지 baseline 학습 (`person` + `stairs` 단일 모델).

선행: `scripts/build_detect_dataset.py` 로 `outputs/datasets/detect_v1/` 생성.

왜 단일 모델인가
    계단 고전 CV 하이브리드가 정량 검증에서 기각(야간 재현율 0.232 / 오경보 100%)돼
    YOLO `stairs` 클래스 통합으로 전환했다 (→ data.md 4장). 클래스 1개 추가는 추론
    비용이 사실상 0 이라 ③ 단계 FPS 예산이 오히려 늘었다 (→ hardware_inference.md 2장).

왜 nano 인가
    배포 타깃과 일치시킨다. hardware_inference.md 2장의 FPS 예산은 **YOLOv8n/v11n +
    INT8** 을 전제로 잡혀 있고, ②가 이미 ~20ms 를 쓰는 상황이라 더 큰 모델은 예산을
    다시 계산해야 한다. baseline 의 목적은 최고 mAP 가 아니라
    **(1) ② 4-arm 판정의 채점기 확보 · (2) 실현가능성 확인** 이다.

클래스 불균형(person:stairs ≈ 8:1)에 가중치를 주지 않는다
    stairs 박스는 크고 쉬우며(프레임의 중앙 47.5%), person 박스는 작고 어렵다 —
    불균형 방향과 난이도 방향이 반대라 단순 가중치는 오히려 교란이다. **먼저 클래스별
    mAP 를 재고, 그 결과를 보고 결정한다** (프로젝트의 가설→측정→결정 절차).

⚠️ 이 val 의 mAP 를 과신하지 말 것 — **두 소스의 도메인이 분리돼 있다**
    LoLI(거리 장면)와 StairNet(계단 사진)은 시각적으로 겹치지 않는다. 모델이
    "계단처럼 생긴 사진이면 stairs" 라는 **장면 단서**만으로도 점수를 얻을 수 있고,
    val 이 같은 구성이라 그 지름길이 검출되지 않는다. 실제 판정은 **한 프레임에 보도·
    계단·보행자가 함께 있는 자체 촬영 야간분(`C5`)** 에서 해야 한다 (→ TODO.md).

★ C4b — NightOwls 를 학습에 투입 (2026-08-02)
    C4 는 NightOwls held-out(rec 34)에서 person mAP50 0.127 · recall 0.195 로 무너졌다.
    학습 데이터를 바꾼 두 런을 **같은 설정으로** 돌려 원인을 가린다.

        c4b_loli6000   LoLI 6,000 + NightOwls 4,819 + StairNet   (실야간 박스 26.9%)
        c4b_loli0      LoLI 없음  + NightOwls 4,819 + StairNet   (실야간 박스 100%)

    두 데이터셋은 **val 이 동일**하고 train 만 다르다. 판정은 개발 val 이 아니라
    held-out 인 NightOwls rec 34 에서 한다 (→ scripts/eval_nightowls.py).

⚠️ **Windows 에서 `--cache ram` 은 학습을 죽인다** (2026-08-02 실측)
    `c4b_loli0`(7,489장)에 `cache='ram'` 을 걸었더니 train 6.5GB + val 1.6GB 를 캐시한 뒤
    **epoch 1 검증 중 `DataLoader worker exited unexpectedly` 로 크래시**했다.
    Windows 는 worker 를 fork 가 아니라 **spawn** 으로 띄우므로 캐시를 담은 데이터셋이
    worker 마다 복제된다 — 8.1GB × (1+4) 로 메모리가 터진다.

    `c4b_loli6000`(14,089장)은 21.5GB 가 필요해 ultralytics 가 캐시를 **거부**한 덕에
    살아남았다. 즉 **데이터가 작을수록 위험하다** — 캐시가 되는 크기가 오히려 함정이다.

    디스크 병목은 캐시가 아니라 **NightOwls PNG→JPG 변환**(13.78 → 1.16 GiB)으로 이미
    해소했다 (→ scripts/build_detect_dataset.py). 기본값은 `none` 이다.

사용:
    uv run python scripts/train_detect.py                    # 기본 (YOLO11n, 100ep)
    uv run python scripts/train_detect.py --model yolo11s.pt
    uv run python scripts/train_detect.py --epochs 50 --batch 8
    uv run python scripts/train_detect.py --data outputs/datasets/detect_v2_loli6000/data.yaml \
        --name c4b_loli6000                                 # C4b 런1
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "outputs/datasets/detect_v1/data.yaml"
OUT = ROOT / "outputs/detect"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--model", default="yolo11n.pt", help="사전학습 가중치 (COCO)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20, help="조기종료 — 수렴하면 멈춘다")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16,
                   help="VRAM 8GB 기준. OOM 이면 8 로 내릴 것")
    p.add_argument("--workers", type=int, default=4, help="Windows 는 과도하면 불안정")
    p.add_argument("--cache", default="none", choices=["ram", "disk", "none"],
                   help="⚠️ Windows 에서 'ram' 을 쓰지 말 것 — 아래 주의 참고")
    p.add_argument("--name", default="c4_baseline")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"데이터셋이 없다: {args.data}\n"
            "먼저 실행할 것:\n"
            "  uv run python scripts/stairnet_to_bbox.py\n"
            "  uv run python scripts/build_detect_dataset.py")

    from ultralytics import YOLO  # noqa: E402 — 무거워서 인자 검증 뒤에 임포트

    print("=" * 78)
    print("C4 — ③ 통합 탐지 baseline 학습")
    print(f"   모델 {args.model} · {args.epochs}ep(patience {args.patience}) "
          f"· imgsz {args.imgsz} · batch {args.batch}")
    print(f"   데이터 {args.data}")
    print("=" * 78)

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        device=0,
        seed=args.seed,
        deterministic=True,
        project=str(OUT),
        name=args.name,
        exist_ok=True,
        plots=True,
        # 야간 도메인이라 색상 증강은 보수적으로 — 색 왜곡은 ②가 이미 만든다
        hsv_h=0.010, hsv_s=0.4, hsv_v=0.3,
        # 계단·보행자 모두 상하 반전이 물리적으로 무의미하다
        flipud=0.0, fliplr=0.5,
    )

    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=0,
                        project=str(OUT), name=f"{args.name}_val", exist_ok=True)

    print("\n" + "=" * 78)
    print("클래스별 성능 — 불균형 가중치 판단의 입력")
    print("=" * 78)
    names = model.names
    print(f"{'class':<10}{'mAP50':>10}{'mAP50-95':>12}{'precision':>12}{'recall':>10}")
    print("-" * 54)
    for i, c in enumerate(metrics.box.ap_class_index):
        print(f"{names[int(c)]:<10}{metrics.box.ap50[i]:>10.3f}"
              f"{metrics.box.ap[i]:>12.3f}{metrics.box.p[i]:>12.3f}"
              f"{metrics.box.r[i]:>10.3f}")
    print(f"{'(전체)':<10}{metrics.box.map50:>10.3f}{metrics.box.map:>12.3f}")

    print(f"\n산출물: {OUT / args.name}")
    print("\n⚠️ 이 mAP 는 **개발용**이다 — LoLI 는 pseudo-label, StairNet 은 84%가 주간,")
    print("   두 소스의 도메인이 분리돼 있어 '장면 단서' 지름길이 검출되지 않는다.")
    print("   야간 실효 성능 판정은 자체 촬영분(C5)에서 한다.")


if __name__ == "__main__":
    main()
