"""NightOwls 라벨 JSON → YOLO 변환 공용 로직.

왜 모듈로 분리하는가
    `eval_nightowls.py`(평가)와 `build_detect_dataset.py`(학습 데이터)가 **GT 를 똑같이
    만들어야** C4b before/after 비교가 성립한다. 두 곳에 같은 코드를 두면 한쪽만 고쳐져
    비교가 조용히 깨진다. 변환 규칙은 여기 한 곳에만 둔다.

프레임 4종 — 학습에서의 취급이 다르다 (2026-08-02 실측, 해제분 13,602장 기준)

    pedestrian 있음      6,236장   ✅ 양성 표본
    ignore 만 있음       3,097장   ⚠️ 사람이 있는데 정답을 알 수 없다 → **학습 제외**
    자전거/오토바이만      156장   ⚠️ 사람이 타고 있다 → **학습 제외**
    주석 자체가 없음     4,113장   ✅ 진짜 빈 프레임 (전량 recording 38)

    `ignore`(category 4)는 군중·불명확 영역이다. GT 에서 빼면 빈 라벨 파일이 되는데,
    그 프레임을 학습에 넣으면 **"사람이 있는데 배경"** 이라고 가르친다. C4b 가 고치려는
    낮은 재현율(0.226)을 오히려 강화하는 방향이라 학습에서는 반드시 빼야 한다.

recording 그룹
    `recordings_id` 가 비어 있는 1,318장은 파일명(ObjectId) 구간이 다른 recording 과
    겹치지 않는 독립 블록이다 (2026-08-02 실측) — `"none"` 이라는 하나의 recording 으로
    다뤄도 누수가 없다. timestamp 는 recording 별 상대 시각이라 그룹 판정에 쓸 수 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

PEDESTRIAN = 1
BICYCLEDRIVER = 2
MOTORBIKEDRIVER = 3
IGNORE = 4

# 통합 클래스 배치 — **여기가 유일한 정의처**다. 데이터셋 빌더·평가·변환기가 전부
# 이것을 임포트해 쓰므로, 소스마다 id 가 어긋나는 사고가 원천적으로 안 난다.
# `bollard` 는 2026-08-03 확정(AIHub 인도보행이 유일 소스 → scripts/aihub_to_yolo.py).
CLASS_NAMES = {0: "person", 1: "stairs", 2: "bollard"}
PERSON_ID = 0
STAIRS_ID = 1
BOLLARD_ID = 2


def rec_key(v) -> str:
    """recordings_id → 문자열 키. 값이 없는 그룹도 하나의 recording 으로 취급한다."""
    return "none" if v is None else str(int(v))


def _yolo_row(bbox, w: int, h: int) -> str | None:
    """COCO bbox(xywh, 절대) → YOLO 행. 프레임 밖으로 나간 좌표는 잘라 낸다.

    NightOwls 에는 이미지 경계를 넘는 박스가 있다. 그대로 두면 ultralytics 가
    'non-normalized or out of bounds' 로 **이미지 통째를 버린다** — 학습 표본이
    조용히 사라지므로 클램프한다.
    """
    x, y, bw, bh = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (f"{PERSON_ID} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


class NightOwlsIndex:
    """라벨 JSON 을 한 번 읽어 프레임 종류·YOLO 행·recording 을 들고 있는다."""

    def __init__(self, json_path: Path):
        meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
        self.info: dict[int, dict] = {i["id"]: i for i in meta["images"]}

        self.rows: dict[int, list[str]] = {}
        self.has_ped: set[int] = set()
        self.has_ignore: set[int] = set()
        self.has_rider: set[int] = set()

        for a in meta["annotations"]:
            iid = a["image_id"]
            cat = a["category_id"]
            if cat == PEDESTRIAN and not a.get("ignore"):
                v = self.info[iid]
                row = _yolo_row(a["bbox"], v["width"], v["height"])
                if row is None:
                    continue
                self.has_ped.add(iid)
                self.rows.setdefault(iid, []).append(row)
            elif cat == IGNORE or a.get("ignore"):
                self.has_ignore.add(iid)
            elif cat in (BICYCLEDRIVER, MOTORBIKEDRIVER):
                self.has_rider.add(iid)

        # 중복 박스 제거 — 원본에 같은 좌표가 두 번 들어간 경우가 있다
        for iid, rs in self.rows.items():
            self.rows[iid] = list(dict.fromkeys(rs))

    def file_name(self, iid: int) -> str:
        return self.info[iid]["file_name"]

    def recording(self, iid: int) -> str:
        return rec_key(self.info[iid].get("recordings_id"))

    def is_unlabeled_person(self, iid: int) -> bool:
        """사람이 있지만 정답을 알 수 없는 프레임 — 학습에 넣으면 안 된다."""
        return iid not in self.has_ped and (iid in self.has_ignore or iid in self.has_rider)

    def select(self, available: set[str], recordings: set[str] | None = None,
               drop_unlabeled_person: bool = False) -> list[int]:
        """조건에 맞는 image_id 목록. `available` 은 해제된 파일명 집합."""
        out = []
        for iid, v in self.info.items():
            if v["file_name"] not in available:
                continue
            if recordings and self.recording(iid) not in recordings:
                continue
            if drop_unlabeled_person and self.is_unlabeled_person(iid):
                continue
            out.append(iid)
        return sorted(out, key=self.file_name)


def write_data_yaml(dst: Path, generator: str, train: str, val: str,
                    extra: str = "") -> Path:
    path = dst / "data.yaml"
    path.write_text(
        f"# {generator} 가 생성한다. 직접 수정하지 말 것.\n"
        + (f"# {extra}\n" if extra else "")
        + f"path: {dst.as_posix()}\n"
        f"train: {train}\n"
        f"val: {val}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(CLASS_NAMES.items())),
        encoding="utf-8")
    return path
