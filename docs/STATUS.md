# 밤마실 — 현재 상태 한 장

> **이 파일만 읽고 작업을 시작할 수 있게** 유지한다. 다른 문서는 *필요할 때만* 연다.
> 작업을 끝낼 때마다 **이 파일을 먼저** 고칠 것.
> 갱신 **2026-08-31** — `C4e` **E3 오염 검증 재실행**(clean val-only + `c4d_11n_640` 대조 —
> 결론 재현, **ROI 크롭 기각 최종 확정**) · 이전: 2026-08-30 `C5` **1차 판정 실행**(자체
> 27장·`c4e_s3_11n` 우세) + `eval_own_night.py` 버그 수정 · 2026-08-28 `C4e` **E3 ROI 크롭
> 기각**(+ 배포 conf 0.35→0.25 절반 정정) · `C10a` ③ ONNX 양자화(**8비트 채택 · 4비트·FP8
> 기각**) + **함정 20**.

야맹증 저시력자의 야간 보행을 돕는 **실시간 AI 시각보조**(스마트폰 단독).
구조는 **표시/탐지 2경로** — 탐지는 **원본**을, 표시는 **①② 처리본**을 쓴다(↓ 1장).
목표 **720p 15FPS**. 해커톤 마감 **9월 초**.
실행은 전부 `uv run python scripts/xxx.py` (→ [README](../README.md#개발-환경)).

---

## 1. 단계별 현재 상태

### ★ 아키텍처 확정 — 표시/탐지 경로 분리 (8/2, `C7`) · **이 절이 정본**

```
프레임 ─┬─→ [탐지 경로] 원본 그대로 ────────→ ③ YOLO ──┐
        │                                              ↓
        └─→ [표시 경로] ② D1A1+bf(+ts) ──────→ ④ 경계선 강조 → 화면
```

②를 탐지 앞단에 두면 mAP 가 내려가고 **`stairs` 야간 오탐이 0.1% → 5.7%** 로 되살아난다
(`C4b` 가 0 으로 만든 결함이 전처리 한 번에 복귀). 분리는 비용이 아니라 이득이다 — 두 경로가
독립이라 병렬화 가능하고 ②의 지연이 ③에 더해지지 않는다 (→ [detection.md 7장](detection.md)).

> ✅ **코드는 이미 이 구조다** — `pipeline_demo.py:128` 이 탐지에 **원본 프레임**을 넣고
> ②는 표시 경로에서만 돈다. 옛 직렬 구조는 `--fused` 플래그로만 재현되며 기본 꺼져 있다.

| 단계 | 채택안 | 확정된 수치 | 상태 |
|------|--------|-------------|------|
| ① 눈부심 억제 | `D1`(Drago) — **표시 경로 전용** · ②에 흡수 예정 | 광원 코어 250→157 @640×360 | 🗣️ 통합 여부 회의 안건 |
| ② 저조도 개선 | **고전 CV 확정**(8/1) · 표시 경로 전용 · `D1A1+bf`+`ts` | 3축 만족 · `+ts` **+1.5ms** | ⚠️ **속도 게이트 경계**(≈20.4ms@640×360) |
| ③ 위험요소 탐지 | **YOLO11n 단일 모델** · **입력은 원본** · 클래스 `0 person` `1 stairs` `2 bollard`(8/22 최종 확정) | ↓ 후보 표 | 🟢 **`C5` 1차 완료(8/30)** — `c4e_s3_11n` 우세 · 정식 판정은 `C2` 정규 촬영 대기 (→ [detection.md 9장](detection.md)) |
| ④ 선택적 강조 | 이중 스트로크 비채움 렌더 · 색 3종 확정(8/24) | 박스당 **0.5ms** · `person` 시안 `#00E6FF` · `stairs` 노랑 `#FFD700` · `bollard` 라임 `#9CFF2E` | ✅ 구현·색 확정 · 🔴 **`P1` 실기기 가독성 대기**(라임↔노랑 적록색약 근접) |
| **파이프라인 통합** | `scripts/pipeline_demo.py` | 데스크톱 end-to-end 확인(8/2) | ✅ 사전 검증 · 앱 이식 대기 |
| **③ 앱 전달물** | ONNX FP32 · 640 고정 · NMS 제외 · **INT8 판 동봉**(8/26) | `bammasil_det_c4e_s3_11n_640.zip` 9.20MB · 출력 **[1,7,8400]** · PT↔ONNX **0.0001px** · **INT8 3.15MB**(계약 동일) | 🔴 계약 변경 **앱팀 공지 완료**(8/24 → [share 7-0](share_yolo_c4b_20260803.md)) · 🟡 INT8 은 실기기 확인 전(`C10`) |
| 앱 | — | — | ❌ **구현 0건** |

**③ 후보 3개** — held-out(NightOwls rec34 · person) 기준 mAP50 / recall / `stairs` 오탐

| 런 | 수치 | 성격 |
|---|---|---|
| **`c4b_loli0`** (현 배포 · 2클래스) | 0.684 / 0.609 / 0.2% | 채택 가중치 `outputs/detect/c4b_loli0/weights/best.pt` |
| `c4d_11n_640` (인수분 · 3클래스) | 0.736 / 0.644 / 0.0% | 별개 계보 · **코드 없음** · 배포 자(square)에서 야간 볼라드 2/2→1/2 |
| **`c4e_s3_11n`** (자체 · 3클래스) | **0.710 / 0.635 / 0.0%** | 야간 볼라드 **2/2** · 실야간 오탐 최저 · **ONNX 로 나가 있는 판** |

⚠️ 요약 수치는 **한 런 기준**으로 적을 것 — `c4b_loli6000`(0.691/0.625/0.0%)과 섞으면
"오탐 0.0%" 같은 과대 표기가 된다 (→ [detection.md 6-7](detection.md)).

### 🟢 `C5` 1차 판정 완료 (8/30)

자체 촬영 27장(`data/own_night`)에서 `c4e_s3_11n`이 mAP·운영점·음성 오탐·볼라드 recall
전 축 우세(FP32 mAP50 0.551 · INT8도 손실 거의 없음 0.526) — 기존 held-out 판단과 방향이
같다. `<32px` 표본이 0장이라 **아직 최종 판정은 아니다**. 상세 표·버그 수정·노트북은
→ [detection.md 9-10](detection.md).

---

## ★ 외부 인수분 — `received/` (8/7 도착 · **판정 완료 8/23**)

> 번호 없는 절이다. 다른 문서가 `R1`~`R5` 로 참조하므로 **소절 제목을 유지**한다.
> 📦 원문(6런 표·오류 사례·경위)은 [archive/received_c4d_2026-08-07.md](archive/received_c4d_2026-08-07.md).

별도 환경의 **`C4d` 3클래스 6런 비교**. 우리 `c4b_*` 와 **별개 계보**(11K 축소셋)이고
**코드는 오지 않았다**(표·CSV·가중치만). 가중치는 `outputs/bammasil_results/bammasil_weights/c4d_*/`.

### R1. 6런에서 확정된 것 (공식 지표 `benchmark_c4d.csv` · 평가셋 `bollard_val_2146`)

| 확정된 것 | 값 |
|---|---|
| **해상도 > 모델 세대** | 동일 50ep 에서 해상도 **+0.065** vs 모델 **+0.022**(3배) · 960 은 볼라드 FP 도 감소(925→792). ⚠️ **person 오탐만은 는다** |
| **배경 이미지 10% 유지** | `noneg`(0%)는 mAP·recall 이 좋아 보이나 **6런 중 오탐 최다**(볼라드 +11% · person +16%) — **효과가 mAP 가 아니라 FP 에 난다** |
| **운영 conf 무릎 0.10** | ⚠️ **주간 판단이다.** 야간에서 뒤집혀 **단일 conf 0.25 확정** (→ [detection.md 9-9](detection.md)) |
| **`<4px`(GT의 22%)** | 960 에서도 recall **0.49 가 상한** — 해상도로 더 못 간다 |

`stairs` 는 이 평가셋에 GT 0개라 FP 2건뿐(**주간 근거** · 실야간에서 뒤집힌다 → 2장 4).
`confusion`(person↔bollard) 1% 미만 — **클래스 설계는 건전하다.**

### R2. ★ 이 수치는 전부 **주간**이다 — `is_night` 은 무효 필드

`is_night` 이 **파일 mtime** 에서 나와(→ 3장 함정 13) `is_night=1` 로 고른 2,146장이
**99.1% 주간**이다. AIHub 실측 야간은 **0.58%**, 누수 없는 야간 볼라드는 **14장** — 통계가
성립하지 않는다.

> **→ 야간 볼라드는 인수분으로 영영 못 잰다.** NightOwls 에 볼라드 라벨이 없고 StairNet 은
> 계단 데이터셋이라 **`bollard` 의 야간 근거는 자체 촬영(`C2`)뿐이다.**

### R3. 기존 서술과 어긋난 것 — 전부 반영 완료

"AIHub 저조도 ~600장"은 야간이 아니었다(→ R2) · `<4px` 상한이 추가됐다 · **`C2` 우선순위는
올라갔다** · c4d 는 별개 계보라 `c4b` 를 자동 대체하지 않는다. 대조표 원문은 archive.

### R4. ✅ 판정 완료 (8/23) — **자체 3클래스 런이 더 낫다**

640 축 4런을 rec34 에 태워 **3런 통과 · `26n_640` 기각 · 주간 순위가 야간에서 뒤집혔다**.
`c4e_s3_11n` 이 c4d 의 두 약점을 지웠다 — `stairs` 실야간 오탐(conf 0.10 에서 21→**9박스**) ·
야간 볼라드 confidence 붕괴(0.595→0.442→0.321 → **붕괴 없음** 0.759~0.804). 차이는 c4d 에
없던 **NightOwls 실야간 + 배경 음성**이다 (→ [detection.md 9·11-3c](detection.md)).

⚠️ 남는 것 — 코드가 없어 **재현 불가** · 인수분 안에서도 `results.csv` ≠ `benchmark_c4d.csv` ·
**셋 다 seed 1개**라 최종 판정은 `C5` · **실기기 속도 미측정**이라 **배포 해상도 640 유지**.

### R5. 회신 대기 3건 (→ `received/README.md` 10장)

1. 🔴 **야간 데이터셋 사양** — 보낸 쪽이 당일 촬영·라벨링 가능하다고 했다.
   **우리 `C2` 와 정면으로 겹친다 — 같은 밤을 두 번 찍지 않도록 분업부터 정할 것.**
2. **ONNX 로 넘길 비교축 런 지정** — ⚠️ YOLO26n 은 NMS-free 라 **앱 계약이 바뀐다**
3. **KPI 비교표** — 실기기 640 계측(p50 32.7 · p95 38.1ms · 11분)이 나오면 960·26n 과 대조

---

## 2. 지금 막혀 있는 것 — 우선순위

1. 🔴 **`C2` 자체 야간 촬영** — `C3`→`C5`→`C7b`→`C8`→`C9` 가 전부 직렬로 매달려 있다.
   ✅ **막는 것이 하나도 없다** — 개인정보(8/3) · `stairs` 라벨 정의(→
   [labeling_stairs.md](labeling_stairs.md)) · 클래스 3종 확정(8/22)으로 전제가 전부 풀렸다.
   - 🟢 **`test_real_data2`(20장·8/30)가 1차분으로 들어왔고 `C5`를 한 번 돌렸다**(→ 1장) —
     아래 두 항목은 여전히 미해결
   - ⚠️ **멀리서 계단에 접근하는 구간**을 반드시 넣을 것 — 없으면 `stairs` 최대 미검증
     항목(원거리 계단)이 이번에도 측정 불가로 남는다
   - 🔴 **볼라드 구간 필수** — AIHub 로는 야간 볼라드를 영영 못 잰다(→ R2) · 8/30분은
     근거리 위주라 작은 볼라드 축은 아직 못 잼
   - ⚠️ **인수처도 야간 촬영 대기 중** — 사양·분업을 먼저 정할 것(→ R5)
   - ✅ **판정 하네스는 준비됐고 8/30 첫 실행에서 버그 1건 고쳤다** —
     `scripts/eval_own_night.py`(8/25 · 5축 · 기본 `square`).
2. 🔴 **`P3` 앱 껍데기** — 구현 0건. `C9` 통합 시점에 없으면 통합할 곳이 없다.
3. ⚠️ **② 속도 게이트** — 720p 를 확실히 통과하는 건 `A1`·`A2`(6ms)뿐인데 **글레어 축
   실격**이다. 채택안 `D1A1+bf+ts` 는 640×360 환산 **≈20.4ms** 로 경계선(판정 불가 밴드
   15~25ms). `bf`→A3 시간축 대체는 8/2 기각(→ lowlight_classical 8장).
   남은 수 — 내부 해상도 하향 · 셰이더 이식 · `C11` 실기기 재판정.
   - ✅ 표시/탐지 분리로 압박이 줄었다(②의 지연이 ③에 안 더해진다) — 🗣️ 회의 안건 1
   - ✅ **③ 프레임 스킵 상한 확정 — `--detect-every` 2**. k=2→3 에서 IoU 0.93→0.88 ·
     miss 0.21→0.30 으로 꺾인다 (→ [detection.md 9-9-2](detection.md))
4. **`stairs` — 저조도가 아니라 *도메인* 문제였고, 실야간 오탐은 8/23 에 풀렸다**
   - 저조도는 문제가 아니다(StairNet 야간 76장 mAP50 **0.993**). 그러나 StairNet 은
     *계단이 화면을 채운 사진*이라 **보행 중 멀리 있는 계단**을 못 잰다(→ [detection.md 8장](detection.md))
   - ✅★ 배경 음성 1,163장을 넣은 `c4e_s3_11n` 이 실야간 음성 15프레임에서 conf 0.10 기준
     **22 → 9박스**(2.4배) · 발화 14/15 → **6/15**. 즉 모델 계보가 아니라 **StairNet 에 음성
     표본이 없다**는 데이터 문제였다 (→ [detection.md 9-6·11-3c](detection.md))
   - 🔴 **남은 미검증은 원거리 계단 하나** — 야간 답은 `C2` 뿐. 주간은 AIHub 노면 마스킹이
     처음으로 잴 수단인데 **아직 안 받았다**(→ TODO `W5` · [detection.md 8-3](detection.md))
5. 🟡 **`bollard` — 데이터·배선·학습은 끝났고 남은 것은 야간 판정 하나**
   - ✅ 좁은 정의 확정(8/22) · AIHub 3클래스 변환본 **113,163장** 확보 · 배선
     `aihub_subset_to_yolo.py`(8/23) · 자체 3클래스 런 완료
   - ✅★ **야간 볼라드 confidence 붕괴가 사라졌다** — c4d 는 0.595→0.442→**0.321** 로
     무너지나 `c4e_s3_11n` 은 0.759→0.785→**0.804**(세 밝기 전부 2/2).
     → 클래스별 conf 카드는 **c4d 를 쓸 때만** 필요하다
   - 🔴 **c4d 의 그 붕괴 수치는 `rect` 로 잰 것 — 배포 자(square)에서는 2/2 → 1/2**
     (maxconf 0.669→0.501 · 세 밝기 전부). `C5` 는 **`--letterbox square`** 로 잰다(→ 함정 18)
   - 🔴 **그래도 미측정 항목이다** — 근거가 `test_real_data/_04` **한 장**이고 라벨이 없어
     recall·FP 를 못 센다. **`C2` 가 유일한 판정 수단.**
   - ⚠️ **박스별로 볼 것** — 장면 `maxconf` 만 보면 먼 쪽(0.67~0.71)에 가려 가까운 쪽의
     붕괴가 안 보인다
   - ⚠️ 남는 한계 2개 — `<4px`(GT 22%)는 960 에서도 recall **0.49 상한** · ghost FP
     (기둥·가로등·표지판 지주) **57~63%** 는 좁은 정의상 **영영 오탐**이라 배경 음성으로 줄인다

---

## 3. ⚠️ 반복해서 발목을 잡은 함정 — 작업 전 반드시 확인

> **번호는 고정이다** — 다른 문서가 `3장 함정 N` 으로 참조한다. 추가는 뒤에만 붙인다.

1. **개발 val 은 야간 실효 성능을 못 본다.** `C4b` 에서 실야간이 5.4배 좋아지는 동안 개발
   val 은 0.892 → 0.892 로 한 자리도 안 움직였다. 판정은 **항상** held-out(rec34) 또는
   자체 촬영분(`C5`)에서 한다.
2. **지표를 영상 통계에 상대적으로 정의하지 말 것.** 백분위·평균 대비로 정의하면 영상이
   바뀔 때 *재는 대상 자체*가 달라진다. 노이즈·글레어·대비 3개가 전부 이 결함으로 육안과
   어긋났다 (→ `scripts/metrics.py`).
3. **이 PC 는 15~25ms 를 판정할 수 없다.** 프로세스 간 10~20% 흔들리고, 다른 작업과 동시에
   재면 **1.7배까지** 부풀어 오른다.
4. **NightOwls `ignore`-only 프레임 3,097장은 학습 금지.** 사람이 있는데 라벨이 비어 있어
   "사람=배경"을 가르친다 (→ `scripts/nightowls_yolo.py`).
5. **NightOwls rec 34·38 학습 금지.** 34 = 판정용 held-out · 38 = ② 시간축 전용.
   `build_detect_dataset.py` 가 실행을 거부한다.
6. **Windows 에서 `--cache ram` 금지.** worker 가 spawn 이라 캐시가 복제돼 죽는다.
7. **Raw 데이터 수정 금지.** 산출물은 전부 `outputs/`(git 비추적), 이미지는 하드링크.
   ⚠️ 이 규칙이 뚫리는 경로가 있다 → 함정 19.
8. ★ **AI Hub 는 해외 IP 다운로드를 차단한다** (8/4 실측). Colab·해외 클라우드에서
   `aihubshell` 을 돌리면 `502 · "해외에서의 데이터 다운로드를 제한"`. 우회는 이용정책
   위반이라 하지 않는다 — **취득은 국내 PC 에서만**, 클라우드 GPU 는 **국내에서 받아 640
   리사이즈해 옮기는** 경로뿐이다(`aihub_pack_for_colab.py` → [data.md 3-1-3](data.md)).
   같은 실측에서 하나 더 — **라벨과 원천이 안 갈려 있어**(태스크별 10GB 통zip) "라벨 먼저
   받아 분포 실측"이 불가능하고 **태스크별 1번 zip 정찰**로 대체한다.
9. **ultralytics AutoUpdate 를 켠 채 ONNX 추론 금지** (8/3 실측). `onnxruntime`(CPU)이
   없으면 제멋대로 설치해 `uv.lock` 밖 패키지를 끼워 넣는데, 이 프로젝트의
   `onnxruntime-gpu` 와 **같은 디렉토리를 공유**해서 나중에 CPU 판을 지우면 통째로 망가진다
   (`module has no attribute 'InferenceSession'`).
   복구 `uv sync --reinstall-package onnxruntime-gpu` · 예방 임포트 전 `YOLO_AUTOINSTALL=false`.
10. **`.pt` 와 ONNX 를 비교할 때 파일 경로를 그대로 넘기지 말 것** (8/3). ultralytics 는
    `.pt` 에 **가변 rect 레터박스**(16:9 → 640×384), 고정 shape ONNX 에는 640×640 을 먹인다.
    비정사각 이미지에서 신뢰도가 0.1 씩 벌어지는데 **변환 손실이 아니라 전처리 차이**다.
    미리 정사각 레터박스한 배열을 양쪽에 넣으면 좌표 오차 **0.0001px** 로 일치한다.
11. ★ **다른 머신으로 옮길 `data.yaml` 에 `path:` 를 넣지 말 것** (8/4 실측). 상대경로면
    `settings['datasets_dir']` 기준으로 풀려(Colab `/content/datasets`) zip 을 옮기는 순간
    `images not found` 로 죽고, 절대경로면 Windows 경로가 박혀 역시 못 쓴다. **빼면**
    ultralytics 가 yaml 위치를 기준으로 잡아 어느 머신에서든 동작한다
    (`aihub_pack_for_colab.make_portable`). 로컬 전용은 절대경로여도 된다.
12. ★ **한국어 Windows 에서 조용히 죽는 것 2개** (8/5 실측). 둘 다 **큰 데이터를 다 훑은
    뒤** 터져서 실행이 통째로 날아간다.
    - 콘솔이 cp949 라 `⚠️`·`★` 출력에서 `UnicodeEncodeError` → `sys.stdout.reconfigure(errors="replace")`
    - **`cv2.imread` 가 비ASCII 경로를 못 연다** — 한글이 섞이면 예외 없이 **전부 `None`**
      → `cv2.imdecode(np.fromfile(path, np.uint8), ...)`. PIL 은 영향 없다
13. ★ **파일 mtime 을 촬영시각으로 쓰지 말 것** (8/7 · 외부 인수분에서 발생 → R2).
    AIHub 야간 선별이 mtime 기반 `is_night` 으로 돌아가 **99.1% 가 주간인 셋을 "야간"으로**
    쓰고 있었다. **야간 판정은 실측 휘도로 한다** — 2번과 뿌리가 같다.
14. ★ **`outputs/datasets/` 의 파생 평가셋은 하드링크가 아니라 *복사본*이다** (8/22).
    원본은 `D:`, `outputs/` 는 `C:` 라 `os.link` 가 볼륨을 못 넘어 통째로 복사된다 —
    장당 ~1MB PNG 라 금방 수십 GB. `eval_nightowls.py`·`eval_stairs_night.py` 양쪽 해당.
    **전부 재생성 가능**하므로 C: 가 부족하면 지웠다 되살린다.
    ```powershell
    uv run python scripts/eval_nightowls.py --dst outputs/datasets/nightowls_eval  # 해제분 전체
    uv run python scripts/eval_nightowls.py --recordings 38 --fp-only              # rec38
    ```
    ⚠️ `nightowls_eval` 은 **레거시 경로**다 — 현재 기본 dst 는 `nightowls_split/all`.
15. ★ **데이터셋 실체는 전부 `D:\datasets\` — Junction 이 없으면 스크립트가 죽는다**
    (8/22 재편 · 8/23 복구). `data/<이름>` 을 하드코딩하는 스크립트가 7개라 Junction 이
    빠지면 **"데이터 없음"으로 조용히 끝난다.** 저장소를 새로 받으면 **먼저 걸 것**:
    ```powershell
    foreach ($n in "NightOwls","LoLI-Street","Stair dataset","ExDark","LOLdataset") {
        if (-not (Test-Path "data\$n")) {
            New-Item -ItemType Junction -Path "data\$n" -Target "D:\datasets\$n"
        }
    }
    ```
    ⚠️ **AIHub 만 예외** — `D:\datasets\bammasil_aihub_subset` 을 `--src` 로 넘기고, 받는
    쪽은 `aihub_to_yolo.py`(CVAT XML 전용)가 **아니라** `aihub_subset_to_yolo.py` 다.
    회귀 확인은 `uv run python scripts/inspect_datasets.py` (→ [data.md 5-2](data.md)).
16. ★ **2클래스 → 3클래스 전환 시 조용히 깨지는 것 3개 — 2개 해소** (8/23 발견 · 8/24 해소).
    - ✅ **④ `bollard` 색** — 라임 `#9CFF2E` 확정. `FALLBACK_COLOR`(흰색)가 이제 **세 클래스
      어느 것도 아닌 값**이라 미지정 클래스가 볼라드로 오인될 여지가 없다.
      고른 자는 [`emphasis_palette_review.ipynb`](../notebooks/emphasis_palette_review.ipynb) ·
      렌더 로직은 `emphasize.py` 한 곳.
      ★★ **한 색이 두 축을 다 이기지 않는다** — 라임은 **배경 대비**(4.18~10.05 · WCAG 3:1
      전부 통과)를 산 선택이고 대가는 **적록색약에서 `stairs` 노랑과 가까워지는 것**
      (녹색맹 최소 ΔE **9.9**). 🔴 이 절충이 `P1` 의 판정 대상이며 되돌아갈 자리
      `violet`·`magenta` 는 **밝기 대비 3:1 미만**이라는 반대급부를 안는다.
      ⚠️ **세 클래스가 동시에 있는 실사진이 로컬에 없다**(`test_real_data` 사람 0장 ·
      AIHub bbox 에 `stairs` 0건) — 🗣️ `C2` 코스 요구 후보다.
    - ✅ **ONNX metadata 의 held-out 지표** — `RUN_METRICS` 등록 후 내보냈고, **미측정(rec38)이
      `0.0%` 로 보이지 않게** README 렌더도 고쳤다. ⚠️ c4d 를 내보낼 일이 생기면 먼저 등록할 것.
    - 🔴 **남은 하나 — 배포 계약의 conf 가 단일 스칼라**(`export_onnx.py`). 클래스별 conf 를
      채택하면 dict 로 바뀌고 README 후처리 절차도 같이 고쳐야 하며 **앱팀 공지 대상**이다.
      🆕 단 `c4e_s3_11n` 을 쓰는 한 손댈 일이 없다(야간 볼라드 conf 0.759~0.843).
17. ★★ **세션 키에 '그룹'을 넣지 말 것 — 누수가 "누수 0" 으로 보고된다** (8/23 실측).
    `bammasil_aihub_subset` 의 그룹 폴더는 무효 필드 `is_night`(→ 13) 기반이라 **같은 촬영
    세션이 그룹마다 쪼개져 있다**(블록 2,296개 중 **1,357개(59%)가 여러 그룹에 걸친다**).
    분할 키를 `(그룹, 블록)` 으로 잡으면 같은 세션이 train/val 로 갈리는데 **검사도 그 키로
    하므로 "누수 0" 이라고 보고된다.** 실측 피해 **val 2,205장 중 568장(26%) 오염.**
    ⚠️ `best.pt` 는 **val fitness 로 선택**되므로 **모델 선택 자체가 왜곡된다.**
    → **블록(세션)을 그룹과 무관한 원자 단위로** 다루고, 검사는 **산출된 파일명에서 되짚어**
    한다(선별 로직이 쓰는 키로 검사하면 같은 버그를 못 잡는다). `aihub_subset_to_yolo.py` 가 그 규칙이다.
18. ★★★ **전처리 모드가 "배치에 뭐가 같이 들어 있느냐"로 정해진다** (8/25 · ✅ 해결됨).
    - **원인** `pre_transform` 이 `auto = same_shapes and rect` 로 레터박스를 고르는데,
      `predict(source=[...])` 처럼 **리스트**를 주면 리스트 전체가 한 배치다(`batch=1` 무시).
      방향이 전부 같으면 **rect**, **하나라도 섞이면 정사각 640** — 같은 이미지의 예측이
      **옆에 무엇이 있느냐로** 달라진다. 실측 — 볼라드 conf **0.669 → 0.501** ·
      계단 **0.352 → 0.476** · 세로 이미지들까지 흔들렸다.
    - ★★ **배포 ONNX 가 고정 640 정사각**이므로(→ 10) **`square` 가 배포와 같은 자**이고
      ultralytics 기본(`rect`)은 **앱이 절대 쓰지 않는 전처리**다.
    - **해결** 두 하네스에 **`--letterbox {square,rect}`**(기본 `square`)를 두고 **한 장씩**
      돌린다 — 배치 구성이 아니라 **플래그가** 모드를 정한다.
19. ★★ **평가셋을 하드링크로 깔면 `model.val()`·학습이 원본을 덮어쓴다** (8/25 · ✅ 해결됨).
    - **원인** `data/utils.check_image` 는 JPEG 의 마지막 2바이트가 EOI(`FFD9`)가 아니면
      **그 경로에 다시 쓴다**("corrupt JPEG restored and saved"). 하드링크면 그 쓰기가
      **원본을 관통한다**(실측 3.37→7.27MB · inode 동일). 함정 7 이 뚫리는 경로다.
      ⚠️ **데이터셋 스캔 경로에서만**(train·val) 일어난다 — `predict` 는 안전하다.
    - **노출** `data/test_real_data/` 7장 전부 대상이고 `C:` 라 `outputs/` 와 같은 볼륨이다.
      AIHub 서브셋은 113,163장 중 0건. 함정 14 대로 `--dst` 를 `D:` 에 두는 순간 활성화된다.
    - **해결** `link_or_copy` **7곳 전부**에 가드를 넣어 **덮어쓸 파일만 복사**한다
      (113k 링크 이득 유지 · 재현 테스트 통과).
20. ★★★ **ONNX 를 통째로 INT8 양자화하면 검출이 0 이 된다 — `Conv` 만 할 것** (8/26 · ✅ 해결됨).
    - **원인** 머리의 마지막 `Concat`(`/model.23/Concat_3`)이 **박스 좌표(0~640)와 클래스
      점수(0~1)를 한 텐서로 합친다.** per-tensor uint8 스케일이 **640/255 ≈ 2.5** 가 되어
      **점수가 전부 0 으로 반올림**된다. 실측 7장 전부 0검출 · 야간 볼라드 **2/2 → 0/2**.
      파일도 shape 도 정상이라 **조용히 죽는다.**
    - **해결** `op_types_to_quantize=["Conv"]` — Conv 88개는 INT8 로 가고 머리의 산술은
      float 로 남는다(볼라드 2/2 복귀). `scripts/quantize_onnx.py` 가 이 설정으로 고정.
    - ⚠️ **곁다리** — 파일 이름이 `*_qnn.onnx` 로 끝나면 ultralytics 가 **QNN 컨텍스트
      바이너리**로 착각해 없는 DLL 을 찾다 죽는다(`autobackend.py:348`) → `_qnn_qdq.onnx`.

---

## 4. 데이터 — 무엇이 어디에

| 데이터셋 | 규모 | 성격 · 주의 |
|---|---|---|
| **NightOwls** | 51,848장 중 **13,602장 해제** | ★ 사람 손 라벨 · **전량 야간**. 유일한 정직한 야간 평가원. **차량 대시캠**이라 보행 시점과 다름 |
| LoLI-Street | 33,000쌍 (+person 라벨) | ⚠️ **주간을 어둡게 만든 합성본** — 포화 광원 **0.00%**(글레어 학습·측정 불가) · 노이즈가 `high` 보다 **적다**(σ 0.97 vs 7.13) · 라벨 불일치. ★ **합성 증강(`W2`) 8/25 기각** (→ [detection.md 6-7](detection.md)) |
| StairNet | 3,094장 (야간 505) | `stairs` 유일 소스. **계단 중심 사진**이라 음성 표본이 없음 |
| ExDark | 7,363장 | 실제 야간이나 **실내가 많음** |
| LOL | 500쌍 | 벤치마크 비교용 |
| **AIHub 인도보행** | bbox 전량 300GB(8/5) | `bollard` 유일 소스 · **주간 데이터셋**(실측 야간 **0.58%** → R2) |

> 📍 **실체는 전부 `D:\datasets\`** 이고 저장소 기준 경로 `data/<이름>` 이 **Junction** 이다
> — 스크립트에 경로 분기가 없다. ✅ 5종 전부 연결됨(8/23). 명령은 → 3장 함정 15.
> AIHub 만 예외로 `--src` 인자이며 **`aihub_subset_to_yolo.py`** 가 받는다(3클래스 YOLO
> 변환본 113,163장 · → [detection.md 11장](detection.md)).

**NightOwls recording 배치 (고정)**: train `36·none·37` / val `35` / **test `34`** / 제외 `38`

**실촬영 테스트 소재** `data/test_real_data/`(8/8·7장+동영상5) + `data/test_real_data2/`
(8/30 신규·20장, `C2`의 1차분 성격 · 사람 표본 최초 포함) — 합쳐 **`data/own_night/`로
라벨을 채워 `C5`에 썼다**. 장면 구성·정정 내역 상세는 → [data.md 2-2-1](data.md).

### 4-1. 사용 / 미사용 구분 (탐지 학습·평가 파이프라인 기준)

| 데이터셋 | 상태 | 근거 |
|---|---|---|
| NightOwls · LoLI-Street(person) · StairNet(stairs) · AIHub(bollard) | ✅ 학습+평가 결합 | `build_detect_dataset.py` 에 전부 배선됨 |
| LOL | ⚠️ 벤치마크 전용 | 저조도 **향상(enhancement)** PSNR/SSIM 비교에만 사용, 탐지 학습엔 미사용 (`compare_lowlight.py`) |
| ExDark | ⚠️ 점검만, 미배선 | `inspect_datasets.py`/`night_eval.py` 에 경로만 있고 `build_detect_dataset.py` 엔 미참조 |
| LSRW | ❌ 기각(미획득) | Baidu Pan 배포만 존재해 접근 불가 → LoLI-Street 로 대체 (→ [archive/data_decisions_2026-07.md](archive/data_decisions_2026-07.md) A-1) |
| BDD100K | ❌ 후보 언급만 | [data.md](data.md) §6 에 보조 후보로만 등장, 코드 참조 없음 |
| COCO | ❌ 데이터셋은 미사용 | `yolo11n.pt` 사전학습 가중치 출처·라벨 인덱스(80-class) 규약으로만 사용 |

---

## 5. 문서 지도 — *필요할 때만* 연다

| 알고 싶은 것 | 문서 | 크기 |
|---|---|---|
| **지금 뭘 해야 하나** (순서·병렬성·일정·담당) | [TODO.md](TODO.md) | 30KB |
| ★ **③ 탐지 전부** — `C4`·`C4b`(3~6장) · 계단 도메인(8장) · **`C4e` S0~S3**(9·11장) · `W2` 기각(6-7) | [detection.md](detection.md) | 75KB |
| ★ **앱팀 인수인계** — 3클래스 계약(출력 2→3) · **INT8 판**(7-0b) · ④ 색 · 연동 주의 | [share 7-0](share_yolo_c4b_20260803.md) | 17KB |
| ★ **양자화 실측** — 파일 표 · FP32 정합 · **붕괴 검사**(INT8 통과 / INT4·FP8 기각) | `outputs/quantization/<pkg>/<pkg>_README.md` · FP8은 `<pkg>-FP8_experiment/result.md` | — |
| ② 고전 arm — 지표·순위·속도·A3 시간축 | [lowlight_classical.md](lowlight_classical.md) | 51KB |
| 데이터 상세 · **AIHub**(3-1) · 계단 방식 근거 · 데이터 위치(5-2) | [data.md](data.md) | 61KB |
| 기획(배경·구성·일정) · 환경 셋업 | [../README.md](../README.md) | 24KB |
| 모바일 런타임·양자화 전략·FPS 병목 · 프레임 스킵 | [hardware_inference.md](hardware_inference.md) | 10KB |
| 스크립트가 뭘 하는지 | [../scripts/README.md](../scripts/README.md) | 8KB |
| ★ **`stairs` 라벨을 어떻게 치는가** (`C3` 작업자용) | [labeling_stairs.md](labeling_stairs.md) | 10KB |
| ★ **외부 리뷰 회신** — 실증 4단 · 계단 오탐 3겹 · 작은 볼라드 & distillation | [review_response_20260825](review_response_20260825.md) | 17KB |
| 🔴 **지식 증류 RunPod 실행 중** — Pod 재접속·진행 확인·재개 방법 | [runpod_distillation_20260831](runpod_distillation_20260831.md) | 6KB |
| ★ **기획서 v20 수치 검증** — 수정 필요 9건 (미반영) | [review_proposal_v20](review_proposal_v20_20260808.md) | 19KB |
| 외부 인수분 1차 자료 (회신 요청 10장) | [../received/README.md](../received/README.md) | 15KB |
| 종결된 실험·결정 원문 — **인수분 `C4d` 원문 포함** | [archive/](archive/) | — |

**노트북 — 판정을 그림으로 확인할 때**

| 보고 싶은 것 | 노트북 |
|---|---|
| ④ `bollard` 색을 어떻게 골랐나 (색각이상 × 배경 대비) | [emphasis_palette_review](../notebooks/emphasis_palette_review.ipynb) |
| c4d 라벨 검수 · 640 추론 · **실야간 추론** | [aihub_subset_review](../notebooks/aihub_subset_review.ipynb) |
| 자체 3클래스 런 `c4e_s3_11n` 추론 | [c4e_infer](../notebooks/c4e_infer.ipynb) |
| `C5` 후보 4종(FP32 3 + INT8) 지표·예측 비교 | [c5_own_night_review](../notebooks/c5_own_night_review.ipynb) |
| 계단 탐지가 완료됐는가 | [stairs_night_review](../notebooks/stairs_night_review.ipynb) |
| AIHub 를 받아 학습까지 (GPU 없는 PC 에서도) | [colab_aihub_train](../notebooks/colab_aihub_train.ipynb) |
| ② arm 육안 비교 | [lowlight_arms_review](../notebooks/lowlight_arms_review.ipynb) · [lowlight_lol_review](../notebooks/lowlight_lol_review.ipynb) |

> 📌 **큰 문서는 통째로 읽지 말 것.** 각 문서 맨 위 `0. 결론 요약`만 읽고, 필요한 절만
> `grep -n "^#" 파일` 로 찾아 `sed -n 'A,Bp'` 로 연다. 목적 축을 **무엇으로 어떻게 재는가**는
> [`scripts/metrics.py`](../scripts/metrics.py) 가 정본이다.

---

## 6. 최근 완료

전체 이력은 → [TODO 8장](TODO.md). 여기는 **아직 머릿속에 있어야 하는 것**만 한 줄로 둔다.

| 항목 | 결과 | 원문 |
|---|---|---|
| ★ **`C4e` E3 — 오염 검증 재실행, 기각 최종 확정** (8/31) | `roi_crop_eval.py`에 `detect_v3`(표준 YOLO 레이아웃) 로더 추가 → val 스플릿(400장·train 미포함)으로 재측정. `c4e_s3_11n`(val-only)·`c4d_11n_640`(이 데이터셋 자체를 학습에 안 쓴 무관 계보) 둘 다 원본과 같은 방향·비슷한 크기로 재현(union `<4px` Δ +0.193·+0.200 vs 원본 +0.220 · FP 증가 방향도 동일). **오염이 8/26 결론을 만들지 않았음을 측정으로 확인** | [detection.md 9-9-3](detection.md) |
| ★★ **`C5` 1차 실행 — `c4e_s3_11n` 우세 확인 + INT8 포함 비교 노트북** (8/30) | 자체 27장에서 후보 4종(FP32 3 + INT8) 비교, `c4e_s3_11n` 전 축 우세. `eval_own_night.py` 버그 1건 수정 + Windows/Jupyter `model.val()` 함정(`workers=0` 필요) 발견 | [detection.md 9-10](detection.md) · `notebooks/c5_own_night_review.ipynb` |
| ★ **`C4e` E3 — ROI 크롭 기각** (8/26) | 신설 `scripts/roi_crop_eval.py`(3-arm — **정보 증가 0 인 arm** 을 끼워 스케일 효과와 해상도 효과를 가른다). 단일 창은 전체 recall **−0.18~−0.24** · 2패스는 `<4px` **+0.220** 이나 **FP 2.26배**(정밀도 0.798→0.654). ★★ 그 +0.220 이 **정보 증가 0** 에서 났다 — `<4px` 미탐은 *정보*가 아니라 **스케일** 문제이고, 이는 **용량 축(`11s`·distillation) 근거를 굳힌다**. ★ 리뷰 문서의 "볼라드는 화면 하단에 몰린다" 전제도 **정정**(작은 볼라드는 cy 중앙 0.48). ~~🟡 평가셋에 학습분 최대 ~9% — 학습 PC 재확인 대기~~ → **8/31 재확인 완료(위 행)** | [detection.md 9-9-3](detection.md) |
| 🟡 **배포 conf 계약 정정 — 절반** (8/26) | `export_onnx.py`·`pipeline_demo.py` 기본값 **0.35 → 0.25**(확정 운영값). 🔴 **패키지 `metadata.json`·`README.md` 는 아직 0.35** — 생성물이라 `best.pt` 이관 후 재내보내기 필요 | [TODO §7](TODO.md) |
| ★★ **`C10a` ③ ONNX 양자화 — 8비트 채택 · 4비트·FP8 기각** (8/26 · FP8 8/28) | **INT8 10.61→3.15MB** · 계약 `[1,7,8400]` 그대로 · 야간 볼라드 **2/2 유지** · 음성 오탐 5→4박스. 🔴 **INT4 는 붕괴**(`qnn` 은 conf 0.01 에서도 0검출 · `generic` 은 볼라드 2/2→0/2 · Conv 는 per-block 스케일 불가라 per-channel 이 상한) · 🔴 **FP8 은 세 경로 다 막힘**(`scripts/fp8_quant_experiment.py` — A·C 는 파일은 생겨도 이 PC(CPU/CUDA, QNN EP 없음)에 `float8e4m3fn` 실행 공급자가 없어 `INVALID_GRAPH` · B 는 캘리브 단계에서 생성 자체가 실패) · 🔴 속도는 이 PC 로 판정 불가(`C11`) | [TODO `C10a`](TODO.md) · 함정 20 · 패키지 README |
| ★ **`C5` 판정 하네스 신설** (8/25) | `scripts/eval_own_night.py` — 5축(mAP · 운영점 · 음성 오탐 · 폭 구간별 recall · 볼라드 박스별 conf) · `ignore` 감점 제외 · **기본 `square`**. 촬영·라벨만 오면 바로 돈다 | [TODO `C4e` S4](TODO.md) |
| ★★★ **함정 18·19 원인 규명 + 해결** (8/25) | 전처리가 **배치 구성으로** 정해져 `eval_real_night.py` 가 한 실행 안에서 자가 두 개였다 · 하드링크가 원본을 관통해 재인코딩. 둘 다 고쳤다. 🔴 **배포 자에서 c4d 야간 볼라드 2/2→1/2** | 함정 18·19 |
| ★ **④ 색 3종 확정 · ③ ONNX 3클래스 공지** (8/24) | `bollard` 라임 `#9CFF2E`(대비를 산 선택 · `P1` 이 판정) · 출력 계약 `[1,6,8400]`→`[1,7,8400]` | 함정 16 · [share 7-0](share_yolo_c4b_20260803.md) |

> **8/23 `C4e` S0~S3 4건**(c4d held-out 판정 · 추론측 두 실험 · 배경 음성 보강 · 첫 자체
> 3클래스 런)과 **8/22 회의(클래스 3종 확정)**, 그 이전은 전부 [TODO 8장](TODO.md) 에 있다.
> 결론은 위 1~4장에 이미 반영돼 있어 여기서 되풀이하지 않는다.
