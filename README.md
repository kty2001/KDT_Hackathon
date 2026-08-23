# KDT Hackathon - 밤마실

> 2026년 제8회 K-디지털 트레이닝 해커톤 (지정과제: AI 전환 시대, 인간의 역량을 확장하는 디지털 서비스 개발)

## 주제
야맹증 저시력자의 야간 보행을 돕는 실시간 AI 시각보조 서비스

**관련 분야**: 인공지능, 컴퓨터 비전

### 문서 안내

| 찾는 것 | 문서 |
|---------|------|
| ★★ **지금 상태가 어떤가 — 가장 먼저 읽을 것** | **[docs/STATUS.md](docs/STATUS.md)** |
| **무엇을 언제 해야 하는가 (전체 TODO·순서·병렬성)** | **[docs/TODO.md](docs/TODO.md)** ★ |
| 무엇을·왜 만드는가 (기획) | **이 문서** 1~5장 |
| 스크립트가 뭘 하는지 (인덱스) | [scripts/README.md](scripts/README.md) |
| 어떤 데이터를 쓰고 모델을 어떻게 정했나 | [docs/data.md](docs/data.md) |
| ③ 탐지 — 데이터 구성·baseline·**야간 검증·C4b 판정** | [docs/detection.md](docs/detection.md) ★ |
| ② 고전 기법 — 지표·arm 순위·속도·**A3 시간축** | [docs/lowlight_classical.md](docs/lowlight_classical.md) |
| ② 목적 축을 **무엇으로 어떻게 재는가** (지표 정의·결함 이력) | [scripts/metrics.py](scripts/metrics.py) — `uv run python scripts/metrics.py` |
| 실시간 성능·양자화·폼팩터 검토 | [docs/hardware_inference.md](docs/hardware_inference.md) |
| ★ **저조도 계단 탐지가 완료됐는가** (야간/주간·도메인 격차·② 오탐) | [notebooks/stairs_night_review.ipynb](notebooks/stairs_night_review.ipynb) |
| 예측·arm 결과를 **눈으로 확인** | `notebooks/` — [detect_c4_review](notebooks/detect_c4_review.ipynb) · [lowlight_arms_review](notebooks/lowlight_arms_review.ipynb) · [lowlight_lol_review](notebooks/lowlight_lol_review.ipynb) · [stair_hybrid_baseline](notebooks/stair_hybrid_baseline.ipynb) |
| 종결된 실험·결정의 원문 | [docs/archive/](docs/archive/) |
| 환경 셋업·실행 방법 | 이 문서 [개발 환경](#개발-환경) |

---

## 실행 방법

### 0. 사전 준비
```powershell
winget install astral-sh.uv     # uv 미설치 시 1회
```

### 1. 환경 생성
```powershell
uv sync                         # .venv 생성 + 의존성 설치 (uv.lock 기준 동일 환경 재현)
```
`uv sync`는 가상환경 활성화 없이 `.venv`를 자동 사용한다. 이후 모든 Python 실행은 `uv run`으로 한다.

### 2. 스크립트 실행
```powershell
uv run python scripts/inspect_datasets.py         # data/ 배치 후 무결성 검증 (최초 1회)
uv run python scripts/probe_classical.py          # ② 고전 기법 속도 프로브
uv run python scripts/metrics.py                  # ② 목적 축 지표 자기검증 (신·구 대조)
uv run python scripts/compare_lowlight.py --dataset lol   # ② arm 비교 (속도·PSNR/SSIM·노이즈)
uv run python scripts/night_eval.py --profile-only        # ② 야간 표본 밝기·포화 프로파일만
uv run python scripts/night_eval.py                       # ② 야간 표본 목적 축 전체 평가
uv run python scripts/resolution_sweep.py                 # ② C6 — 해상도 × arm 속도·목적 축 스윕
uv run python scripts/extract_nightowls.py                # NightOwls 선택 해제 계획 (--run 으로 실행)
uv run python scripts/stairnet_to_bbox.py                 # C1 계단 선분 → BBox 변환
uv run python scripts/build_detect_dataset.py             # C4 준비 — 통합 탐지 데이터셋
uv run python scripts/build_detect_dataset.py --nightowls # C4b 준비 — NightOwls 투입
uv run python scripts/train_detect.py                     # C4 — ③ 탐지 baseline 학습
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person
                                                          # ③ 야간 held-out 정직 평가
uv run python scripts/compare_detect.py                   # C4b — before/after 일괄 판정
uv run python scripts/arm_detect_eval.py                  # C7 — ② arm × ③ 탐지 mAP 판정
uv run python scripts/temporal_eval.py                    # W1 — ② A3 시간축(플리커·고스팅)
uv run python scripts/pipeline_demo.py                    # ②→③→④ end-to-end 데모 영상
uv run python scripts/emphasize.py                        # ④ 강조 렌더 데모 (②→③→④ 연결)
```

### 3. 노트북 실행
VS Code에서 `.ipynb`를 열고 커널로 **`.venv`(Python 3.13)** 를 선택한다.
> ⚠️ 작업 디렉토리를 **노트북이 든 폴더(`notebooks/`)로 열 것** — 데이터 경로를 `../data`로 잡는다.

| 노트북 | 필요 데이터 |
|--------|-------------|
| [lowlight_lol_review.ipynb](notebooks/lowlight_lol_review.ipynb) | **LOL** (1~9절) + `data/sample_image.png` (10절) |
| [lowlight_arms_review.ipynb](notebooks/lowlight_arms_review.ipynb) | ExDark · StairNet · LoLI-Street |
| [stair_hybrid_baseline.ipynb](notebooks/stair_hybrid_baseline.ipynb) | StairNet |

JupyterLab UI가 필요하면 `uv add --dev jupyterlab` 후 `uv run jupyter lab`.

> 세부 사항(uv를 쓰는 이유·의존성 변경 방법·검증된 버전 구성)은 [개발 환경](#개발-환경) 참고.

## 현재 진행 상황

> 📌 **현재 상태는 [docs/STATUS.md](docs/STATUS.md) 한 장에 정리돼 있다.**
> 단계별 채택안·확정 수치·막힌 것·반복된 함정·문서 지도가 모두 거기 있다.
> 7~8월 진행 로그 원문은 [docs/archive/progress_log_2026-07~08.md](docs/archive/progress_log_2026-07~08.md).

**요약 (2026-08-05)** — 환경·데이터·② 방식·③ 탐지가 정리됐고, 남은 병목은 **촬영과 앱**이다.

| 단계 | 상태 |
|------|------|
| ① 눈부심 억제 | `D1`(Drago) 하이라이트 압축으로 후보 확보 — ②에 통합할지 🗣️ 회의 안건 |
| ② 저조도 개선 | **고전 CV 확정**(8/1) · **표시 경로 전용**(8/2). 잠정 1위 `D1A1+bf+ts`. ⚠️ **속도 게이트 미해결** |
| ③ 위험요소 탐지 | ✅ **3클래스 완료**(8/23 `C4e` S3) — `0 person` `1 stairs` `2 bollard`. 실야간 held-out recall **0.195 → 0.609(`C4b`) → 0.635(`c4e_s3_11n`)** · `stairs` 야간 오탐 **7.7% → 0.2% → 0.0%** · 🆕 **`bollard` 추가**(개발 val mAP50 0.806 · 야간 볼라드 confidence 붕괴 없음). ONNX 3클래스 패키지 앱팀 전달(8/24 · 출력 계약 `[1,6,8400]`→`[1,7,8400]`). 🔴 최종 채택 판정은 `C5`(자체 야간 촬영) |
| ④ 선택적 강조 | ✅ 렌더 구현 완료 — 실기기 가독성 검증 대기 |
| 앱 | ❌ **구현 0건 — 8월 4주 통합의 병목** |

**다음 3수**: 🗣️ 회의 안건 7 → **`C2` 자체 야간 촬영**(크리티컬 패스 시작점) ·
**`P3` 앱 껍데기** · **`C4c` 3클래스 학습**(병렬).
자세한 순서·병렬성은 [docs/TODO.md](docs/TODO.md).

---

## 1. 추진 배경

### 제안 아이디어
- KDT 훈련과 프로젝트 경험을 통해 학습한 AI·컴퓨터비전 기술을 활용해, 야맹증 저시력자의 야간 보행을 돕는 실시간 시각보조 서비스를 개발.
- 어두운 밤길에서 위험 요소(계단·단차·보행자 등)를 실시간으로 식별·강조해 **'잔존시력'** 으로 안전하게 걷도록 지원.

### 발굴 배경 및 문제
- 시각장애인의 다수는 전맹이 아닌 저시력임 (등록 시각장애인 중 심한 장애 약 20% — Kim et al., 2022). 야맹증이 있으면 야간에 눈부심과 암흑이 겹쳐 계단·보행자를 인지하지 못해 보행이 어려워 외출을 회피하게 됨.
- 기존 보조기술은 음성 대체 중심이거나 고가 착용형이라 접근성이 낮음. 잔존시력을 활용한 야간 보조는 임상 연구 단계에 머물러 있어, 일반 스마트폰으로 즉시 쓸 수 있는 서비스는 **공백** 상태.

### 적용 분야 및 활용 방안
- **산업 분야**: 보조공학·복지(저시력 재활), 모바일 서비스
- **비즈니스 모델**: 개인용 앱(구독형) → 복지·재활기관 B2B 및 장애인 보조기기 정부지원제도 연계 보급 → 디바이스 탑재 라이선스로 확장
- **운영 시나리오**: 야간 보행 시 스마트폰으로 전방을 비추면 위험 요소가 실시간 강조되어 안전 보행 지원 (별도 장비 불요)

---

## 2. 개발 내용

### 구현 목표
스마트폰 기반 실시간 야간 보행 시각보조 — 어두운 밤길 영상이 실시간으로 '걸을 수 있는 화면'으로 바뀌는 것을 시연.
- **목표 성능**: 720p 기준 **15FPS 이상**
  - 시각장애인 보행 보조 연구의 실사용 검증 처리 속도 대역(11~15fps — Rodríguez et al., 2012)이자 임베디드 비전의 통상 실시간 요구 범위(15~30fps — Velez & Otaegui, 2015)에 해당.
  - 탐지 모델을 매 2~3프레임 주기로 실행해 확보하며, 미달 시 해상도·탐지 주기를 조정.

### 4단계 파이프라인
| 단계 | 기능 | 설명 |
|------|------|------|
| ① 눈부심 억제(글레어 제어) | 국소 강광원 검출·밝기 저감 | 가로등·전조등 등 강한 광원을 검출해 밝기를 낮춰, 야맹 환자 특유의 광과민으로 인한 시야 방해를 완화 |
| ② 저조도 개선 | 밝기 복원·대비 강화 | 어두운 영역을 단순히 밝히는 것이 아니라 '대비' 중심으로 처리 |
| ③ 위험 요소 탐지 | AI 객체탐지(YOLO 계열) — **단일 모델** | 보행자·계단을 하나의 YOLO 모델로 검출(`stairs` 클래스 통합). 저조도·저해상도 환경에 맞춰 최적화 |
| ④ 선택적 강조 | 경계선 대비 색 표시 | 검출된 위험 요소의 경계선만 대비 색으로 표시하고 내부는 원본 유지 — 잔존시력을 가리지 않으면서 '봐야 할 것'만 강조 |

> **③의 설계 근거**: 계단은 '평행한 수평 엣지의 반복'이라는 구조적 사전정보가 강해 고전 CV(엣지+기하 검증)로 라벨링 없이 처리하는 안을 먼저 검토하고 **정량 검증했다**. 결과는 야간 재현율 0.232 · 오경보율 상한 100%로 안전 기준에 미달해 **기각**했고, 딥러닝 단일 모델로 전환했다. (→ [data.md 4장](docs/data.md))

> **학습 전략**: 공개 야간·저조도 데이터셋(NightOwls, ExDark, LoLI-Street 등) 사전학습 + 자체 촬영 야간 데이터 파인튜닝의 2단 구성으로, 소규모 자체 데이터로도 저조도 실효 성능 확보.

### 개발 범위
- 대표 위험 요소 2~3종(계단·보행자 중심)에 대한 실시간 파이프라인 프로토타입 구현.
- 카메라 입력 → 처리 → 출력의 현장 시연 목표.
- **(확장)** 일정 여유 시 NVIDIA Jetson 엣지 이식, 디스플레이형 글래스 미러링 시연.

### 시스템 구성 및 아키텍처
```
                ┌─→ [탐지 경로] 원본 그대로 ──→ ③ 위험 요소 탐지 ─┐
                │                              AI 객체탐지(YOLO)   │  탐지 박스
카메라 입력 ────┤                                                  ↓
 (단안·센서 불요)│                                                  │
                └─→ [표시 경로] ① 눈부심 억제 → ② 저조도 개선 → ④ 선택적 강조 → 화면 출력
                                 글레어 제어     암부 복원·대비    대비 윤곽선(비채움)  (온-디스플레이)

  전 단계 온디바이스 추론 — 모바일: ONNX Runtime / QNN · NNAPI · Core ML
```
> ★ **①②는 탐지 앞단에 두지 않는다** (2026-08-02 `C7` 판정). 전처리를 탐지 앞에 붙이면
> mAP 가 내려가고 **`stairs` 야간 오탐이 0.1% → 5.7%** 로 되살아난다. 그래서 **탐지는
> 원본을 먹고 ①②는 표시 경로 전용**이다. 분리는 비용이 아니라 이득이다 — 두 경로가
> 독립이라 병렬화할 수 있고 ②의 지연이 ③에 더해지지 않는다
> (→ [detection.md 7장](docs/detection.md) · 데모 구현 `scripts/pipeline_demo.py`).
- **입력부**: 스마트폰 후면 카메라의 실시간 영상 스트림. 단안 카메라만으로 동작.
- **처리부**: **표시 경로와 탐지 경로의 2갈래**. 눈부심 억제(①)·저조도 개선(②)은 GPU 영상처리로 **표시 경로에만** 걸리고, 위험 요소 탐지(③)는 **원본 프레임**을 받는 경량 객체탐지 모델이다. ④ 강조는 표시 경로의 결과 위에 ③의 박스를 그린다. 추론은 모바일 런타임(ONNX Runtime + QNN·NNAPI·Core ML EP)으로 온디바이스 최적화. *(확장) NVIDIA Jetson 엣지 이식 단계에서는 TensorRT 사용.*
- **출력부**: 처리된 영상을 스마트폰 디스플레이에 실시간 표시.
- **구현 환경**: 추가 하드웨어 없이 스마트폰 단독 동작. 영상 처리·추론이 모두 기기 내에서 수행(**서버 전송 없음**). 모델 경량화(양자화·입력 해상도 조정)로 실시간 성능 확보.
- **하드웨어 폼팩터(프로토타입)**: 상용 스마트폰을 **골판지 VR 하우징**(구글 카드보드형)에 삽입한 pass-through 방식으로 시연. 별도 디바이스 제작 없이 즉시 착용형 데모 구현. 발열·지연·양자화 등 실시간 처리 검토는 [hardware_inference.md](docs/hardware_inference.md) 참고.

---

## 3. 주요 특징 및 핵심 기술

### 아이디어 컨셉
> **'균일하게 밝히기'가 아닌 선택적 시각증강** — 무엇이 위험한지 AI가 판단해 강조

스마트폰 카메라로 비춘 밤길 영상에서 ⓐ 눈부심을 억제하고 ⓑ 어두운 영역의 대비를 높이며 ⓒ 계단·보행자 등 위험 요소를 실시간 탐지해 ⓓ 그 경계만 강조함으로써, 잔존시력을 가리지 않으면서 안전 보행을 지원. 별도 장비 없이 개인 스마트폰만으로 즉시 이용 가능.

### 활용 기술
1. 딥러닝 저조도 개선(톤매핑 기반)으로 어두운 장면의 대비를 복원
2. AI 객체탐지(YOLO 계열)로 보행 위험 요소를 실시간 인식
3. 추론 최적화(모바일: ONNX Runtime + QNN·NNAPI·Core ML / 엣지 확장: Jetson·TensorRT)로 실시간 처리 성능 확보

### 기술적 특징
단순히 밝히는 영상 처리가 아니라 **'무엇이 위험한지'를 판단하는 AI가 핵심**. 저조도 개선(고전 영상처리로도 가능)을 넘어, 위험 요소를 인식·선별하는 데 딥러닝이 필수적으로 활용됨. 또한 밝기가 아닌 **'대비' 중심 설계**로 야맹·저시력의 눈부심 취약성과 대비감도 저하를 함께 고려한 증상 기반 접근.

### 유사 서비스 대비 차별성
| 기존 서비스 | 접근 방식 | 본 서비스와의 차이 |
|-------------|-----------|--------------------|
| 음성 안내형 (설리번플러스·Seeing AI 등) | 시각을 음성으로 대체 | **잔존시력을 직접 활용** |
| 화면 확대·대비형 (확대독서기 등) | 밝은 환경·정지 대상 '읽기' 특화 | **'야간×이동(보행)' 특화 + 위험 우선 판단** |
| 카메라 야간모드 | 심미 목적의 균일 증폭 | 저시력자의 '인지' 목적 **선택 강조** |

**독창성**: 시각을 음성으로 대체하지 않고 '남은 시각을 직접 강화'. 잔존시력 활용의 이동성 개선 효과가 입증된 방향(Angelopoulos et al., 2019; Rubegni et al., 2025)을 실시간·모바일로 구현.

---

## 4. 기대효과 및 활용방안

- **사회적**: 야맹은 특정 질환(망막색소변성증·녹내장·당뇨망막병증 등)부터 고령층 야간 시력 저하까지 폭넓게 발생. 취약계층의 야간 이동권을 회복해 사회활동 참여 확대에 기여.
- **기술적**: GPU 영상처리 기반 저조도 개선, AI 객체탐지, 온디바이스 추론 최적화를 스마트폰 환경에서 통합. 저사양 환경에서 실시간 성능을 확보하는 통합·최적화 구조는 다른 온디바이스 실시간 영상 AI 응용으로 확장 가능.
- **경제적**: 고가의 기존 저시력 보조기기(80만~300만원대 확대독서기 등 — Kim et al., 2022) 대비 접근성이 높아, 경제적 부담 없이 저시력 보조를 대중화.
- **구현 가능성**: 이미 검증된 공개 기술(저조도 개선 딥러닝·YOLO 계열)의 조합으로, 신규 기술 개발이 아닌 **통합·최적화 과제**. 추가 센서 없이 스마트폰 카메라만으로 동작.
- **발전 단계**: 1단계 모바일 앱 → 2단계 엣지 디바이스 최적화(저지연·저전력) → 3단계 스마트글래스 폼팩터 탑재.
- **개인정보**: 영상은 기기 내에서 처리·소멸되어 서버 전송·저장이 없음.

---

## 5. 추진 일정

### 최종 결과물
야간 보행 영상에서 눈부심을 억제하고 위험 요소(계단·보행자 등)를 실시간으로 식별·강조하는 시각보조 프로토타입. 본선 현장에서 카메라 입력 → 처리 → 화면 출력의 실시간 라이브 시연. 모의 야간 코스 보행 비교로 개선 효과의 정량 확인 병행.

### 팀원별 역할
| 역할 | 담당 | 수행 내용 |
|------|------|-----------|
| AI 파이프라인·총괄 | 팀장 | 4단계 파이프라인 설계·통합, 추론 최적화(ONNX)·실시간 성능 확보, 학습 환경·모델 검증 자동화, 프로젝트 총괄·발표 |
| 모델 | 팀원1 | 계단·보행자 검출 모델 설계·학습·튜닝(로컬 GPU), 생성형 증강 실험, 저조도 정확도 개선·모델 경량화 |
| 앱 | 팀원2 | 데모 앱 구조 개발, 카메라 입력~처리~출력 연동·UI, 실기기 성능 안정화, 서비스 인터페이스 구성 |
| 데이터·시연 | 팀원3 | 위험 요소 야간 촬영·데이터셋 구축·라벨링, 야간 시연 영상·발표자료 제작, 사전 녹화 입력 데모 모드 구성 |

### 개발 일정 (7월 ~ 9월)
- **7월 초·중**: 아키텍처 설계·인터페이스 정의, 모델 선정·베이스라인 학습, 앱 구조 설계·환경 셋업, 야간 촬영·데이터셋 구축
- **8월**: 파이프라인 개발·추론 최적화, 계단·보행자 검출 모델 학습·튜닝, 입력~처리~출력 연동·UI, 데이터 라벨링·정리
- **9월 초**: 4단계 통합·프로토타입 완성, 저조도 정확도 개선·경량화, 성능 안정화·현장 데모, 시연영상·발표자료·녹화 데모

---

## 개발 환경

**패키지 관리는 [uv](https://docs.astral.sh/uv/)를 사용한다.** 의존성은 `pyproject.toml`에 선언하고 `uv.lock`으로 버전을 고정한다.

### 셋업 (팀원 공통)
```powershell
winget install astral-sh.uv     # uv 미설치 시 1회
uv sync                         # .venv 생성 + 의존성 설치 (lock 기준 동일 환경 재현)
```

### 실행
```powershell
uv run python scripts/xxx.py                # 권장 — 가상환경 자동 사용
uv run python scripts/inspect_datasets.py   # 데이터 무결성 검증 (데이터 추가 시마다)
```

**노트북**은 `ipykernel`만 설치돼 있다. VS Code에서 `.ipynb`를 열고 커널로 **`.venv`(Python 3.13)** 를 선택하면 된다.
JupyterLab UI가 필요하면 `uv add --dev jupyterlab` 후 `uv run jupyter lab`.
> ⚠️ 노트북은 데이터 경로를 `작업 디렉토리의 상위/data`로 잡는다. **노트북이 들어 있는 폴더(`scripts/` 또는 `notebooks/`)를 작업 디렉토리로** 열지 않으면 첫 셀의 경로 assert가 실패한다.

### 의존성 변경
```powershell
uv add <패키지>                  # pyproject.toml + uv.lock 자동 갱신
```
> `pip install`을 직접 쓰지 말 것 — `uv.lock`과 실제 환경이 어긋나 팀원 간 환경 재현이 깨진다.

### uv를 쓰는 이유
- **CUDA 빌드 고정**: `pyproject.toml`의 `[tool.uv.sources]`에서 `torch`·`torchvision`을 PyTorch 전용 인덱스(`cu130`)에 `explicit = true`로 못박아 둠. 설치 순서와 무관하게 CPU 빌드가 섞이지 않는다. (pip은 `ultralytics`를 먼저 설치하면 CPU판 torch가 딸려와 GPU 학습이 죽는다)
- **환경 재현**: `uv.lock`을 커밋해 4인 팀이 동일한 버전 조합을 공유.
- **속도**: torch CUDA 휠(약 1.8GB) 등 대용량 의존성 설치가 빠름.

### 검증된 구성 (2026-07-26)
| 항목 | 버전 |
|------|------|
| Python | 3.13 (`requires-python = "==3.13.*"`) |
| torch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 |
| ultralytics | 8.4.106 |
| opencv-python | 5.0.0 |
| numpy | 2.5.1 |
| onnxruntime-gpu | 1.28.0 — CUDA·TensorRT EP 인식 확인 (**로컬 검증용**, 폰 배포 경로와 무관) |
| 학습 GPU | NVIDIA RTX 4060 Ti (VRAM 8GB) |

> **VRAM 8GB 제약**: ② 저조도 개선을 720p 풀해상도로 학습하면 OOM 위험. 패치 학습(256~384 크롭) 전제로 설계할 것 — [hardware_inference.md](docs/hardware_inference.md)의 "내부 처리 해상도 하향" 방침과도 일치.

---

## 저장소 구조

```
├── README.md                       기획 + 진행 현황 + 환경 (이 문서)
├── CLAUDE.md                       AI 어시스턴트용 프로젝트 규칙
├── docs/
│   ├── STATUS.md              ★ 현재 상태 한 장 — **여기부터 읽는다**
│   ├── TODO.md                     전체 작업 TODO — 크리티컬 패스·병렬 트랙·일정
│   ├── data.md                     데이터 현황 · AI Hub · 계단 방식 결정 근거
│   ├── detection.md                ③ 탐지 — 데이터 구성 · C4/C4b 판정 · 계단 도메인
│   ├── lowlight_classical.md       ② 고전 기법 지형도 · arm 실측 · 지표 재설계
│   ├── labeling_stairs.md          `stairs` 라벨 작업 가이드 (C3 작업자용)
│   ├── hardware_inference.md       모바일 런타임·양자화 · FPS 병목 · 폼팩터
│   ├── share_yolo_c4b_*.md         팀 공유 — ③ 결과 + 앱팀 인수인계
│   ├── archive/                    📦 종결된 실험·결정 원문 (삭제하지 않고 옮긴다)
│   └── 아이디어기획서_밤마실_20260729.pdf    원본 기획서 (수정 금지)
├── scripts/                        → 목록·용도는 **scripts/README.md** 에 있다
├── notebooks/
│   ├── detect_c4_review.ipynb      ③ 예측 결과 육안 검토 (지름길·야간 실패 분석)
│   ├── stairs_night_review.ipynb   ③ 계단 야간 탐지가 완료됐는가 (그림으로)
│   ├── colab_aihub_train.ipynb     ★ AIHub → Colab 학습 (GPU 없는 PC 에서도)
│   ├── lowlight_arms_review.ipynb  ② arm 육안 검토 (톤커브·강광원/암부 확대)
│   ├── lowlight_lol_review.ipynb   ② arm 차이 설명 + LOL 비교 + 자체 촬영본
│   └── stair_hybrid_baseline.ipynb 계단 고전 CV 정량 평가 (→ 기각 근거)
├── data/                           raw 데이터 (git 비추적, **원본 미수정**)
├── outputs/                        실험 산출물 (git 비추적)
```

### `data/` 배치

`.gitignore`로 추적하지 않으므로 팀원이 각자 아래 구조로 내려받아야 한다. 배치 후 `inspect_datasets.py`로 검증할 것.

```
data/
├── LoLI-Street/LoLI-Street Dataset/   Train·Val(low/high) + Test + YOLO Annotations
├── LOLdataset/                        our485/ · eval15/ (각 low/high)
├── Stair dataset/                     train/ · val/ (각 images/labels)
├── ExDark/                            ExDark_data/ · ExDark_Annno/ (클래스별 12개 하위 폴더)
└── NightOwls/  → D:\datasets\NightOwls  (Junction)   Validation 51,848장 + 라벨 JSON
```

> **NightOwls 는 D드라이브에 있다.** 53.5 GiB 라 C 여유(77 GiB)로는 다운로드+해제를 감당할 수 없어
> `D:\datasets\NightOwls\` 에 두고 Junction 으로 `data/NightOwls` 에 연결했다. 스크립트 경로 분기는 불필요하다.
>
> ```powershell
> New-Item -ItemType Junction -Path "<저장소>\data\NightOwls" -Target "D:\datasets\NightOwls"
> ```
>
> 취득 명령·라벨 구성·선택 해제 방침은 [data.md 5-2](docs/data.md) 참고.
