# RunPod 지식 증류 학습 — Pod 재접속 가이드

## 0. 결론 요약

- ✅ **지식 증류(`docs/distillation_plan_20260829.md`) 실행 완료** (2026-08-31) — 교사
  (`c4f_11s_640_teacher`) · 학생 증류(`c4f_distill_11n_640`) 학습, ONNX 변환, INT8 양자화,
  `C5`(own_night) 4자 비교, 정성 확인 노트북까지 끝냈다. **Pod는 `stop` 상태**(과금 중지 ·
  볼륨은 유지)다. 상세 결과는 → **5장**.
- ✅ **NightOwls rec34 held-out 비교 완료** (2026-08-31, 같은 날 후속). `own_night`(27장)
  비교와 **방향이 어긋난다** — rec34(1,001장·박스 1,576)에서는 **학생이 교사를 이긴다**
  (mAP50 0.717 vs 0.689). own_night 단독 결과("교사가 낫다")를 최종 판정으로 쓰지 말라던
  경고가 실제로 맞아떨어졌다. 상세 표·해석은 → **5-6장**.
- ✅ **stairs 외부 야간 테스트셋(14장) 비교 완료** (2026-08-31, 같은 날 후속). rec34는
  person만 재므로, own_night 외 도메인에서 `stairs`를 보려고 CC 라이선스 외부 이미지로
  별도 held-out을 만들었다. 표본이 매우 작고(GT 16박스) **AI가 직접 그린 바운딩박스**라
  own_night·rec34와 같은 신뢰 등급이 아니다 — 방향 참고용. **다섯 후보 전부 recall이
  낮다**(0.19~0.25) — 다양한 촬영 스타일에 대한 일반화가 own_night·StairNet보다 어렵다는
  신호. 상세 → **5-7장**.
- 이하는 Pod 재접속 절차(원본 내용, Pod를 다시 켤 때 유효):
  - 가장 쉬운 재접속: **RunPod 웹 콘솔 → Pods → `bammasil-distill` → Connect → Web
    Terminal**(브라우저 로그인만 있으면 SSH 키 없이 접속 가능).
  - 학습은 `tmux` 세션 안에서 돌았다 — **SSH가 끊겨도 학습 자체는 안 죽는다.**
  - ⚠️ **Pod를 다시 켜서 쓴다면 다 쓰고 반드시 종료할 것** — On-Demand A100 SXM은
    **$1.59/hr** 과금이다(잔액 $120에서 차감).

## 1. Pod 정보

| 항목 | 값 |
|---|---|
| Pod ID | `a8u04go7tbztk0` |
| Pod 이름 | `bammasil-distill` |
| GPU | NVIDIA A100-SXM4-80GB · On-Demand · $1.59/hr |
| 데이터센터 | `EUR-IS-1` (네덜란드) |
| SSH | `ssh -i <개인키경로> root@157.157.221.29 -p 16515` |
| 프로젝트 경로(Pod 안) | `/workspace/bammasil` |
| 컨테이너 이미지 | `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster` (torch 2.9.1+cu13.0 프리인스톨 · 실제 학습은 `uv sync`가 만든 `.venv`의 torch 2.13.0+cu130 사용) |

IP·포트는 Pod를 재시작하면 바뀔 수 있다 — 아래 2장 방법으로 최신 값을 다시 조회할 것.

## 2. 다른 PC에서 Pod 상태·접속 정보 다시 조회하기

### 2-1. RunPod API 키 확보

이 저장소 루트의 `.env`(`.gitignore`에 등록돼 있어 커밋되지 않음)에
`RUNPOD_API_KEY=...` 한 줄이 있다. 다른 PC에서 이어가려면:
- 이 저장소를 그대로 가져가면 `.env`도 로컬 파일로 같이 옴(git으로는 안 옴 — 별도 복사
  필요), 또는
- RunPod 콘솔 → **Settings → API Keys**에서 같은 키를 다시 확인해 새 `.env`에 적는다.

### 2-2. `runpodctl` 설치 (새 PC)

```powershell
# Windows 예시 — GitHub 릴리스에서 windows-amd64 바이너리 받기
curl -L -o runpodctl.exe https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-windows-amd64.exe
```

```bash
# 리눅스/맥
curl -L -o runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
chmod +x runpodctl
```

### 2-3. 인증 + Pod 조회

```bash
runpodctl config --apiKey <RUNPOD_API_KEY>   # ~/.runpod/config.toml 에 저장
runpodctl pod list                            # 켜져 있는지, IP·포트 확인
runpodctl pod get a8u04go7tbztk0 -o json       # 상세 (ssh_command 필드에 최신 접속 명령 있음)
```

⚠️ `runpodctl config`는 **로컬에 SSH 키가 없으면 새로 만들어 계정에 자동 등록**한다
(`runpodctl doctor`와 동일 동작). 이 새 키는 **이미 떠 있는 이 Pod의 `authorized_keys`에는
자동으로 안 들어간다**(Pod 부팅 시점에 등록된 계정 키만 심어짐) — 아래 2-4·2-5 중 하나로
풀어야 한다.

### 2-4. (권장, 키 불필요) 웹 콘솔 Web Terminal

RunPod 콘솔(https://www.runpod.io/console/pods) 로그인 → 해당 Pod 카드의
**Connect → Web Terminal** 클릭 — SSH 키 설정 없이 바로 브라우저에서 쉘 접속된다.
아래 3장의 확인·재개 명령을 여기서 그대로 실행하면 된다.

### 2-5. (SSH로 직접 붙고 싶을 때) 새 PC에서 키를 새로 만드는 경우 — 권장

⚠️ **RunPod 계정의 "SSH Public Keys"에 새 키를 추가해도, 이미 떠 있는 이 Pod에는 자동
반영되지 않을 가능성이 높다** — 그 목록은 보통 **Pod 생성 시점**에 주입되는 값이라
실행 중인 Pod가 계정 설정 변경을 실시간으로 따라가는지는 확인되지 않았다(실측 안 함).
그래서 새 PC에서 `ssh-keygen`을 새로 해도 되지만, **아래 순서**를 따라야 확실하다.

1. 새 PC에서 키 생성: `ssh-keygen -t ed25519 -f ~/.ssh/runpod_key -N ""`
2. **2-4의 Web Terminal**(브라우저, 키 불필요)로 먼저 Pod에 들어간다.
3. Web Terminal 안에서 새 PC의 **공개키**를 `authorized_keys`에 추가:
   ```bash
   echo "<새 PC에서 만든 ssh-ed25519 공개키 전체>" >> ~/.ssh/authorized_keys
   ```
4. 이후 새 PC에서 바로 SSH 가능:
   ```bash
   ssh -i ~/.ssh/runpod_key -p 16515 root@<현재 IP>
   ```

키를 새로 만들지 않고 **기존 개인키를 그대로 재사용**하고 싶다면, 이 PC의
`C:\Users\USER\.runpod\ssh\runpodctl-ssh-key`(개인키)를 안전한 방법으로 옮겨도 된다
(USB 등 — 채팅·이메일 평문 전달 금지). 이 경우 3번(authorized_keys 추가) 없이 바로
`ssh -i runpodctl-ssh-key -p 16515 root@<IP>`로 접속된다 — 이미 등록돼 있는 키라서다.

## 3. 접속 후 — 진행 상황 확인·재개

### 3-1. 지금까지 해 둔 것

1. `uv sync` 완료 — `/workspace/bammasil/.venv`에 `torch==2.13.0+cu130`,
   `ultralytics==8.4.106` 등 설치됨(`onnxruntime-gpu`는 Linux 휠이 없어 제외 — 학습에는
   불필요, ONNX 내보내기는 로컬 Windows에서 진행).
2. 데이터셋 — `/workspace/bammasil/data/detect_v3/`에 `detect_v3`(train 16,291장 ·
   val 3,781장) 업로드·압축해제·무결성 확인 완료. `data.yaml`에 `path:` 없음(이식성
   확보, → 함정 11).
3. `scripts/train_detect.py`에 `--distill-model`/`--dis` 인자 추가됨(로컬 저장소에도
   동일하게 반영돼 있음 — 커밋 여부는 사용자 판단).
4. **교사(`YOLO11s`) 학습**을 `tmux` 세션 `teacher`에서 실행 중:
   ```
   .venv/bin/python scripts/train_detect.py --model yolo11s.pt \
       --data data/detect_v3/data.yaml --imgsz 640 --batch -1 \
       --name c4f_11s_640_teacher
   ```
   - `--batch -1`(AutoBatch)이 GPU 60% 기준으로 **batch 75**를 자동 선택함(기본값 16은
     8GB급 GPU 기준이라 A100 80GB에서 너무 작았음 — 이 프로젝트의 다른 GPU 학습을
     RunPod 대형 GPU에서 돌릴 때도 **`--batch -1` 사용을 권장**).
   - 로그: `/workspace/bammasil/teacher_train.log`
   - 산출물: `/workspace/bammasil/outputs/detect/c4f_11s_640_teacher/`

### 3-2. 학습 상태 확인

```bash
tmux list-sessions                       # 'teacher'(또는 'student') 세션이 살아있는지
tmux attach -t teacher                   # 실시간으로 붙어서 보기 (나갈 땐 Ctrl+B, D)
tail -50 /workspace/bammasil/teacher_train.log
grep -E "Results saved to|EarlyStopping|Traceback|CUDA out of memory" \
    /workspace/bammasil/teacher_train.log   # 완료/실패 여부만 빠르게
nvidia-smi
```

### 3-3. 교사 학습이 끝나 있다면 → 학생 증류 학습 시작

```bash
cd /workspace/bammasil
tmux new-session -d -s student \
  '.venv/bin/python scripts/train_detect.py --model yolo11n.pt \
   --distill-model outputs/detect/c4f_11s_640_teacher/weights/best.pt --dis 6.0 \
   --data data/detect_v3/data.yaml --imgsz 640 --batch -1 \
   --name c4f_distill_11n_640 2>&1 | tee /workspace/bammasil/student_train.log'
```

### 3-4. 둘 다 끝났다면 → 결과 회수 (로컬로 다운로드)

로컬(작업 PC)에서:

```bash
scp -i <개인키> -P <포트> -r \
  root@<IP>:/workspace/bammasil/outputs/detect/c4f_11s_640_teacher \
  root@<IP>:/workspace/bammasil/outputs/detect/c4f_distill_11n_640 \
  D:/kdt_hackathon_outputs/   # 원하는 로컬 경로로
```

받은 뒤 로컬에서 `docs/distillation_plan_20260829.md` 5-4·5-5(평가·배포 정합성)를
그대로 진행한다 — `scripts/eval_own_night.py`, `scripts/export_onnx.py`,
`scripts/quantize_onnx.py`는 전부 로컬(Windows) 스크립트다.

### 3-5. 다 끝나면 — Pod 종료 (과금 중지)

```bash
runpodctl pod stop a8u04go7tbztk0     # 정지(과금은 멈추나 볼륨은 유지 — 재시작 가능)
runpodctl pod remove a8u04go7tbztk0   # 완전 삭제(볼륨까지 제거 — 결과 다 받은 뒤에만)
```

또는 웹 콘솔에서 Pod 카드의 Stop/Terminate 버튼.

## 4. 참고

- 전체 증류 계획·근거: `docs/distillation_plan_20260829.md`
- 이번 실행의 플랜 파일(로컬 전용, 저장소 밖): `C:\Users\USER\.claude\plans\floofy-mixing-coral.md`
- RunPod 템플릿 `a9dk3g7cny`(`Runpod Pytorch 2.9.0 for clusters`)를 쓴 이유: CUDA 13.0이
  프리인스톨돼 있어 이 프로젝트의 `pyproject.toml` `pytorch-cu130` 인덱스와 바로 맞음.
- Pod 생성 시 `--min-cuda-version 13.0`을 줘서 호스트 드라이버가 CUDA 13 미만인 곳에는
  배정되지 않도록 했다(실측: 드라이버 580.159.04, CUDA 13.0 확인됨).

## 5. 결과 (2026-08-31 완료)

### 5-1. 학생 증류 학습

3장의 명령대로 Pod에서 실행하되, **GPU 사용량을 60%(기본 `--batch -1`) 대신 75%로
올려서** 돌렸다. `scripts/train_detect.py`의 `--batch` 인자가 원래 `type=int`라 소수 비율
(`0.75`)을 못 받아, **로컬·Pod 양쪽에서 `type=float`로 한 줄 수정**했다(ultralytics
AutoBatch는 `--batch`에 0~1 사이 float를 주면 그 값을 목표 메모리 비율로 쓴다 —
`autobatch.py`의 `fraction=batch if 0.0<batch<1.0 else 0.6`).

```bash
.venv/bin/python scripts/train_detect.py --model yolo11n.pt \
    --distill-model outputs/detect/c4f_11s_640_teacher/weights/best.pt --dis 6.0 \
    --data data/detect_v3/data.yaml --imgsz 640 --batch 0.75 \
    --name c4f_distill_11n_640
```

- AutoBatch가 **batch 165**를 잡았다(59.06G/79.25G · 75% 정확히 반영, 교사 때는 60%에서
  batch 75였음).
- **100/100 epoch 끝까지 완주**(조기종료 없음), 총 소요 약 3시간 20분(교사는 95epoch·
  3시간10분).
- 최종 val(개발용, `detect_v3`): mAP50 **0.832** · mAP50-95 **0.601**
  (person 0.714/0.439 · stairs 0.987/0.887 · bollard 0.796/0.478).
  교사(`c4f_11s_640_teacher`, mAP50 0.861 · mAP50-95 0.630)보다 소폭 낮다 — 예상된 방향
  (11n < 11s).
- 가중치를 로컬로 회수: `outputs/detect/c4f_distill_11n_640/weights/{best,last}.pt`
  (+ `results.csv`·`args.yaml`).

### 5-2. ONNX 변환 · INT8 양자화 (로컬, CPU 전용 PC)

기존 스크립트 그대로 재사용, 코드 변경 없음.

```powershell
uv run python scripts/export_onnx.py `
  --weights outputs/detect/c4f_distill_11n_640/weights/best.pt `
  --val-images outputs/datasets/aihub_colab_rehearsal/images/val
uv run python scripts/quantize_onnx.py `
  --onnx outputs/export/bammasil_det_c4f_distill_11n_640_640/bammasil_det_c4f_distill_11n_640_640.onnx
```

- ONNX: PT↔ONNX 정합성 ✅ 통과(개수 불일치 0/8 · 좌표 최대 0.0001px). 산출물
  `outputs/export/bammasil_det_c4f_distill_11n_640_640/`.
  ⚠️ `export_onnx.py`는 산출 폴더를 `OUT/<런이름>`(접두사 없음)으로 만드는데, 기존
  교사·`c4e_s3_11n` 산출물은 `bammasil_det_<런이름>_<imgsz>/`(zip과 같은 이름)로 존재해
  현재 스크립트 동작과 어긋난다(`git log`로 확인 — 이 로직은 바뀐 적이 없어 **기존 폴더가
  생성 후 수동 개명된 것**으로 보임). 같은 규칙을 맞추려고 **`Move-Item`으로 개명**했다.
- INT8: `outputs/quantization/bammasil_det_c4f_distill_11n_640_640-INT8/`. `generic`
  3.15MB(3.37배 축소)·`qnn` 3.04MB(3.49배). FP32↔INT8 검출수 불일치 **0/7 둘 다** —
  함정 20(전체 양자화 시 검출 0 붕괴) 재발 없음(`Conv`만 양자화 그대로 유효).
  CPU 참고 지연: FP32 39.7ms → generic 45.2ms · qnn 63.1ms(⚠️ 참고치일 뿐 — 함정 3).

### 5-3. `C5` own_night 4자 비교 — 교사·학생·`c4e_s3_11n`(FP32)·`c4e_s3_11n`-INT8

`c4e_s3_11n`의 `.pt`가 이 PC에 없어 **FP32 ONNX**(`outputs/export/bammasil_det_c4e_s3_11n_640/`)
로 대체했다. 결과는 `outputs/detect/c5_own_night_distill.json`(기존
`c5_own_night_review.ipynb`가 참조하는 `c5_own_night.json`과는 별도 파일 — 안 건드림).

```powershell
uv run python scripts/eval_own_night.py `
  --weights outputs/detect/c4f_11s_640_teacher/weights/best.pt `
  --weights outputs/detect/c4f_distill_11n_640/weights/best.pt `
  --weights outputs/export/bammasil_det_c4e_s3_11n_640/bammasil_det_c4e_s3_11n_640.onnx `
  --weights outputs/quantization/bammasil_det_c4e_s3_11n_640-INT8/bammasil_det_c4e_s3_11n_640-INT8_generic.onnx `
  --device cpu --out outputs/detect/c5_own_night_distill.json
```

conf 0.25 · own_night 27장(GT person 4·stairs 6·bollard 13) 기준:

| | mAP50(종합) | recall | precision | F1 | 음성오탐(전체) |
|---|---|---|---|---|---|
| 교사 `c4f_11s_640_teacher` | 0.600 | 0.652 | 0.625 | 0.638 | 2 |
| 학생 `c4f_distill_11n_640` | 0.554 | 0.652 | 0.556 | 0.600 | 5 |
| `c4e_s3_11n`(FP32) | 0.523 | 0.696 | 0.593 | 0.640 | 4 |
| `c4e_s3_11n`-INT8 | 0.525 | 0.739 | 0.654 | 0.694 | 4 |

- person mAP50 전부 0.995(공통) · stairs는 GT 6개뿐이라 전 후보가 약함(교사·학생은
  recall 0).
- 학생은 교사 대비 recall은 유지(0.652)했지만 precision이 떨어져(음성 오탐 2→5) 종합
  지표가 낮다.
- `c4e_s3_11n`-INT8이 FP32보다 이 27장에서는 오히려 근소 우세 — **표본이 너무 작아
  (GT 23개) 통계적 의미를 두지 말 것.**
- ⚠️ **이 표는 `own_night` 단독 결과다** — 위 0장 요약대로 NightOwls rec34 교차 확인
  전이라 "학생/교사 중 무엇이 낫다"의 최종 근거로 쓰지 말 것.

### 5-4. 정성 확인 노트북

`notebooks/c4f_distill_test_real_viz.ipynb` 신설 — `c5_own_night_review.ipynb`와 같은
셀 구성(마크다운 개요 → 설정 → 성능 비교 표(5-3 재사용) → 예측 시각화 → GT 라벨 감사).
`data/own_night`에서 `test_real_data` 유래 7장만 골라 학생 모델 단독으로 예측·GT를
그린다. `jupyter_client`로 커널을 직접 띄워 **실행 결과(이미지 포함)까지 저장**했다
(프로젝트에 `nbconvert`가 없어 새 의존성 추가 없이 이 방식을 씀).

### 5-5. 다음에 할 것

1. ✅ ~~`data/NightOwls`(또는 최소 rec34분)를 이 PC로 옮겨 `scripts/eval_nightowls.py
   --recordings 34 --drop-unlabeled-person`으로 교사·학생·`c4e_s3_11n`를 재비교.~~ →
   **완료(5-6장)** — `data/NightOwls`는 이미 이 PC에 Junction으로 연결돼 있었다(8/31 재확인).
2. `docs/distillation_plan_20260829.md` 5-6절 채택 기준(`C11` 속도 + `C5` 야간 실측)을
   마저 통과해야 배포 후보로 올릴 수 있다 — 이번 결과만으로는 아직 미달.
3. 🔴 **own_night과 rec34가 교사/학생 우열을 뒤집는다** — 왜 뒤집히는지(표본 크기·
   촬영 시점 차이·클래스 구성 차이 등) 아직 원인 규명 안 됨. `C2` 추가 촬영으로
   own_night 표본을 키우기 전까지는 어느 쪽도 확정적 근거가 아니다.
4. ✅ ~~stairs를 own_night 외 데이터로도 확인~~ → **완료(5-7장)** — 단 AI가 그린 라벨
   14장뿐이라 참고용. `C2`가 stairs의 유일한 정식 판정 수단이라는 결론은 그대로다.

### 5-6. NightOwls rec34 교차 확인 (2026-08-31 완료)

`docs/STATUS.md` 3장 함정 1의 관례대로 own_night(27장)과 별개 도메인인 NightOwls
held-out(rec34, 학습 미사용)에서 같은 5개 후보를 재측정했다. `data/NightOwls`는 이미
Junction으로 로컬에 연결돼 있었고(`c4e_s3_11n`의 `.pt`도 로컬에 있어 ONNX 대체 불필요),
`scripts/eval_nightowls.py`에 `--device` 옵션을 추가해(기존 GPU 고정 → `--device cpu`로
INT8 ONNX 평가 가능, `eval_own_night.py`의 기존 패턴과 동일) 5건을 각각 실행했다.

```powershell
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person `
  --weights outputs/detect/c4f_11s_640_teacher/weights/best.pt --device 0
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person `
  --weights outputs/detect/c4f_distill_11n_640/weights/best.pt --device 0
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person `
  --weights outputs/quantization/bammasil_det_c4f_distill_11n_640_640-INT8/bammasil_det_c4f_distill_11n_640_640-INT8_generic.onnx `
  --device cpu --name c4f_distill_11n_640_INT8__rec34_labeled
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person `
  --weights outputs/detect/c4e_s3_11n/weights/best.pt --device 0
uv run python scripts/eval_nightowls.py --recordings 34 --drop-unlabeled-person `
  --weights outputs/quantization/bammasil_det_c4e_s3_11n_640-INT8/bammasil_det_c4e_s3_11n_640-INT8_generic.onnx `
  --device cpu --name c4e_s3_11n_INT8__rec34_labeled
```

rec34(`--drop-unlabeled-person`) 이미지 1,001장 · pedestrian 박스 1,576개 기준:

| | mAP50 | mAP50-95 | precision | recall | stairs 오탐 |
|---|---|---|---|---|---|
| 교사 `c4f_11s_640_teacher` | 0.689 | 0.360 | 0.824 | 0.633 | 0.0% |
| 학생 `c4f_distill_11n_640`(FP32) | **0.717** | 0.361 | 0.836 | **0.668** | 0.0% |
| 학생 INT8 | 0.692 | 0.362 | 0.850 | 0.631 | 0.0% |
| `c4e_s3_11n`(FP32) | 0.710 | 0.354 | 0.787 | 0.635 | 0.0% |
| `c4e_s3_11n` INT8 | 0.649 | 0.332 | 0.794 | 0.582 | 0.0% |

(`c4e_s3_11n` FP32 행은 STATUS 후보표의 기존 수치 0.710/0.635/0.0%와 정확히 일치 — 하네스
재현성 확인됨.)

**own_night과 방향이 어긋난다.**

- own_night: 교사(0.600) > 학생(0.554) > `c4e_s3_11n` INT8(0.525) ≈ FP32(0.523).
- rec34: **학생(0.717) > `c4e_s3_11n` FP32(0.710) > 학생 INT8(0.692) > 교사(0.689)** >
  `c4e_s3_11n` INT8(0.649).

학생이 own_night에서는 교사보다 못했지만(정밀도 하락·음성오탐 2→5), 표본이 27장(GT 23개)
뿐이라 통계적으로 얇았다. rec34는 GT 1,576개로 표본이 훨씬 크고, 여기서는 **교사가 오히려
5개 후보 중 최하위권**(FP32 중 최저)이다. 즉 이번 4/5자 비교에서 "학생이 교사보다 나쁘다"는
own_night만의 결론이었고, 서로 다른 도메인에서 방향이 반대로 나왔다 — `docs/STATUS.md`
3장 함정 1("판정은 항상 held-out 또는 자체 촬영분에서 하되, 한쪽만 보고 결론 내리지 말 것")이
경고한 바로 그 상황이다.

양자화(INT8) 효과는 두 도메인에서 **방향이 일치**한다 — 학생·`c4e_s3_11n` 둘 다 INT8이
FP32보다 mAP50이 낮다(학생 −0.025 · `c4e_s3_11n` −0.061). own_night에서 `c4e_s3_11n`
INT8이 FP32보다 근소 우세로 나온 건 표본이 작아서(GT 23개) 생긴 노이즈였을 가능성이 크다 —
rec34(GT 1,576개)의 하락 방향이 더 신뢰할 만하다. stairs 오탐은 5건 전부 0.0%로 own_night과
같은 방향(NightOwls엔 계단 GT가 없어 오탐만 셈).

⚠️ 남는 한계 — NightOwls는 **차량 대시캠 시점**이라 보행 시점과 다르고, `bollard`·`stairs`
GT가 없어 person만 잰다(→ 함정 없음, 기존 하네스 설계 그대로). 교사/학생 순위가 왜
도메인마다 뒤집히는지는 원인 미상 — `C2` 자체 촬영이 늘어야 own_night 쪽 표본 신뢰도가
개선된다.

### 5-7. stairs 외부 야간 테스트셋 (2026-08-31 완료)

rec34는 person만 잰다 — `stairs`는 own_night(GT 6개)에서만 측정된 채였다. 기존 프로젝트
자산인 StairNet 야간 76장(`scripts/eval_stairs_night.py`)은 `detect_v3`의 val 스플릿에
그대로 들어가 있어(→ `build_detect_dataset.py:add_stairs`) teacher/student `best.pt`
체크포인트 선택(val fitness)에 이미 관여했고, "계단이 화면을 채운 사진"이라 원거리 접근
구간도 못 잰다. 그래서 **완전히 새로운 외부 이미지**로 별도 held-out을 만들었다.

**소싱** — Openverse API(`api.openverse.org`, Flickr·Wikimedia Commons의 CC0/CC-BY 이미지
통합 검색)로 "stairs at night" 등 6개 검색어를 돌려 79장을 모으고, 육안 검수로 **14장**을
채택했다(실내 장식적 나선계단·필름 스캔 아티팩트·과도한 포스터라이즈·극단적 어안왜곡·계단이
실제로 안 보이는 사진 등은 제외). 근거리·원거리·다중 플라이트(계단참 너머 2번째 플라이트)·
사람 포함·순수 배경(계단이 안 보이는 오탐 검증용 네거티브 1장)을 섞었다. 출처·라이선스는
`data/external_stairs_night/images_raw/sources.csv`에 기록(전부 CC BY 2.0/3.0).

**라벨링** — `docs/labeling_stairs.md` 기준(2단 이상만 `stairs`, 계단 전체를 1박스, 별개
플라이트는 각각 1박스)을 따르되, **클릭 드래그 도구가 아니라 이미지를 보고 바운딩박스
좌표를 시각적으로 추정**했다. own_night·rec34와 달리 사람 손 라벨이 아니므로 정밀도가
낮다 — 대표 이미지 4장을 박스 오버레이로 재검토해 대략적으로는 맞음을 확인했지만, 픽셀
단위 정확도는 보장 못 한다. **이 결과는 방향 참고용이지 최종 판정 근거가 아니다.**

**평가** — `scripts/eval_stairs_night.py`는 소스 경로가 고정돼 있어 외부 데이터를 못 받는다.
새 스크립트 대신 ultralytics 기본 CLI로 5종을 재사용 데이터셋(`data/external_stairs_night/
data.yaml`)에 대해 돌렸다:

```powershell
uv run yolo val model=<가중치> data=data/external_stairs_night/data.yaml imgsz=640 device=<0|cpu>
```

이미지 14장(양성 13 · 음성 1) · GT 박스 16개 기준:

| | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| 교사 `c4f_11s_640_teacher` | **0.228** | 0.063 | 0.729 | 0.188 |
| 학생 `c4f_distill_11n_640`(FP32) | 0.176 | 0.057 | 0.398 | 0.250 |
| 학생 INT8 | 0.148 | 0.033 | 0.411 | 0.188 |
| `c4e_s3_11n`(FP32) | 0.162 | 0.036 | 0.354 | 0.250 |
| `c4e_s3_11n` INT8 | 0.221 | 0.046 | 0.647 | 0.250 |

음성 이미지(계단 없는 배경 1장) FP — 5종 전부 **0박스**(conf 0.25 기준, 함정 20 재발 없음).

**읽는 법** — GT 16개뿐이라 recall 한 칸 차이가 0.0625 단위로 뛴다. 정밀 순위 판정에
쓰지 말고 방향만 본다:

- mAP50은 own_night과 같은 방향(교사 > 학생)이나, **recall은 학생이 교사보다 오히려
  높다**(0.250 vs 0.188) — precision 차이가 mAP를 가른다. own_night의 "학생이 정밀도에서
  밀린다"는 관찰과 결이 같다.
- **다섯 후보 전부 recall이 0.19~0.25로 낮다** — own_night(stairs recall은 GT가 6개뿐이라
  단정 못 하지만) · StairNet 야간(mAP50 0.993, → STATUS 2장 4번)보다 뚜렷이 나쁘다. 학습
  분포(StairNet·자체 촬영)와 다른 촬영 스타일(다양한 카메라·구도·조명)에 대한 일반화가
  약하다는 신호로 읽을 수 있다 — 다만 라벨 정밀도가 낮은 점을 감안해야 한다.
- INT8 방향은 후보마다 다르다(학생은 하락, `c4e_s3_11n`은 오히려 상승) — GT 16개에서는
  한 검출 차이가 방향을 뒤집을 수 있어 이 결과만으로 INT8 stairs 성능을 판정하지 않는다.

⚠️ **이 결과는 `C2` 자체 야간 촬영을 대체하지 않는다** — 라벨이 AI 추정치이고 표본이
14장뿐이다. StairNet·own_night과 달리 사람 손 검수를 거치지 않았으므로, 배포 판단에는
쓰지 말고 "방향이 own_night과 비슷한지" 정도의 참고로만 쓴다.
