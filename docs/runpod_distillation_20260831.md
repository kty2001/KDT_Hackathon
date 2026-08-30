# RunPod 지식 증류 학습 — Pod 재접속 가이드

## 0. 결론 요약

- 지식 증류(`docs/distillation_plan_20260829.md`) 실행을 **RunPod A100 SXM 80GB Pod**에서
  진행 중이다. 이 세션이 끊겨도(다른 PC에서든) 아래 정보로 **같은 Pod에 다시 붙어** 이어갈
  수 있다 — Pod는 세션과 무관하게 RunPod 서버에서 계속 돈다.
- 가장 쉬운 재접속: **RunPod 웹 콘솔 → Pods → `bammasil-distill` → Connect → Web
  Terminal**(브라우저 로그인만 있으면 SSH 키 없이 접속 가능).
- 학습은 `tmux` 세션 안에서 돌고 있어 **SSH가 끊겨도 학습 자체는 안 죽는다.**
- ⚠️ **다 쓰면 반드시 Pod를 종료할 것** — On-Demand A100 SXM은 **$1.59/hr** 과금 중이다
  (잔액 $120에서 차감).

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
