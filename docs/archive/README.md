# 📦 archive — 종결된 실험·결정의 원문

여기 있는 것은 **더 이상 현재 방침이 아니다.** 삭제하지 않고 옮겨 둔다 (CLAUDE.md 규칙).
현재 상태는 [../STATUS.md](../STATUS.md), 할 일은 [../TODO.md](../TODO.md).

> **왜 남기는가** — 이 프로젝트의 발표 서사가 *"가설 → 정량 측정 → 기각 → 재설계"* 라서,
> 기각의 근거가 곧 결과물이다. 그리고 같은 함정을 두 번 밟지 않기 위한 기록이기도 하다.

| 문서 | 무엇 | 왜 종결됐나 | 지금의 답 |
|---|---|---|---|
| [lowlight_method_decision_2026-07.md](lowlight_method_decision_2026-07.md) | ② 방식 A/B/C 비교 · 가설 H1~H3 · paired 촬영 존폐 논의 | ② 는 **고전 CV 로 확정**(8/1) · 촬영은 **존폐 결정 불요**로 재정의(7/29) | [data.md 2-3·2-4](../data.md) |
| [detect_superseded_2026-08.md](detect_superseded_2026-08.md) | ② 전처리가 재현율을 "절반"으로 떨어뜨린다는 프리뷰 · 30클래스 방침 | ⚠️ **붕괴한 모델을 채점기로 써서 영향이 과장됐다** · 클래스는 3종으로 좁혀졌다 | [detection.md 7장](../detection.md) · [data.md 3-1-1](../data.md) |
| [stair_classical_rejection_2026-07-26.md](stair_classical_rejection_2026-07-26.md) | 계단 고전 CV 하이브리드(Canny→Hough+기하검증) | 야간 재현율 **0.232** · 오경보 상한 100% → 안전 기준 미달로 **기각** | YOLO `stairs` 클래스 통합 |
| [lowlight_arms_2026-07-29.md](lowlight_arms_2026-07-29.md) | ② arm 초기 실측 (구 지표 기준) | 지표를 **절대 기준으로 재설계**(7/31)해 수치가 갈렸다 | [lowlight_classical.md 6-4](../lowlight_classical.md) |
| [lowlight_lol_notebook_guide.md](lowlight_lol_notebook_guide.md) | LOL 기준 arm 육안 비교 노트북 안내 | 노트북이 신 지표로 재실행됨(8/1) | `notebooks/lowlight_lol_review.ipynb` |
| [data_decisions_2026-07.md](data_decisions_2026-07.md) | data.md 부록 A — 초기 데이터 결정 이력 | 본문에 흡수 | [data.md](../data.md) |
| [storage_hold_2026-07-27.md](storage_hold_2026-07-27.md) | 저장공간 부족으로 AI Hub 보류 | D드라이브 확인으로 **해소**(8/1). 이후 300GB 취득(8/5) | [data.md 3-1](../data.md) |
| [progress_log_2026-07~08.md](progress_log_2026-07~08.md) | 7~8월 진행 로그 원문 (README 에서 분리) | 요약이 STATUS·TODO 로 이관 | [../STATUS.md](../STATUS.md) |

## ⚠️ 여기서 배운 것 중 아직 유효한 함정

전체 목록은 [STATUS 3장](../STATUS.md)에 있다. 그중 archive 가 근거인 것:

- **붕괴한 모델을 채점기로 쓰면 전처리의 영향이 과장된다** — 채점기의 건강 상태를 먼저
  확인하지 않으면 어떤 전처리 실험도 해석할 수 없다 (`detect_superseded`).
- **지표를 영상 통계에 상대적으로 정의하지 말 것** — 영상이 바뀌면 *재는 대상 자체*가
  달라진다. 노이즈·글레어·대비 3개가 전부 이 결함으로 육안과 어긋났다 (`lowlight_arms`).
- **판정 기준은 측정 전에 못박는다** — 계단 고전 CV 를 "재현율 ≥ 0.8" 로 미리 정하고
  기각했던 절차가 이후 ②·③ 판정의 골격이 됐다 (`stair_classical_rejection`).
