# MediFlow 문서 안내

**처음 읽을 문서:** [최종 상세 보고서](FINAL_DETAILED_REPORT_20260928.md). 초기 데이터 수집·증강·학습, 데이터 감사, 논문 기반 실험, Hair 6클래스, 실행 비용, 모델 선정과 향후 계획을 한 파일에서 읽을 수 있다.

**최신 점검과 진행 방향:** [LLM·VLM 착수 전 종합 점검](research/PRE_LLM_VLM_PROJECT_AUDIT_20261026.md).
팀원의 실제 학습 코드와 연결 방법은 [LLM·VLM 저장소 분석](research/TEAM_LLM_VLM_REPOSITORY_REVIEW_20261026.md),
[결과 설명 LLM과 독립 질환 분류 VLM 가이드](research/LLM_VLM_PROJECT_INTEGRATION_GUIDE_20261026.md)에 있다.
발표의 1~4차 순서로 다시 쓴 상세 원고는 [단계별 보고서](MEDIFLOW_4_PHASE_DETAILED_REPORT_20260930.md),
Word 파일은 [단계별 Word 보고서](MEDIFLOW_4_PHASE_DETAILED_REPORT_20260930.docx)다.

| 목적 | 문서 |
|---|---|
| 선정 모델 7개의 실제 파일 경로와 현재 사용 상태 | [선정 모델 경로 안내](../results/SELECTED_MODEL_PATHS.md) |
| 최신 파일·평가·추론 확인과 LLM·VLM 준비 | [2026-10-26 종합 점검](research/PRE_LLM_VLM_PROJECT_AUDIT_20261026.md) |
| 팀원 학습 결과와 우리 분류 모델 연결 | [저장소 분석](research/TEAM_LLM_VLM_REPOSITORY_REVIEW_20261026.md), [실행 가이드](research/LLM_VLM_PROJECT_INTEGRATION_GUIDE_20261026.md) |
| 9월 연구 마감 당시 끝난 일과 보류한 일 | [2026-09-28 상태](PROJECT_STATUS_20260928.md) |
| 다음 실험·검증의 순서와 완료 기준 | [실행 로드맵](ROADMAP.md) |
| 선정 모델과 원본 성능·속도 | [모델 최종 선정](../results/FINAL_MODEL_SELECTION_20260928.md) |
| 팀원이 모델을 불러 쓰는 방법 | [모델 사용법](../results/MODEL_USAGE.md) |
| 발표에 쓸 그림과 단계별 설명 | [발표 자료실](../presentation/README.md) |
| 학습 방법별 이유와 결과 | [연구 기록 안내](research/README.md), [전체 실험 기록](research/ALL_EXPERIMENTS_RATIONALE_AND_RESULTS_20260924.md) |
| Colab 노트북의 역할과 실행 상태 | [노트북 사용 안내](../notebooks/사용안내.md) |

`research/`에는 각 실험의 계획·결과와 감사 근거를 남겼다. 이 기록은 [최종 상세 보고서](FINAL_DETAILED_REPORT_20260928.md)가 수치를 추적할 때 사용하므로 삭제하지 않았다. 현재 실행 지침이 아닌 옛 문서 13개는 [과거 문서 ZIP](zip/README.md)에 SHA-256 목록과 함께 보존했다. 옛 기록의 당시 ‘최종’ 표현보다 최신 모델 선정과 로드맵을 우선한다.

모델·원본 결과 JSON/CSV·이미지·노트북은 기존 `results/`, `data_examples/`, `notebooks/` 위치를 유지한다. 이 문서 정리 과정에서 학습이나 실제 장비 측정을 새로 실행하지 않았다.

`docs/`의 시작 문서는 이 README, 최종 상세 보고서, 현재 상태, 로드맵이다. `research/`는 실험별 원문 근거, `presentations/`는 초기 발표 당시의 기록, `zip/`은 과거 문서 보관본이다. 모델 파일·평가 원본은 [results 안내](../results/README.md)에서 찾는다.
