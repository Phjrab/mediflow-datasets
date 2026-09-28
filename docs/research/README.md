# 연구 기록 찾기

이 폴더는 **실험을 왜 했는지와 실제 결과의 근거**를 남긴 곳이다. 현재 상태와 최종 결론은 [문서 첫 화면](../README.md), [최종 상세 보고서](../FINAL_DETAILED_REPORT_20260928.md), [모델 선정](../../results/FINAL_MODEL_SELECTION_20260928.md)을 먼저 본다. 아래의 `계획` 문서는 실행 결과가 아니며, 과거 문서의 당시 “최종” 표현이 현재 모델을 뜻하지는 않는다.

## 전체 흐름·공통 기준

- [연구 배경](PROJECT_BACKGROUND.md), [전체 실험 이유·방법·결과](ALL_EXPERIMENTS_RATIONALE_AND_RESULTS_20260924.md), [논문 실험 최종 분석](PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md)
- [저장 평가 기록 재현 점검](EVALUATION_REPRODUCTION_AUDIT_20260927.md), [34개 저장 모델 실행 시간 검토](RUNTIME_BENCHMARK_REVIEW_20260928.md)
- [공통 Colab 안내](COMMON_COLAB_NOTEBOOKS_GUIDE.md), [Hair·Web Skin 학습 설정 해설](HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md)
- [논문 기반 강화 계획](PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md), [후속 개선 방향 계획](HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md)
- [2026-09-23 당시 종합 기록](MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md) — 이후 Hair 6클래스·실행 시간 결과는 포함하지 않는 시점 기록
- [팀원 키오스크 저장소 검토](KIOSK_CORE_REVIEW_20260913.md) — 연결 계획이며 통합 완료 기록이 아님

## Skin: 현미경 피부

- [데이터 감사와 정제](SKIN_DATA_AUDIT_20260909.md) → [원본·증강 비교](SKIN_ORIGINAL_VS_AUGMENTED_20260909.md) → [후보 선정까지 종합](SKIN_FULL_SUMMARY_20260909.md)

## Web Skin: 웹캠 얼굴

- [초기 데이터·6조건·v1](WEB_SKIN_FULL_SUMMARY_20260908.md)
- [WS-DAN·PMG·MixStyle 실험 계획](WEB_SKIN_WSDAN_EXPERIMENT_PLAN_20260922.md) → [세 방법 결과](WEB_SKIN_PAPER_METHOD_RESULT_20260922.md)
- [PMG B1·384 계획](WEB_SKIN_PMG_B1_384_EXPERIMENT_PLAN_20260922.md) → [PMG B1·384 결과](WEB_SKIN_PMG_B1_384_RESULT_20260923.md)
- [PMG 배포 후보 선택](WEB_SKIN_PMG_FINAL_SELECTION_20260923.md) → [현재 v2 최종 평가·패키징](WEB_SKIN_FINAL_CANDIDATE_V2_20260923.md)
- [MedSigLIP 선별 계획](WEB_SKIN_MEDSIGLIP_LINEAR_PLAN_20260924.md) → [미채택 결과](WEB_SKIN_MEDSIGLIP_LINEAR_RESULT_20260924.md)

## Hair: 현미경 두피 5클래스

- [Clean 재학습과 초기 비교](HAIR_EXPERIMENT_SUMMARY_20260907.md)
- [SupCon 결과](HAIR_SUPCON_RESULT_20260917.md) → [DINOv2·EfficientNetV2-S·B1 384 결과](HAIR_REMAINING_METHODS_RESULT_20260917.md) → [후보 오답 보완성 검토](HAIR_CANDIDATE_COMPLEMENTARITY_20260921.md) → [SAM 결과](HAIR_SAM_RESULT_20260921.md)
- [현재 v2 최종 평가·패키징](HAIR_FINAL_CANDIDATE_V2_20260921.md)
- [MedSigLIP 선별 계획](HAIR_MEDSIGLIP_LINEAR_PLAN_20260924.md) → [미채택 결과](HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md)

## Hair: 별도 6클래스

- [양호 원본 ZIP 1차 점검](HAIR_GOOD_UPLOAD_AUDIT_20260928.md) → [기존 데이터와 교차 검사](HAIR_GOOD_CROSS_AUDIT_REVIEW_20260928.md)
- [네 조건 실험 계획](HAIR_SIX_CLASS_FOUR_EXPERIMENTS_PLAN_20260928.md) → [네 조건 결과](HAIR_SIX_CLASS_FOUR_RESULTS_20260928.md) → [5·6클래스 종합 분석](HAIR_5CLASS_6CLASS_EXPERIMENT_ANALYSIS_20260928.md)
- [성능 우선·경량 후보 비교](HAIR_SIX_CLASS_LIGHT_HEAVY_COMPARISON_20260928.md)
- [후속 양호 데이터 검증 계획](HAIR_NORMAL_V2_DATA_PLAN_20260928.md) — 아직 실행 결과가 아님

정확한 모델 파일 경로와 현재 사용 상태는 [선정 모델 경로 안내](../../results/SELECTED_MODEL_PATHS.md), 원본 수치는 각 실행의 `results/` JSON·CSV를 따른다.
