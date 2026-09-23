# MediFlow 문서 안내

이 폴더는 현재 사용하는 문서와 완료된 작업 기록을 구분한다. 처음 보는 사람은 아래 순서로
읽으면 된다.

## 현재 사용하는 핵심 문서

1. [프로젝트 최종 종합 정리](research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md):
   데이터 수집·정제·기본 학습·논문 실험·세 최종 후보와 한계까지의 현재 기준
2. [문서·코드·모델 위치 지도](PROJECT_ARTIFACT_MAP_20260923.md):
   현재 사용할 파일과 과거 기록의 위치 및 역할
3. [2026-09-22 프로젝트 종합 기록](research/MEDIFLOW_PROJECT_MASTER_SUMMARY_20260922.md):
   Web Skin PMG 최종화 전 시점의 이전 종합 기록
4. [논문 기반 Hair 실험 상세 분석](research/PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md):
   Hair에서 방법을 고른 이유, MediFlow 적용, 측정 결과와 채택 판단
5. [문서·노트북 상태 색인](DOCUMENT_STATUS_INDEX_20260922.md):
   현재 기준, 상세 근거, 과거 기록과 노트북 실행 상태 구분
6. [논문 기반 모델 강화 계획](research/PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md):
   기존 세 분류 과제를 유지한 고급 학습법의 검증 순서와 후속 정상 클래스 계획
7. [Hair SupCon 선별 결과](research/HAIR_SUPCON_RESULT_20260917.md):
   기준선과 SupCon의 Validation 성능, 클래스별 F1, 혼동 변화와 후속 검증 조건
8. [Hair 남은 방법 선별 결과](research/HAIR_REMAINING_METHODS_RESULT_20260917.md):
   DINOv2·EfficientNetV2-S·B1 384의 학습 방식과 Validation 결과
9. [Hair 후보 오답 보완성과 ensemble 검토](research/HAIR_CANDIDATE_COMPLEMENTARITY_20260921.md):
   SupCon B1·256, B1·384와 1:1 확률 평균의 동일 Validation 비교
10. [Hair B1·384 Adam 대 SAM 결과](research/HAIR_SAM_RESULT_20260921.md):
   동일 Stage 1에서 optimizer update만 바꾼 Validation 결과와 채택 판단
11. [Hair 최종 공개 데이터 후보 v2](research/HAIR_FINAL_CANDIDATE_V2_20260921.md):
   B1·384 최종 Test, 클래스별 결과, 패키지 검증과 전처리 계약
12. [프로젝트 배경](research/PROJECT_BACKGROUND.md): 세 전문 모델의 목적과 연구 배경
13. [실행 로드맵](ROADMAP.md): 현재 단계와 이후 개발 순서
14. [공통 Colab 노트북 안내](research/COMMON_COLAB_NOTEBOOKS_GUIDE.md): 데이터 검증과 재학습 방법
15. [Hair 실험 총정리](research/HAIR_EXPERIMENT_SUMMARY_20260907.md): 두피 실험 과정과 결과
16. [Web Skin 실험 총정리](research/WEB_SKIN_FULL_SUMMARY_20260908.md): 웹캠 피부 실험과 후보 모델
17. [Hair·Web Skin 실험 해설](research/HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md):
   미세조정, 256 해상도, Loss, B1을 실험한 이유
18. [후속 개선 계획](research/HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md): 다음 성능 개선 방향
19. [팀 모델 사용 안내](guides/TEAM_MODEL_QUICKSTART.md): 저장 모델을 코드에서 사용하는 방법
20. [Skin 데이터 감사](research/SKIN_DATA_AUDIT_20260909.md): 중복 발견과 clean v1 재구성 기준
21. [Skin 원본·증강 비교](research/SKIN_ORIGINAL_VS_AUGMENTED_20260909.md):
    정제 데이터의 비교 학습과 선정 모델 결과
22. [Skin 전체 총정리](research/SKIN_FULL_SUMMARY_20260909.md):
    재학습 이유, 데이터 정제, 후보 선정, 생략한 실험과 이후 방향
23. [Web Skin 논문 기반 3개 방법 실험 계획](research/WEB_SKIN_WSDAN_EXPERIMENT_PLAN_20260922.md):
    WS-DAN·PMG·MixStyle의 근거, 구현과 Validation 판단 기준
24. [Web Skin 논문 기반 3개 방법 결과](research/WEB_SKIN_PAPER_METHOD_RESULT_20260922.md):
    동일 Validation에서 세 방법을 비교하고 PMG를 최종 Test 후보로 고정한 근거
25. [Web Skin PMG·B1·384 추가 실험 계획](research/WEB_SKIN_PMG_B1_384_EXPERIMENT_PLAN_20260922.md):
    현재 PMG를 Hair 최종 scale로 확장하는 마지막 Validation 비교 설계
26. [Web Skin PMG·B1·384 결과](research/WEB_SKIN_PMG_B1_384_RESULT_20260923.md):
    기존 PMG 대비 성능 변화와 학습곡선 해석
27. [Web Skin PMG 최종 배포 후보 선정](research/WEB_SKIN_PMG_FINAL_SELECTION_20260923.md):
    B1·384와 B0·256의 성능·비용 비교, B0·256 Test 후보 고정과 패키징 계약
28. [Web Skin 최종 공개 데이터 후보 v2](research/WEB_SKIN_FINAL_CANDIDATE_V2_20260923.md):
    PMG·B0·256 최종 Test, 클래스별 결과, 패키지 검증과 다중 출력 추론 계약

[1차 발표 정리](presentations/1차.md)는 발표 자료를 만들 때 참고한다.

## 보관 문서

`archive`에는 완료된 작업의 중간 기록을 보존한다. 현재 작업 지침으로 사용하지 않는다.

- `archive/planning`: 과거 실행 계획, 데이터 평가 준비, 보류 실험
- `archive/analysis`: 이전 분석과 세부 실험 결과
- `archive/verification`: 코드 대조 및 후보 모델 검증 기록
- `archive/process`: 후보 모델 패키징 과정
- `archive/setup`: 과거 작업 프롬프트 양식

보관 문서는 당시 판단의 근거를 추적하거나 발표·보고서 수치를 확인할 때만 참고한다.
