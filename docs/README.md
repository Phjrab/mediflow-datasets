# MediFlow 문서 안내

이 폴더는 현재 사용하는 문서와 완료된 작업 기록을 구분한다. 처음 보는 사람은 아래 순서로
읽으면 된다.

## 현재 사용하는 핵심 문서

1. [프로젝트 배경](research/PROJECT_BACKGROUND.md): 세 전문 모델의 목적과 연구 배경
2. [실행 로드맵](ROADMAP.md): 현재 단계와 이후 개발 순서
3. [공통 Colab 노트북 안내](research/COMMON_COLAB_NOTEBOOKS_GUIDE.md): 데이터 검증과 재학습 방법
4. [Hair 실험 총정리](research/HAIR_EXPERIMENT_SUMMARY_20260907.md): 두피 실험 과정과 결과
5. [Web Skin 실험 총정리](research/WEB_SKIN_FULL_SUMMARY_20260908.md): 웹캠 피부 실험과 후보 모델
6. [Hair·Web Skin 실험 해설](research/HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md):
   미세조정, 256 해상도, Loss, B1을 실험한 이유
7. [후속 개선 계획](research/HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md): 다음 성능 개선 방향
8. [팀 모델 사용 안내](guides/TEAM_MODEL_QUICKSTART.md): 저장 모델을 코드에서 사용하는 방법
9. [Skin 데이터 감사](research/SKIN_DATA_AUDIT_20260909.md): 중복 발견과 clean v1 재구성 기준
10. [Skin 원본·증강 비교](research/SKIN_ORIGINAL_VS_AUGMENTED_20260909.md):
    정제 데이터의 비교 학습과 선정 모델 결과
11. [Skin 전체 총정리](research/SKIN_FULL_SUMMARY_20260909.md):
    재학습 이유, 데이터 정제, 후보 선정, 생략한 실험과 이후 방향

[1차 발표 정리](presentations/1차.md)는 발표 자료를 만들 때 참고한다.

## 보관 문서

`archive`에는 완료된 작업의 중간 기록을 보존한다. 현재 작업 지침으로 사용하지 않는다.

- `archive/planning`: 과거 실행 계획, 데이터 평가 준비, 보류 실험
- `archive/analysis`: 이전 분석과 세부 실험 결과
- `archive/verification`: 코드 대조 및 후보 모델 검증 기록
- `archive/process`: 후보 모델 패키징 과정
- `archive/setup`: 과거 작업 프롬프트 양식

보관 문서는 당시 판단의 근거를 추적하거나 발표·보고서 수치를 확인할 때만 참고한다.
