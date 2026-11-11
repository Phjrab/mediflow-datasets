# MediFlow 문서 시작점

현재는 **1~4차 학습과 기존 데이터 Source Test 비교 완료**다. 4차 신규 15개 평가와 기본~3차 36개 결과를 확인했다. 재설정 4차는 실행하지 않기로 했다. 실제 장비 질환 평가는 정답 자료가 필요하며 LLM·VLM과 장비 통합은 이후 작업이다.

| 찾을 내용 | 안내 |
| --- | --- |
| 지금까지 한 작업과 자료 위치 | [프로젝트 정리](PROJECT_SUMMARY_20261111.md) |
| 적응 후보 사용 ZIP·실행 안내 | [피부 4차·얼굴 4차·두피 2차 패키지](../results/camera_adaptation_packages_20261111/README.md) |
| 최신 기본~4차 Test 결과 | [4차 추가 Test 분석](research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md) |
| 지금 할 일과 완료 기준 | [로드맵](ROADMAP.md) |
| 조정 전 4차 결과 | [학습 결과 분석](research/camera_adaptation/ADAPTATION_STAGE4_BEFORE_REFIT_RESULT_REVIEW_20261111.md) |
| 완료한 Test 계획·파일 위치 | [4차 추가 Test 안내](research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md) |
| 실제 사용 코드·완료 결과·미실행 준비본 구분 | [실행 코드 대조표](../notebooks/EXECUTION_INDEX.md) |
| 문서 하나씩의 역할과 상태 | [전체 문서 대조표](DOCUMENT_INDEX.md) |
| 촬영 수집부터 1·2·3차·Test까지 | [카메라 적응 과정](research/camera_adaptation/CAMERA_ADAPTATION_RETROSPECTIVE_20261110.md) |
| 미실행 재설정 분석 보관 | [실제 촬영 재대조](research/camera_adaptation/CAMERA_REFIT_20261111.md) |
| 실제 촬영·분석 입력 폴더 | [camera_capture/_review 안내](../camera_capture/_review/README.md) |
| 9월 공개 데이터 연구 전체 설명 | [최종 상세 보고서](FINAL_DETAILED_REPORT_20260928.md) |
| 선정 모델과 원본 결과 | [results 안내](../results/README.md) |
| 발표·보고서 그림 | [자료실](../presentation/README.md), [보고서 그림](../presentation/report_assets/README.md) |
| 최신 이동·삭제 내역 | [완료 후 정리 기록](research/maintenance/PROJECT_WRAPUP_CLEANUP_20261111.md) |
| 이전 전체 문서 재분류 | [전체 정리 기록](research/maintenance/PROJECT_ORGANIZATION_20261111.md) |

## 문서 폴더

`research/camera_adaptation`은 촬영·분석·현재 및 완료 학습 기록이며, 그 안의 `archive`는 대체된 계획이다. `research/public_data`는 기존 공개 데이터 연구이며 피부·얼굴·두피로 구분한다. `research/integration`은 키오스크·LLM·VLM의 검토와 향후 설계, `research/maintenance`는 검사·폴더 정리 기록이다.

`PROJECT_BACKGROUND.md`는 기존 연구 배경의 기준 경로를 유지했다. 루트의 긴 보고서와 Word 두 개도 기존 상대 링크와 문서 내용을 보존하기 위해 위치를 유지했다. [9월 28일 상태](PROJECT_STATUS_20260928.md), [8주차 보고서](MEDIFLOW_8_WEEK_PROGRESS_REPORT_20260929.md), [발표 단계별 보고서](MEDIFLOW_4_PHASE_DETAILED_REPORT_20260930.md)는 해당 날짜의 기록이다. 현재 작업 순서는 위 로드맵을 따른다.

과거 문서를 읽었다는 사실만으로 그 안의 모든 제안을 실행한 것은 아니다. 실행 완료 여부는 실제 결과 JSON/CSV·체크포인트·코드 사본에 대조한다. 실제 촬영 124장은 질환 정답이 없는 환경 조사 자료이므로 실제 장비 질환 정확도는 아직 측정하지 않았다.
