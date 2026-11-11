# 촬영 환경 적응 노트북

현재는 **1~4차 학습과 Source Test 비교 완료** 단계다. [4차 추가 Test 분석](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md)을 확인한다. 재설정 4차는 사용자 결정으로 실행하지 않고 보관한다.

| 구분 | 파일·폴더 | 상태 |
| --- | --- | --- |
| 완료한 Test 재현 | [4차만 추가 Test](completed/MediFlow_Adaptation_Stage4_Test_20261111.ipynb) | 4차 신규 15개·기존 36개 재사용 완료, [결과](../../results/camera_adaptation_stage4_test_20261111/README.md) |
| 완료한 4차 | [4차 재현 노트북](completed/MediFlow_Adaptation_Stage4_20261111.ipynb) | [모델·회수 결과](../../results/camera_adaptation_stage4_before_refit_20261111/README.md), 설정 492ee01afb45 |
| 완료 실험 재현 | [completed](completed/README.md) | 1·2·3·4차와 이전 Test |
| 미실행·중단 기록 | [archive](archive/README.md) | 재설정 4차 미실행 보관, 9개 통합 부분 이력과 삭제 기록 |

Test는 기존 Drive의 공통 ZIP v3·원본 데이터 ZIP·1차 분할 목록과 완료한 4차 모델을 사용한다. 기본~3차의 기존 Test 점수는 노트북에 포함돼 있고 다시 추론하지 않는다. GPU 선택 → 모두 실행 → Drive 연결 순서이며 새 ZIP 업로드나 Target 복호화 키 입력은 필요 없다.

[현재 Test 계획·파일 위치](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md) · [완료한 4차 분석](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_BEFORE_REFIT_RESULT_REVIEW_20261111.md) · [실제 실행 코드·결과 대조표](../EXECUTION_INDEX.md) · [로컬 분석 입력](../../camera_capture/_review/README.md).

실제 분석에 사용한 세 노트북과 완료 실험의 코드·셀·출력은 보존했다. 학습·Test가 끝난 노트북 7개는 completed에, 미실행 재설정·중단 기록 2개는 archive에 모았다. 기존 완료 결과를 확인하기 위해 다시 실행할 필요는 없다. 재현할 때만 해당 노트북을 사용한다.
