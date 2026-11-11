# 촬영 환경 적응 노트북

현재는 **스마트폰 최종 Source Test·연구용 패키징 완료**다. 휴대폰 총 6회와 기존 장비 15회 학습을 확인했다. 전체 과정은 [최종 정리](../../docs/research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md)를 따른다. 아래 노트북은 완료 실험의 재현용이며 실제 장비 질환 정답 평가는 별도로 남아 있다. 기존 재설정 4차는 사용자 결정으로 실행하지 않고 보관한다.

| 구분 | 파일·폴더 | 상태 |
| --- | --- | --- |
| 스마트폰 최종 Test · 실행 완료 | [고정 최종 Test](MediFlow_Smartphone_Final_Test_20261112.ipynb) | 실제 400장·6모델×5조건·예측 검산 완료. [결과](../../results/smartphone_final_test_20261112/README.md), [재현 안내](../../output/smartphone_final_test_20261112/README.md) |
| 휴대폰 근접 후보 · 실행 완료 | [추가 1회 전용](MediFlow_Smartphone_Joint_Training_20261112.ipynb) | 5 epoch 완료·최적 3 epoch. [결과·모델](../../results/smartphone_joint_adaptation_20261112/README.md), [재현 안내](../../output/smartphone_joint_training_20261112/README.md) |
| 스마트폰 얼굴 · 실행 완료 | [스마트폰 1~4차](MediFlow_Smartphone_WebSkin_Stages_20261111.ipynb) | 실제 5회 학습 결과 회수 완료. [모델·원본 결과](../../results/smartphone_web_skin_adaptation_20261112/README.md), [재현 ZIP·안내](../../output/smartphone_web_skin_20261111/README.md) |
| 완료한 Test 재현 | [4차만 추가 Test](completed/MediFlow_Adaptation_Stage4_Test_20261111.ipynb) | 4차 신규 15개·기존 36개 재사용 완료, [결과](../../results/camera_adaptation_stage4_test_20261111/README.md) |
| 완료한 4차 | [4차 재현 노트북](completed/MediFlow_Adaptation_Stage4_20261111.ipynb) | [모델·회수 결과](../../results/camera_adaptation_stage4_before_refit_20261111/README.md), 설정 492ee01afb45 |
| 완료 실험 재현 | [completed](completed/README.md) | 1·2·3·4차와 이전 Test |
| 미실행·중단 기록 | [archive](archive/README.md) | 재설정 4차 미실행 보관, 9개 통합 부분 이력과 삭제 기록 |

Test는 기존 Drive의 공통 ZIP v3·원본 데이터 ZIP·1차 분할 목록과 완료한 4차 모델을 사용한다. 기본~3차의 기존 Test 점수는 노트북에 포함돼 있고 다시 추론하지 않는다. GPU 선택 → 모두 실행 → Drive 연결 순서이며 새 ZIP 업로드나 Target 복호화 키 입력은 필요 없다.

[현재 Test 계획·파일 위치](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md) · [완료한 4차 분석](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_BEFORE_REFIT_RESULT_REVIEW_20261111.md) · [실제 실행 코드·결과 대조표](../EXECUTION_INDEX.md) · [로컬 분석 입력](../../camera_capture/_review/README.md).

실제 분석에 사용한 세 노트북과 완료 실험의 코드·셀·출력은 보존했다. 학습·Test가 끝난 노트북 7개는 completed에, 미실행 재설정·중단 기록 2개는 archive에 모았다. 기존 완료 결과를 확인하기 위해 다시 실행할 필요는 없다. 재현할 때만 해당 노트북을 사용한다.
