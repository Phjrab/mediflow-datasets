# 실행 준비·분석 재현 자료

이 폴더는 실제 사용한 분석 노트북 3개, 스마트폰 실행 준비본 3개와 PDF를 보관한다. 학습 모델·Test 결과는 [results](../results/README.md), 완료한 학습 노트북은 [completed](../notebooks/camera_adaptation/completed/README.md)에 있다.

| 폴더 | 실제 사용한 노트북 | 역할 |
| --- | --- | --- |
| [smartphone_final_test_20261112](smartphone_final_test_20261112/README.md) | [스마트폰 고정 최종 Test](../notebooks/camera_adaptation/MediFlow_Smartphone_Final_Test_20261112.ipynb) | 실행 완료한 재현 자료. 실제 30개 Test 결과는 [results](../results/smartphone_final_test_20261112/README.md)에 보관 |
| [smartphone_joint_training_20261112](smartphone_joint_training_20261112/README.md) | [휴대폰 근접 후보 추가 1회](../notebooks/camera_adaptation/MediFlow_Smartphone_Joint_Training_20261112.ipynb) | 실행 완료한 재현 준비본. 5 epoch 결과·모델은 [results](../results/smartphone_joint_adaptation_20261112/README.md)에 회수 |
| [smartphone_web_skin_20261111](smartphone_web_skin_20261111/README.md) | [스마트폰 전용 노트북](../notebooks/camera_adaptation/MediFlow_Smartphone_WebSkin_Stages_20261111.ipynb) | 휴대폰 분석·실행 준비본. 실제 5회 학습 결과는 [results](../results/smartphone_web_skin_adaptation_20261112/README.md)에 회수 완료 |
| [domain_adaptation_20261108_final](domain_adaptation_20261108_final) | [Domain_Adaptation_Colab.ipynb](domain_adaptation_20261108_final/Domain_Adaptation_Colab.ipynb) | 얼굴 Source Train·실제 촬영본 분석 |
| [hair_skin_batch_20261109](hair_skin_batch_20261109) | [Hair_Skin_Analysis_Colab.ipynb](hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.ipynb) | 피부·두피 Source Train·실제 촬영본 분석 |
| [adaptation_preflight_v2_20261109](adaptation_preflight_v2_20261109) | [Adaptation_Preflight_Colab.ipynb](adaptation_preflight_v2_20261109/Adaptation_Preflight_Colab.ipynb) | 세 분야 분할·유사 후보·변환·특징 거리 사전 점검 |
| [pdf](pdf) | PDF 자료 | 기존 출력 문서 보관 |

실제 분석 결과·당시 코드·입력의 연결은 [실행 코드 대조표](../notebooks/EXECUTION_INDEX.md)와 [로컬 분석 입력 안내](../camera_capture/_review/README.md)를 확인한다. 폴더 안의 기존 ZIP은 분석 재현에 쓰는 고정 실행 자료이며 다운로드 폴더의 중복 결과 ZIP과 구분한다.

## 용량 정리 후 과거 분석 ZIP 복원

완료한 과거 분석용 `mediflow_da_runtime.zip`과 `mediflow_preflight_runtime.zip`은 중복 모델을 포함해 정리했다. 노트북·준비 기록과 모든 고유 코드·설정은 보존했다. 현재 학습 결과를 볼 때는 복원이 필요 없다. 과거 생성기나 분석 노트북을 다시 실행할 때만 [복원 안내](../camera_capture/_review/history/project_cleanup_20261112/README.md)의 `domain_analysis` 또는 `preflight_v2` 명령을 먼저 사용한다. 최신 공통 ZIP v3와 스마트폰 실행 ZIP은 유지했다.
