# 완료한 촬영 환경 적응 실험

아래 파일은 모두 완료한 실험의 실행 코드와 재현 자료다. 최신 [프로젝트 정리](../../../docs/PROJECT_SUMMARY_20261111.md)와 [4차 추가 Test 결과](../../../docs/research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md)를 확인한다.

| 파일 | 역할 | 결과 근거 |
| --- | --- | --- |
| [4차 추가 Source Test](MediFlow_Adaptation_Stage4_Test_20261111.ipynb) | 신규 4차 15개 평가·기본~3차 36개 결과 재사용 | [4차 Test 결과](../../../results/camera_adaptation_stage4_test_20261111/README.md) |
| [완료한 4차](MediFlow_Adaptation_Stage4_20261111.ipynb) | 조정 전 4차 492ee01afb45, 세 분야 각 5 epoch | [4차 결과](../../../results/camera_adaptation_stage4_before_refit_20261111/README.md) |
| [1차 저장 모델 적응](MediFlow_Adaptation_Stage1_Saved_20261110.ipynb) | 세 분야 × 초기·선정 저장본, 조명 적응 6회 | [1차 결과](../../../results/camera_adaptation_stage1_20261110) |
| [2차 확장 적응](MediFlow_Adaptation_Stage2_Expanded_20261110.ipynb) | 선택한 세 출발 모델에 색·미세 대비·흐림 추가 | [2차 결과](../../../results/camera_adaptation_stage2_20261110) |
| [2차 학습곡선 추가본](MediFlow_Adaptation_Stage2_Curves_20261110.ipynb) | 같은 2차 실행 및 저장된 이력의 곡선 출력 | [2차 실행 안내](../../../docs/research/camera_adaptation/ADAPTATION_STAGE2_EXPANDED_20261110.md) |
| [3차 특징 정렬](MediFlow_Adaptation_Stage3_Aligned_20261110.ipynb) | 2차와 같은 출발점·변환에 MMD·CORAL 손실 추가 | [3차 결과](../../../results/camera_adaptation_stage3_20261110) |
| [최종 Source Test](MediFlow_Adaptation_Final_Test_20261110.ipynb) | 학습 전·1차·2차·3차를 원본·조명·확장 조건으로 평가 | [최종 Test 결과](../../../results/camera_adaptation_final_test_20261110) |

완료한 결과를 확인하기 위해 다시 학습할 필요는 없다. 2026-11-11에 위치만 옮겼고 노트북 파일 내용은 보존했다. 결과 폴더의 위치와 보고된 수치는 변경하지 않았다.
