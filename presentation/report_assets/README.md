# 보고서용 그림과 분석 근거

기존 하위 폴더와 그림은 원래 위치에 보존했다. 보고서에 사용할 때는 실제 학습·평가 결과와 변환 예시·코드 검증용 그림을 구분한다.

완료한 4차는 [조정 전 설정](../../configs/camera_revised_20261111.json)을 따른다. 재설정 분석은 미실행 보관 자료다. [현재 Test 계획](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md)을 확인한다.

| 구분 | 폴더 | 용도 |
| --- | --- | --- |
| 조절값·장비 유사성·성능 수치 비교 | [adaptation_numeric_review_20261112](adaptation_numeric_review_20261112/README.md) | 완료 설정값·107개 분포 거리·특징 거리·81개 Source Test, 엑셀 7개 시트·비교 그림·원문 정밀도 |
| 완료한 어댑테이션 최신 전체표 | [adaptation_completion_20261112](adaptation_completion_20261112/README.md) | 실제 완료 21회·17개 프로필, 최신 휴대폰 1~5차 포함·정확한 조절값·모델 기록 JSON/CSV |
| 휴대폰 공동 유사성 검토 | [smartphone_joint_review_20261112](smartphone_joint_review_20261112) | 196개 후보·12개 전체 표본 재생의 검토 기록, 후속 추가 5차 완료·보호 기준 미통과 유지 |
| 실험 조절값 전체표 | [adaptation_parameters_20261111](adaptation_parameters_20261111) | 완료 15회·휴대폰 예정 5회, 조명·색·미세 대비·흐림·MMD/CORAL 수치와 학습 설정, Excel·CSV·JSON |
| 미실행 재설정 분석 | [camera_refit_20261111](camera_refit_20261111) | 원본 재측정·후보·비교 그림·미실행 설정의 검사 근거 |
| 기존 공개 데이터 연구 | [data_figures_20261026](data_figures_20261026), [validation_comparisons_20261027](validation_comparisons_20261027) | 데이터·초기 학습·Validation 비교 그림과 수치 |
| 1·2·3차 변환 설명 | [adaptation_examples_20261110](adaptation_examples_20261110) | 적용 방식 예시. 그 자체가 성능 개선의 증거는 아님 |
| 확대 후보 분석 | [measured_stronger_adaptation_20261110](measured_stronger_adaptation_20261110), [camera_similarity_check_20261110](camera_similarity_check_20261110) | 이전 확대 후보와 실제 촬영 비교 |
| 세 분야 재검토 | [camera_profile_revision_20261110](camera_profile_revision_20261110) | 현재 수정본 설정의 측정·방향 근거 |
| 현재 4차 예시 | [stage4_strength_examples_20261111](stage4_strength_examples_20261111) | 이전 2차와 수정본의 시각적 차이 |
| 학습 직전 코드 검사 | [revised_pretraining_check_20261111](revised_pretraining_check_20261111) | 로컬 배치 학습·복원·그래프/ZIP 구성 검사. 정식 4차 학습 결과가 아님 |

실제 1·2·3차 학습곡선과 최종 Test 그래프는 각각 [1차](../../results/camera_adaptation_stage1_20261110/plots), [2차](../../results/camera_adaptation_stage2_20261110/plots), [3차](../../results/camera_adaptation_stage3_20261110), [최종 Test](../../results/camera_adaptation_final_test_20261110/plots)에 있다. 수치는 해당 `results` 폴더의 JSON/CSV를 기준으로 쓴다.

완료한 [4차 학습곡선](../../results/camera_adaptation_stage4_before_refit_20261111/plots/learning_curves.png)과 [최신 기본~4차 Test 그래프](../../results/camera_adaptation_stage4_test_20261111/plots/test_comparison.png), [4차 혼동행렬](../../results/camera_adaptation_stage4_test_20261111/plots/stage4_confusion_matrices.png)도 실제 결과다. [프로젝트 정리](../../docs/PROJECT_SUMMARY_20261111.md)에서 분석·학습·평가를 구분해 확인한다.

실제 참여자 사진이 포함된 일부 비교 그림은 로컬 전용이며 공유 범위를 확인한다. 촬영 자료는 질환 정답이 없는 환경 조사 자료다. [촬영 적응 과정](../../docs/research/camera_adaptation/CAMERA_ADAPTATION_RETROSPECTIVE_20261110.md)과 [4차 재검토](../../docs/research/camera_adaptation/CAMERA_PROFILE_REVISION_20261111.md)에 해석과 한계가 있다.
