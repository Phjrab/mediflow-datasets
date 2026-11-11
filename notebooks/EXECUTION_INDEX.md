# 실행 코드와 실제 결과 대조표 · 2026-11-11

코드가 있다는 사실과 실행 완료를 구분했다. 출력 셀이 비었다는 이유로 미실행으로 판단하지 않았다. 공개 데이터 표의 ‘대응 결과 확인’은 문서·results 내용에 대응한다는 뜻이며, 모든 Colab 탭의 실행 버전을 바이트 단위로 입증했다는 뜻은 아니다.

[완료한 기본~4차 Test](camera_adaptation/README.md) · [문서 대조표](../docs/DOCUMENT_INDEX.md)

| 노트북 | 확인 상태 | 근거 |
| --- | --- | --- |
| [4차 추가 Test](camera_adaptation/completed/MediFlow_Adaptation_Stage4_Test_20261111.ipynb) | 실제 4차 15개 평가·기존 36개 재사용·결과 회수 완료 | [결과 분석](../docs/research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md) |
| [완료한 4차](camera_adaptation/completed/MediFlow_Adaptation_Stage4_20261111.ipynb) | 세 분야 학습 완료, 설정 492ee01afb45 | [결과](../results/camera_adaptation_stage4_before_refit_20261111/README.md) |
| [01_common_dataset_audit_colab.ipynb](01_common_dataset_audit_colab.ipynb) | 공개 데이터 감사 기록 있음; 사람·세션 누수 전부 검증 아님 | [근거](../docs/research/public_data/skin/SKIN_DATA_AUDIT_20260909.md) |
| [02_common_original_vs_augmented_colab.ipynb](02_common_original_vs_augmented_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/skin/experiments/comparison_20260909_075056_72d865bf) |
| [03_common_six_experiments_colab.ipynb](03_common_six_experiments_colab.ipynb) | Hair·Web Skin 6조건 결과 확인; Skin은 이 6조건 미실행 | [근거](../docs/research/public_data/HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md) |
| [04_hair_three_seed_baseline_colab.ipynb](04_hair_three_seed_baseline_colab.ipynb) | 3개 seed 반복 완료 결과 미확인; seed 42 기준선과 혼동 금지 | [근거](사용안내.md) |
| [05_hair_paper_experiment_suite_colab.ipynb](05_hair_paper_experiment_suite_colab.ipynb) | 미실행 구현·참고 코드로 보존; 완료 실험에 포함하지 않음 | [근거](사용안내.md) |
| [06_hair_supcon_comparison_colab.ipynb](06_hair_supcon_comparison_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/experiments/supcon_compare_20260917_120447_38d75491) |
| [07_hair_supcon_repeat_seeds_colab.ipynb](07_hair_supcon_repeat_seeds_colab.ipynb) | 미실행 구현·참고 코드로 보존; 완료 실험에 포함하지 않음 | [근거](사용안내.md) |
| [08_hair_remaining_methods_screen_colab.ipynb](08_hair_remaining_methods_screen_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/experiments/paper_screen_20260917_141641_874f03f1) |
| [09_hair_b1_384_sam_screen_colab.ipynb](09_hair_b1_384_sam_screen_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/experiments/sam_screen_20260921_145602_a5e8c406) |
| [10_hair_b1_384_final_test_package_colab.ipynb](10_hair_b1_384_final_test_package_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/selected_models/v2/selection.json) |
| [11_web_skin_wsdan_attention_colab.ipynb](11_web_skin_wsdan_attention_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/web_skin/experiments/web_skin_paper_suite_20260922_124758_717b465e) |
| [12_web_skin_pmg_b1_384_colab.ipynb](12_web_skin_pmg_b1_384_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/web_skin/experiments/web_skin_pmg_b1_384_20260922_144631_ac7e5c8b) |
| [13_web_skin_pmg_b0_256_final_test_package_colab.ipynb](13_web_skin_pmg_b0_256_final_test_package_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/web_skin/selected_models/v2/selection.json) |
| [14_web_skin_medsiglip_linear_probe_colab.ipynb](14_web_skin_medsiglip_linear_probe_colab.ipynb) | Validation 실행·미채택, Test 미실행 | [근거](../results/web_skin/experiments/web_skin_medsiglip_linear_20260924_072442_f7176e07) |
| [15_hair_medsiglip_linear_probe_colab.ipynb](15_hair_medsiglip_linear_probe_colab.ipynb) | Validation 실행·미채택, Test 미실행 | [근거](../results/hair/experiments/hair_medsiglip_linear_20260924_081708_9479fbb4) |
| [16_hair_good_cross_audit_colab.ipynb](16_hair_good_cross_audit_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../docs/research/public_data/hair/HAIR_GOOD_CROSS_AUDIT_REVIEW_20260928.md) |
| [17_hair_six_class_four_experiments_colab.ipynb](17_hair_six_class_four_experiments_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../docs/research/public_data/hair/HAIR_SIX_CLASS_FOUR_RESULTS_20260928.md) |
| [18_hair_six_class_package_colab.ipynb](18_hair_six_class_package_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/selected_models/6class_performance_v1/selection.json) |
| [19_hair_six_class_light_package_colab.ipynb](19_hair_six_class_light_package_colab.ipynb) | 내용에 대응하는 공개 데이터 실험·평가 결과 확인 | [근거](../results/hair/selected_models/6class_light_v1/selection.json) |
| [20_four_cohort_runtime_benchmark_colab.ipynb](20_four_cohort_runtime_benchmark_colab.ipynb) | 저장 모델 34개 Colab 실행 시간 측정 완료, 실제 배포 장비 측정 아님 | [근거](../results/runtime_benchmark_20260928_051423/README.md) |
| [MediFlow_Adaptation_Final_Test_20261110.ipynb](camera_adaptation/completed/MediFlow_Adaptation_Final_Test_20261110.ipynb) | 실행·결과 회수 확인 | [근거](../results/camera_adaptation_final_test_20261110/summary.json) |
| [MediFlow_Adaptation_Stage1_Saved_20261110.ipynb](camera_adaptation/completed/MediFlow_Adaptation_Stage1_Saved_20261110.ipynb) | 실행·결과 회수 확인 | [근거](../results/camera_adaptation_stage1_20261110/summary.json) |
| [MediFlow_Adaptation_Stage2_Curves_20261110.ipynb](camera_adaptation/completed/MediFlow_Adaptation_Stage2_Curves_20261110.ipynb) | 실행·결과 회수 확인 | [근거](../results/camera_adaptation_stage2_20261110/summary.json) |
| [MediFlow_Adaptation_Stage2_Expanded_20261110.ipynb](camera_adaptation/completed/MediFlow_Adaptation_Stage2_Expanded_20261110.ipynb) | 실행·결과 회수 확인 | [근거](../results/camera_adaptation_stage2_20261110/summary.json) |
| [MediFlow_Adaptation_Stage3_Aligned_20261110.ipynb](camera_adaptation/completed/MediFlow_Adaptation_Stage3_Aligned_20261110.ipynb) | 실행·결과 회수 확인 | [근거](../results/camera_adaptation_stage3_20261110/summary.json) |
| [미실행 재설정 4차](camera_adaptation/archive/MediFlow_Camera_Refit_Not_Run_20261111.ipynb) | 사용자 결정으로 미실행 보관, 학습 대상 제외 | [근거](../docs/research/camera_adaptation/CAMERA_REFIT_20261111.md) |
| [MediFlow_9_Experiments_Colab_20261109.ipynb](camera_adaptation/archive/MediFlow_9_Experiments_Colab_20261109.ipynb) | 두피 조명 2회차 확인 후 중단, 9개 완료 아님 | [근거](../camera_capture/_review/history/executed_code/user_supplied_hair_lighting_history.json) |
| [Domain_Adaptation_Colab.ipynb](../output/domain_adaptation_20261108_final/Domain_Adaptation_Colab.ipynb) | 사용자 실행 분석 결과 확인, 이 실행의 학습 완료를 뜻하지 않음 | [근거](../docs/research/camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md) |
| [Hair_Skin_Analysis_Colab.ipynb](../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.ipynb) | 사용자 실행 분석 결과 확인, 이 실행의 학습 완료를 뜻하지 않음 | [근거](../docs/research/camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md) |
| [Adaptation_Preflight_Colab.ipynb](../output/adaptation_preflight_v2_20261109/Adaptation_Preflight_Colab.ipynb) | 사용자 실행 분석 결과 확인, 이 실행의 학습 완료를 뜻하지 않음 | [근거](../docs/research/camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md) |

## 보고서에 넣을 실제 분석 코드

초기 두피·피부 분석의 `source_code`, 사전 점검의 `source_code`, 완료 학습·평가 기록의 코드 식별값과 보존 사본을 기준으로 한다. 현재 `src`는 후속 수정이 있어 과거 결과의 코드 버전과 항상 같지는 않다.

- [사용자 실행 분석·코드 48개 대조 기록](../docs/research/camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md)
- [사전 점검 당시 코드](../camera_capture/_review/preflight_v2_fbe37d44/hair/source_code)
- [최종 Test 당시 코드](../results/camera_adaptation_final_test_20261110/code)
- [중단 실험과 삭제한 준비본](camera_adaptation/archive/README.md)

생성기(`scripts/build_*`)는 실행본을 포장하는 도구다. 분석·학습 계산은 `src/mediflow_datasets` 또는 결과에 보존된 당시 모듈이 담당한다. 구현·단위 검사 통과를 전체 질환 데이터 학습 완료로 세지 않는다.
