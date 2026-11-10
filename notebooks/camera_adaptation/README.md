# 촬영 환경 적응 노트북 상태 — 2026-11-10

**현재 실행본** [3차 MMD·CORAL 특징 정렬 노트북](MediFlow_Adaptation_Stage3_Aligned_20261110.ipynb)과
[3차 실행 안내](../../docs/research/ADAPTATION_STAGE3_ALIGNED_20261110.md).
1차 6개 실험은 [1차 노트북](MediFlow_Adaptation_Stage1_Saved_20261110.ipynb)으로 실행을 마쳤다.
2차는 [확장 적응 노트북](MediFlow_Adaptation_Stage2_Curves_20261110.ipynb)에서 세 분야의 학습을 마쳤다.
3차는 기존 공통 ZIP v3와 1·2차 완료 기록을 사용한다.
[1차 학습곡선](../../results/camera_adaptation_stage1_20261110/plots/stage1_learning_curves.png)은
로컬에 저장한 실제 1차 기록으로 생성했다.

정리 시점에 이전 학습 계획을 보관용으로 전환했다.
이전 9개 통합 학습과 별도 두피·피부·얼굴 학습 준비본은 보관용이다.
1차 v1·v2 노트북은 당시 준비 이력으로 보관한다.

실제로 분석·점검 결과를 확보한 실행본은 다음 세 위치에 보존한다.

- [얼굴 초기 분석](../../output/domain_adaptation_20261108_final/Domain_Adaptation_Colab.ipynb)
- [두피·피부 일괄 분석](../../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.ipynb)
- [세 분야 사전 점검](../../output/adaptation_preflight_v2_20261109/Adaptation_Preflight_Colab.ipynb)

이 세 노트북도 당시 분석 재현용이다. 특히 초기 분석 노트북의 후반부에는
학습·평가 칸이 함께 있으므로 전체 실행을 현재의 새 학습 지시로 사용하지 않는다.

이전 준비본의 큰 실행 ZIP은 정리하고 코드·설정·노트북 이력은
[보관 기록](../../camera_capture/_review/training_preparation_archive_20261110)에 남긴다.
기존 입력값은 [참고 입력 자료](../../camera_capture/_review/training_reference_inputs_20261110)에
따로 보존한다. 이 자료가 다음 학습의 설정을 확정한다는 뜻은 아니다.

[실제 코드·결과 대응표](../../docs/research/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md)와
[정리 기록](../../docs/research/OUTPUT_REORGANIZATION_20261110.md)을 함께 확인한다.
