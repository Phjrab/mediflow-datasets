# 휴대폰 공동 근접 후보 · 실제 추가 학습 결과

후속 [최종 Source Test 30개 비교](../smartphone_final_test_20261112/README.md)를 완료했고,
Validation 선정 [4차 패키지](../smartphone_adaptation_package_20261112/README.md)를 유지했다.
아래는 추가 5차 학습 수신 시점의 기록이며 원본 JSON·모델은 보존했다.

사용자가 전달한 `MediFlow_Smartphone_Joint_Results_20261112.zip`에서 필요한 원본 28개를 보관했다.
독립 추가 1회(5차), 5 epoch 완료·최적 3 epoch다. 기존 4차 가중치를 이어 학습한 모델이 아니다.

[상세 결과·기존 1~4차와 비교](../../docs/research/camera_adaptation/SMARTPHONE_JOINT_RESULT_REVIEW_20261112.md).

| 찾을 것 | 경로 |
| --- | --- |
| 새 PMG·B0·256 최적 모델 | [stage5/best.keras](stage5/best.keras) |
| 최적 epoch·원본/새 변환 검증 수치 | [stage5/complete.json](stage5/complete.json) |
| 전체 5 epoch와 학습곡선 | [이력](stage5/history.json), [그림](stage5/learning_curves.png) |
| 공통 6모델×5조건 Validation | [수치](validation_comparison.json), [그림](validation_comparison.png) |
| 클래스별 혼동행렬 | [그림](stage5_confusion_matrices.png) |
| 실제 42장 특징 거리·예측 변화 | [calibration_diagnostics.json](calibration_diagnostics.json) |
| 실제 적용 설정·환경 | [stage5/run_configuration.json](stage5/run_configuration.json) |
| 고정 분할과 점검 | [source_manifest.json](source_manifest.json), [data_audit.json](data_audit.json) |
| 실행 코드·설정·해시 | [reproduction](reproduction) |
| 결과 완료 상태·선정 | [summary.json](summary.json), [selection.json](selection.json) |
| 수신·원본 보존 검사 | [receipt_verification.json](receipt_verification.json) |
| 모델 실제 로드·합성 추론 | [model_load_verification.json](model_load_verification.json) |
| 사용한 노트북 | [추가 1회 노트북](../../notebooks/camera_adaptation/MediFlow_Smartphone_Joint_Training_20261112.ipynb) |

새 변환 조건의 정답 수는 적응 전 424/499, 4차 418/499, 추가 5차 421/499다.
원본은 각각 424/499, 419/499, 418/499다. 공통 Validation 기준 후보는 여전히 4차다.
Source Test는 미실행이고 실제 휴대폰 질환 정답이 없어 Target Accuracy는 없다.
영상 통계 7/8 근접 후보라도 모든 장비 유사성 기준을 통과한 것은 아니며 모델 특징 거리도 줄지 않았다.

클래스 인덱스는 **건선·아토피·여드름·정상·주사**다. 입력은 중앙 정사각형 256px RGB float32 0–255이며
내부 Rescaling을 포함하므로 외부 `/255`를 추가하지 않는다. PMG 4개 출력의 logit을 더한 뒤 softmax한다.
합성 입력으로 로드를 확인한 연구 모델이며 실제 진단·범위 밖 입력·보정된 정답 확률을 검증한 배포본이 아니다.

원본 이미지·키·기존 모델은 복사하지 않았다. ZIP 안의 원래 JSON·모델·그림·코드는 바꾸지 않았다.
기존 1~4차 결과와 공식 후보·패키지는 보존했다. Downloads의 원본 ZIP도 삭제하지 않았다.
