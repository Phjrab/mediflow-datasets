# Hair 6-class 경량 대안 v1

- 상태: 공개 데이터 **경량 대안**. 실제 USB 현미경 사진·장비 검증 전.
- 모델: EfficientNet-B0, 256×256 RGB float32, 픽셀 0–255, 내부 `Rescaling(1/255)`.
- 학습: 증강 이미지, Categorical Crossentropy, head-only 15 epoch + 부분 미세조정 15 epoch.
- 선택 기준: 같은 6-class Validation에서 B0 두 조건 중 Macro F1 선두. 성능 우선 B1·384보다 이번 Colab 실행 시간이 짧음.
- Validation Accuracy `0.7777777777777778`, Macro F1 `0.7819142733310591`.
- Test Accuracy `0.7869731800766283`, Macro F1 `0.8021613486071133` (같은 6-class Test 1,305장).
- Colab End-to-end p50 `288.73898249958074 ms`, p95 `296.42769299971405 ms`. 모델 로드와 카메라 촬영은 제외.
- 모델 SHA-256 `38e91fee1c514b75165d07f5dfbb0c52fdb7ac92313b40d47abb50c71367d432`.
- 원본 ZIP: [`../../candidates/public_candidate_6class_light_b0_256_augmented_v1.zip`](../../candidates/public_candidate_6class_light_b0_256_augmented_v1.zip).

이 폴더의 `class_names.json` 배열 순서가 출력 인덱스다. `preprocessing.json`을 사용하고 외부 `/255`를 적용하지 않는다. 경량 대안은 별도 선택 항목이며 기존 Hair 5-class v2를 자동 교체하지 않는다. [종합 선정 문서](../../../FINAL_MODEL_SELECTION_20260928.md)에 성능 우선 후보와의 비교가 있다.
