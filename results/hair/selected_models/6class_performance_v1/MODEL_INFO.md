# Hair 6-class 성능 우선 후보 v1

- 상태: 공개 데이터 **연구상 성능 우선 후보**. 실제 USB 현미경 사진·장비 검증 전.
- 모델: EfficientNet-B1, 384×384 RGB float32, 픽셀 0–255, 내부 `Rescaling(1/255)`.
- 학습: 원본 이미지, Label Smoothing 0.05, Adam, head-only 15 epoch + 부분 미세조정 15 epoch.
- 선택 기준: 같은 6-class Validation에서 네 조건 중 Macro F1 1위.
- Validation Accuracy `0.7992337164750958`, Macro F1 `0.8144942091318823`.
- Test Accuracy `0.7915708812260537`, Macro F1 `0.8091500773460419` (같은 6-class Test 1,305장).
- Colab End-to-end p50 `409.83297050024703 ms`, p95 `422.71190300016315 ms`. 모델 로드와 카메라 촬영은 제외.
- 모델 SHA-256 `5ee54f51e65d8656fc81e88de96ad82a6d119991aa74684f921e4cd7e8b7efbb`.
- 원본 ZIP: [`../../candidates/public_candidate_6class_v1_b1_384_original_20260928_024655.zip`](../../candidates/public_candidate_6class_v1_b1_384_original_20260928_024655.zip).

이 폴더의 `class_names.json` 배열 순서가 출력 인덱스다. `preprocessing.json`을 사용하고 외부 `/255`를 적용하지 않는다. 기존 Hair 5-class v2를 덮어쓴 버전이 아니다. 경량 대안과의 선택 근거는 [종합 선정 문서](../../../FINAL_MODEL_SELECTION_20260928.md)를 본다.
