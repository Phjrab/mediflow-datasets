# Web Skin selected model v1

- 상태: **논문 강화 전 최종 선정 모델 — 이후 v2로 교체**
- 원본 후보 ID: `public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e`
- 구조: EfficientNet-B0
- 입력: 256×256 RGB float32, 0–255
- Loss: Categorical Crossentropy
- Validation Accuracy: `0.796`
- Test Accuracy: `0.8825`
- Test Macro F1: `0.8813822927981046`
- 출력 처리: `direct_softmax`
- 모델 SHA-256: `d4c834a7fd47480c1c45ddbdc09257942b38b2408d5b28effc34ee68c4fad97b`

## 선정 이유

논문 기반 강화 실험을 시작하기 전, 여섯 학습 설정을 같은 Validation에서 비교했다. B0·256·CE가 최고 Validation Accuracy를 기록했고 동률 선택 규칙을 적용해 당시 최종 Web Skin 후보로 선정한 뒤 Test 평가와 패키징까지 완료했다.

## 버전 설명

논문 강화 전 최종 선정 모델이다. 이후 PMG가 Validation과 고정 Test 성능을 높여 v2로 교체됐지만, v1도 당시 정식 선정·패키징된 후보로 보존한다.

## 사용 파일

- 모델: `web_skin_model.keras`
- 클래스 순서: `class_names.json`
- 전처리 계약: `preprocessing.json`
- 구조화된 기록: `selection.json`

전체 원본 보고서와 배포 ZIP은 `results/web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e`에서 보존한다.
