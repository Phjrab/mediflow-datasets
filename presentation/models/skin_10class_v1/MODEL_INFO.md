# Skin selected model v1

- 상태: **현재 사용 모델**
- 학습 데이터: Augmented
- 원본 후보 ID: `public_candidate_v1_b0_224_ce_augmented_20260909_075056`
- 구조: EfficientNet-B0
- 입력: 224×224 RGB float32, 0–255
- Loss: Categorical Crossentropy
- Validation Accuracy: `0.982`
- Validation Macro F1: `0.9820008885656606`
- Test Accuracy: `0.9871428571428571`
- Test Macro F1: `0.9871314132317626`
- 출력 처리: `direct_softmax`
- 모델 SHA-256: `0b2a757320a113dbc1b285adab8c7745b2f7edc2cb29cc402d06f7882edc9f7a`

## 선정 이유

Original과 Augmented를 같은 Clean 분할과 B0·224·CE 조건에서 비교했을 때 Augmented의 Validation Accuracy가 `0.982`로 Original `0.973`보다 높아 선정했다.

## 버전 설명

후보 고정 뒤 Test 700장에서 최종 평가한 Skin의 최초이자 현재 모델이다. Original과 Augmented는 별도 모델 버전이 아니라 선정 비교 조건이다.

## 사용 파일

- 모델: `skin_model.keras`
- 클래스 순서: `class_names.json`
- 전처리 계약: `preprocessing.json`
- 구조화된 기록: `selection.json`

전체 원본 보고서와 배포 ZIP은 `results/skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056`에서 보존한다.
