# Skin 원본 대 증강 비교 결과

실행 ID: `comparison_20260909_075056_72d865bf`. 데이터는
`skin_datasets.zip`, SHA-256은
`db6cf59087b68d4534ff857bdb0eacc1a1a09ed5703772fefc75f038d4570e4b`이다.

## 실험 조건

검증할 가설은 저장된 Train 증강본을 사용하면 원본 Train만 사용할 때보다 Validation 성능이
높아지는가이다. 변경 변수는 학습 데이터 종류 하나다. 두 실험 모두 ImageNet EfficientNet-B0,
224×224, CE Loss, Dropout 0.3, Adam 1e-4, 분류층만 15 epoch, seed 42, batch 32를 사용했다.
Validation 1,000장은 두 실험에서 동일하다. Original Train은 7,249장, Augmented Train은 원본을
포함한 14,498장이므로 epoch당 업데이트 수와 연산량은 같지 않다.

별도 공통 감사를 생략하고 정제 실행에서 확인한 ZIP SHA-256을 대조했다. 정제 실행은 완전 동일
이미지 중복 제거와 새 분할 자체 검사를 수행했지만, 사람·병변·촬영 세션과 유사 장면 누수는
확인할 수 없었다.

## Validation 비교와 선정

| 학습 데이터 | Accuracy | Macro F1 | 학습 시간(초) | 선정 단계 |
|---|---:|---:|---:|---|
| Original | 0.973 | 0.9730455448862949 | 213.11485821299993 | stage1 |
| Augmented | 0.982 | 0.9820008885656606 | 268.81120826899996 | stage1 |

Augmented가 Accuracy에서 0.009, Macro F1에서 0.0089553436793657 높아 선정됐다. Validation
오분류는 Original 27장, Augmented 18장이다. Augmented의 클래스별 F1은 모든 클래스에서
Original과 같거나 높았다. 가장 큰 변화는 지루각화증이
0.9587628865979382에서 0.9797979797979798, 흑색점이 0.9569377990430622에서
0.9803921568627451로 높아진 것이다. 지루각화증을 흑색점으로 분류한 사례는 7장에서 3장으로
줄었다.

두 학습 곡선 모두 15 epoch까지 Validation Accuracy가 오르고 Validation Loss가 내려갔다.
이 범위에서는 뚜렷한 과적합 반전이 관찰되지 않았다.

## 선정 모델의 Test 결과

Validation으로 Augmented를 고정한 뒤 이 모델만 Test 700장에 평가했다.

| 지표 | 기록값 |
|---|---:|
| Accuracy | 0.9871428571428571 |
| Macro F1 | 0.9871314132317626 |
| 정답 수 | 691 / 700 |
| 모델 SHA-256 | `0b2a757320a113dbc1b285adab8c7745b2f7edc2cb29cc402d06f7882edc9f7a` |

Test 오분류 9장은 광선각화증 3장, 기저세포암 1장, 보웬병 1장, 사마귀 3장,
편평세포암 1장이다. 지루각화증·표피낭종·피부섬유종·혈관종·흑색점은 각 70장을 모두 맞혔다.
클래스별 F1 최저값은 광선각화증의 0.9571428571428572다.

## 산출물 확인

보고서 ZIP과 후보 모델 ZIP의 CRC 검사는 통과했다. 후보 패키지 `manifest.json`에 기록된 파일
해시는 모두 일치했고, 결과 ZIP과 후보 ZIP의 핵심 JSON도 서로 일치했다. 로컬 보관 위치는 다음과
같다.

- 보고서: `results/skin/experiments/comparison_20260909_075056_72d865bf/`
- 후보 모델: `results/skin/selected_models/comparison_20260909_075056_72d865bf_c2d5923c/`
- 원본 ZIP: `results/skin/archives/`

주요 그림은 `all_training_curves.png`, `validation_performance_dashboard.png`,
`all_validation_confusion_matrices.png`, `final_test_confusion_matrix.png`다.

## 해석과 다음 실험

이 실험에서는 저장된 증강본 사용이 같은 Validation에서 일관되게 더 좋았다. 따라서 공통 ③의
`TRAIN_VARIANT`는 `augmented`로 고정한다. 공통 ③에서는 이 데이터 종류를 유지하고 해상도,
Loss, Backbone, 미세조정 단계의 가설을 순서대로 비교한다.

이 후보는 공개 데이터 기준 결과이며 실제 USB 현미경 데이터로 검증되지 않았다. 단일 seed의
차이이므로 0.009 개선을 반복 실행에서도 유지되는 효과로 확정할 수 없다. 범위 밖 입력 거부 기능이
없고, 출력 점수는 정답일 확률로 보정되지 않았다.
