# Skin 데이터 정제·재학습·후보 모델 총정리

작성일: 2026-09-09. 이 문서는 Skin(USB 현미경 피부 병변) 분류기의 기존 결과 확인부터
데이터 정제, 원본·증강 재학습, 후보 모델 선정과 패키징까지 한 번에 설명한다. 기록된 성능은
공개 데이터 기준이며 실제 장비 및 임상 성능을 의미하지 않는다.

## 1. 작업을 다시 한 이유

기존 Skin 모델의 보고된 Test Accuracy는 Original 0.9942857027053833, Augmented
0.9957143068313599로 높았다. 그러나 데이터 감사에서 Train–Validation 10그룹,
Train–Test 5그룹, Validation–Test 1그룹의 RGB 픽셀이 완전히 같은 사진을 확인했다.
학습에서 본 사진이 평가 분할에도 있으면 Test가 새로운 사진에 대한 평가가 아니게 되므로,
기존 점수를 최종 후보 선정 근거로 그대로 사용할 수 없었다.

## 2. 데이터를 어떻게 정리했고 왜 그렇게 했는가

기존 증강본을 다시 분할하면 같은 원본에서 나온 파생 사진이 여러 분할에 섞일 수 있다. 따라서
Original 9,000장만 다시 모은 뒤 RGB 픽셀 해시가 같은 파일은 한 장만 남겼다. 중복 51장을
제거해 고유 원본 8,949장을 만들었다.

클래스별로 seed 42와 픽셀 해시를 사용해 재현 가능한 순서로 정렬한 뒤 Test 70장,
Validation 100장, 나머지를 Train으로 배정했다. 결과는 다음과 같다.

| 구분 | Train | Validation | Test |
|---|---:|---:|---:|
| Original | 7,249 | 1,000 | 700 |
| Augmented | 14,498 | 1,000 | 700 |

Validation과 Test는 증강하지 않고 Original과 Augmented 실험에서 같은 사진을 사용했다. 그래야
두 학습 데이터 방식의 차이만 비교할 수 있다. Train 원본마다 11개 변환 중 하나를 적용해 새
증강본 한 장을 만들고, 각 증강본이 어느 원본에서 왔는지 `augmentation_lineage.csv`에 기록했다.
기존 방식처럼 원본을 임의로 다시 뽑지 않고 모든 Train 원본을 한 번씩 사용했다.

생성 단계 검사에서 분할 간 완전 동일 이미지 0그룹, 라벨 충돌 0그룹, 고유 증강 이미지
7,249장을 확인했다. 새 데이터 ZIP의 SHA-256은
`db6cf59087b68d4534ff857bdb0eacc1a1a09ed5703772fefc75f038d4570e4b`이다.

## 3. 어떤 재학습을 했고 왜 했는가

먼저 복잡한 모델을 여러 개 돌리지 않고 원본과 증강 중 어느 데이터 방식이 좋은지만 비교했다.
실험 가설은 저장된 Train 증강본이 같은 Validation에서 일반화 성능을 높이는가였다.

두 실험에서 고정한 조건은 ImageNet EfficientNet-B0, 입력 224×224, CE Loss, Dropout 0.3,
Adam 1e-4, 분류층만 15 epoch, seed 42, batch 32다. 변경한 것은 Train 데이터 종류 하나다.
Augmented는 이미지 수가 두 배이므로 같은 15 epoch라도 업데이트 수와 연산량은 더 많다.

## 4. 원본과 증강 결과

| 학습 데이터 | Validation Accuracy | Validation Macro F1 | 오분류 | 학습 시간(초) |
|---|---:|---:|---:|---:|
| Original | 0.973 | 0.9730455448862949 | 27 / 1,000 | 213.11485821299993 |
| Augmented | 0.982 | 0.9820008885656606 | 18 / 1,000 | 268.81120826899996 |

Augmented가 Accuracy에서 0.009, Macro F1에서 0.0089553436793657 높아 선정됐다. 10개 클래스의
Validation F1은 모두 Original과 같거나 높았다. 지루각화증 F1은 0.9587628865979382에서
0.9797979797979798, 흑색점 F1은 0.9569377990430622에서 0.9803921568627451로 높아졌다.
지루각화증을 흑색점으로 분류한 사례도 7장에서 3장으로 줄었다.

두 학습 곡선 모두 15 epoch까지 Validation Accuracy가 오르고 Validation Loss가 내려갔다.
이 구간에서는 Validation 성능이 나빠지는 뚜렷한 과적합 반전이 없었다.

## 5. 최종 Test와 후보 모델

모델 선택에는 Validation만 사용했다. Augmented를 고정한 다음 이 모델 하나만 Test 700장에
평가했다.

| 지표 | 기록값 |
|---|---:|
| Test Accuracy | 0.9871428571428571 |
| Test Macro F1 | 0.9871314132317626 |
| 정답 수 | 691 / 700 |
| 모델 SHA-256 | `0b2a757320a113dbc1b285adab8c7745b2f7edc2cb29cc402d06f7882edc9f7a` |

Test 오분류는 광선각화증 3장, 기저세포암 1장, 보웬병 1장, 사마귀 3장,
편평세포암 1장으로 총 9장이다. 지루각화증·표피낭종·피부섬유종·혈관종·흑색점은 각 70장을
모두 맞혔다. 클래스별 F1 최저값은 광선각화증의 0.9571428571428572다.

현재 후보는 EfficientNet-B0 / 224×224 / CE / Augmented다. 후보 ZIP과 내부 manifest,
모델 SHA-256을 대조했고 실제 모델 로드 및 224×224 입력, 10개 출력도 확인했다.

## 6. Hair·Web Skin과 달리 추가 6개 실험을 생략한 이유

Hair는 Validation과 Test 성능이 약 0.78 수준이어서 해상도, Loss, Backbone, 미세조정 실험으로
개선 가능성을 비교할 필요가 컸다. Web Skin도 클래스 간 혼동이 남아 같은 실험 묶음을 수행했다.

Skin은 중복을 제거한 새 분할에서도 Validation Accuracy 0.982, Test Accuracy
0.9871428571428571이 나왔고 증강 효과도 확인됐다. 추가 6개 실험에서 얻을 수 있는 개선 폭보다
Test를 반복 확인하며 설정을 고르는 위험과 계산 비용이 더 크다고 판단해, 사용자 결정에 따라
공통 ③을 실행하지 않고 현재 모델을 공개 데이터 후보 v1으로 확정했다. 이는 모든 가능한 설정을
시험했다는 뜻이 아니라 현재 목적에 충분한 후보를 확보해 실험을 멈췄다는 뜻이다.

## 7. 현재 파일 위치

- 전체 후보 목록: `results/CANDIDATE_INDEX.json`
- Skin 후보 ZIP: `results/skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056.zip`
- 풀어놓은 후보: `results/skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056/`
- 비교 보고서: `results/skin/experiments/comparison_20260909_075056_72d865bf/`
- 원본 다운로드 ZIP: `results/skin/archives/`
- 학습 곡선: `results/skin/experiments/comparison_20260909_075056_72d865bf/all_training_curves.png`
- Validation 비교: `results/skin/experiments/comparison_20260909_075056_72d865bf/validation_performance_dashboard.png`
- Test 혼동행렬: `results/skin/experiments/comparison_20260909_075056_72d865bf/final_test_confusion_matrix.png`

## 8. 해석할 때 지켜야 할 범위

정제 과정에서 완전 동일 이미지 중복은 제거했지만 사람·병변·촬영 세션 식별 정보가 없으므로
그 단위의 누수는 확인하지 못했다. 재압축·밝기 변화·비슷한 장면 같은 근접 중복과 폴더 라벨의
임상적 정확성도 별도 검증하지 않았다. 단일 seed 실험이므로 0.009 개선이 반복 실행에서도
같이 나타나는지는 확인되지 않았다.

Skin 후보에는 정상 클래스와 범위 밖 입력 거부 기능이 없다. 출력 softmax 점수는 정답일 확률로
보정되지 않았다. 실제 USB 현미경과 실제 환자 데이터가 확보되면 데이터 수정 없이 먼저 외부
평가해 Domain Gap을 측정해야 한다.

## 9. 이후 개발 순서

현재 후보 모델을 공통 추론 모듈에 연결하고 0–255 범위의 RGB 입력, 224×224 크기, 클래스 순서와
오류 처리를 고정한다. 장비 정보와 촬영 부위를 함께 사용해 Hair와 Skin 모델을 구분해야 한다.
실제 장비 데이터가 생기면 현재 후보를 그대로 평가하고, 결과가 부족할 때만 미세조정·256 해상도·
Loss·Backbone 실험을 새 가설로 진행한다.

세부 근거는 [Skin 데이터 감사 결과](SKIN_DATA_AUDIT_20260909.md)와
[Skin 원본 대 증강 비교 결과](SKIN_ORIGINAL_VS_AUGMENTED_20260909.md)에 보존한다.
