# Hair 남은 논문 기반 방법 seed 42 선별 결과

작성일: 2026-09-17  
실행 ID: `paper_screen_20260917_141641_874f03f1`

## 실험 범위

기존 기준선과 SupCon을 다시 학습하지 않고 DINOv2 Small, EfficientNetV2-S·256,
EfficientNet-B1·384를 같은 Hair Clean 데이터와 seed 42에서 선별했다. 모든 평가는 original
Validation 1,252장을 사용했고 Test는 열지 않았다.

## Epoch 수가 다른 이유

| 모델 | Stage 1 | Stage 2 | 총 Epoch | 학습 범위 |
|---|---:|---:|---:|---|
| DINOv2 Small·224 | 15 | 0 | 15 | 사전학습 encoder 고정, 새 분류층만 학습 |
| EfficientNetV2-S·256 | 15 | 15 | 30 | 분류층 학습 후 Backbone 후반부 부분 미세조정 |
| EfficientNet-B1·384 | 15 | 15 | 30 | 분류층 학습 후 Backbone 후반부 부분 미세조정 |

DINOv2는 계획한 첫 단계인 frozen linear probe로 전이 가능성을 선별했다. 30 epoch를 분류층에만
사용하는 것은 CNN의 15+15 미세조정과 같은 실험이 아니다. DINOv2가 유망할 때만 별도 실험에서
마지막 transformer block 일부를 낮은 학습률로 미세조정하도록 계획했다.

## 원본 Validation 결과

| 모델 | Accuracy | Macro F1 | 학습 시간(초) |
|---|---:|---:|---:|
| DINOv2 Small·224 | `0.7699680511182109` | `0.7705341251195019` | `504.30501069499996` |
| EfficientNetV2-S·256 | `0.7691693290734825` | `0.7681559382466019` | `704.8830730919999` |
| EfficientNet-B1·384 | `0.7963258785942492` | `0.7957146810115047` | `773.3170999580002` |

## 저장된 seed 42 결과와의 위치

| 조건 | Validation Macro F1 |
|---|---:|
| B1·256 기준선 | `0.7746028470442391` |
| SupCon B1·256 | `0.7910880666407202` |
| DINOv2 Small·224 | `0.7705341251195019` |
| EfficientNetV2-S·256 | `0.7681559382466019` |
| B1·384 | `0.7957146810115047` |

B1·384가 현재 seed 42 Validation Macro F1 선두다. 다만 SupCon과의 차이가 작고 모두 단일
seed 결과이므로 최종 우위로 확정하지 않는다. 다음에는 두 후보의 예측 오류가 보완적인지 먼저
확인하고, 최선 단일 모델에 SAM을 선별 적용한 뒤 최종 후보만 반복 검증한다.

## 보존 위치

- 보고서 ZIP: `results/hair/experiments/paper_screen_20260917_141641_874f03f1_results_f024f72d.zip`
- 압축 해제 보고서: `results/hair/experiments/paper_screen_20260917_141641_874f03f1/`
- ZIP SHA-256: `42F03C336418489D375B5D07C88943B516A208DD74DEDA0301ED4F30996B1E0F`
