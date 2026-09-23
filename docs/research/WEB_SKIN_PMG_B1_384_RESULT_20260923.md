# Web Skin PMG·B1·384 Validation 결과

## 결론

`PMG·EfficientNet-B1·384·CE`가 동일한 Web Skin Validation 500장에서 기존
`PMG·EfficientNet-B0·256·CE`보다 높은 Accuracy와 Macro F1을 기록했다. 사전에 정한
Validation Macro F1 기준에서는 새 모델이 성능 선두다. 이후 입력 비용과 모델 크기를 포함한
배포 판단에서 PMG·B0·256을 최종 Test 후보로 선택했다. 이 B1·384 모델은 Test를 평가하지
않고 연구 후보로 보존한다.

## 비교 조건

| 항목 | 기존 후보 | 새 실험 |
|---|---|---|
| PMG | 동일 | 동일 |
| Backbone | EfficientNet-B0 | EfficientNet-B1 |
| 입력 | 256×256 | 384×384 |
| Loss | CE | CE |
| Train | augmented | augmented |
| Validation | 원본 500장 | 동일 원본 500장 |
| Epoch | 15 + 10 | 15 + 10 |
| Seed | 42 | 42 |
| Batch | 32 | 16 |
| Test | 미평가 | 미평가 |

데이터 SHA-256은
`f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d`로 일치한다.
Batch 16은 B1·384와 PMG 다중 branch의 GPU 메모리를 위한 필수 동반 변경이다. 따라서 이번
차이를 해상도 하나만의 순수한 효과로 해석하지 않고, PMG scale 조합 전체의 결과로 해석한다.

## Validation 결과

| 모델 | Accuracy | Macro F1 | 정답/500 | 오분류/500 |
|---|---:|---:|---:|---:|
| PMG·B0·256 | 0.8500 | 0.8473279632397033 | 425 | 75 |
| **PMG·B1·384** | **0.8660** | **0.8643216544321994** | **433** | **67** |
| 변화 | **+0.0160** | **+0.0169936911924961** | **+8** | **-8** |

새 모델은 Accuracy가 1.6%p, Macro F1이 약 1.70%p 상승했다.

## 클래스별 F1

| 클래스 | PMG·B0·256 | PMG·B1·384 | 변화 |
|---|---:|---:|---:|
| 건선 | 0.8252427184466019 | 0.8240740740740741 | -0.0011686443725278 |
| 아토피 | 0.7835051546391754 | 0.8021390374331552 | +0.0186338827939798 |
| 여드름 | 0.7634408602150538 | 0.8172043010752689 | +0.0537634408602151 |
| 정상 | 0.9615384615384615 | 0.9708737864077670 | +0.0093353248693055 |
| 주사 | 0.9029126213592233 | 0.9073170731707316 | +0.0044044518115083 |

가장 큰 개선은 여드름 약 5.38%p이며, 아토피도 약 1.86%p 개선됐다. 건선 F1은 약 0.12%p
낮아졌지만 나머지 네 클래스와 전체 Macro F1이 개선됐다.

새 모델의 클래스별 정답 수는 건선 89장, 아토피 75장, 여드름 76장, 정상 100장, 주사
93장이다. 여전히 아토피를 건선으로 분류한 12건과 여드름을 건선으로 분류한 11건이 주요
혼동이다.

## 학습곡선 해석

- Stage 1 최고 Validation Accuracy: `0.8560000061988831`
- Stage 2 최고 Validation Accuracy: `0.8659999966621399`
- 선택 checkpoint: `stage2_best.keras`
- 최고점: 전체 21번째 epoch, Stage 2의 6번째 epoch
- 마지막 epoch Validation Accuracy: `0.8539999723434448`
- 마지막 epoch Train Accuracy: `0.9995833039283752`

Train Accuracy는 거의 1.0까지 상승했지만 Validation Accuracy와 loss는 흔들렸다. 이는 PMG의
무작위 jigsaw와 높은 모델 용량에 따른 과적합 신호다. 마지막 모델을 사용하지 않고 Stage 2의
6번째 epoch에서 저장된 최고 checkpoint를 사용하는 이유다. 추가 epoch 학습은 권장하지 않는다.

## Validation 성능 선두 연구 후보

- 후보 ID: `pmg_b1_384_ce_seed_42`
- 실행: `web_skin_pmg_b1_384_20260922_144631_ac7e5c8b`
- attempt: `attempt_46ccdb012543`
- 모델: `stage2_best.keras`
- 모델 SHA-256: `33da0d89b8fbd4ec20453b865735f4fccd20d850533e709f3f3f9c0a41e3a66d`
- 파라미터 수: `11,398,043`
- 모델 크기: `46,828,176 bytes`
- 학습 시간: `2453.2392815050002초`
- 입력: 384×384 RGB float32, 0–255
- 클래스 순서: 건선, 아토피, 여드름, 정상, 주사
- 추론: 세 branch logit과 fusion logit을 합산한 뒤 softmax
- Test: 미평가

Drive 모델 경로:

```text
/content/drive/MyDrive/mediflow_Project/2_results/web_skin/
web_skin_pmg_b1_384_20260922_144631_ac7e5c8b/
pmg_b1_384_ce_seed_42/attempt_46ccdb012543/stage2_best.keras
```

## 무결성 확인

- Validation 예측 CSV: 500행
- support 합계: 500
- 혼동행렬 합계: 500
- 혼동행렬 대각선 합계: 433
- 보고서 ZIP에 포함된 6개 실험 파일 해시: 모두 일치
- 보고서 ZIP에는 `.keras` 모델 4개가 용량 정책에 따라 포함되지 않음
- 결과 ZIP SHA-256:
  `DF18AFEEE3BBAB3266DE82F209C0B2AB2901044E1CCA3FC130109EDFFBB3CF78`

## 후속 결정

배포 후보는 실행 비용을 함께 고려해 기존 PMG·B0·256 `stage2_best.keras`로 고정했다.
B1·384는 Test를 열거나 배포 ZIP을 만들지 않는다. 자세한 비교와 고정 파일 해시는
`WEB_SKIN_PMG_FINAL_SELECTION_20260923.md`를 따른다.

독립 데이터 감사가 생략되어 사람·병변·촬영 세션 단위 누수, perceptual near-duplicate 및
임상 라벨 정확성은 확인되지 않았다. 최종 Test 결과에도 이 한계를 함께 기록한다.
