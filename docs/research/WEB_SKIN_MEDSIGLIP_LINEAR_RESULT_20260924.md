# Web Skin MedSigLIP-448 Linear Probe 결과

기준일: 2026-09-24  
상태: **Validation 실행 완료, 미채택, Test 미실행**

## 1. 결론

동일한 Web Skin clean 데이터와 Validation 500장에서 frozen MedSigLIP-448 embedding과
Linear 분류기를 평가했다. Validation Accuracy는 `0.824`, Macro F1은
`0.8195774288599683`이었다. 현재 PMG·B0·256 v2의 Accuracy `0.85`, Macro F1
`0.8473279632397033`보다 각각 `-0.026`, `-0.027750534379735` 낮았다.

사전에 정한 Validation Macro F1 우선 규칙을 통과하지 못했으므로 MedSigLIP Linear는
최종 Test 대상이나 v3 후보로 올리지 않는다. 현재 Web Skin 배포 후보 v2를 유지한다.

## 2. 실험 계약

| 항목 | 값 |
|---|---|
| 실험 | `medsiglip_448_frozen_linear_seed_42` |
| 주요 변경 변수 | PMG·ImageNet EfficientNet 대신 frozen MedSigLIP-448 embedding |
| 분류기 | Linear 층 하나 |
| 학습 데이터 | Augmented Train 7,200장 |
| 선정 데이터 | Original Validation 500장 |
| 클래스 순서 | 건선, 아토피, 여드름, 정상, 주사 |
| seed | 42 |
| Linear 학습 | AdamW, lr `0.001`, weight decay `0.0001`, 50 epoch |
| 데이터 SHA-256 | `f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d` |
| 환경 | Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.53.2, NVIDIA L4 |
| Test | 사용하지 않음 |

MedSigLIP encoder는 학습하지 않았다. 따라서 결과는 MedSigLIP 전체 fine-tuning이 아니라
고정 의료 embedding의 선형 분리 성능을 의미한다.

## 3. 기존 PMG v2 비교

| 모델 | Validation Accuracy | Validation Macro F1 | 판단 |
|---|---:|---:|---|
| PMG·B0·256 v2 | 0.8500 | 0.8473279632 | 유지 |
| MedSigLIP-448 Frozen Linear | 0.8240 | 0.8195774289 | 미채택 |
| 변화 | -0.0260 | -0.0277505344 | 기준 미달 |

## 4. 클래스별 분석

| 클래스 | PMG v2 F1 | MedSigLIP F1 | 변화 |
|---|---:|---:|---:|
| 건선 | 0.8252427184 | 0.8181818182 | -0.0070609003 |
| 아토피 | 0.7835051546 | 0.7078651685 | -0.0756399861 |
| 여드름 | 0.7634408602 | 0.7878787879 | +0.0244379277 |
| 정상 | 0.9615384615 | 0.8940092166 | -0.0675292449 |
| 주사 | 0.9029126214 | 0.8899521531 | -0.0129604682 |

MedSigLIP은 여드름 정답을 PMG의 71장에서 78장으로 늘려 여드름 F1이 개선됐다. 반면
아토피 Recall은 `0.76`에서 `0.63`으로 낮아졌다. 아토피 100장 중 15장을 여드름으로,
9장을 정상으로 분류했다. 건선 10장을 주사로 분류한 것도 주요 오류였다. 정상과 주사는 각각
97장과 93장을 맞혔지만 PMG의 정상 100장 정답과 전체 균형 성능을 넘지 못했다.

## 5. 학습곡선 해석

Train loss는 1 epoch `1.5251`에서 50 epoch `0.4192`까지 안정적으로 감소했다. Validation
Macro F1도 `0.6133`에서 `0.8196`까지 대체로 상승했고 최고값이 마지막 50 epoch에서 나왔다.
따라서 학습 실패나 발산으로 성능이 낮았던 것은 아니다. 다만 현재 실험의 목적은 사전에 정한
50 epoch Linear Probe의 선별이므로, 결과를 본 뒤 epoch만 늘리는 것은 별도의 Validation
튜닝 실험으로 구분해야 한다.

곡선이 끝까지 상승한 점은 더 긴 Linear 학습의 가능성을 남기지만, 현재 기준선과의 Macro F1
차이는 약 2.78%p이고 MedSigLIP은 448 입력과 큰 encoder가 필요하다. 배포 비용까지 고려하면
이번 결과만으로 추가 Test나 후보 교체를 진행할 근거가 부족하다.

## 6. 보존 위치

- 재현 결과:
  `results/web_skin/experiments/web_skin_medsiglip_linear_20260924_072442_f7176e07/`
- 원본 보고서 ZIP:
  `results/web_skin/archives/web_skin_medsiglip_linear_20260924_072442_f7176e07_reports_5d8524bf.zip`
- 보고서 ZIP SHA-256:
  `120a8b2d066a56cda89ed340aad46e59a12e7cc209be86beb8d2f3fe5d50ec`

보고서에는 Linear head, Validation 예측, 지표, 곡선, 혼동행렬, 실행 설정과 내장 소스가
포함된다. 원본 MedSigLIP 가중치는 Hugging Face에서 받는 외부 기반 모델이므로 ZIP에 포함하지
않는다.

## 7. 다음 판단

Web Skin은 PMG·B0·256 v2를 그대로 사용한다. MedSigLIP을 Web Skin에서 최종 Test하거나
패키징하지 않는다. 이후 Hair에서도 같은 frozen MedSigLIP Linear 가설을 독립적으로 검증했고
현재 Hair v2보다 낮아 미채택했다. Hair 결과는
[`HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md`](HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md)에
기록했다.
