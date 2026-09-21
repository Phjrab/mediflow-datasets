# MediFlow 논문 기반 모델 강화 실험 최종 분석

기준일: 2026-09-22  
대상: Hair 5-class 공개 데이터 모델  
선정 기준: 동일 Validation 분할의 Accuracy와 Macro F1  

이 문서는 논문이 제안한 원리, MediFlow가 그 원리를 시험한 방식, 실제 측정 결과를 구분한다.
논문의 벤치마크 성능이 MediFlow 데이터에서도 그대로 재현된다고 가정하지 않는다.

## 1. 출발점과 연구 질문

Hair의 핵심 문제는 `미세각질`과 `비듬`처럼 시각적으로 유사한 클래스였다. 데이터의 완전
동일 중복을 제거하고 2단계 전이학습을 적용한 뒤에도 초기 후보의 Test Macro F1은
`0.7885764577048346`이었다. 이후 실험은 다음 질문을 하나씩 확인했다.

1. 더 큰 입력이 작은 각질·모낭 특징을 보존하는가?
2. 같은 클래스를 특징 공간에서 모으는 학습이 유사 클래스 구분을 돕는가?
3. 다른 사전학습 표현이나 더 효율적인 구조가 더 나은가?
4. 평평한 손실 영역을 찾는 최적화가 일반화를 높이는가?
5. 서로 다른 모델의 오답이 보완되어 앙상블 이득이 생기는가?

## 2. 논문 근거와 MediFlow 적용

| 방법 | 논문에서 얻은 아이디어 | MediFlow에서 시험한 이유 | 실제 적용 |
|---|---|---|---|
| EfficientNet 전이학습 | 깊이·너비·해상도를 함께 확장하는 compound scaling과 높은 전이 성능 | 제한된 의료 이미지에서 처음부터 학습하지 않고 일반 시각 특징을 재사용 | ImageNet EfficientNet-B0/B1, 분류층 학습 후 후반부 부분 미세조정 |
| 224→256→384 | EfficientNet의 해상도도 모델 용량의 한 축 | 각질·모낭처럼 작은 특징이 축소 과정에서 사라지는지 확인 | 나머지 조건을 유지하며 입력 크기 비교 |
| Label Smoothing | 정답 라벨을 1.0으로 과도하게 확신하는 학습을 완화 | 경계가 모호한 두피 클래스에서 과신을 줄일 가능성 | Categorical Crossentropy, smoothing `0.05` |
| Focal Loss | 쉬운 표본의 손실 기여를 낮추고 어려운 표본에 집중 | 균형 데이터에서도 유사 클래스의 어려운 예에 더 집중하는지 확인 | gamma `1.5`, alpha `1.0` |
| Supervised Contrastive Learning | 같은 클래스 표현은 가깝게, 다른 클래스 표현은 멀게 배치 | 미세각질·비듬의 특징 군집을 더 분리할 가능성 | B1·256에서 SupCon 표현 학습 후 분류기 학습 |
| DINOv2 | 다양한 데이터로 학습한 범용 자기지도 시각 특징 | ImageNet CNN과 다른 표현이 두피 질감에 유리한지 확인 | DINOv2 Small·224 encoder 고정, 새 분류층 15 Epoch |
| EfficientNetV2-S | 학습 속도와 파라미터 효율을 함께 고려한 CNN | 기존 EfficientNet 계열보다 효율적인 구조의 전이 가능성 확인 | 동일 Hair 분류 문제의 후보로 선별 실험 |
| SAM | 현재 지점만이 아니라 주변에서도 손실이 낮은 파라미터 탐색 | 작은 데이터 변화에 덜 민감한 해를 찾는지 확인 | B1·384 동일 Stage 1에서 Adam과 SAM `rho=0.05` 비교 |
| 확률 평균 앙상블 | 서로 다른 오답이 보완되면 단일 모델보다 개선될 수 있음 | SupCon B1·256과 B1·384의 오답 보완성 확인 | 같은 Validation 이미지의 두 softmax를 1:1 평균 |

## 3. 초기 6개 실험

| 실험 | 변경한 핵심 변수 | Validation Accuracy | Test Accuracy | Test Macro F1 | 결론 |
|---|---|---:|---:|---:|---|
| B0·224·CE | Clean 기준선 | 0.7643769979 | 0.7763578275 | 0.7757408068 | 데이터 정제 후 기준점 |
| B0·256·CE | 입력 224→256 | 0.7699680328 | 0.7891373802 | 0.7886477230 | 해상도 증가가 유리한 방향 |
| B0·256·LS 0.05 | CE→Label Smoothing | 0.7739616632 | 0.7867412141 | 0.7863134458 | Validation은 상승, Test는 CE보다 낮음 |
| B0·256·Focal 1.5 | CE→Focal | 0.7675718665 | 0.7907348243 | 0.7905122040 | 이 묶음의 Test 지표 최고, Validation 선두는 아님 |
| B1·256·LS·Stage2 10 | B0→B1 | 0.7763578296 | 0.7787539936 | 0.7788123272 | 더 큰 backbone만으로 이득 없음 |
| B1·256·LS·Stage2 15 | 미세조정 5회 연장 | 0.7771565318 | 0.7883386581 | 0.7885764577 | 초기 공개 후보 v1 |

이 단계에서 확인한 핵심은 “큰 모델이 자동으로 더 좋다”가 아니었다. 256 입력은 224보다
좋은 방향을 보였고, Loss 변경과 B1 확대의 이득은 선택 지표에 따라 작거나 일관되지 않았다.

## 4. 고급 방법의 동일 Validation 비교

| 방법 | 입력 | Validation Accuracy | Validation Macro F1 | 기준 B1·256 대비 | 판단 |
|---|---:|---:|---:|---:|---|
| B1·256 분류 기준선 | 256 | 0.7755591054 | 0.7746028470 | 기준 | 비교용 기준선 |
| SupCon B1·256 | 256 | 0.7915335463 | 0.7910880666 | **+0.0164852196 F1** | 효과 확인, 후보 유지 |
| DINOv2 Small linear probe | 224 | 0.7699680511 | 0.7705341251 | -0.0040687219 F1 | 현재 고정 encoder 설정은 불리 |
| EfficientNetV2-S | 256 | 0.7691693291 | 0.7681559382 | -0.0064469088 F1 | 채택하지 않음 |
| EfficientNet-B1 | 384 | **0.7963258786** | **0.7957146810** | **+0.0211118340 F1** | Validation 최종 선두 |

SupCon은 정답을 971장에서 991장으로 20장 늘렸고 다섯 클래스 F1이 모두 상승했다.
`미세각질↔비듬` 양방향 혼동도 74건에서 69건으로 줄었다. 논문의 표현 분리 아이디어가 이
데이터에서 도움이 될 가능성을 실제 수치로 확인했다.

DINOv2는 범용 표현이 강한 모델이지만, 이번 실험은 encoder를 고정한 linear probe였다.
따라서 결과는 “DINOv2가 두피에 부적합하다”가 아니라 “현재 224 입력·고정 encoder·15 Epoch
설정이 B1 미세조정보다 낮았다”로 해석한다.

B1·384가 가장 높았다는 결과는 두피의 미세 질감에서 입력 해상도가 중요한 변수였음을
시사한다. 동시에 B1 확대만 했던 256 실험은 뚜렷한 이득이 없었으므로, 개선을 backbone
크기 하나의 효과로 분리해서 주장하지 않는다.

## 5. 앙상블과 SAM 후속 검증

| 비교 | Accuracy | Macro F1 | 결과 |
|---|---:|---:|---|
| SupCon B1·256 | 0.7915335463 | 0.7910880666 | 보조 후보 |
| B1·384 Adam | **0.7963258786** | **0.7957146810** | 선두 단일 모델 |
| 두 모델 1:1 확률 평균 | 0.7947284345 | 0.7942285284 | 단일 B1·384보다 낮아 미채택 |
| B1·384 SAM | 0.7963258786 | 0.7954748921 | Accuracy 동일, F1 `-0.0002397889`; 미채택 |

두 단일 모델 중 하나라도 맞힌 Validation 이미지는 1,052장으로 oracle accuracy는
`0.8402555911`이었다. 보완성 자체는 있었지만 단순 평균은 그 이득을 실현하지 못했다.
SAM은 핵심 `미세각질↔비듬` 혼동을 71건에서 68건으로 줄였으나 세 클래스 F1이 낮아져 전체
Macro F1 개선으로 이어지지 않았다.

## 6. 최종 후보와 실제 얻은 결과

최종 선택은 `EfficientNet-B1 · 384×384 · Label Smoothing 0.05 · Adam`이다. Validation에서
설정을 고정한 뒤 Test 1,252장을 한 번 평가했다.

| 지표 | Hair v1 B1·256 | Hair v2 B1·384 | 변화 |
|---|---:|---:|---:|
| Test Accuracy | 0.7883386581 | **0.8003194888** | **+0.0119808307** |
| Test Macro F1 | 0.7885764577 | **0.8002222283** | **+0.0116457706** |

v2는 Test 1,252장 중 1,002장을 맞혔다. 클래스별 F1은 모낭사이홍반 `0.8275862069`,
미세각질 `0.7807692308`, 비듬 `0.75`, 탈모 `0.8442622951`, 피지과다 `0.7984934087`이다.
미세각질과 비듬이 여전히 가장 어려운 축이므로 이후 개선도 이 두 클래스의 데이터 품질과
특징 보존을 우선해야 한다.

## 7. 최종 해석

- 가장 큰 실질 개선은 논문 아이디어를 많이 쌓은 결과가 아니라, 가설을 분리해 검증한 뒤
  **B1과 384 입력의 조합**을 선택한 데서 나왔다.
- SupCon은 유사 클래스 표현 분리에 실제 이득을 보였지만 최종 단일 후보보다는 낮았다.
- DINOv2, EfficientNetV2-S, SAM, 단순 앙상블은 유명한 방법이라는 이유만으로 채택하지 않았다.
- 공개 데이터 성능은 확정했지만 USB 현미경 실사용 성능과 의료적 진단 성능은 아직 측정하지 않았다.

## 8. 논문과 원문 링크

1. Tan & Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html), ICML 2019.
2. Szegedy et al., [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567), CVPR 2016.
3. Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), ICCV 2017.
4. Khosla et al., [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), NeurIPS 2020.
5. Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), 2023.
6. Tan & Le, [EfficientNetV2: Smaller Models and Faster Training](https://proceedings.mlr.press/v139/tan21a.html), ICML 2021.
7. Foret et al., [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/), ICLR 2021.

## 9. 근거 파일

- 초기 6개 실험: `HAIR_EXPERIMENT_SUMMARY_20260907.md`
- SupCon: `HAIR_SUPCON_RESULT_20260917.md`
- DINOv2·EfficientNetV2-S·B1 384: `HAIR_REMAINING_METHODS_RESULT_20260917.md`
- 앙상블: `HAIR_CANDIDATE_COMPLEMENTARITY_20260921.md`
- SAM: `HAIR_SAM_RESULT_20260921.md`
- 최종 Test와 패키지: `HAIR_FINAL_CANDIDATE_V2_20260921.md`
