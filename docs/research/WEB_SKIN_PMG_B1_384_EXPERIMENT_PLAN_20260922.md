# Web Skin PMG·B1·384 마지막 Validation 실험 계획

## 목적

현재 Web Skin Validation 선두인 `PMG·B0·256·CE`의 미세 특징 학습 방식을 유지하면서,
Hair에서 가장 좋았던 `EfficientNet-B1·384` 규모를 적용했을 때 성능이 더 개선되는지 확인한다.
새 모델 하나만 학습하고 Test는 후보가 결정될 때까지 열지 않는다.

## 근거

- PMG 논문은 `8×8`, `4×4`, `2×2` jigsaw와 원본 특징을 점진적으로 결합하여 세밀한
  클래스 차이를 학습한다.
  - Du et al., [Progressive Multi-Granularity Training of Jigsaw Patches](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf)
- EfficientNet 논문은 네트워크의 깊이·너비·입력 해상도를 함께 고려하는 scaling의 효율을
  제시한다.
  - Tan and Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html)
- `B1·384`라는 정확한 조합이 논문에 정답으로 제시된 것은 아니다. MediFlow Hair에서 같은
  조합이 Validation 선두였다는 내부 결과를 Web Skin에 이전하는 추가 가설이다.

## 비교 설계

| 항목 | 저장된 기준선 | 새 실험 |
|---|---|---|
| 모델 | PMG·EfficientNet-B0 | PMG·EfficientNet-B1 |
| 입력 | 256×256 | 384×384 |
| Loss | CE | CE |
| Train | augmented | augmented |
| Validation | 원본 500장 | 동일 원본 500장 |
| 학습 단계 | 15 + 10 epoch | 15 + 10 epoch |
| Seed | 42 | 42 |
| Batch | 32 | 16 |
| Test | 미사용 | 미사용 |

주요 실험 변수는 현재 PMG 모델의 scale을 `B0·256`에서 `B1·384`로 확장하는 것이다. Batch
16은 384 입력과 PMG 다중 branch의 Colab GPU 메모리를 위한 필수 동반 변경이며 결과 설정에
명시한다. 이 차이 때문에 결과를 해상도만의 순수한 효과로 해석하지 않는다.

## 고정 계약

- 데이터 SHA-256: `f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d`
- 클래스 순서: 건선, 아토피, 여드름, 정상, 주사
- PMG jigsaw: `8×8 → 4×4 → 2×2 → 원본`
- Optimizer: Adam, Stage 1 `1e-4`, Stage 2 `1e-5`
- 부분 미세조정: EfficientNet 후반 30개 레이어, BatchNormalization 제외
- 추론: 원본 한 장에서 세 branch logit과 fusion logit 합산 후 softmax
- 후보 선택: Validation Macro F1

## 판단 기준

- 새 모델 Macro F1이 `0.8473279632397033`보다 높으면 새 모델을 최종 Test 후보로 고정한다.
- 같거나 낮으면 기존 `PMG·B0·256·CE`를 유지한다.
- 선택된 한 모델만 고정 Test에서 한 번 평가하고 패키징한다.

## 실행 파일

- `notebooks/12_web_skin_pmg_b1_384_colab.ipynb`
- 실행 모드: `web_skin_pmg_b1_384`
- 신규 학습 모델: `pmg_b1_384_ce_seed_42` 하나

