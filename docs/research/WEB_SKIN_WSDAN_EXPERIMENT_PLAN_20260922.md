# Web Skin 논문 기반 3개 방법 통합 실험

## 목적과 비교 원칙

기존 `B0·256·CE`의 저장된 Validation Accuracy `0.796`, Macro F1
`0.7915394647566528`을 기준선으로 사용한다. 기준선은 다시 학습하지 않는다. 같은 Web Skin
데이터, augmented Train, original Validation 500장과 seed 42에서 세 방법을 독립적으로
실행한다. 후보를 정하기 전이므로 Test는 사용하지 않는다.

| 실험 | 논문 가설 | 학습 중 변경 | 실제 사용 입력 |
|---|---|---|---|
| WS-DAN | 중요한 국소 영역을 찾아 확대하면 세밀한 클래스 구분에 도움이 되는가 | 다중 Attention Pooling, Attention Crop/Drop | 원본과 Attention crop 예측 평균 |
| PMG | 서로 다른 크기의 조각에서 학습한 특징을 점진적으로 결합하면 세부 차이를 구분하는가 | `8×8 → 4×4 → 2×2 → 원본` jigsaw 순차 학습 | 원본 한 장, 네 분류 logit 합산 |
| MixStyle | 색감·질감 통계 변화에 덜 의존하면 촬영 환경 변화에 강해지는가 | 초기 CNN 특징의 평균·표준편차를 확률적으로 혼합 | 원본 한 장, MixStyle 비활성화 |

세 방법을 서로 합치지 않고 각각 독립적으로 비교한다.

## 구현 근거

### WS-DAN

Hu et al., [See Better Before Looking Closer](https://arxiv.org/abs/1901.09891)는 이미지 수준
라벨로 Attention Map을 만들고 Attention Cropping과 Dropping을 사용한다. MediFlow는
EfficientNet-B0에 Bilinear Attention Pooling을 적용한 변형이다. 추론에서는 원본과
결정적인 Attention crop의 예측을 평균한다.

### PMG

Du et al., [Progressive Multi-Granularity Training of Jigsaw
Patches](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650154.pdf)는 작은
granularity부터 큰 granularity까지 점진적으로 학습한다. 공식 코드의 `8×8`, `4×4`, `2×2`
jigsaw와 원본 결합 순서를 유지하고, 특징 추출기는 EfficientNet-B0의 세 단계로 바꾼다.

### MixStyle

Zhou et al., [Domain Generalization with
MixStyle](https://openreview.net/forum?id=6xHJ37MVxxp)은 초기 CNN 특징의 instance-level
평균과 표준편차를 batch 안에서 혼합해 가상의 스타일을 만든다. `alpha=0.1`, 적용 확률
`0.5`를 사용하며 Validation과 실제 추론에서는 자동으로 꺼진다.

## 실행·보존

- 실행 파일: `notebooks/11_web_skin_wsdan_attention_colab.ipynb`
- 순서: WS-DAN → PMG → MixStyle
- 각 실험 완료 후 별도 모델·지표·예측 CSV·해시 저장
- 중단 후 `RESUME_DIR`을 지정하면 완료 실험은 검증 후 건너뜀
- 결과 ZIP에는 모델을 제외하고 표·JSON·CSV·그래프를 포함
- 최종 비교: Accuracy, Macro F1, 클래스별 F1, 혼동행렬, 학습곡선
