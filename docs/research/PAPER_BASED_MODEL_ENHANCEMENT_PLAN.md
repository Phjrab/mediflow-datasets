# 논문 기반 모델 강화 계획

## 1. 현재 연구 방향

현재 우선순위는 기존 공개 데이터 과제를 유지한 채 학습 방법을 바꾸어 모델을 강화하는 것이다.

- `skin`: 현재 Clean 10-class 유지
- `web_skin`: 현재 5-class 유지
- `hair`: 현재 Clean 5-class 유지
- 장비 연결, LLM, VLM 통합: 현재 범위에서 보류

기존 후보 모델과 결과는 기준선으로 보존한다. 논문에 소개된 방법을 한 번에 여러 개 섞지 않고 하나씩 적용해 Validation 결과가 실제로 좋아지는지 확인한다.

논문에 방법이 있다는 사실만으로 “성능이 향상됐다”고 쓰지 않는다. 보고서에는 다음 세 내용을 분리해서 기록한다.

1. 논문이 제안한 원리
2. MediFlow 데이터에 맞게 적용한 부분
3. 새 실험에서 직접 측정한 결과

---

## 2. 강화 대상과 현재 기준선

| 도메인 | 현재 과제 | 현재 후보 | 저장된 Test 결과 | 우선 연구 문제 |
|---|---|---|---|---|
| Hair | 두피 상태 5-class | B1, 384, Label Smoothing 0.05, Adam | Accuracy `0.8003194888178914`, Macro F1 `0.8002222282821301` | 미세각질·비듬 등 유사 클래스 분리 |
| Web Skin | 얼굴 피부 5-class | B0, 256, CE | Accuracy `0.8825`, Macro F1 `0.8813822927981046` | 건선·아토피·여드름 혼동 완화 |
| Skin | 확대 피부 병변 10-class | B0, 224, CE, Augmented | Accuracy `0.9871428571428571`, Macro F1 `0.9871314132317626` | 높은 평균 성능의 안정성과 세부 오류 개선 |

세 Test 결과는 이미 확인된 값이므로 새 설정을 고르는 데 다시 사용하지 않는다. 새 실험은 기존 Validation 분할에서 비교하고, 최종 후보를 하나 고른 뒤 Test를 한 번 평가한다.

---

## 3. 공통 실험 원칙

### 3.1 기존 데이터 고정

- 현재 사용한 Clean ZIP과 분할을 그대로 사용한다.
- 클래스 순서와 Train/Validation/Test는 바꾸지 않는다.
- 기존 증강 파일도 그대로 고정한다.
- 새 학습 중 온라인 증강을 추가하는 실험은 별도 항목으로 구분한다.
- 각 결과에 데이터 ZIP SHA-256을 저장한다.

### 3.2 한 실험에서 하나의 주요 가설만 확인

예를 들어 DINOv2와 Supervised Contrastive Learning, SAM을 동시에 적용하면 좋아지거나 나빠진 원인을 알 수 없다. 먼저 Backbone만 바꾸고, 다음 실험에서 Loss만 바꾸는 식으로 순서를 나눈다.

### 3.3 동일 seed에서 순차 선별

- 새로운 방법은 seed 42의 동일한 조건에서 선별
- Validation Accuracy와 클래스별 F1도 함께 기록
- 성능이 같으면 모델이 단순하고 계산 비용이 낮은 쪽 선택
- 최종 후보 1개만 Test 평가

모든 방법은 같은 데이터 분할과 seed를 사용하여 변경한 학습 방법의 차이를 비교한다.

### 3.4 결과가 낮아도 보존

논문 방법이 현재 데이터에서 효과가 없다는 결과도 연구 결과다. 실패 실험의 설정, 곡선과 지표를 삭제하지 않고 “왜 채택하지 않았는지”를 기록한다.

---

## 4. 실험 0 — 기존 후보 학습법 기준선

고급 방법과 비교할 현재 최선 설정을 같은 데이터와 seed 42에서 사용한다. 반복 실행용 노트북도
구현했지만 현재 학습 계획에서는 사용하지 않는다.

| 도메인 | 기준선 설정 |
|---|---|
| Hair | EfficientNet-B1, 256, Label Smoothing 0.05, Stage 1 15 + Stage 2 15 |
| Web Skin | EfficientNet-B0, 256, CE, Stage 1 15 + Stage 2 10 |
| Skin | EfficientNet-B0, 224, CE, head-only 15 |

**왜 먼저 하는가:** 데이터와 학습 조건이 다른 과거 수치를 그대로 비교하면 새 방법의 효과를
분리할 수 없기 때문이다.

**얻게 될 결과:** Validation Accuracy·Macro F1, 클래스별 F1, 학습 시간과 모델 크기다.

### 구현 상태

- Hair 실행 노트북: `notebooks/04_hair_three_seed_baseline_colab.ipynb`
- Hair 통합 실행 노트북: `notebooks/05_hair_paper_experiment_suite_colab.ipynb`
- 현재 실행할 SupCon 비교 노트북: `notebooks/06_hair_supcon_comparison_colab.ipynb`
- SupCon 반복 확인 노트북: `notebooks/07_hair_supcon_repeat_seeds_colab.ipynb`
- 남은 방법 단일 seed 선별 노트북: `notebooks/08_hair_remaining_methods_screen_colab.ipynb`
- B1·384 Adam 대 SAM 선별 노트북: `notebooks/09_hair_b1_384_sam_screen_colab.ipynb`
- B1·384 최종 Test·패키징 노트북:
  `notebooks/10_hair_b1_384_final_test_package_colab.ipynb`
- 실행 코드: `src/mediflow_datasets/paper_baseline.py`
- 통합 실행 코드: `src/mediflow_datasets/paper_suite.py`
- SupCon 비교·보고서 코드: `src/mediflow_datasets/hair_supcon.py`
- 현재 상태: seed 42 방법 선별·후보 보완성·B1·384 SAM 비교 완료
- 결과 위치: `mediflow_Project/2_results/hair/baseline3_<실행 ID>/`
- 통합 결과 위치: `mediflow_Project/2_results/hair/paper_suite_<실행 ID>/`
- SupCon 결과 위치: `results/hair/experiments/supcon_compare_20260917_120447_38d75491/`
- 상세 분석: `docs/research/HAIR_SUPCON_RESULT_20260917.md`
- 이 단계에서는 Test 평가와 후보 모델 패키징을 수행하지 않음

기존 15회 통합 노트북은 구현 기록으로 보존하지만 실행하지 않는다. `08`번 실행에서 B1·384가
Validation Macro F1 `0.7957146810`으로 seed 42 선두였다. SupCon B1·256과 B1·384의 동일
Validation 예측을 비교한 1:1 확률 평균은 `0.7942285284`로 B1·384보다 낮아 ensemble을
채택하지 않았다. 상세 근거는 `HAIR_CANDIDATE_COMPLEMENTARITY_20260921.md`에 보존했다.

`09`번 노트북으로 B1·384의 동일한 Stage 1 체크포인트에서 기존 Adam 부분 미세조정과 SAM
부분 미세조정을 비교했다. Validation 결과에 따라 B1·384 Adam을 최종 후보로 고정한다.

### SAM seed 42 선별 결과 — 2026-09-21

동일한 B1·384 Stage 1 체크포인트에서 15 epoch 부분 미세조정을 비교했다. 기존 Adam과
SAM의 Validation Accuracy는 모두 `0.7963258785942492`였다. Macro F1은 Adam
`0.7957146810115047`, SAM `0.7954748920799393`으로 SAM이 `0.00023978893156539893`
낮았다. 미세각질↔비듬 오류는 71건에서 68건으로 줄었지만 세 클래스 F1이 낮아져 전체
Macro F1 개선으로 이어지지 않았다. 따라서 SAM은 채택하지 않고 B1·384 Adam을 최종
후보로 고정한다. Test는 열지 않았다. 상세 내용은
`HAIR_SAM_RESULT_20260921.md`에 기록했다.

고정된 후보의 마지막 단계는 `10`번 노트북에서 수행한다. 이 노트북은 기존 B1·384
`stage2_best.keras`의 SHA-256을 확인하고 original Test를 한 번 평가한 뒤 모델, 클래스
순서, 384×384 전처리 계약, 결과와 재현 코드를 후보 ZIP으로 보존한다. 재학습이나 후보
재선정은 수행하지 않는다.

### 최종 후보 v2 결과 — 2026-09-21

`10`번 노트북 실행을 완료했다. original Test 1,252장에서 Accuracy
`0.8003194888178914`, Macro F1 `0.8002222282821301`을 기록했다. 모델과 패키지의 모든
artifact 해시, 384×384 입력, 5개 출력, 내부 `Rescaling(1/255)`과 가상 입력 실행을
검증했다. 현재 Hair 통합 후보는 `public_candidate_v2_b1_384_ls005_adam`이다.

---

## 5. 실험 1 — Hair Supervised Contrastive Learning

Hair의 미세각질과 비듬처럼 시각적으로 비슷한 클래스는 Cross Entropy만으로 특징 공간이 충분히 분리되지 않을 수 있다. Supervised Contrastive Learning은 같은 클래스 표현을 가깝게 모으고 다른 클래스 표현을 멀게 배치하도록 학습한다.[^1]

### 적용 방법

1. 같은 클래스 이미지가 batch에 최소 두 장 이상 포함되도록 class-aware sampler를 사용한다.
2. 한 이미지에서 약한 두 개 view를 만들고 encoder를 SupCon Loss로 학습한다.
3. encoder에 분류 head를 붙여 현재 Label Smoothing 설정으로 미세조정한다.
4. Backbone, 입력 256, 데이터 분할과 총 계산 예산은 가능한 한 기준선과 맞춘다.

### 확인할 결과

- Hair 전체 Validation Macro F1 평균
- 미세각질·비듬 클래스 F1
- 미세각질↔비듬 혼동 수
- embedding UMAP과 클래스 중심 간 거리
- 기준선 대비 학습 시간

논문에서 ImageNet 성능이 좋아졌다는 사실을 MediFlow 결과로 옮겨 쓰지 않는다. MediFlow의
동일 Validation 조건에서 기준선을 넘을 때만 Hair에서 효과가 있었다고 결론낸다.

### seed 42 선별 결과 — 2026-09-17

동일한 Hair Clean 데이터, Validation 1,252장, B1·256 입력과 seed 42를 사용했다. 기준선의
Validation Accuracy는 `0.7755591054313099`, Macro F1은 `0.7746028470442391`이었다.
SupCon의 Validation Accuracy는 `0.7915335463258786`, Macro F1은
`0.7910880666407202`였다. 미세각질↔비듬 양방향 혼동은 `74`건에서 `69`건으로 줄었고,
정답 수는 `971`건에서 `991`건으로 늘었다. 다섯 클래스의 F1이 모두 상승했다.

이 결과를 이후 B1·384, DINOv2와 EfficientNetV2-S 후보 비교에 포함했다. Test는 열지
않았다. 상세 수치는 `HAIR_SUPCON_RESULT_20260917.md`에 기록했다.

---

## 6. 실험 2 — DINOv2 전이학습

DINOv2는 1억 장이 넘는 다양한 이미지에서 라벨 없이 학습한 Vision Transformer로, 여러 과제에 사용할 수 있는 시각 특징을 만드는 것을 목표로 했다.[^2] ImageNet 분류로 사전학습한 EfficientNet과 다른 특징 표현을 제공하므로 세밀한 피부·두피 패턴에서 비교할 가치가 있다.

### 적용 순서

1. DINOv2 ViT-S/14 encoder를 고정하고 linear classifier만 학습한다.
2. 기준선에 근접하거나 더 좋으면 마지막 transformer block 일부만 낮은 학습률로 미세조정한다.
3. Hair에서 먼저 확인하고, 효과가 있으면 Web Skin에 적용한다.
4. Skin은 현재 성능이 높으므로 Hair·Web Skin 결과를 본 뒤 실행한다.

### 고정·변경 조건

- 고정: 데이터, 분할, 클래스, seed 세 개와 선정 지표
- 변경: 사전학습 Backbone과 그에 필요한 입력 전처리
- 별도 기록: DINOv2의 PyTorch normalization, resize, 파라미터 수와 모델 크기

DINOv2는 현재 `.keras` 후보와 전처리가 다르다. 모델 안에 Rescaling이 있는 기존 EfficientNet 규칙을 그대로 적용하지 않는다.

---

## 7. 실험 3 — EfficientNetV2

EfficientNetV2는 EfficientNet보다 빠른 학습과 높은 파라미터 효율을 목표로 설계됐고, 입력 크기가 커질 때 정규화 강도도 함께 조절하는 progressive learning을 제안했다.[^3]

### 적용 방법

- EfficientNetV2-S의 ImageNet 사전학습 가중치 사용
- 먼저 기존 입력 해상도에서 Backbone만 교체
- progressive resizing은 Backbone 비교가 끝난 다음 별도 실험
- 기존 데이터와 Loss, seed, 평가 규칙 고정

### 이유

현재 B0/B1만 비교했으므로 같은 EfficientNet 계열의 최신 구조가 더 효율적인 특징을 만들 수 있는지 확인할 수 있다. 모델이 더 크다는 이유만으로 채택하지 않고 Validation 평균, 시간과 모델 크기를 함께 비교한다.

---

## 8. 실험 4 — Multi-resolution 학습과 평가

ISIC 2019 피부 병변 분류 연구에서는 서로 다른 입력 해상도와 cropping을 사용한 EfficientNet들을 결합해 다양한 병변 크기와 원본 해상도를 다뤘다.[^4]

MediFlow에서는 바로 큰 ensemble을 만들지 않고 다음처럼 나눈다.

1. 같은 Backbone을 Hair/Web Skin에서는 256과 384, Skin에서는 224와 384로 각각 학습한다.
2. 해상도 외 조건을 고정한다.
3. 클래스별 Validation 결과가 서로 보완적인지 확인한다.
4. 단일 모델 중 우수 모델을 먼저 선정한다.
5. 두 해상도가 서로 다른 오답을 낼 때만 확률 평균 ensemble을 별도 실험한다.

Skin은 이미 224에서 높은 성능이므로 전체 Accuracy보다 광선각화증 등 현재 오답 클래스가 개선되는지 본다. 고해상도가 항상 유리하지 않으며 배경이나 압축 흔적을 더 잘 외울 가능성도 함께 확인한다.

---

## 9. 실험 5 — SAM optimizer

Sharpness-Aware Minimization은 현재 가중치 한 점의 손실만 낮추기보다 주변 가중치에서도 손실이 낮은 해를 찾아 일반화를 개선하려는 방법이다.[^5]

### 적용 조건

- 앞선 실험에서 정한 최선 단일 Backbone 하나에만 적용
- 데이터, Loss, 해상도와 seed 고정
- Adam/AdamW 기준선과 SAM 적용 결과 비교
- 한 update에 계산이 늘어나는 만큼 학습 시간 기록

정상 데이터나 라벨 문제를 optimizer로 해결할 수는 없으므로 첫 실험으로 사용하지 않는다. 앞선 표현 학습과 Backbone 비교가 끝난 뒤 마지막 일반화 실험으로 둔다.

---

## 10. 보조 실험 — Class-Balanced Loss

현재 Hair Clean 데이터는 클래스별 원본 수가 완전히 같지는 않지만 차이가 크지 않다. 따라서 Class-Balanced Loss를 자동으로 적용하지 않는다. 클래스별 수량이나 오류가 뚜렷하게 치우친 경우에만 effective number of samples 기반 가중치를 별도 비교한다.[^6]

이 방법은 데이터가 적은 클래스를 무조건 복사하는 대신 손실 기여도를 조절한다. 적용 시 전체 Macro F1과 각 클래스 Recall이 함께 좋아지는지 확인한다.

---

## 11. 권장 실행 순서

```text
현재 Clean 데이터와 기존 후보 보존
        ↓
기존 Hair seed 42 결과를 선별 기준으로 보존
        ↓
Hair 실험 1: Supervised Contrastive Learning seed 42
        ↓
효과가 부족하면 DINOv2 seed 42 단독 실험
        ↓
필요할 때 EfficientNetV2, 384 해상도를 각각 seed 42로 실험
        ↓
최선 단일 모델에만 SAM, 필요할 때만 ensemble
        ↓
Validation Macro F1으로 후보 고정
        ↓
최종 후보 하나만 Test 평가 및 패키징
```

모든 도메인에 모든 실험을 반복하지 않는다. Hair에서 가장 큰 개선 가능성을 먼저 확인하고, 효과가 있는 방법만 Web Skin과 Skin으로 확장한다.

---

## 12. 실험별 저장 결과

| 구분 | 저장 내용 |
|---|---|
| 가설 | 해결하려는 오류와 논문 근거 |
| 고정 조건 | 데이터 해시, split, 클래스, 입력, seed, epoch |
| 변경 조건 | Backbone, Loss, optimizer 또는 해상도 중 주요 변수 하나 |
| 학습 | epoch별 Train/Validation Accuracy·Loss, 최고 체크포인트 |
| 평가 | Accuracy, Macro F1, 클래스별 F1·Recall, 혼동행렬 |
| 비용 | Colab GPU, 학습 시간, 파라미터와 모델 파일 크기 |
| 시각화 | 전체 곡선 비교, 성능 대시보드, 오류 이미지, embedding |
| 결론 | 채택 또는 제외 이유와 제한 사항 |

발표 문장은 다음 틀을 사용한다.

> Hair의 미세각질과 비듬이 자주 혼동되는 문제를 줄이기 위해, 같은 클래스 특징을 가깝게 하고 다른 클래스 특징을 멀게 학습하는 Supervised Contrastive Learning을 적용했다. 다른 조건을 고정하고 기존 기준선과 비교했으며, Validation Macro F1이 ___에서 ___로 변했다. 따라서 이 데이터에서는 효과가 ___했다고 판단했다.

빈칸은 실제 실험이 끝난 뒤 원본 결과값으로만 채운다.

---

## 14. 성공 기준

새 방법은 다음 조건을 만족할 때 채택한다.

- Validation Macro F1이 기준선보다 높음
- 일부 클래스만 좋아지고 다른 핵심 클래스가 크게 나빠지지 않음
- 더 큰 모델이면 성능 증가가 학습·추론 비용을 정당화함
- 클래스 순서와 전처리 계약, 데이터 해시가 보존됨
- 후보 선정에 Test 결과를 사용하지 않음
- 논문 원형과 MediFlow 구현의 차이를 문서화함

한 방법이 효과가 없으면 다음 방법으로 넘어간다. 여러 방법을 계속 더해 Test에 맞추는 방식은 사용하지 않는다.

---

## 주석 및 논문 근거

[^1]: Prannay Khosla et al., “Supervised Contrastive Learning,” *NeurIPS*, 2020.
[^2]: Maxime Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” 2023/2024.
[^3]: Mingxing Tan and Quoc V. Le, “EfficientNetV2: Smaller Models and Faster Training,” *ICML*, 2021.
[^4]: Nils Gessert et al., “Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data,” *MethodsX*, 2020.
[^5]: Pierre Foret et al., “Sharpness-Aware Minimization for Efficiently Improving Generalization,” *ICLR*, 2021.
[^6]: Yin Cui et al., “Class-Balanced Loss Based on Effective Number of Samples,” *CVPR*, 2019.
[^7]: AI Hub, “유형별 두피 이미지,” 데이터셋 번호 216. 통계에 `양호` 811건을 기록한다.

## Sources

1. [Supervised Contrastive Learning — NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
2. [DINOv2](https://arxiv.org/abs/2304.07193)
3. [EfficientNetV2 — ICML/PMLR](https://proceedings.mlr.press/v139/tan21a)
4. [Multi-resolution EfficientNets for skin lesions](https://www.sciencedirect.com/science/article/pii/S2215016120300832)
5. [Sharpness-Aware Minimization — Google Research/ICLR](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)
6. [Class-Balanced Loss — CVPR](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)
7. [AI Hub 유형별 두피 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=216)
8. [AI Hub 피부종양 이미지 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71864)
