# MediFlow 전체 실험: 선택 이유·방법·결과 대장

기준일: 2026-09-24  
범위: Skin, Web Skin, Hair에서 실제로 수행했거나 구조 검토 후 중단한 실험  
용도: 발표·보고서·후속 실험 설계의 기준 기록

이 문서는 각 실험을 **왜 했는가 → 방법은 무엇인가 → 프로젝트에서는 어떻게 적용했는가 →
무엇을 얻었는가 → 채택했는가** 순서로 정리한다. 논문에 나온 방법을 사용했다는 사실과 실제
MediFlow 데이터에서 좋아졌다는 사실을 구분한다. 결과가 낮은 실험도 삭제하지 않고, 현재 설정에서
채택하지 않은 근거로 보존한다.

## 1. 먼저 해결한 문제: 평가를 믿을 수 있는가

### 1.1 분할 간 완전 동일 이미지 검사

**왜 했나:** Train에서 본 사진이 Validation이나 Test에 다시 들어가면 새 사진을 잘 분류하는
능력보다 암기 효과가 섞인다. 파일명이 달라도 내용이 같을 수 있으므로 파일 SHA-256과 RGB 픽셀을
확인했다.

**무엇을 했나:** Hair에서는 Train–Validation 195개, Train–Test 178개, Validation–Test 16개의
완전 동일 파일 겹침을 발견했다. 일부 라벨 충돌도 있어 기존 증강본을 재사용하지 않고 원본 전체를
합쳐 정리한 뒤 seed 42로 다시 분할했다. 이 세 수는 쌍별 집계라 단순 합계가 제거 장수는 아니다.
Skin은 완전 동일 이미지 51장을 제거했다. Web Skin은 파일·RGB 픽셀이 완전히 같은 분할 간 중복이
발견되지 않아 기존 분할을 유지했다.

**얻은 결과:** Hair는 Original 12,536장과 `원본 10,032 + 새 증강 5,015 = Train 15,047장`의
Clean 데이터를 만들었다. Skin은 8,949장을 남겨 Train 7,249, Validation 1,000, Test 700으로
재구성했다. 이후 모델 성능은 이 정제된 분할을 기준으로 해석한다.

**한계:** 사람·병변·촬영 세션 식별자가 충분하지 않아 같은 사람이나 같은 부위의 유사 사진까지
분할 사이에서 완전히 배제했다고 주장하지 않는다.

### 1.2 Train에만 증강을 넣은 이유

**방법 설명:** 밝기, 대비, 색, 미세 회전, 반전, 확대·crop, blur, noise 같은 변환으로 실제 조명과
촬영 위치 변화를 연습시킨다. Validation과 Test까지 증강하면 평가 기준 자체가 달라지므로 원본만
사용한다.

**실제 적용:** `Augmented Train = 원본 Train + 증강본`, `Validation/Test = 원본`으로 구성했다.
Hair의 새 증강본은 출처 원본과 변환 기록을 남겼다. 증강은 새로운 환자나 병변을 추가하는 것이
아니므로 데이터 다양성의 한계를 완전히 해결하지는 않는다.

## 2. 모든 EfficientNet 실험의 공통 기반

### 2.1 ImageNet 전이학습

**왜 했나:** 공개 의료 데이터만으로 모델 전체를 처음부터 학습하는 것보다 ImageNet에서 모양,
경계, 색과 질감을 배운 가중치를 시작점으로 쓰는 편이 제한된 데이터에서 안정적일 가능성이 높다.

**방법 설명:** [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)은 네트워크 깊이, 너비와
입력 해상도를 함께 조절하는 compound scaling을 제안했다. B0는 가볍고 B1은 더 큰 표현 용량을
가진다. MediFlow에서는 ImageNet EfficientNet-B0/B1을 피부·두피 클래스에 맞게 전이했다.

### 2.2 Head-only와 부분 미세조정

**왜 했나:** 처음부터 backbone을 모두 바꾸면 사전학습 특징이 급격히 무너질 수 있다. 먼저 새
클래스 경계를 학습하고 나중에 고수준 특징 일부만 의료 영상에 맞추려 했다.

**방법 설명:** Stage 1은 backbone을 고정하고 새 분류 head만 학습한다. Stage 2는 마지막 일부
레이어를 낮은 learning rate로 열어 부분 미세조정한다. Batch Normalization은 작은 batch의 통계로
흔들리지 않도록 고정했다. Stage 1→2 변화에는 레이어 해제, 낮은 학습률과 추가 학습량이 함께
포함되므로 상승분을 한 요소만의 효과로 설명하지 않는다.

### 2.3 해상도 224·256·384

**왜 했나:** 각질, 모낭과 작은 피부 병변은 축소 과정에서 정보가 사라질 수 있다. 반면 큰 입력은
메모리, 학습 시간과 실제 추론 비용을 늘린다.

**방법 설명:** 같은 계열 모델에서 224→256→384를 비교해 세부 정보 보존 이득과 비용을 확인했다.
[Gessert et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7150512/)의 피부 병변 다중 해상도
EfficientNet 연구는 이 가설을 검토한 근거다. MediFlow는 그 논문의 ensemble 전체를 재현한 것이
아니라 입력 해상도 가설만 가져왔다.

### 2.4 세 Loss의 목적

| Loss | 원리 | 왜 시험했나 |
|---|---|---|
| Cross Entropy | 정답 클래스의 손실을 직접 줄이는 기본 다중 분류 손실 | 모든 비교의 기준선 |
| Label Smoothing 0.05 | 정답 목표를 1.0에 몰지 않아 과도한 확신을 완화 | Hair처럼 시각 경계가 모호한 클래스의 과신 감소 가능성 |
| Focal Loss gamma 1.5 | 쉬운 예의 손실 비중을 낮추고 어려운 예에 집중 | 균형 데이터에서도 유사 클래스의 어려운 표본에 도움이 되는지 확인 |

[Label Smoothing](https://arxiv.org/abs/1512.00567)과
[Focal Loss](https://arxiv.org/abs/1708.02002)를 근거로 했다. Focal Loss 원 논문은 dense
detection의 불균형 문제를 다루므로, MediFlow 적용은 원 논문 재현이 아니라 어려운 분류 표본에
집중하는 확장 실험이다.

## 3. Skin 실험

### 3.1 Original Train과 Augmented Train 비교

**연구 질문:** 정제된 같은 데이터에서 Train 증강본을 포함하면 원본만 사용할 때보다 새 원본
Validation 분류가 좋아지는가?

**고정 조건:** EfficientNet-B0, 224, CE, head-only 15 epoch, Adam, batch 32, seed 42,
동일 Validation 1,000장. 변경 변수는 Train 데이터 종류다. Original Train 7,249장과
Augmented Train 14,498장은 epoch당 업데이트 수가 다르므로 계산량은 완전히 같지 않다.

| 조건 | Validation Accuracy | Validation Macro F1 | 판단 |
|---|---:|---:|---|
| Original | 0.973 | 0.9730455448862949 | 비교 조건 |
| Augmented | **0.982** | **0.9820008885656606** | 채택 |

Augmented는 오분류를 27장에서 18장으로 줄였고 모든 클래스 F1이 Original과 같거나 높았다.
Validation으로 고정한 Augmented 모델만 Test 700장에서 평가해 Accuracy `0.9871428571428571`,
Macro F1 `0.9871314132317626`을 얻었다. 이미 공개 데이터 성능이 높았으므로 더 복잡한 고급 방법을
반복해 Test에 맞출 위험을 늘리지 않고 Skin v1으로 패키징했다. 실제 USB 현미경 환자 데이터에서
같은 성능이 난다는 뜻은 아니다.

## 4. Web Skin 초기 실험

### 4.1 초기 Original·Augmented 비교

**왜 했나:** 웹캠 조명, 얼굴 거리와 색감 변화에 대응하도록 증강이 일반화를 높이는지 먼저 봤다.

| 초기 모델 | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| Original·B0·224·head-only | 0.6679999828338623 | 0.7850000262260437 | 0.7833125866587888 |
| Augmented·B0·224·head-only | **0.6880000233650208** | **0.8199999928474426** | **0.8195818185502844** |

증강 조건이 높았지만 이 결과는 이후 2단계 재학습 전의 초기 결과다. 뒤의 실험은 검사된 같은 ZIP과
Augmented Train을 사용하고 부분 미세조정을 추가한 별도 비교다.

### 4.2 논문 강화 전 여섯 설정

**왜 했나:** 낮은 초기 성능의 원인이 해상도, 과신, 어려운 표본, backbone 용량 또는 미세조정
부족 중 무엇인지 한 변수씩 확인했다.

| 실험 | 바꾼 핵심 변수 | Validation Accuracy | Macro F1 | 얻은 결론 |
|---|---|---:|---:|---|
| B0·224·CE | 2단계 학습 기준선 | 0.794 | 0.7917009497301795 | 부분 미세조정 기준 확보 |
| B0·256·CE | 입력 224→256 | **0.796** | 0.7915394647566528 | 정답 1장 증가, 당시 v1 선정 |
| B0·256·LS 0.05 | CE→Label Smoothing | 0.794 | 0.7892883497449976 | 과신 완화가 전체 성능 개선으로 연결되지 않음 |
| B0·256·Focal 1.5 | CE→Focal | 0.792 | 0.7898148816790076 | 어려운 표본 집중 이득 확인 안 됨 |
| B1·256·LS 0.05 | B0→B1 | 0.796 | **0.7931535064807589** | F1은 높지만 Accuracy 동률 |
| B1 Stage 2 +5 | 미세조정 연장 | 0.796 | 0.7931535064807589 | 부모 최고를 넘지 못해 연장 미채택 |

선정 규칙이 Validation Accuracy 우선이고 동률이면 앞선 실험 유지였으므로 B0·256·CE를 Web Skin
v1으로 고정했다. B1이 모든 면에서 나빴다는 뜻은 아니다. 고정 후 Test 400장에서 Accuracy
`0.8825`, Macro F1 `0.8813822927981046`을 기록했다.

## 5. Web Skin 논문 기반 강화 실험

Web Skin의 문제는 얼굴 전체에서 질환별 중요 부위, 서로 다른 크기의 국소 질감과 웹캠 색감 차이를
함께 다루는 것이었다. 저장된 v1 지표를 기준으로 사용하고 Test는 새 후보를 고정할 때까지 열지
않았다.

### 5.1 WS-DAN Attention

**방법 설명:** [WS-DAN](https://arxiv.org/abs/1901.09891)은 이미지 라벨만으로 attention map을
만들고 중요한 부위를 확대하는 attention crop과 일부를 가리는 attention drop으로 세부 특징을
학습한다.

**왜 했나:** 얼굴 전체에서 건선, 아토피와 여드름이 나타난 핵심 영역을 모델이 자동으로 찾도록
하려 했다. 8개 attention map과 원본·attention crop 예측 평균으로 현재 코드에 맞게 적용했다.

**결과:** Validation Accuracy `0.764`, Macro F1 `0.7546949301`로 v1 Macro F1보다
`0.0368445346` 낮아 미채택했다. 이는 WS-DAN 전체가 효과 없다는 결론이 아니라 현재
backbone·threshold·데이터 adaptation 결과다.

### 5.2 PMG

**방법 설명:** [PMG](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php)는
fine-grained classification에서 서로 다른 크기의 jigsaw 조각을 사용해 여러 세밀도의 특징을
단계적으로 학습하고 fusion한다.

**왜 했나:** 얼굴 전체 분포는 비슷해도 질환마다 국소 피부 질감과 병변 크기가 다르므로 전역 정보와
8·4·2 grid의 국소 정보를 함께 사용하려 했다. 실제 후보는 네 branch logit을 합산한 뒤 softmax를
적용한다.

**결과:** B0·256은 Validation Accuracy `0.85`, Macro F1 `0.8473279632397033`으로 v1보다
각각 `+0.054`, `+0.0557884984830505` 높았다. 세 논문 방법 중 선두여서 v2 후보가 됐다.

### 5.3 MixStyle

**방법 설명:** [MixStyle](https://arxiv.org/abs/2107.02053)은 학습 중 얕은 CNN feature의
평균과 표준편차를 다른 샘플과 섞어 새로운 style domain을 만든다. 추론 때는 추가 처리가 없다.

**왜 했나:** 웹캠 기기, 조명과 색감이 달라져도 질환 형태를 보도록 만들려 했다. block2b 뒤에서
alpha 0.1, 적용 확률 0.5로 사용했다.

**결과:** Validation Macro F1 `0.8033767540`으로 v1보다 `+0.0118372892` 높았지만 PMG보다
낮았다. 공개 데이터 안의 개선이며 실제 다른 웹캠 domain 강건성은 아직 측정하지 않았다.

### 5.4 PMG B1·384

**왜 했나:** PMG가 선두였으므로 더 큰 backbone과 입력이 국소 병변을 더 잘 보존하는지 확인했다.
GPU 메모리 때문에 batch 32→16은 필요한 동반 변경이었다.

**결과:** Validation Accuracy `0.866`, Macro F1 `0.8643216544321994`로 PMG B0·256보다
각각 `+0.016`, `+0.0169936911924961` 높았다. 그러나 입력 픽셀은 2.25배, 모델 파일은 약 29%
커지고 기록된 학습 시간은 약 2.49배였다. Validation 연구 선두로 보존하되 배포 비용과 성능의
균형을 위해 B0·256을 v2로 고정했다.

### 5.5 MedSigLIP-448 Frozen Linear Probe

**방법 설명:** [MedSigLIP-448](https://huggingface.co/google/medsiglip-448)은 의료 이미지·텍스트로
사전학습된 큰 vision-language 기반 모델이다. 이번 실험은 encoder를 고정하고 embedding 위에
Linear 층 하나만 학습해 의료 특징의 선형 분리 가능성을 확인했다. full fine-tuning이 아니다.

**왜 했나:** ImageNet 일반 특징보다 의료 사전학습 embedding이 얼굴 피부질환을 더 잘 분리하는지
단일 가설로 검증했다. Augmented Train 7,200장, Original Validation 500장, 448 입력, AdamW
50 epoch를 사용했다.

**결과:** Validation Accuracy `0.824`, Macro F1 `0.8195774288599683`으로 PMG v2보다 각각
`-0.026`, `-0.027750534379735` 낮았다. 여드름 F1은 `+0.0244379277` 개선됐지만 아토피와
정상 F1이 크게 하락해 전체 균형 성능이 낮았다. Test와 패키징은 진행하지 않았다.

### 5.6 Web Skin 최종 판단

PMG·B0·256만 고정 Test에서 평가해 Accuracy `0.915`, Macro F1 `0.9141049081029712`를
기록했다. WS-DAN, MixStyle, B1·384와 MedSigLIP은 각각 실패 또는 비용 비교 자료로 보존한다.

## 6. Hair 초기 실험

Hair의 핵심 문제는 미세각질·비듬·피지처럼 작은 질감이 서로 비슷하다는 점이었다. Clean 데이터와
Augmented Train 15,047장, Original Validation/Test 각 1,252장을 사용했다.

| 실험 | 왜 했나·바꾼 변수 | Validation Accuracy | Test Accuracy | Test Macro F1 | 판단 |
|---|---|---:|---:|---:|---|
| B0·224·CE | Clean 2단계 기준선 | 0.7643769979 | 0.7763578275 | 0.7757408068 | 기준점 |
| B0·256·CE | 작은 질감 보존, 224→256 | 0.7699680328 | 0.7891373802 | 0.7886477230 | 해상도 증가가 유리한 방향 |
| B0·256·LS 0.05 | 유사 클래스 과신 완화 | 0.7739616632 | 0.7867412141 | 0.7863134458 | Validation 상승, Test는 CE보다 낮음 |
| B0·256·Focal 1.5 | 어려운 표본 집중 | 0.7675718665 | **0.7907348243** | **0.7905122040** | Test 선두지만 Validation 선정 기준 미달 |
| B1·256·LS·Stage2 10 | 더 큰 backbone 표현 용량 | 0.7763578296 | 0.7787539936 | 0.7788123272 | B1 자체의 확실한 이득 없음 |
| B1·256·LS·Stage2 15 | 미세조정 5 epoch 연장 | **0.7771565318** | 0.7883386581 | 0.7885764577 | 논문 강화 전 v1 |

Focal 결과가 Test에서 높았지만 Test를 보고 후보를 고르면 평가 세트에 맞추게 된다. 사전 기준인
Validation으로 B1·256 연장 모델을 v1으로 선정했다. Label Smoothing은 평균 최대 예측 확률을
낮추는 방향으로 작동했지만 단독으로 큰 분류 성능 개선을 만들지는 못했다.

## 7. Hair 논문 기반 강화 실험

SupCon 비교에서는 새 방법의 차이를 같은 실행 코드와 데이터 조건에서 판단하기 위해 B1·256 일반
분류 기준선을 함께 학습했다. 이 기준선은 새 후보를 하나 더 만들기 위한 것이 아니라 공정한 대조군이다.
기준선은 Validation Accuracy `0.7755591054`, Macro F1 `0.7746028470`이었다.

### 7.1 Supervised Contrastive Learning

**방법 설명:** [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)은
같은 클래스 표현은 가깝게, 다른 클래스 표현은 멀게 배치한다.

**왜 했나:** 미세각질과 비듬처럼 픽셀 모양이 비슷한 클래스를 특징 공간에서 더 분리하려 했다.
B1·256에서 두 view를 사용한 표현 학습 15 epoch 뒤 분류기 15 epoch를 학습했다.

**결과:** Validation Accuracy `0.7915335463`, Macro F1 `0.7910880666`으로 분류 기준선보다
Macro F1 `+0.0164852196` 높았고 다섯 클래스 F1이 모두 상승했다. 미세각질↔비듬 혼동도
74건에서 69건으로 줄었다. 가설의 효과는 확인했지만 이후 B1·384보다 낮아 최종 후보는 아니다.

### 7.2 DINOv2 Small Linear Probe

**방법 설명:** [DINOv2](https://arxiv.org/abs/2304.07193)는 라벨 없이 대규모 데이터에서 범용
시각 표현을 학습한다. encoder를 고정하고 Linear head만 학습했다.

**왜 했나:** ImageNet CNN과 다른 자기지도 표현이 두피 질감에 유리한지 확인했다. DINOv2는
encoder를 고정한 Linear Probe라 15 epoch였고, EfficientNetV2-S와 B1·384는 head 학습 15회와
부분 미세조정 15회를 합쳐 30 epoch였다. epoch 수 차이는 같은 모델을 불공정하게 덜 학습시킨 것이
아니라 학습 가능한 범위가 다른 방법 구조에서 생긴 차이다.

**결과:** 224 입력에서 Validation Accuracy `0.7699680511`, Macro F1 `0.7705341251`로 기준선보다
낮았다. DINOv2 전체의 한계가 아니라 현재 고정 encoder·224·Linear Probe 설정의 결과다.

### 7.3 EfficientNetV2-S

**방법 설명:** [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html)는 더 빠른 학습과
파라미터 효율을 목표로 설계됐다.

**왜 했나:** 기존 EfficientNet-B1보다 새로운 CNN backbone의 전이 특성이 두피 분류에 유리한지
확인했다.

**결과:** Validation Accuracy `0.7691693291`, Macro F1 `0.7681559382`로 기준선보다 낮아
미채택했다.

### 7.4 EfficientNet-B1·384

**왜 했나:** B1·256만으로는 큰 이득이 없었지만, 두피의 작은 각질·모낭 정보가 256 축소에서
손실될 가능성을 분리해서 확인했다. B1, Label Smoothing 0.05와 Adam을 유지하고 입력을 384로
확대했다.

**결과:** Validation Accuracy `0.7963258785942492`, Macro F1 `0.7957146810115047`로 단일 모델
선두가 됐다. 후보 고정 후 Test에서 Accuracy `0.8003194888178914`, Macro F1
`0.8002222282821301`을 기록해 Hair v2로 선정했다. B1 하나가 아니라 B1·384 조합의 결과로
해석한다.

### 7.5 SupCon+B1·384 단순 확률 평균

**방법 설명:** 두 모델이 서로 다른 사진을 맞힌다면 softmax 확률을 1:1로 평균해 오류를 보완할 수
있다.

**왜 했나:** SupCon과 B1·384 중 하나라도 맞힌 Validation 이미지는 1,052장으로 oracle accuracy가
`0.8402555911`이어서 실제 결합 가능성을 확인했다.

**결과:** 평균 모델 Accuracy `0.7947284345`, Macro F1 `0.7942285284`로 B1·384 단일 모델보다
낮았다. 보완 가능성은 있었지만 단순 평균이 그것을 실현하지 못해 미채택했다.

### 7.6 SAM

**방법 설명:** [SAM](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)은
현재 가중치뿐 아니라 주변에서도 손실이 낮은 flat minimum을 찾는 최적화 방법이다.

**왜 했나:** 데이터가 조금 달라져도 덜 민감한 해를 찾아 일반화를 높이는지 B1·384의 같은 Stage 1
조건에서 Adam과 비교했다.

**결과:** Accuracy `0.7963258786`으로 Adam과 같았고 Macro F1 `0.7954748921`로 Adam보다
`0.0002397889` 낮았다. 미세각질↔비듬 혼동은 71건에서 68건으로 줄었지만 다른 클래스 F1이
낮아졌고 계산이 더 복잡해 미채택했다.

### 7.7 MedSigLIP-448 Frozen Linear Probe

**왜 했나:** 피부과 영상을 포함한 의료 사전학습 특징이 ImageNet 기반 B1보다 두피 클래스를 더 잘
선형 분리하는지 확인했다. Augmented Train 15,047장, Original Validation 1,252장, frozen
MedSigLIP embedding과 Linear head, AdamW 50 epoch를 사용했다.

**결과:** Validation Accuracy `0.7699680511182109`, Macro F1 `0.7687534259685547`로 Hair v2보다
각각 `-0.0263578274760383`, `-0.02696125504295` 낮았다. 다섯 클래스 F1이 모두 낮았다.
미세각질↔비듬 혼동은 71건에서 67건으로 줄었지만 다른 혼동이 증가했다. Test와 패키징은 진행하지
않았다. 이 판단은 frozen Linear 설정에 한정하며 MedSigLIP full fine-tuning 성능을 의미하지 않는다.

## 8. 수치 없는 구조 검토와 중단 기록

### 8.1 YOLO 병변 crop

**처음 생각:** 얼굴에서 병변을 YOLO로 찾고 crop한 뒤 기존 확대 피부 분류기에 넣으려 했다.

**중단 이유:** 홍반이나 피부색 변화는 넓게 퍼지고 경계가 명확하지 않을 수 있다. 병변 하나만
자르면 주변 분포와 얼굴 전체 맥락을 잃는다. 사용할 detection bbox의 신뢰 가능한 라벨과 독립
평가도 부족했다. 따라서 성능 수치를 만든 완료 실험으로 표현하지 않고 구조 검토 후 중단으로
기록한다. 이후 확대 병변 Skin과 얼굴 전체 Web Skin 모델을 분리했다.

### 8.2 Hair 6-class→5-class

초기 모낭사이홍반과 모낭홍반농포가 사람이 보기에도 매우 유사해 label ambiguity가 큰 것으로
확인됐다. 모낭홍반농포를 제외하고 나머지 다섯 클래스를 재구성했다. 이것은 모델 기법 실험이 아니라
분류 문제 정의와 데이터 선별 변경이며, 해당 초기 6-class 수치가 현재 후보 성능의 기준은 아니다.

## 9. 전체 실험 결론

| 도메인 | 실제로 도움이 된 핵심 | 채택하지 않은 주요 방법 | 현재 후보 |
|---|---|---|---|
| Skin | Clean 분할과 Train 증강 | 추가 고급 방법은 반복하지 않음 | B0·224·CE·Augmented v1 |
| Web Skin | PMG의 다중 세밀도 학습 | WS-DAN, MixStyle 단독, B1·384 배포안, MedSigLIP Linear | PMG·B0·256 v2 |
| Hair | 고해상도 B1·384, SupCon은 보조 개선 확인 | DINOv2 Linear, EfficientNetV2-S, 앙상블, SAM, MedSigLIP Linear | B1·384·LS 0.05·Adam v2 |

유명한 논문 방법을 많이 적용하는 것 자체가 목표가 아니었다. 각 데이터에서 발생한 문제와 논문의
아이디어를 연결해 한 가설씩 검증했고, 같은 Validation에서 실제로 좋아진 방법만 후보 선정 근거로
사용했다. Test는 가능한 한 후보를 고정한 뒤 평가했다. 초기 탐색 자료에 여러 Test 결과가 남아 있는
경우는 후기의 엄격한 최종 후보 프로토콜과 구분한다.

## 10. 발표·보고서에서 사용할 해석

- “논문의 아이디어를 현재 데이터에 맞게 적용하고 같은 Validation에서 비교했다.”
- “Skin은 증강, Web Skin은 PMG, Hair는 고해상도 B1·384가 실제 개선으로 이어졌다.”
- “SupCon과 MixStyle은 기준선보다 좋아졌지만 최종 선두는 아니었다.”
- “DINOv2, EfficientNetV2-S, SAM, WS-DAN, 단순 앙상블과 MedSigLIP frozen Linear는 현재
  설정에서 채택 기준을 넘지 못했다.”
- “공개 데이터 결과이며 실제 장비·실제 사용자 성능은 아직 검증하지 않았다.”

“논문 방법이므로 좋아졌다”, “모든 데이터 누수를 제거했다”, “softmax 점수가 질환일 실제 확률이다”
같은 표현은 사용하지 않는다.

## 11. 앞으로 모든 실험에 남길 기록

새 실험은 결과 숫자만 추가하지 않고 다음 항목을 함께 기록한다.

1. 해결하려는 문제와 실험 가설
2. 논문·공식 문서에서 가져온 방법의 원리와 출처
3. 기존 기준선, 고정 조건, 하나의 주요 변경 변수
4. 데이터 버전·분할·클래스 순서·해시
5. 실제 구현 범위와 원 방법에서 달라진 부분
6. Validation 결과, 클래스별 결과와 주요 혼동
7. 기존 후보 대비 변화량과 비용
8. 채택·미채택 이유와 Test 실행 여부
9. 코드·환경·seed·checkpoint·원본 결과 위치
10. 확인하지 못한 항목과 실제 장비 적용 한계

## 12. 근거 파일

- 전체 프로젝트: `MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md`
- Hair 초기 실험: `HAIR_EXPERIMENT_SUMMARY_20260907.md`
- Hair 논문 실험: `PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md`
- Web Skin 초기 실험: `WEB_SKIN_FULL_SUMMARY_20260908.md`
- Web Skin 논문 실험: `WEB_SKIN_PAPER_METHOD_RESULT_20260922.md`
- Skin 증강 비교: `SKIN_ORIGINAL_VS_AUGMENTED_20260909.md`
- Web Skin MedSigLIP: `WEB_SKIN_MEDSIGLIP_LINEAR_RESULT_20260924.md`
- Hair MedSigLIP: `HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md`
- 현재 모델 버전: `../../results/MODEL_VERSION_COMPARISON.md`
- 정확한 후보 ID와 해시: `../../results/CANDIDATE_INDEX.json`

표의 값은 위 원본 JSON·CSV와 기존 검증 문서에서 옮겼으며 반올림 표기와 전체 정밀도 표기를
구분했다.
