# MediFlow 피부·두피 분류 프로젝트 최종 종합 정리

기준일: 2026-09-23  
범위: Skin, Web Skin, Hair 공개 데이터 분류 모델  
현재 상태: 데이터 정제·재학습·논문 기반 실험·최종 Test·후보 패키징 완료

이 문서는 발표, 보고서, 팀 인계에 사용하는 최종 기준 문서다. 논문의 원래 주장, MediFlow가
실제로 적용한 범위, 측정 결과를 구분한다. 논문 벤치마크 성능을 MediFlow 성능으로 옮겨 적지
않으며, 실제 결과는 `results/`의 JSON·CSV와 후보 패키지를 기준으로 한다.

## 1. 프로젝트가 해결한 문제

초기 목표는 안구 질환 AI에 피부·두피 분석을 더하는 것이었다. 데이터를 직접 확인하면서 같은
피부 분야라도 촬영 거리와 영상 특징이 크게 달라 하나의 모델로 합치기 어렵다고 판단했다.

| 도메인 | 입력 장비·부위 | 모델이 보는 특징 | 현재 출력 |
|---|---|---|---|
| Skin | USB 현미경·확대 피부 병변 | 병변 표면, 경계, 색과 미세 형태 | 10종, 정상 없음 |
| Web Skin | 웹캠·얼굴 정면 | 얼굴 전체의 색, 홍반, 분포와 국소 특징 | 5종, 정상 있음 |
| Hair | USB 현미경·두피 확대 | 각질, 피지, 모낭, 홍반과 모발 | 5종, 정상 없음 |

따라서 최종 구조는 입력 장비와 촬영 부위로 세 전문 분류기를 선택하는 방식이다. 현재 결과는
의료 진단 시스템이나 하나의 Medical VLM이 아니라 세 개의 이미지 분류 후보 모델이다.

## 2. 전체 진행 과정

1. Kaggle과 AI Hub에서 피부·두피 데이터를 조사했다.
2. 최종 학습 원천은 AI Hub 데이터로 확인하고 도메인별 클래스를 직접 선별했다.
3. Original·Augmented 데이터셋과 EfficientNet-B0 초기 학습 코드를 만들었다.
4. 확대 병변 모델을 얼굴 웹캠에 연결하기 위한 YOLO crop을 검토했지만, 넓게 퍼지는 홍반과
   주변 분포 정보가 사라질 수 있어 중단했다.
5. Skin, Web Skin, Hair의 역할과 입력 장비를 분리했다.
6. 초기 학습에서 Skin은 높았지만 Web Skin과 Hair는 상대적으로 낮아 학습법 개선이 필요했다.
7. Train·Validation·Test 사이 완전 동일 이미지 중복을 검사했다.
8. Hair와 Skin에서 중복을 제거한 Clean 데이터셋을 새로 만들었다.
9. ImageNet 전이학습, 2단계 부분 미세조정, 입력 해상도, Loss와 Backbone을 비교했다.
10. Hair의 유사 클래스와 Web Skin의 얼굴 세부 특징에 맞춰 논문 기반 방법을 별도로 실험했다.
11. Validation으로 후보를 고정한 뒤 최종 Test를 평가하고 모델·전처리·클래스 계약을 ZIP으로
    패키징했다.

## 3. 데이터 출처와 클래스 선정

| 도메인 | 원천 | 원천 규모 | MediFlow 선택 | 현재 학습 데이터 |
|---|---|---:|---|---:|
| Skin | [AI Hub 피부종양 이미지 합성 데이터 71864](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71864) | 15종×1,000=15,000 | 직접 검수 후 10종 | Clean 원본 8,949 |
| Web Skin | [AI Hub 안면부 피부질환 이미지 합성 데이터 71863](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71863) | 6종×정면·측면×1,000=12,000 | 지루·측면 제외, 정면 5종 | 원본 4,500 |
| Hair | [AI Hub 유형별 두피 이미지 216](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=216) | 고유 이미지 101,027 | 중등도·중증 5증상, 각 2,900 | Clean 원본 12,536 |

### Skin 10종

광선각화증, 기저세포암, 보웬병, 사마귀, 지루각화증, 편평세포암, 표피낭종, 피부섬유종,
혈관종, 흑색점이다. 초기에는 선택한 10종의 클래스당 1,000장 가운데 900장씩 총 9,000장을
분할에 사용했다. 나머지 100장을 제외한 세부 기준은 현재 기록에 남아 있지 않다.

### Web Skin 5종

출력 순서는 건선, 아토피, 여드름, 정상, 주사다. 얼굴 전체 입력과 맞지 않는 측면은 제외했고,
직접 검수에서 주사와 시각적으로 혼동하기 쉬운 지루성 피부염은 제외했다. 원천 기준 선택 대상은
5,000장이지만 실제 분할에는 클래스당 900장, 총 4,500장을 사용했다.

### Hair 5종

출력 순서는 모낭사이홍반, 미세각질, 비듬, 탈모, 피지과다다. 양호·경증은 제외하고 중등도·중증을
질환명 단위로 합쳤다. 초기 6종 중 모낭홍반농포는 모낭사이홍반과 사람이 보기에도 유사한 사진이
많아 제외했다. 가장 적은 비듬 수량에 맞춰 각 2,900장으로 균형화했다.

## 4. 데이터 증강과 평가 분할

증강의 목적은 단순히 장수를 늘리는 것이 아니라 밝기, 대비, 색, 회전, 크기, blur, noise처럼
촬영 조건이 바뀌어도 특징을 찾게 하는 것이었다. 기본 원칙은 다음과 같다.

```text
Original Train = 원본
Augmented Train = 원본 + 학습용 증강본
Validation = 원본
Test = 원본
```

평가 사진을 증강하지 않아 Original과 Augmented 학습을 같은 Validation·Test에서 비교했다.

| 도메인 | Original Train | Augmented Train | Validation | Test |
|---|---:|---:|---:|---:|
| Skin Clean | 7,249 | 14,498 | 1,000 | 700 |
| Web Skin | 3,600 | 7,200 | 500 | 400 |
| Hair Clean | 10,032 | 15,047 | 1,252 | 1,252 |

Skin은 각 Train 원본에서 새 증강본을 한 장씩 생성했다. Hair는 10,032개 원본 Train에 5,015개
증강본을 더했다. Web Skin은 3,600개 원본 Train과 3,600개 증강본을 사용했다.

## 5. 중복 발견과 Clean 데이터 재구성

Hair 원본 분할에서 RGB 픽셀 내용이 완전히 같은 파일이 Train–Validation 195개,
Train–Test 178개, Validation–Test 16개 발견됐다. 파일명이 아니라 이미지 내용의 해시를
비교한 결과다. 중복을 제거하고 분할을 다시 만들어 원본 14,500장에서 서로 구분되는 12,536장을
남겼다. 세 중복 수에는 세 분할에 함께 있던 파일이 반복 집계될 수 있으므로 단순 합계를 제거
수로 해석하지 않는다.

Skin은 원본 9,000장에서 완전 동일 이미지 51장을 제거해 8,949장을 남겼다. 새 분할은 Train
7,249, Validation 1,000, Test 700이며 생성 후 분할 간 완전 동일 이미지와 라벨 충돌이 0임을
검사했다. Web Skin의 기존 감사에서는 파일·RGB 픽셀이 완전히 같은 분할 간 중복이 발견되지 않아
기존 분할을 유지했다.

이 검사는 완전 동일 이미지 중복에 대한 검사다. 사람 ID, 병변 ID, 촬영 세션 정보가 충분하지
않아 같은 사람·같은 부위의 유사 사진이 분할 사이에 있는지는 확인하지 못했다. 재압축, 밝기 변화,
근접 장면과 임상 라벨 정확성도 별도 검증이 필요하다.

## 6. 공통 학습 방법과 선택 이유

### ImageNet EfficientNet 전이학습

[EfficientNet](https://proceedings.mlr.press/v97/tan19a.html)은 깊이, 너비, 입력 해상도를 함께
확장하는 compound scaling을 제안했다. MediFlow는 제한된 의료 이미지로 처음부터 모든 시각
특징을 학습시키는 대신 ImageNet에서 일반 모양·색·질감을 배운 B0/B1을 시작점으로 사용했다.

### 2단계 부분 미세조정

Stage 1에서는 pretrained backbone을 고정하고 새 분류층을 학습했다. Stage 2에서는 후반부
30개 레이어만 낮은 학습률로 열고 Batch Normalization은 고정했다. 먼저 클래스 경계를 만들고,
그다음 고수준 특징 일부만 피부·두피에 맞추어 사전학습 특징이 한꺼번에 무너지는 위험을 줄이려는
설계다.

### 224·256·384 해상도

해상도를 높이면 작은 각질·모낭·홍반 특징을 덜 잃을 수 있지만 픽셀 수, 메모리와 추론 시간이
늘어난다. 224와 256을 공통 비교했고 Hair에서는 384를 추가했다. 피부 병변 분야에서 여러 입력
해상도의 EfficientNet을 사용한 [Gessert et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7150512/)
도 고해상도·다중 해상도 검토의 근거로 삼았다. MediFlow는 그 논문의 ensemble·metadata 전체를
재현한 것이 아니라 B1·384 단일 모델의 해상도 가설만 시험했다.

### Label Smoothing과 Focal Loss

[Label Smoothing](https://arxiv.org/abs/1512.00567)은 정답 클래스에 1.0을 몰아주는 학습을
완화한다. 시각적 경계가 모호한 클래스에서 과신을 줄일 가능성을 보기 위해 `0.05`를 시험했다.
[Focal Loss](https://arxiv.org/abs/1708.02002)는 잘 맞히는 쉬운 예의 손실을 낮춰 어려운 예에
집중하도록 제안됐다. 원 논문은 dense detection의 심한 불균형 문제를 다루었으므로 MediFlow의
균형 분류 적용은 어려운 유사 클래스에 집중하는지 보는 확장 실험이며 원 논문 재현은 아니다.

### 모델 선택 규칙

후기 논문 기반 실험은 같은 데이터 ZIP과 클래스 순서, 같은 Validation을 사용하고 Test를 후보
선택에 사용하지 않았다. Validation Accuracy와 Macro F1로 후보를 고정한 뒤 선택된 모델만 최종
Test에 평가했다. 초기 6개 실험 자료에는 여러 후보의 Test 지표가 함께 기록되어 있으므로 초기
탐색 결과와 후기 최종 평가 프로토콜을 구분해 해석한다.

## 7. Hair 논문 기반 실험

Hair의 핵심 연구 질문은 미세각질과 비듬처럼 유사한 클래스의 특징을 더 잘 분리할 수 있는가였다.

| 방법 | 논문의 아이디어 | MediFlow 적용 이유·범위 | Validation Macro F1 | 판단 |
|---|---|---|---:|---|
| B1·256 기준선 | EfficientNet 전이학습 | 고급 방법 비교 기준 | 0.7746028470 | 기준 |
| SupCon B1·256 | 같은 클래스는 가깝게, 다른 클래스는 멀게 | 두 view를 만든 표현 학습 15회 후 분류 학습 15회 | 0.7910880666 | 개선 확인 |
| DINOv2 Small | 대규모 자기지도 범용 특징 | 224 encoder 고정, linear probe 15회 | 0.7705341251 | 현재 설정 미채택 |
| EfficientNetV2-S | 빠른 학습과 파라미터 효율 | 새 CNN backbone의 전이 가능성 확인 | 0.7681559382 | 미채택 |
| B1·384 | 고해상도에서 미세 특징 보존 | B1·384 부분 미세조정 | **0.7957146810** | 단일 모델 선두 |
| B1·384 SAM | 주변에서도 손실이 낮은 flat minimum | 같은 Stage 1에서 Adam과 SAM 비교 | 0.7954748921 | Adam보다 낮아 미채택 |
| SupCon+B1·384 평균 | 서로 다른 오답의 보완 | 두 softmax 1:1 평균 | 0.7942285284 | 단일 B1·384보다 낮음 |

[Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)은
같은 클래스 표현을 모으고 다른 클래스를 분리한다. MediFlow에서 기준선보다 Accuracy가
`0.0159744409`, Macro F1이 `0.0164852196` 높았고 다섯 클래스 F1이 모두 상승했다. 이 결과는
유사 클래스 표현 분리 가설과 맞지만 최종 B1·384보다 낮아 최종 후보로 쓰지 않았다.

[DINOv2](https://arxiv.org/abs/2304.07193)는 다양한 데이터에서 자기지도 방식으로 범용 시각
특징을 학습한다. 이번 결과는 고정 encoder linear probe 설정이 낮았다는 뜻이며 DINOv2 전체가
두피에 부적합하다는 결론은 아니다. [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html)도
현재 설정에서 선두가 아니었다.

[SAM](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/)은
손실값과 주변의 sharpness를 함께 줄이는 최적화다. Adam과 Accuracy는 같았지만 Macro F1이
`0.0002397889` 낮아 추가 계산을 정당화하지 못했다. SupCon과 B1·384는 서로 다르게 맞힌 사진이
있었지만 단순 평균은 B1·384 단일 모델보다 낮았다.

최종 Hair 후보는 `EfficientNet-B1·384·Label Smoothing 0.05·Adam`이다. Test 1,252장에서
Accuracy `0.8003194888`, Macro F1 `0.8002222283`을 기록했다. 미세각질 F1은 `0.7807692308`,
비듬 F1은 `0.75`로 여전히 주요 개선 대상이다.

## 8. Web Skin 논문 기반 실험

Web Skin은 얼굴 전체에서 질환별로 중요한 영역, 서로 다른 크기의 국소 특징, 촬영 색감 차이를
학습하는 방법을 비교했다. 기존 저장 B0·256·CE 기준선은 다시 학습하지 않고 같은 Validation
500장의 기록을 사용했다.

| 방법 | 논문의 아이디어 | MediFlow 실제 구현 | Validation Macro F1 | 기준선 대비 | 판단 |
|---|---|---|---:|---:|---|
| B0·256·CE 기준선 | 일반 부분 미세조정 | 저장 지표 재사용 | 0.7915394648 | 기준 | 기준 |
| WS-DAN adaptation | attention으로 중요한 부분을 crop/drop | 8 attention map, 원본·attention crop 예측 평균 | 0.7546949301 | -0.0368445346 | 미채택 |
| PMG B0·256 | 여러 크기의 jigsaw로 multi-granularity 학습 | 8·4·2 grid branch와 fusion logit 합산 | **0.8473279632** | **+0.0557884985** | 채택 |
| MixStyle | 얕은 feature 통계 혼합으로 새 style 생성 | block2b 뒤, alpha 0.1, 확률 0.5 | 0.8033767540 | +0.0118372892 | 개선했지만 PMG보다 낮음 |

[WS-DAN](https://arxiv.org/abs/1901.09891)은 약한 지도만으로 attention map을 만들고 attention
crop과 drop으로 중요한 부위를 더 보게 한다. 얼굴 질환의 중요한 영역을 자동으로 찾는 가설은
타당했지만 현재 adaptation은 기준선보다 낮았다. 이는 WS-DAN 원 논문의 효과가 없다는 뜻이
아니며 현재 데이터·Backbone·threshold 조합에서 개선되지 않았다는 뜻이다.

[PMG](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php)는
미세 분류에서 서로 다른 granularity의 jigsaw 정보를 단계적으로 결합한다. Web Skin의 건선,
아토피, 여드름처럼 전체 얼굴에서 비슷해 보이지만 국소 질감과 분포가 다른 클래스를 구분하는
가설로 적용했다. B0·256 PMG는 기준선보다 Accuracy `0.054`, Macro F1 `0.0557884985` 높아 세
방법 중 선두였다.

[MixStyle](https://arxiv.org/abs/2107.02053)은 학습 중 얕은 CNN feature의 평균·표준편차를
샘플 사이에서 섞어 색감·질감 style의 새 domain을 만든다. 웹캠 조명·색감 변화에 대응하는
가설로 사용했고 기준선보다 소폭 높았지만 PMG보다 낮았다. 실제 환자 웹캠 domain에서 강건성이
개선됐는지는 아직 측정하지 않았다.

PMG를 B1·384로 확장했을 때 Validation Accuracy `0.866`, Macro F1 `0.8643216544`로 B0·256보다
높았다. 그러나 입력 픽셀이 2.25배, 모델 파일이 약 29% 크고 기록된 학습 시간도 약 2.49배였다.
현재 배포 목표에서는 비용과 성능 균형을 택해 B0·256을 최종 Test 후보로 고정했고 B1·384는
Validation 성능 선두 연구 후보로 보존했다.

최종 PMG B0·256은 Test 400장에서 366장을 맞혀 Accuracy `0.915`, Macro F1
`0.9141049081`을 기록했다. 건선 Recall `0.825`가 가장 낮고 정상 Recall은 `1.0`이었다. 정상
80장을 모두 맞혔지만 다른 클래스 3장을 정상으로 예측했으므로 정상 판정이 완전하다는 뜻은 아니다.

## 9. Skin 실험과 추가 고급 실험을 생략한 이유

Skin은 완전 동일 중복 51장을 제거한 Clean 데이터에서 Original과 Augmented만 같은 조건으로
비교했다. B0·224·CE head-only 15 epoch에서 Augmented Validation Accuracy `0.982`, Macro F1
`0.9820008886`으로 Original `0.973`, `0.9730455449`보다 높았다. 후보를 고정한 뒤 Test
700장에서 Accuracy `0.9871428571`, Macro F1 `0.9871314132`를 기록했다.

이미 높은 공개 데이터 성능이 나왔기 때문에 Hair·Web Skin과 같은 고급 방법을 반복하면 계산량을
늘리고 Test를 반복해서 보게 될 위험이 더 컸다. 따라서 Skin은 데이터 중복 정제와 증강 효과 확인,
후보 패키징에서 멈췄다. 이 높은 수치는 실제 USB 현미경 환자 데이터 성능을 보장하지 않는다.

## 10. 현재 최종 후보 3개

| 도메인 | 현재 후보 | 입력 | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|---:|
| Skin | EfficientNet-B0·CE·Augmented | 224 | 0.982 | 0.9871428571 | 0.9871314132 |
| Web Skin | PMG·EfficientNet-B0·CE | 256 | 0.85 | 0.915 | 0.9141049081 |
| Hair | EfficientNet-B1·LS 0.05·Adam | 384 | 0.7963258786 | 0.8003194888 | 0.8002222283 |

정확한 후보 ID, 로컬 경로, ZIP·모델 SHA-256은 `results/CANDIDATE_INDEX.json`이 기계 판독
기준이다. 세 후보 모두 원본 ZIP과 압축 해제본을 `results/<domain>/candidates/`에 보존했다.

| 도메인 | 후보 ZIP SHA-256 | 모델 SHA-256 |
|---|---|---|
| Skin | `9b3e3893a6cd89d58bd8ae3a4b85f4214f6a32a21586094ea416c552312f1686` | `0b2a757320a113dbc1b285adab8c7745b2f7edc2cb29cc402d06f7882edc9f7a` |
| Web Skin | `38bd47493ec06bcf2be351a7d12b8f889489fd5d8205e4fcb39557a400471db9` | `83e659dd09a9355135ee0de0197037e81ece962777f59dedae8d9a37b49a3fc4` |
| Hair | `870356a8b079539e8e6963b937964f790653a1b39994ee2b6b7c7bdfed0134d5` | `9faa33b8f79f45ef6e7be415473e2ee5225f6195fff04d098828733db23ed3c2` |

## 11. 모델 사용 계약

1. 장비 정보와 촬영 부위를 함께 사용해 모델을 고른다.
2. `class_names.json`의 배열 순서를 바꾸지 않는다.
3. RGB float32 0–255를 후보별 크기로 TensorFlow bilinear resize한다.
4. 현재 EfficientNet 계열 모델은 내부에 `Rescaling(1/255)`이 있으므로 외부 `/255`를 하지 않는다.
5. Hair와 Skin은 단일 softmax 출력을 사용한다.
6. Web Skin PMG는 네 logit 출력을 합산한 뒤 softmax를 한 번 적용한다. 패키지의
   `inference.py` 또는 `candidate_reproduction.py`를 사용한다.
7. softmax 점수를 실제 정답 확률이나 의료적 위험도로 해석하지 않는다.
8. Hair와 Skin은 정상 클래스가 없으므로 낮은 점수를 정상으로 바꾸지 않는다.

공통 추론 모듈은 세 후보 모델을 실제로 불러와 샘플 입력, 모델 해시, 입력 shape, 클래스 수와
출력 합이 1인지 검사했다. Web Skin의 네 출력 합산 계약도 로컬 dummy·샘플 추론을 통과했다.

## 12. 코드와 노트북 구조

- `notebooks/01~03`: 공통 데이터 감사, Original/Augmented 비교, 6개 기본 실험
- `notebooks/04~10`: Hair 논문 기반 실험과 최종 패키징 기록
- `notebooks/11~13`: Web Skin WS-DAN·PMG·MixStyle, B1·384, 최종 패키징
- `src/mediflow_datasets/`: 재사용 가능한 감사·학습·평가·추론 코드
- `scripts/`: 노트북 생성기와 전처리 도구
- `results/<domain>/experiments/`: 실험 기록과 그림
- `results/<domain>/candidates/`: 팀 통합용 현재·과거 후보
- `docs/research/`: 연구 근거와 결과 해석
- `docs/archive/`: 당시 계획과 중간 검토 기록

현재 실행 기준은 `docs/ROADMAP.md`, 이 문서, `results/CANDIDATE_INDEX.json`,
`docs/guides/TEAM_MODEL_QUICKSTART.md`다. 과거 문서는 실험 당시 기록을 보존하며 현재 후보를
고르는 기준으로 사용하지 않는다.

## 13. 완료한 일과 완료하지 않은 일

### 완료

- 세 도메인 역할과 클래스 계약 확정
- 데이터 구조·완전 동일 중복 검사
- Hair·Skin Clean 데이터 재구성
- Original/Augmented와 2단계 미세조정 비교
- Hair·Web Skin 논문 기반 강화 실험
- Validation 기반 후보 선택과 최종 Test
- 모델 ZIP, 해시, 모델 카드, 전처리·추론 계약
- 로컬 모델 load와 공통 추론 검사

### 보류 또는 미검증

- 실제 USB 현미경·웹캠 환자 데이터 평가
- 사람·병변·촬영 세션 단위 누수 확인
- 재압축·유사 장면 near-duplicate 전수 검사
- 임상 전문가 라벨 검토
- 범위 밖 입력과 저품질 이미지 거부
- 점수 calibration
- Hair·Skin 정상 클래스 추가
- Jetson 변환·지연시간·메모리 측정
- LLM/VLM 및 장비 통합

## 14. 결과를 발표할 때의 해석

발표의 핵심은 “논문 방법을 사용했기 때문에 무조건 좋아졌다”가 아니다. 데이터 중복 문제를
먼저 고치고, 각 방법이 해결하려는 문제와 현재 데이터의 문제를 연결해 하나씩 검증했다.
SupCon과 MixStyle은 개선됐지만 최종 선두는 아니었고, DINOv2·EfficientNetV2·SAM·WS-DAN과
단순 앙상블은 채택하지 않았다. Hair에서는 고해상도 B1, Web Skin에서는 multi-granularity PMG,
Skin에서는 Clean 데이터와 저장 증강본이 실제로 선택됐다.

다음 표현을 사용한다.

- “완전 동일 이미지의 분할 간 중복을 제거했다.”
- “같은 공개 Validation에서 후보를 비교했다.”
- “공개 Test 기준 성능이며 실제 장비 성능은 미검증이다.”
- “논문의 아이디어를 현재 데이터와 코드에 맞게 적용한 실험이다.”

다음 표현은 피한다.

- “모든 데이터 누수를 제거했다.”
- “91.5% 확률로 진단한다.”
- “논문을 완전히 재현했다.”
- “정상 클래스가 없는 모델에서 낮은 점수는 정상이다.”

## 15. 최종 결론

이 프로젝트의 성과는 세 개의 숫자만 높인 것이 아니다. 촬영 방식이 다른 문제를 세 모델로
분리하고, 데이터 중복을 발견해 Clean 분할을 만들고, 전이학습과 부분 미세조정의 재현 가능한
기준을 세웠다. 이후 논문의 아이디어를 문제별 가설로 바꾸어 같은 Validation에서 검증하고,
개선되지 않은 방법을 제외한 뒤 세 후보를 해시와 추론 계약까지 포함해 패키징했다.

공개 데이터 기준 최종 결과는 Skin `98.71%`, Web Skin `91.5%`, Hair `80.03%` Accuracy다.
이제 모델 연구 산출물은 팀에 전달할 수 있는 상태다. 다음 성능 주장을 강화하려면 새 학습법보다
먼저 실제 장비·실제 사용자 데이터를 독립 검증 세트로 확보하는 것이 가장 중요하다.

## 16. 주요 논문

1. Tan & Le, [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html), ICML 2019.
2. Szegedy et al., [Label Smoothing을 포함한 Inception 재검토](https://arxiv.org/abs/1512.00567), CVPR 2016.
3. Lin et al., [Focal Loss](https://arxiv.org/abs/1708.02002), ICCV 2017.
4. Khosla et al., [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), NeurIPS 2020.
5. Oquab et al., [DINOv2](https://arxiv.org/abs/2304.07193), 2023/2024.
6. Tan & Le, [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html), ICML 2021.
7. Foret et al., [SAM](https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/), ICLR 2021.
8. Gessert et al., [Multi-resolution EfficientNets for skin lesion classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC7150512/), MethodsX 2020.
9. Hu et al., [WS-DAN](https://arxiv.org/abs/1901.09891), 2019.
10. Du et al., [PMG](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3399_ECCV_2020_paper.php), ECCV 2020.
11. Zhou et al., [MixStyle](https://arxiv.org/abs/2107.02053), 2021.

## 17. 실제 근거 파일

- 현재 후보: `results/CANDIDATE_INDEX.json`
- 모델 사용: `results/MODEL_USAGE.md`
- Hair 논문 분석: `docs/research/PAPER_EXPERIMENTS_FINAL_ANALYSIS_20260922.md`
- Web Skin 세 방법: `docs/research/WEB_SKIN_PAPER_METHOD_RESULT_20260922.md`
- Web Skin B1·384: `docs/research/WEB_SKIN_PMG_B1_384_RESULT_20260923.md`
- Hair 최종 후보: `docs/research/HAIR_FINAL_CANDIDATE_V2_20260921.md`
- Web Skin 최종 후보: `docs/research/WEB_SKIN_FINAL_CANDIDATE_V2_20260923.md`
- Skin 최종 결과: `docs/research/SKIN_ORIGINAL_VS_AUGMENTED_20260909.md`
- 데이터·파일 위치: `docs/PROJECT_ARTIFACT_MAP_20260923.md`

## 18. 현재 후보 기계 판독 기준

아래 값은 발표용 반올림 값이 아니라 `results/CANDIDATE_INDEX.json`에 기록된 전체 값이다.
후보를 복사하거나 팀 코드에 연결할 때는 표시 이름 대신 후보 ID를 사용한다.

| 도메인 | 후보 ID | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|
| Skin | `public_candidate_v1_b0_224_ce_augmented_20260909_075056` | `0.982` | `0.9871428571428571` | `0.9871314132317626` |
| Web Skin | `public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de` | `0.85` | `0.915` | `0.9141049081029712` |
| Hair | `public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4` | `0.7963258785942492` | `0.8003194888178914` | `0.8002222282821301` |

이 표는 후보 식별과 수치 대조를 위한 부록이다. 모델 선택 과정과 각 수치의 의미는 앞 절의
실험 설계·결과 해석을 함께 따른다.
