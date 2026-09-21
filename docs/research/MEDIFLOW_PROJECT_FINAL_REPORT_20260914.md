# MediFlow 피부·두피 AI 연구 개발 최종 정리

## 1. 이 문서의 목적과 읽는 방법

이 문서는 MediFlow 프로젝트에서 피부·두피 이미지 분류를 시작한 때부터 공개 데이터 후보 모델 3종을 만든 뒤 팀원의 키오스크 저장소와 연결 방향을 검토한 시점까지를 한 문서로 정리한다. 나중에 발표 자료와 최종 보고서를 만들 때 결과 숫자뿐 아니라 **왜 그런 결정을 했는지**, **어떤 데이터와 코드가 근거인지**, **아직 무엇을 검증하지 못했는지**를 함께 설명하기 위한 기록이다.

이 문서의 사실은 다음 세 종류로 구분한다.

- **저장 결과로 확인됨**: 현재 저장소의 JSON, CSV, 모델 패키지 또는 실행 코드로 다시 확인한 내용이다.
- **당시 문서에 기록됨**: 당시 작성한 계획·분석 문서에서 확인했지만 원본 실행 로그가 완전하지 않은 내용이다.
- **향후 제안**: 아직 실행하거나 검증하지 않은 다음 단계다.

공개 데이터 평가 성능은 실제 USB 현미경이나 키오스크 환경의 성능을 보장하지 않는다. 세 후보는 모두 **공개 데이터 후보 모델**이며 의료 진단기기로 검증된 모델이 아니다.

---

## 2. 전체 과정 한눈에 보기

| 단계 | 실제로 한 일 | 판단 또는 결과 | 다음 단계로 넘어간 이유 |
|---|---|---|---|
| 1. 데이터 수집 | Kaggle·AI Hub 등을 조사해 피부와 두피 이미지를 수집 | 세 사용 환경에 맞는 후보 클래스를 구성 | 한 모델로 모든 사진을 처리하기보다 촬영 장비와 신체 부위에 따라 나눌 필요가 생김 |
| 2. 초기 전처리·학습 | Train/Validation/Test 분할, 오프라인 증강, EfficientNet-B0 전이학습 코드 작성 | `skin`은 매우 높은 저장 성능, `web_skin`과 `hair`는 상대적으로 낮은 성능 | 낮은 두 모델의 학습법 개선이 필요했고 Skin의 높은 수치도 데이터 중복 여부를 확인할 필요가 생김 |
| 3. YOLO 검토 | 얼굴 피부 병변을 먼저 검출한 뒤 잘라서 분류하는 방식을 검토 | 라벨링 작업이 실패·중단되었고, 넓게 퍼지는 증상과 얼굴 전체 분포를 작은 박스로 표현하기 어렵다고 판단 | 전체 이미지 분류 중심 구조로 정리 |
| 4. 서비스 이야기 재구성 | `web_skin`, `skin`, `hair`로 역할을 분리 | 웹캠의 일상적 상태, USB 현미경의 피부 병변, USB 현미경의 두피 상태라는 흐름 수립 | 데이터와 모델의 입력 조건을 서비스 화면과 일치시키기 위함 |
| 5. 세 데이터셋 증강·재학습 | 기존 방식으로 원본·증강 학습 | Skin은 좋았지만 Web Skin·Hair는 기대보다 낮음 | 단순히 이미지를 늘리는 것만으로는 미세한 클래스 차이를 충분히 학습하지 못함 |
| 6. 학습 방법 변경 | 누수 검사, Clean 데이터 재구성, 2단계 미세조정, 224/256, Loss, B0/B1 실험 | Hair와 Web Skin의 검증 성능이 개선되고 각 도메인 후보 모델을 패키징 | 동일 조건 비교와 재현 가능한 후보 선정 체계 확보 |
| 7. 공통화·통합 준비 | 데이터 감사·비교·6실험을 공통 노트북 3개로 통합하고 팀원 GitHub 분석 | 모델 파일을 전달할 준비는 됐지만 팀원 저장소의 피부·두피 추론은 아직 미연결 | 공통 추론 계약과 LLM 입력 형식부터 맞춰야 함 |

### 현재 후보 모델

| 도메인 | 촬영 방식과 역할 | 클래스 수 | 선정 모델 | 입력 | 저장 Test 결과 |
|---|---|---:|---|---:|---|
| `skin` | USB 현미경으로 확대 피부 병변 확인 | 10 | EfficientNet-B0, CE, 증강 학습 | 224 | Accuracy `0.9871428571428571`, Macro F1 `0.9871314132317626` |
| `web_skin` | 웹캠 얼굴 사진으로 일상적 피부 상태 확인 | 5 | EfficientNet-B0, CE, 2단계 미세조정 | 256 | Accuracy `0.8825`, Macro F1 `0.8813822927981046` |
| `hair` | USB 현미경으로 두피 상태 확인 | 5 | EfficientNet-B1, Label Smoothing 0.05, Adam | 384 | Accuracy `0.8003194888178914`, Macro F1 `0.8002222282821301` |

세 숫자의 평가 데이터와 클래스 난이도가 다르므로 모델끼리 우열을 직접 비교하면 안 된다.

---

## 3. 서비스 스토리와 세 모델의 역할

초기에는 피부 관련 이미지를 하나의 문제처럼 다뤘지만 실제 서비스에서는 촬영 장비와 사진의 범위가 크게 다르다. 이 차이를 기준으로 다음과 같이 분리했다.

### 3.1 `web_skin`: 웹캠으로 보는 일상적 얼굴 피부 상태

- 대상: 사용자가 키오스크 웹캠 앞에서 얼굴을 촬영하는 상황
- 클래스 순서: `건선`, `아토피`, `여드름`, `정상`, `주사`
- 정상 클래스: 있음
- 핵심 특징: 얼굴 전체의 붉음, 병변 분포, 위치와 넓은 문맥
- 서비스 의미: 눈에 띄는 일상적 피부 상태를 분류하고 안내 문구의 입력 자료를 제공

### 3.2 `skin`: USB 현미경으로 확대해서 보는 피부 병변

- 대상: 사용자가 이미 이상을 느껴 특정 부위를 확대 촬영하는 상황
- 클래스 순서: `광선각화증`, `기저세포암`, `보웬병`, `사마귀`, `지루각화증`, `편평세포암`, `표피낭종`, `피부섬유종`, `혈관종`, `흑색점`
- 정상 클래스: 없음
- 핵심 특징: 병변의 표면, 색, 경계, 각질과 확대 질감
- 서비스 의미: 정상 여부를 가리는 1차 선별기가 아니라, 이미 관찰 대상이 된 병변을 10개 학습 클래스 중 하나로 분류하는 후보 모델

### 3.3 `hair`: USB 현미경으로 보는 두피 상태

- 대상: 두피를 확대 촬영하는 상황
- 클래스 순서: `모낭사이홍반`, `미세각질`, `비듬`, `탈모`, `피지과다`
- 정상 클래스: 없음
- 핵심 특징: 모발 밀도, 모공 주변의 붉음, 각질 크기, 피지와 반사광
- 서비스 의미: 정상 여부 판단보다 관찰된 두피 상태를 다섯 범주로 분류

`skin`과 `hair`에 정상 클래스가 없는 것은 현재 서비스 이야기와 모순되지 않는다. 사용자가 이상을 느껴 현미경 촬영 단계로 들어온다는 전제가 있기 때문이다. 다만 모델 점수가 낮다는 이유만으로 `정상`이라고 바꾸면 안 된다. Softmax 최고 점수는 “정상일 확률”이 아니며, 학습하지 않은 정상·엉뚱한 사진도 기존 질환 클래스 중 하나로 출력할 수 있다. 시스템에서는 `판정 보류`, `입력 오류`, `모델 범위 밖`을 별도 상태로 설계해야 한다.

---

## 4. 데이터 수집과 클래스 설계

### 4.1 출처 기록 상태

Skin과 Web Skin의 실제 원천 데이터는 다음 AI Hub 합성 데이터셋으로 확인되었다. 초기에 Kaggle 등 여러 자료를 조사했지만, 최종 학습 데이터의 출처를 Kaggle로 적지 않는다.

| 도메인 | 실제 원천 데이터 | AI Hub 번호 | 사이트 원천 규모 | MediFlow 선택 |
|---|---|---:|---:|---:|
| Skin | [피부종양 이미지 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71864) | 71864 | 15종 × 각 1,000장 = 15,000장 | 10종, 원천 기준 10,000장; 초기 분할에는 9,000장 사용 |
| Web Skin | [안면부 피부질환 이미지 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71863) | 71863 | 6종·정면/측면 각 1,000장 = 12,000장 | 지루 제외, 정면 5종, 원천 기준 5,000장; 초기 분할에는 4,500장 사용 |

사이트의 `클래스당 1,000장`과 프로젝트의 `클래스당 900장`은 서로 다른 단계의 수량이다. 1,000장은 AI Hub가 제공한 원천 클래스 수량이고, 900장은 이 프로젝트의 Train·Validation·Test 분할에 실제로 넣은 수량이다. 각 클래스의 나머지 100장을 제외한 세부 기준과 다운로드 날짜는 기록에 남아 있지 않다.

확인된 AI Hub 관련 필드는 식별자, 진단명, 신체 부위, 사진 경로, 이미지 크기, 경계 상자와 병변 영역 등이다. Web Skin에서는 정면 얼굴 사진을 사용했고, 좌우 측면 사진이 같은 사람의 쌍인지 확인할 식별 정보가 충분하지 않아 함께 사용하지 않았다. 지루성 피부염은 주사 등과 시각적 구분이 불명확하다고 판단해 최종 다섯 클래스에서 제외했다.

### 4.2 초기 데이터 구성

| 도메인 | 초기 클래스 구성 | 초기 분할 또는 균형화 기록 |
|---|---|---|
| Skin | 71864의 15종 중 10개 확대 피부 병변 | 원천 클래스당 1,000장; 실제 분할 Train 730, Validation 100, Test 70 |
| Web Skin | 71863의 정면 건선·아토피·여드름·정상·주사 | 원천 클래스당 1,000장; 실제 분할은 클래스당 900장 |
| Hair | 처음 6개 후보에서 `모낭홍반농포` 제외 후 5개 | 클래스당 2,900장으로 맞추고 Train 2,320/Validation 290/Test 290 |

Hair에서 `모낭홍반농포`를 뺀 이유는 `모낭사이홍반`과 사진상 특징이 매우 비슷해 분류 경계가 불안정했기 때문이다. 중증도는 중등도와 중증 이미지를 사용하고, 최종 분류에서는 중증도보다 상태 종류를 하나의 클래스로 합쳤다.

---

## 5. 초기 전처리와 증강을 정확히 무엇을 했는가

증강은 Colab 학습 중 매 epoch마다 새로 만드는 온라인 증강이 아니라, 전처리 스크립트가 변형 이미지를 파일로 저장하는 **오프라인 증강**이었다. 따라서 `augmented/train`에는 원본 Train 이미지와 새로 만든 증강본이 함께 들어가고 Validation/Test에는 원본만 사용했다.

### 5.1 Skin 초기 증강

현재 [`scripts/preprocess_skin.py`](../../scripts/preprocess_skin.py)는 seed 42로 클래스당 원본 Train 730장에 새 증강 730장을 더해 `augmented/train`을 1,460장으로 만든다. 증강 이미지 한 장에는 다음 11개 중 정확히 하나를 무작위 적용한다.

| 변형 | 설정 |
|---|---|
| 좌우 반전 | 수평 반전 |
| 회전 | -10°~+10°, 반사 테두리 |
| 밝기 | 픽셀 밝기 -25~+25 |
| 대비 | 0.85~1.15배 |
| 화이트밸런스 | R/B 채널 0.95~1.05 범위 조정 |
| 감마 | 0.85~1.15 |
| 가우시안 블러 | 커널 3 또는 5 |
| 모션 블러 | 커널 3 또는 5 |
| 가우시안 노이즈 | 표준편차 5 |
| JPEG 열화 | 품질 60~85 |
| 원근 변형 | 최대 약 3% 이동 |

주의할 기록 차이가 있다. [`PROJECT_BACKGROUND.md`](PROJECT_BACKGROUND.md)의 초기 설명은 클래스당 원본 730장에 증강 470장을 더해 1,200장이라고 적었지만, 현재 실행 스크립트는 1,460장이다. 원래 실행 당시의 데이터 manifest가 남아 있지 않아 어떤 수량이 초기 저장 모델에 실제 사용됐는지 확정할 수 없다. 보고서에서는 이 차이를 숨기지 말고 “초기 실행 manifest 미보존”으로 기록한다.

### 5.2 Web Skin 초기 증강

[`scripts/preprocess_web_skin.py`](../../scripts/preprocess_web_skin.py)는 AI Hub의 클래스당 1,000장 중 프로젝트 입력으로 준비된 Train 800장과 Validation 100장을 사용한다. Train 800장 중 80장을 Test로 옮기고 720장을 Train으로 사용한다. 증강 Train은 원본 720장과 새 증강 720장, 총 1,440장이다. 각 새 이미지에는 서로 다른 변형 1~3개를 적용하며 개수 선택 확률은 1개 45%, 2개 40%, 3개 15%다.

| 변형 | 설정 |
|---|---|
| 좌우 반전 | 수평 반전 |
| 회전 | -7°~+7° |
| 밝기 | 0.75~1.25배 |
| 대비 | 0.80~1.20배 |
| 채도 | 0.85~1.15배 |
| 색온도 | R/B 채널 ±8 수준 조정 |
| 가우시안 블러 | 반경 0.2~0.7 |
| 노이즈 | 표준편차 1~3 |
| 감마 | 0.85~1.15 |
| JPEG 열화 | 품질 75~95 |
| 확대·잘라내기 | 1.02~1.08배 확대 후 잘라내기 |

이 설정은 조명, 웹캠 압축, 작은 자세 차이에 대한 적응을 의도했다. 그러나 사람 ID나 촬영 세션 ID가 없으면 같은 사람 또는 같은 병변이 분할을 넘어갔는지 완전히 확인할 수 없다.

### 5.3 Hair 초기 증강

[`scripts/preprocess_hair.py`](../../scripts/preprocess_hair.py)는 각 클래스 2,900장을 Train 2,320/Validation 290/Test 290으로 나눈다. 증강 Train은 원본 2,320장에 50%인 1,160장을 새로 만들어 클래스당 3,480장으로 구성한다. 새 이미지는 Train 원본을 복원추출해 선택한 뒤 다음 변형을 조합한다.

| 변형 | 적용 확률과 범위 |
|---|---|
| 회전 | 65%, -8°~+8° |
| 좌우 반전 | 25% |
| 밝기 | 75%, 0.82~1.18배 |
| 대비 | 65%, 0.85~1.15배 |
| 색상 | 35%, 0.90~1.10배 |
| 블러 | 25%, 반경 0.15~0.45 |
| 가우시안 노이즈 | 25%, 표준편차 1.5~4 |
| 임의 잘라내기·확대 | 30%, 원본의 92~100% 영역을 잘라 원래 크기로 복원 |

회전과 조명 변화는 현미경을 대는 각도와 조명 차이를 흉내 내려는 목적이었다. 변화 범위를 작게 둔 이유는 작은 각질과 모공 구조를 망가뜨리지 않기 위해서다. 데이터 증강은 작은 데이터에서 일반화를 돕는 널리 쓰이는 방법이지만, 현실에 없는 변형이나 지나친 변형은 오히려 분류 신호를 손상시킬 수 있다.[^1]

---

## 6. 초기 모델과 결과

초기 공통 구조는 ImageNet으로 사전학습한 EfficientNet-B0의 특징 추출부를 고정하고 새 분류층만 학습하는 방식이었다. EfficientNet은 깊이·너비·입력 해상도를 함께 조절해 정확도와 연산량의 균형을 맞추도록 설계된 모델군이다.[^2] 의료 영상처럼 라벨 데이터가 제한된 환경에서는 처음부터 전부 학습하는 것보다 일반 이미지 특징을 배운 모델을 가져와 미세조정하는 방식이 유효할 수 있다.[^3]

초기 저장 기록의 공통 조건은 대체로 입력 224, batch 32, 15 epoch였다. Skin의 과거 노트북은 Adam 학습률 `1e-3`, Web Skin과 Hair 기록은 Adam `1e-4`로 남아 있어 모든 도메인의 초기 조건이 완전히 같았다고 보면 안 된다.

| 도메인·데이터 | Validation Accuracy | Test Accuracy | Test Macro F1 | 해석 |
|---|---:|---:|---:|---|
| Skin Original | `0.9950000047683716` | `0.9942857027053833` | 기록 없음 | 매우 높았으나 나중에 분할 중복 발견 |
| Skin Augmented | `0.9950000047683716` | `0.9957143068313599` | 기록 없음 | 당시에는 좋은 결과로 판단했으나 최종 근거로 사용하기 어려움 |
| Web Skin Original | `0.6679999828338623` | `0.7850000262260437` | `0.7833125866587888` | 일반 얼굴 피부 클래스 간 혼동 존재 |
| Web Skin Augmented | `0.6880000233650208` | `0.8199999928474426` | `0.8195818185502844` | 증강이 개선했지만 Validation은 낮음 |
| Hair Original | `0.7062069177627563` | `0.6841379310344827` | `0.6827345904185209` | 미세한 두피 특징 구분 한계 |
| Hair Augmented | `0.704827606678009` | `0.6910344827586207` | `0.6894861696407794` | 개선 폭이 작음 |

여기서 “두 모델 성능이 떨어졌다”는 표현은 Web Skin과 Hair를 가리킨다. Web Skin은 Test가 약 0.82였지만 Validation이 약 0.688이었고, Hair는 Test가 약 0.691이었다. 증강만으로는 ImageNet 특징 추출부가 피부·두피의 세밀한 질감에 적응하지 못했다는 가설을 세웠다.

---

## 7. YOLO 라벨링 검토와 중단

YOLO는 이미지 안에서 병변 위치를 경계 상자로 먼저 찾고 그 부분을 분류하는 방안으로 검토했다. 프로젝트 소유자의 회고에는 “YOLO 라벨링 실패”로 남아 있지만, 현재 저장소에는 몇 장을 라벨링했고 어떤 오류로 실패했는지 보여 주는 원본 라벨·학습 로그가 없다. 따라서 기술 보고서에서는 **라벨링 작업이 실패·중단되었다는 사실**과 **설계상 전체 얼굴 분류가 더 적합하다고 판단한 이유**를 구분해야 한다.

중단 판단의 근거는 다음과 같다.

- 여드름처럼 작은 개별 병변은 상자로 표시할 수 있지만, 아토피·건선·주사는 붉음이나 질감이 넓게 퍼질 수 있다.
- 병변 하나만 자르면 얼굴에서의 분포와 위치 정보가 사라진다.
- 모든 학습 이미지에 정확한 상자 라벨을 새로 만드는 비용이 크고, 작업자별 경계 기준도 달라질 수 있다.
- 현재 서비스 목적은 병변 수를 세거나 위치를 표시하는 것보다 촬영 이미지 전체를 상태 범주로 분류하는 데 가까웠다.

따라서 Web Skin은 얼굴 전체 이미지 분류를 유지했다. 앞으로 위치 설명이 꼭 필요해지면 YOLO를 바로 재도입하기보다 Grad-CAM 같은 설명 지도를 사용해 모델이 어느 영역을 참고했는지 점검하거나,[^4] 얼굴을 여러 구역으로 나눠 정보를 합치는 다중 인스턴스 학습을 연구 후보로 둘 수 있다.[^5] 두 방법 모두 현재 후보에 적용해 성능을 검증한 것은 아니다.

---

## 8. 데이터 누수 발견과 Clean 데이터 재구성

### 8.1 왜 데이터 누수가 중요한가

같은 사진이나 같은 사람의 매우 비슷한 사진이 Train과 Validation/Test에 동시에 들어가면 모델이 새로운 사례를 분류한 것이 아니라 본 이미지를 기억해 높은 점수를 낼 수 있다. 의료 영상에서는 환자 단위 분리가 지켜지지 않을 때 성능이 과대평가될 수 있다는 연구가 보고돼 있다.[^6]

검사 기준은 파일 이름만 비교하는 것이 아니라 이미지 픽셀을 읽고 RGB 내용의 SHA-256 해시를 계산해 **완전히 같은 이미지**를 찾는 방식이었다. 파일 이름이 달라도 픽셀이 같으면 중복으로 판정한다.

이 검사는 완전 동일 사진은 찾지만 다음 항목까지 자동으로 보장하지는 않는다.

- 같은 사람을 다른 각도나 시간에 촬영한 사진
- 같은 병변을 조금 잘라내거나 압축한 근사 중복
- 같은 촬영 세션에서 나온 연속 프레임
- 증강 원본의 신원 관계가 manifest에 없는 과거 자료

### 8.2 Hair Clean v1

Hair 원본 분할에서 완전 동일 이미지가 Train–Validation 195개, Train–Test 178개, Validation–Test 16개 발견됐고 일부는 서로 다른 클래스 라벨 충돌도 있었다. 원본을 하나로 합친 뒤 중복과 라벨 충돌을 제거하고 seed 42로 다시 80:10:10에 가깝게 분할했다.

| 구성 | Train | Validation | Test | 합계 |
|---|---:|---:|---:|---:|
| Clean Original | 10,032 | 1,252 | 1,252 | 12,536 |
| Clean Augmented | 15,047 | 1,252 | 1,252 | 17,551 |

증강 Train은 원본 10,032장 전부과 중복 없이 고른 원본의 약 50%인 5,015장에 새 변형을 적용해 만들었다. 각 증강본에는 회전 -8°~+8°와 밝기 0.82~1.18을 반드시 적용하고, 좌우 반전 25%, 대비 65%, 색상 35%, 블러 25%, 노이즈 25%를 추가로 적용했다. `augmentation_manifest.csv`에는 증강본과 원본의 연결 관계를 저장했다.

초기 총 14,500장과 Clean 원본 12,536장의 차이는 1,964장이다. 이 숫자를 분할 간 중복 개수 195+178+16의 단순 합으로 설명하면 안 된다. 중복 그룹, 같은 분할 안의 중복, 라벨 충돌과 중복 관계가 서로 겹치기 때문이다.

### 8.3 Skin Clean v1

Skin 감사에서는 원본 Train–Validation 10개 그룹, Train–Test 5개 그룹, Validation–Test 1개 그룹의 완전 동일 RGB 중복이 확인됐다. 원본 Train 내부에는 32개 중복 그룹과 34장의 초과 복사본, Validation 내부에는 1개가 있었다. 과거 Augmented Train에서도 130개 그룹, 133장의 초과 복사본이 확인됐다.

원본 9,000장에서 51장의 중복·충돌 항목을 정리해 8,949장을 만들고 다시 분할했다.

| 구성 | Train | Validation | Test | 합계 |
|---|---:|---:|---:|---:|
| Clean Original | 7,249 | 1,000 | 700 | 8,949 |
| Clean Augmented | 14,498 | 1,000 | 700 | 16,198 |

Clean Augmented Train은 원본 7,249장 각각에 새 증강본 한 장을 만들었다. Skin 초기 증강의 11개 변형 중 하나를 적용하고, 최대 12번 다시 시도해 이미 생성된 픽셀 해시와 겹치지 않게 했다. 생성 검사에서는 분할 간 완전 동일 중복 0, 라벨 충돌 0, 고유 증강본 7,249장으로 기록됐다. 별도의 공통 감사 노트북 재실행은 프로젝트 소유자의 결정으로 생략했으므로 사람·병변·촬영 세션 수준의 누수가 없다고 주장할 수는 없다.

### 8.4 Web Skin 감사

저장된 감사 보고서에서는 읽을 수 없는 이미지, 완전 동일 파일/RGB 중복과 라벨 충돌이 발견되지 않았다. Original과 Augmented의 Validation/Test가 동일한 원본 세트라는 점도 확인됐다. 사용한 데이터 ZIP SHA-256은 `f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d`다. 다만 사람·병변·촬영 세션 식별자가 충분하지 않아 그 수준의 누수와 근사 중복은 확인하지 못했다.

---

## 9. Codex와 함께 바꾼 학습 방법

### 9.1 기존 저장 모델을 이어 학습하지 않고 ImageNet에서 다시 시작한 이유

Hair의 기존 모델은 나중에 새 Validation/Test가 된 이미지 일부를 과거 학습에서 봤을 가능성이 있었다. 이 모델을 그대로 이어 학습하면 Clean 분할을 만든 의미가 약해진다. 또한 실험마다 출발점이 달라져 224와 256, Loss와 Backbone 차이를 공정하게 비교하기 어렵다. 그래서 모든 실험은 같은 ImageNet 사전학습 가중치에서 새로 시작했다.

이 선택은 기존 모델이 쓸모없다는 뜻이 아니다. 일반 이미지의 선·색·질감 특징을 배운 ImageNet 가중치는 사용하되, 과거 MediFlow 학습 상태는 계승하지 않는다는 뜻이다.

### 9.2 2단계 미세조정

새 학습법은 두 단계로 나뉜다.

1. **Stage 1, head-only 15 epoch**: EfficientNet 특징 추출부를 고정하고 `GlobalAveragePooling → Dropout(0.3) → Dense(클래스 수, softmax)`만 Adam `1e-4`로 학습한다.
2. **Stage 2, partial fine-tuning 10 epoch**: Stage 1의 가장 좋은 Validation 체크포인트를 불러와 Backbone 마지막 30개 층을 풀고, BatchNormalization 층은 고정한 채 새 Adam `1e-5`로 학습한다.

전이학습 안내에서도 먼저 새 분류층을 안정화한 뒤 낮은 학습률로 사전학습 층 일부를 미세조정하고, BatchNormalization 상태를 조심해서 다루는 방식을 권한다.[^7] 학습률을 10분의 1로 낮춘 이유는 이미 배운 유용한 특징이 큰 업데이트로 무너지는 것을 줄이기 위해서다.

### 9.3 공통 입력과 재현 설정

- RGB `float32`, 픽셀 범위 0~255
- EfficientNet 모델 안의 `Rescaling(1/255)` 사용
- 모델 앞에서 다시 `/255.0` 하지 않음
- TensorFlow bilinear resize, `antialias=False`
- seed 42, batch 32, dropout 0.3
- Stage 1 Adam `1e-4`, Stage 2 Adam `1e-5`
- 각 실험의 가장 좋은 체크포인트는 Validation Accuracy로 선택
- Validation Accuracy가 정확히 같으면 더 이른 안정적 체크포인트를 유지
- Test는 Validation으로 설정과 모델을 고른 다음 최종 후보에 사용

현재 `.keras` 후보는 내부에 Rescaling 층이 있어 0~255 입력을 기대한다. 팀원 코드의 안구 PyTorch 모델처럼 외부에서 `/255`와 ImageNet 평균·표준편차 정규화를 한 뒤 넣으면 전처리가 중복되어 결과가 달라진다.

### 9.4 한 번에 비교한 6개 실험

| 순서 | 변경 조건 | 확인하려는 가설 |
|---:|---|---|
| 1 | B0, 224, Cross Entropy | Clean 데이터의 공통 기준선 |
| 2 | B0, 256, Cross Entropy | 더 큰 입력이 작은 질감 차이를 보존하는가 |
| 3 | B0, 256, Label Smoothing 0.05 | 지나친 확신과 유사 클래스 혼동을 완화하는가 |
| 4 | B0, 256, Focal Loss, alpha 1, gamma 1.5 | 어려운 샘플의 손실 비중을 높이면 개선되는가 |
| 5 | B1, 256, Label Smoothing 0.05 | 용량이 큰 Backbone이 세밀한 특징을 더 잘 배우는가 |
| 6 | B1 실험의 Stage 2를 5 epoch 연장 | 아직 개선 중인 미세조정을 조금 더 지속하면 좋아지는가 |

해상도 실험은 B0·CE·데이터를 고정해 224와 256만 비교했다. Loss 실험은 B0·256·데이터를 고정했다. Backbone 실험은 256·Label Smoothing·데이터를 고정했다. 단, Stage 1에서 Stage 2로 갈 때는 층 해제, 학습률, 학습 epoch가 함께 바뀌는 묶음 절차이므로 결과 차이를 “층 해제 하나의 순수 효과”라고 말할 수 없다.

Label Smoothing은 정답 클래스 목표를 정확히 1로 두는 대신 조금 완화해 지나친 확신을 줄이려는 방법으로 Inception 연구에서 사용됐다.[^8] Focal Loss는 쉬운 예시의 손실 기여를 줄이고 어려운 예시에 더 집중하도록 제안된 손실 함수다.[^9] 두 방법 모두 논문상 장점이 있다는 이유만으로 채택한 것이 아니라 실제 Validation 비교 대상으로 넣었다.

---

## 10. Hair 재학습 실험과 결론

Hair는 Clean Augmented Train 15,047장, Validation 1,252장, Test 1,252장을 사용했다.

| 실험 | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| B0 · 224 · CE | `0.7643769979476929` | `0.7763578274760383` | `0.7757408067550379` |
| B0 · 256 · CE | `0.7699680328369141` | `0.7891373801916933` | `0.7886477230068456` |
| B0 · 256 · Label Smoothing 0.05 | `0.7739616632461548` | `0.786741214057508` | `0.7863134458429594` |
| B0 · 256 · Focal Loss 1.5 | `0.7675718665122986` | `0.7907348242811502` | `0.7905122039602036` |
| B1 · 256 · Label Smoothing · Stage 2 총 10 | `0.7763578295707703` | `0.7787539936102237` | `0.778812327205678` |
| B1 · 256 · Label Smoothing · Stage 2 총 15 | `0.7771565318107605` | `0.7883386581469649` | `0.7885764577048346` |

후보는 Test 최고값이 아니라 Validation 최고값인 마지막 B1 확장 모델로 정했다. 클래스별 Test F1은 클래스 순서대로 다음과 같다.

| 클래스 | F1 |
|---|---:|
| 모낭사이홍반 | `0.8373015873015873` |
| 미세각질 | `0.744466800804829` |
| 비듬 | `0.7325102880658436` |
| 탈모 | `0.8463157894736842` |
| 피지과다 | `0.7822878228782287` |

### Hair에서 무엇이 실제로 좋아졌는가

- 초기 Augmented Test Accuracy `0.6910344827586207`에서 후보 `0.7883386581469649`로 저장 수치가 상승했다.
- 그러나 과거 학습 당시 데이터 ZIP 해시와 실행 환경 기록이 충분하지 않고 Clean 분할도 달라졌으므로, 이 차이를 완전히 같은 조건에서 학습법만 바꾼 순수 향상으로 표현하면 안 된다.
- 224→256에서 같은 B0·CE Validation은 `0.7643769979476929`→`0.7699680328369141`로 상승해 고해상도 가설을 일부 지지했다.
- Label Smoothing은 B0·256 Validation을 `0.7739616632461548`까지 올렸고 B1 실험의 기반으로 채택됐다.
- Focal Loss는 Test가 가장 높았지만 Validation은 B1보다 낮아 후보로 선택하지 않았다.
- B1 Stage 2 연장은 Validation을 `0.7763578295707703`에서 `0.7771565318107605`로 올렸다. 차이는 Validation 1,252장 중 정답 약 1장에 해당하므로 큰 개선이라고 과장하지 않는다.

Hair에서 미세각질과 비듬 F1이 상대적으로 낮은 것은 둘 다 각질성 패턴을 공유하기 때문일 가능성이 있다. 이는 혼동행렬과 오류 이미지에서 확인할 연구 가설이지, 현재 숫자만으로 원인을 확정한 것은 아니다.

또한 Hair 실험 과정에서는 여러 후보의 Test 결과를 이미 확인했다. 따라서 현재 Test는 완전히 손대지 않은 최종 시험 세트라기보다 탐색 과정에서 반복 관찰된 평가 세트다. 실제 장비 자료나 새 외부 세트가 필요하다.

---

## 11. Web Skin 재학습 실험과 결론

Web Skin은 감사에 통과한 Augmented Train 7,200장, Validation 500장, Test 400장을 사용했다.

| 실험 | Validation Accuracy | Test Macro F1 |
|---|---:|---:|
| B0 · 224 · CE | `0.794` | `0.7917009497301795` |
| B0 · 256 · CE | `0.796` | `0.7915394647566528` |
| B0 · 256 · Label Smoothing 0.05 | `0.794` | `0.7892883497449976` |
| B0 · 256 · Focal Loss 1.5 | `0.792` | `0.7898148816790076` |
| B1 · 256 · Label Smoothing | `0.796` | `0.7931535064807589` |
| B1 확장 후 유지된 부모 결과 | `0.796` | `0.7931535064807589` |

B0 256 CE와 B1 Label Smoothing의 최고 Validation Accuracy가 `0.796`으로 같았다. 선택 규칙에 따라 더 단순하고 먼저 안정적으로 도달한 B0 256 CE를 후보로 정했다. 후보의 최종 Test 결과는 Accuracy `0.8825`, Macro F1 `0.8813822927981046`이며 클래스별 F1은 다음과 같다.

| 클래스 | F1 |
|---|---:|
| 건선 | `0.8395061728395061` |
| 아토피 | `0.8456375838926175` |
| 여드름 | `0.8407643312101911` |
| 정상 | `0.9815950920245399` |
| 주사 | `0.8994082840236686` |

Stage 1 최고 Validation은 `0.704`였고 Stage 2에서 `0.796`까지 올라, Web Skin에서는 특징 추출부 일부를 실제 데이터에 맞게 조정한 효과가 뚜렷하게 관찰됐다. B1 추가 5 epoch의 자체 최고 Validation은 `0.790`으로 부모의 `0.796`을 넘지 못해 연장 체크포인트를 채택하지 않았다.

초기 Augmented Test Accuracy `0.8199999928474426`와 새 후보 `0.8825`를 비교하면 저장 수치는 상승했다. 다만 과거 모델을 동일 ZIP과 현재 코드로 재평가하지 않았고 과거 데이터 해시가 없으므로 완전한 동조건 재현 개선으로 단정하지 않는다.

---

## 12. Skin 정제·비교 학습과 결론

Skin은 Clean 데이터에서 B0, 224, CE, head-only 15 epoch 조건을 고정하고 Original과 Augmented만 비교했다. 이미 높은 Validation을 얻었으므로 Hair·Web Skin처럼 6개 실험을 반복하면 Test 반복 사용과 Colab 비용만 늘어날 수 있다고 판단했다.

| 학습 데이터 | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Clean Original | `0.973` | `0.9730455448862949` |
| Clean Augmented | `0.982` | `0.9820008885656606` |

Validation으로 Augmented를 선택한 뒤 Test 700장을 한 번 평가해 Accuracy `0.9871428571428571`, Macro F1 `0.9871314132317626`, 정답 691장을 기록했다. 초기 약 99.6%보다 숫자는 낮지만 중복을 제거한 더 엄격한 평가이므로 성능이 나빠졌다고만 해석하면 안 된다. 높은 초기 수치 일부가 중복의 영향을 받았을 가능성을 줄이고, 데이터 버전과 선택 절차를 남긴 결과라는 점이 더 중요하다.

Skin은 10개 병변 간 성능은 높지만 정상 클래스가 없고 실제 USB 현미경 사진으로 외부 검증하지 않았다. 공개 데이터의 배경, 조명과 촬영 배율이 실제 기기와 다르면 성능이 달라지는 **도메인 갭**이 남아 있다.

---

## 13. 실험 결과를 해석할 때 지켜야 할 기준

### 13.1 Accuracy와 Macro F1

- Accuracy는 전체 이미지 중 맞힌 비율이다.
- Macro F1은 각 클래스 F1을 같은 비중으로 평균해, 특정 클래스만 잘 맞히는 문제를 더 잘 드러낸다.
- 클래스 수, 데이터 난이도와 Test 구성이 다른 세 도메인의 Accuracy를 직접 비교해 “Skin 모델이 Hair보다 우수하다”고 말하면 안 된다.

### 13.2 Validation과 Test

- Validation은 모델과 설정을 선택하는 데 사용한다.
- Test는 선택이 끝난 뒤 최종 평가에 사용한다.
- Test를 보고 다시 설정을 고르면 Test도 사실상 Validation처럼 되어 최종 성능을 낙관적으로 만들 수 있다.
- Hair는 탐색 중 Test를 여러 번 확인했다는 제한을 명시해야 한다.

### 13.3 점수는 진단 확률이 아니다

현재 모델의 Softmax 점수는 보정된 임상 확률로 검증되지 않았다. 신경망은 정확도가 높아도 자신감이 실제 정답 가능성과 맞지 않을 수 있으며, 온도 스케일링 같은 별도 보정과 신뢰도 평가가 필요할 수 있다.[^10] 따라서 LLM에는 `confidence 0.90 = 질병일 확률 90%`로 전달하지 않는다.

### 13.4 숨은 하위집단과 외부 검증

전체 평균이 높아도 특정 피부색, 촬영기기, 연령대, 병변 크기처럼 기록되지 않은 하위집단에서 성능이 낮을 수 있다. 이를 숨은 층화 문제라고 한다.[^11] 피부 이미지 모델은 데이터 출처나 외부 환경이 바뀔 때 성능이 크게 달라질 수 있으므로 외부·실장비 검증이 필요하다.[^12]

---

## 14. 공통 코드와 노트북으로 정리한 내용

많아진 개별 노트북을 계속 복사하면 도메인마다 전처리와 평가 코드가 달라지고, 수정 사항을 빠뜨리기 쉽다. 그래서 검증된 기능을 공통 설정으로 모아 세 노트북으로 정리했다.

| 노트북 | 역할 | 학습 여부 |
|---|---|---|
| [`01_common_dataset_audit_colab.ipynb`](../../notebooks/01_common_dataset_audit_colab.ipynb) | ZIP 구조, 이미지 읽기, 분할, 클래스, exact duplicate, 라벨 충돌, 증강 원본 관계 감사 | 안 함 |
| [`02_common_original_vs_augmented_colab.ipynb`](../../notebooks/02_common_original_vs_augmented_colab.ipynb) | 동일한 B0·224·CE·head-only 조건으로 Original과 Augmented 비교 | 함 |
| [`03_common_six_experiments_colab.ipynb`](../../notebooks/03_common_six_experiments_colab.ipynb) | 2단계 미세조정, 224/256, Loss, B0/B1, 추가 5 epoch 실험 | 함 |

설정에서 `DOMAIN`을 `skin`, `web_skin`, `hair` 중 하나로 바꾸고 해당 ZIP 경로와 해시를 넣어 사용한다. `TRAIN_VARIANT='augmented'`는 6개 실험의 Train 종류를 뜻하고 Original 대 Augmented 비교 노트북은 두 종류를 자동으로 실행한다.

공통 엔진은 다음 결과를 남긴다.

- 실행 설정, 클래스 순서, 데이터 ZIP 해시, 코드·환경 정보와 seed
- epoch별 Train/Validation Accuracy와 Loss
- 각 실험의 최고 체크포인트와 완료 상태
- Validation/Test 예측 CSV, 분류 지표, 혼동행렬과 오류 이미지
- 전체 실험 비교 CSV·JSON과 발표용 그래프
- 후보 `.keras`, 메타데이터, 사용 안내와 SHA-256
- 중단 후 재실행할 때 완료된 실험을 signature로 확인하는 resume 정보

실험 도중 중단된 학습은 epoch 중간부터 그대로 잇지 않고 새 attempt로 해당 실험을 다시 시작한다. 이미 완료되어 설정·데이터 signature가 맞는 실험은 재사용할 수 있다.

과거 개별 노트북 17개는 [`notebooks/legacy_notebooks_20260909.zip`](../../notebooks/legacy_notebooks_20260909.zip)에 보존했다. 현재 실행 기준은 공통 3개와 [`COMMON_COLAB_NOTEBOOKS_GUIDE.md`](COMMON_COLAB_NOTEBOOKS_GUIDE.md)다.

---

## 15. 후보 패키지와 팀 전달 계약

현재 `results`에는 세 후보 모델과 원본 보고서를 도메인별로 보존한다. 팀원에게는 폴더 전체를 무작정 섞어 보내기보다 후보 모델, 클래스 순서, 입력 전처리, 모델 해시와 제한 사항이 함께 있는 패키지를 전달해야 한다.

| 도메인 | 후보 식별 | 팀원이 반드시 지킬 사항 |
|---|---|---|
| Skin | `public_candidate_v1` B0 224 CE Augmented | 10개 클래스 순서 고정, RGB 0~255, 정상 클래스 없음 |
| Web Skin | `public_candidate_v1_b0_256_ce` | 5개 클래스 순서 고정, 정상 클래스 있음, 얼굴 웹캠용 |
| Hair | `public_candidate_v2_b1_384_ls005_adam` | 5개 클래스 순서 고정, 정상 클래스 없음, 두피 현미경용 |

[`docs/guides/TEAM_MODEL_QUICKSTART.md`](../guides/TEAM_MODEL_QUICKSTART.md)와 각 `results` 사용 안내가 현재 전달 기준이다. 모델 파일 이름만 알려 주면 로딩은 가능할 수 있지만, 클래스 배열의 인덱스가 바뀌면 예측 질환명이 전부 틀어질 수 있으므로 메타데이터도 함께 전달해야 한다.

공통 결과 형식에는 최소한 다음 필드가 필요하다.

```json
{
  "domain": "web_skin",
  "device": "webcam",
  "body_region": "face",
  "model_id": "public_candidate_v1_b0_256_ce",
  "model_sha256": "...",
  "status": "ok",
  "class_names": ["건선", "아토피", "여드름", "정상", "주사"],
  "scores": [0.0, 0.0, 0.0, 0.0, 0.0],
  "predicted_class": "...",
  "has_normal_class": true,
  "score_is_calibrated_probability": false
}
```

`status`는 적어도 `ok`, `no_result`, `invalid_input`, `out_of_scope`, `model_error`처럼 구분해야 한다. LLM이 빈 결과를 정상으로 바꾸거나 낮은 점수를 정상으로 해석하지 못하게 하기 위해서다.

---

## 16. 팀원 GitHub와 연결 가능성

분석 대상은 팀원 저장소 [`Phjrab/mediflow-kiosk-core`](https://github.com/Phjrab/mediflow-kiosk-core)의 `main` 커밋 `2877e65f99bb1a3f9837b56c0736de57a4f66902`였다. 상세 근거는 [`KIOSK_CORE_REVIEW_20260913.md`](KIOSK_CORE_REVIEW_20260913.md)에 있다.

### 16.1 현재 연결되는 부분

- 팀 저장소에는 안구 분석, 웹 화면, 검사 세션, DB, PDF, 채팅 관련 구조가 있다.
- 검사 흐름은 `eye`, `skin`, `scalp` 세 ID를 이미 사용한다.
- 로컬 `web_skin`은 팀 저장소의 얼굴 피부 `skin` 화면에 대응한다.
- 로컬 `hair`는 팀 저장소의 `scalp`에 대응한다.
- 로컬 현미경 `skin` 10-class 모델은 팀 저장소에 별도 모드나 장비·신체부위 구분을 추가해야 한다.

### 16.2 아직 연결되지 않은 부분

- 팀 설정에서 안구만 `ready`이고 피부·두피는 `not_configured`, 클래스는 임시 `ex1`, `ex2`다.
- 실제 운영 경로는 PyTorch `.pth` 안구 모델을 사용한다. 우리 후보는 TensorFlow/Keras `.keras`다.
- 피부·두피 촬영 화면은 있지만 실제 추론 호출과 결과 표시가 확인되지 않았다.
- 브라우저가 이미지를 최대 변 1280, JPEG 품질 0.84로 재인코딩하므로 학습 파일과 입력 특성이 달라질 수 있다.
- 안구 코드는 외부 `/255`와 ImageNet 평균·표준편차 정규화를 사용한다. 이를 우리 모델에 그대로 복사하면 안 된다.

따라서 구조적으로는 연결할 수 있지만 모델 파일을 폴더에 복사하는 것만으로 끝나지 않는다. 팀원이 Keras 로더 또는 별도 추론 서비스를 만들고, 도메인별 전처리와 클래스 메타데이터를 읽어 공통 결과 JSON으로 바꾸어야 한다.

### 16.3 팀 저장소에서 먼저 고쳐야 할 의미 오류

- 가장 높은 confidence가 80 이상이면 무조건 `danger`로 만드는 로직은 정상 95도 위험으로 만들 수 있다.
- 데이터가 없을 때 `safe` 또는 `Normal`로 기본 처리하면 검사 실패가 정상으로 기록될 수 있다.
- 보고서 코드의 LLM 주석과 달리 실제 일부 경로는 고정 문자열 템플릿이다.
- 현재 채팅 UI는 안구 결과 키만 읽으며 피부·두피 결과를 읽지 않는다.
- 모델 점수와 의학적 위험도는 별도 필드여야 한다.

---

## 17. LLM과 VLM의 다음 설계

세 분류 모델이 있어야 LLM 연구를 시작할 수 있는 것은 아니다. 다만 LLM이 실제 검사 결과를 설명하려면 먼저 모델 출력 형식이 확정되어야 한다. 현재 우선순위는 이미지 VLM을 바로 학습하는 것이 아니라, 전문 분류기의 구조화 결과를 LLM이 안전하게 설명하도록 연결하는 것이다.

### 17.1 LLM 1단계

1. 세 모델의 공통 결과 JSON을 서버가 직접 생성하고 저장한다.
2. LLM은 서버가 검증한 결과, 문진 답변과 제한된 교육용 지식만 받는다.
3. 없는 검사 결과, 정상 클래스 없음, 낮은 점수와 오류 상태를 구분한다.
4. 질환명, 점수, 촬영 부위와 모델 버전을 임의로 바꾸지 못하게 한다.
5. “진단” 대신 관찰 결과 설명, 일반적 안내와 진료 권고 범위를 출력한다.
6. 환각, 숫자 보존, 클래스 일치, 금지 표현, 응답 지연과 실패 대체문을 테스트한다.

LLM의 결과는 새 정답 라벨이 아니며 분류기 학습 데이터로 자동 환류하지 않는다.

### 17.2 VLM 2단계

VLM은 이미지와 텍스트를 함께 이해해 설명을 만들 수 있지만 현재 전문 분류기와 역할을 섞으면 평가 기준이 흐려진다. 다음 조건이 갖춰진 뒤 별도 연구로 진행한다.

- 실제 장비 이미지와 전문가 또는 신뢰할 수 있는 설명 자료
- 이미지·텍스트 쌍의 출처와 사용 허가
- 분류 정확도와 설명 사실성을 따로 측정하는 평가셋
- 전문 분류기 결과와 VLM 설명이 충돌할 때의 처리 규칙
- 근거 없는 진단 문장과 보이지 않는 특징 생성을 잡는 평가

---

## 18. 아직 하지 않은 일과 다음 연구 방향

### 18.1 공통 최우선 과제

- 실제 키오스크 웹캠과 USB 현미경으로 촬영한 소규모 검증 세트 확보
- 사람·병변·촬영 세션 ID를 가진 새 분할 구성
- 실제 장비와 공개 데이터의 밝기, 색, 확대, 압축 차이 분석
- 잘못된 입력과 학습 범위 밖 이미지에 대한 거절 정책
- 모델 점수 보정과 신뢰도 평가
- 성별·연령·피부색·장비 조건별 하위집단 성능 점검

### 18.2 Hair 개선 후보

- 미세각질·비듬 중심의 오류 이미지 재검토와 라벨 기준 정리
- 한 장의 두피 사진을 여러 패치로 나눠 합치는 다중 인스턴스 학습
- 모발 밀도와 각질 질감을 서로 다른 해상도에서 보는 다중 스케일 모델
- metric learning 또는 supervised contrastive learning으로 유사 클래스 간 표현 거리 개선
- ConvNeXt, EfficientNetV2 등 Backbone 비교는 같은 Clean split과 Validation 규칙으로 한 번에 하나의 가설만 시험
- 반복 Test 사용을 멈추고 새 외부/장비 평가셋 확보

### 18.3 Web Skin 개선 후보

- 얼굴 정렬과 피부 영역 마스킹을 하되 눈·입·머리카락 같은 편향 신호 영향을 별도로 비교
- 조명·화이트밸런스 강건성 테스트와 카메라별 색 보정
- 정면과 측면이 같은 사람임을 확인할 수 있을 때 다중 시점 학습
- 아토피·건선·여드름 오류군을 사람 단위로 재검토
- 실제 웹 화면의 JPEG 재인코딩을 포함한 end-to-end 평가

### 18.4 Skin 개선 후보

- 정상 또는 기타/OOD 클래스를 서비스 범위에 맞춰 새로 수집할지 결정
- 실제 USB 현미경 배율·조명별 검증
- 병변 경계 중심 crop과 전체 확대 사진의 비교
- 높은 공개 데이터 점수가 출처·배경 단서에 기대지 않는지 Grad-CAM과 출처별 평가로 점검

이 기법들은 가능성 목록이다. 현재 후보보다 좋아졌다고 쓰면 안 된다. 먼저 오류 분석으로 구체적인 실패 원인을 찾고, 한 실험에서 하나의 가설만 검증한다.

---

## 19. 발표와 보고서 구성 예시

### 19.1 10장 발표 흐름

1. **문제 정의**: 한 장비·한 모델로 처리하기 어려운 피부·두피 촬영 환경
2. **서비스 구조**: Web Skin, 확대 Skin, Hair의 역할과 장비
3. **데이터 수집·클래스 설계**: Kaggle·AI Hub 조사, 클래스 선택과 제외 이유
4. **초기 전처리·증강**: 도메인별 변형과 원본/증강 구조
5. **첫 결과와 문제**: Skin 고성능, Web Skin·Hair 저성능, YOLO 중단
6. **데이터 누수 발견**: 이름이 아닌 픽셀 해시로 중복 확인, Clean 재분할
7. **학습법 변경**: ImageNet 재시작, 2단계 미세조정과 6개 비교 실험
8. **최종 결과**: 세 후보 성능 대시보드와 클래스별 F1
9. **시스템 연결**: 팀원 키오스크 구조, 공통 결과 JSON과 LLM 역할
10. **한계와 다음 단계**: 실제 장비 검증, OOD, 점수 보정과 VLM 연구

### 19.2 발표에서 사용할 정확한 표현

- “중복 제거 후 Clean 데이터로 다시 분할하고 재학습했다.”
- “모델 선택에는 Validation을 사용했고, 선정 후 Test를 평가했다.”
- “Hair의 최종 공개 데이터 후보 v2는 Test Accuracy `0.8003194888178914`를 기록했다.”
- “실제 장비 성능은 아직 검증하지 않았다.”
- “Skin과 Hair에는 정상 클래스가 없으며 낮은 점수를 정상으로 해석하지 않는다.”

피해야 할 표현은 “의료 진단 정확도 98.7%”, “실제 환자에게도 동일한 정확도”, “AI가 질환 확률을 계산한다”, “중복을 완전히 제거해 데이터 누수가 없다”이다.

---

## 20. 기록 간 차이와 확인이 필요한 항목

| 항목 | 현재 확인된 차이 또는 누락 | 보고서 처리 방법 |
|---|---|---|
| 정확한 원천 데이터 | Skin 71864, Web Skin 71863, Hair 216으로 확인 | 다운로드 날짜·라이선스·클래스당 제외 100장의 선별 기준은 추가 확인 필요 |
| Skin 초기 증강 수량 | 배경 문서 1,200장/클래스, 현재 스크립트 1,460장/클래스 | 둘 다 명시하고 원본 manifest 미보존으로 기록 |
| YOLO 실패 | 소유자 회고는 있으나 라벨·실행 로그 없음 | 실패 사실과 설계 전환 이유만 기술, 오류 세부사항 창작 금지 |
| 과거 모델 데이터 해시 | Web/Hair 초기 모델에 완전한 데이터 해시 없음 | 새 결과와 숫자 차이는 관찰값, 순수 동조건 향상으로 단정 금지 |
| 사람·세션 누수 | ID가 없어 exact RGB 외에는 검증 불가 | “완전 동일 이미지 중복 검사”로 범위를 제한 |
| 실제 장비 검증 | USB 현미경·환자·키오스크 자료 없음 | 공개 데이터 후보라고 표시 |
| Hair Test 반복 관찰 | 여러 실험에서 Test 확인 | 새로운 외부/장비 세트를 최종 평가로 확보 |

---

## 21. 파일별 근거 지도

### 현재 핵심 문서

- [`PROJECT_BACKGROUND.md`](PROJECT_BACKGROUND.md): 연구 목적, 데이터와 클래스 설계, 초기 흐름
- [`ROADMAP.md`](../ROADMAP.md): 완료 단계와 앞으로의 순서
- [`COMMON_COLAB_NOTEBOOKS_GUIDE.md`](COMMON_COLAB_NOTEBOOKS_GUIDE.md): 공통 노트북 실행법과 결과 구조
- [`HAIR_EXPERIMENT_SUMMARY_20260907.md`](HAIR_EXPERIMENT_SUMMARY_20260907.md): Hair Clean 재학습 전체 결과
- [`HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md`](HAIR_WEB_SKIN_EXPERIMENT_EXPLAINED_20260908.md): 256, Loss, B1과 미세조정 이유
- [`HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md`](HAIR_WEB_SKIN_IMPROVEMENT_PLAN_20260908.md): 후속 고급 방법 후보
- [`WEB_SKIN_FULL_SUMMARY_20260908.md`](WEB_SKIN_FULL_SUMMARY_20260908.md): Web Skin 감사·실험·후보
- [`SKIN_DATA_AUDIT_20260909.md`](SKIN_DATA_AUDIT_20260909.md): Skin 중복과 Clean 재구성
- [`SKIN_ORIGINAL_VS_AUGMENTED_20260909.md`](SKIN_ORIGINAL_VS_AUGMENTED_20260909.md): Skin 비교 실험
- [`SKIN_FULL_SUMMARY_20260909.md`](SKIN_FULL_SUMMARY_20260909.md): Skin 최종 판단과 제한
- [`KIOSK_CORE_REVIEW_20260913.md`](KIOSK_CORE_REVIEW_20260913.md): 팀원 GitHub와 LLM 연계 검토
- [`TEAM_MODEL_QUICKSTART.md`](../guides/TEAM_MODEL_QUICKSTART.md): 팀 모델 사용 계약

### 보관 문서

- `docs/archive/analysis/TRAINING_RESULTS_ANALYSIS.md`: 초기 세 모델 결과 비교
- `docs/archive/analysis/WEB_SKIN_EXPERIMENT_RESULTS_20260908.md`: Web Skin 6실험 세부값
- `docs/archive/planning/EVALUATION_READINESS.md`: 평가 준비 상태
- `docs/archive/planning/HAIR_ADVANCED_METHODS_BACKLOG_20260907.md`: Hair 고급 실험 후보
- `docs/archive/planning/WEB_SKIN_RETRAINING_PLAN_20260907.md`: Web Skin 재학습 가설
- `docs/archive/planning/WEB_SKIN_AUTO_SUITE_20260908.md`: 자동 실험 계획
- `docs/archive/planning/DRIVE_ORGANIZATION_PLAN_20260908.md`: 당시 Drive 정리 계획
- `docs/archive/process/WEB_SKIN_CANDIDATE_PACKAGING_20260908.md`: 후보 패키징 과정
- `docs/archive/verification/WEB_SKIN_CANDIDATE_VERIFIED_20260908.md`: 후보 패키지 검증
- `docs/archive/verification/WEB_SKIN_HAIR_PARITY_REVIEW_20260908.md`: 두 도메인 코드 동등성 검토
- `docs/archive/setup/TASK_PROMPT_TEMPLATE.md`: 과거 작업용 안내 양식

### 실행 코드

- `scripts/preprocess_skin.py`, `preprocess_web_skin.py`, `preprocess_hair.py`: 초기 도메인별 분할·증강
- `scripts/build_skin_clean_notebook.py`: Skin Clean builder 노트북 생성
- `scripts/build_common_notebooks.py`: 공통 3개 Colab 노트북 생성
- `src/mediflow_datasets/common_audit.py`: 공통 데이터 감사
- `src/mediflow_datasets/common_workflow.py`: 공통 실행 흐름
- `src/mediflow_datasets/common_engine.py`: 학습·평가 엔진
- `src/mediflow_datasets/experiment_suite.py`: 6개 실험과 resume
- `src/mediflow_datasets/skin_clean_builder.py`: Skin Clean 데이터 생성
- `src/mediflow_datasets/inference.py`: 현재 후보 추론 공통 처리
- 후보 패키징·재현 모듈: 결과 폴더의 모델·메타데이터·해시 검증에 사용

### 결과 자료

- `results/skin`: Skin 초기 결과, Clean 비교, 후보, 발표용 그림
- `results/web_skin`: 원본/증강 결과, 6실험, 후보, 발표용 그림
- `results/hair`: Clean 6실험, 후보, 전체 학습곡선과 성능 대시보드

원본 JSON/CSV가 최종 숫자의 기준이고 PNG 그래프는 발표용 표현이다. 그래프 숫자와 원본 파일이 다를 때는 원본 JSON/CSV를 우선한다.

---

## 22. 결론

이 프로젝트의 핵심 성과는 세 모델을 학습했다는 사실만이 아니다. 촬영 환경에 따라 문제를 세 도메인으로 분리하고, 초기 결과의 한계를 확인한 뒤, 픽셀 해시 중복 검사와 Clean 분할, 재현 가능한 2단계 미세조정, Validation 중심 후보 선정, 모델·메타데이터 패키징으로 연구 과정을 다시 세운 데 있다.

현재 완성된 것은 **공개 데이터 후보 3종과 재현·전달 자료**다. 아직 완성되지 않은 것은 **실제 장비 성능, 학습 범위 밖 입력 처리, 점수 보정, 팀원 키오스크의 실제 피부·두피 추론 연결, LLM/VLM 품질 평가**다. 따라서 다음 실질 작업은 새 학습을 반복하는 것보다 팀원에게 후보 계약을 전달하고 공통 결과 JSON을 연결한 뒤, 가능한 시점에 실제 장비 소규모 검증 데이터를 확보하는 것이다.

---

## 주석 및 연구 근거

[^1]: Connor Shorten and Taghi M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” *Journal of Big Data*, 2019. 증강 방법과 데이터·과제에 따라 효과가 달라지는 점을 폭넓게 정리했다.
[^2]: Mingxing Tan and Quoc V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *ICML*, 2019. 깊이·너비·해상도를 함께 조절하는 compound scaling을 제안했다.
[^3]: Nima Tajbakhsh et al., “Convolutional Neural Networks for Medical Image Analysis: Full Training or Fine Tuning?” *IEEE Transactions on Medical Imaging*, 2016. 여러 의료 영상 과제에서 사전학습 모델의 미세조정을 처음부터 학습하는 방식과 비교했다.
[^4]: Ramprasaath R. Selvaraju et al., “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization,” *ICCV*, 2017. 분류 결정에 영향을 준 공간 영역을 기울기로 시각화한다.
[^5]: Maximilian Ilse, Jakub Tomczak, and Max Welling, “Attention-based Deep Multiple Instance Learning,” *ICML*, 2018. 여러 인스턴스의 정보를 attention으로 합치는 학습 방식을 제안했다.
[^6]: Pouria Rouzrokh et al., “Mitigating Bias in Radiology Machine Learning: 1. Data Handling,” *Radiology: Artificial Intelligence*, 2022. 같은 환자의 여러 영상이 분할을 넘으면 평가가 과대평가될 수 있으므로 환자 수준 분리가 필요하다고 설명한다.
[^7]: Keras, “Transfer learning & fine-tuning.” 새 head 학습 후 낮은 학습률로 일부 또는 전체 층을 미세조정하는 구현 원칙과 BatchNormalization 주의점을 설명한다.
[^8]: Christian Szegedy et al., “Rethinking the Inception Architecture for Computer Vision,” *CVPR*, 2016. Label Smoothing을 정규화 방법으로 제시했다.
[^9]: Tsung-Yi Lin et al., “Focal Loss for Dense Object Detection,” *ICCV*, 2017. 쉬운 예시의 비중을 줄이고 어려운 예시에 집중하는 손실을 제안했다. 원래 검출 연구이므로 이 프로젝트의 분류 적용 효과는 별도 실험으로 확인했다.
[^10]: Chuan Guo et al., “On Calibration of Modern Neural Networks,” *ICML*, 2017. 현대 신경망의 confidence가 잘 보정되지 않을 수 있고 temperature scaling이 유효할 수 있음을 보였다.
[^11]: Luke Oakden-Rayner et al., “Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging,” *ACM CHIL*, 2020. 전체 라벨 성능이 숨은 임상 하위집단의 실패를 가릴 수 있음을 설명한다.
[^12]: “Validation of artificial intelligence prediction models for skin cancer diagnosis using dermoscopy images: the 2019 International Skin Imaging Collaboration Grand Challenge,” *The Lancet Digital Health*, 2022. 학습에 없던 질환, 다른 기관과 영상 인공물이 성능을 낮출 수 있음을 대규모 외부 평가로 보여 준다. 이 프로젝트에는 독립 외부/장비 평가가 아직 없다.

## Sources

1. [EfficientNet — ICML/PMLR](https://proceedings.mlr.press/v97/tan19a.html)
2. [Medical image transfer learning — PubMed](https://pubmed.ncbi.nlm.nih.gov/26978662/)
3. [Image data augmentation survey — Journal of Big Data](https://doi.org/10.1186/s40537-019-0197-0)
4. [Transfer learning and fine-tuning — Keras](https://keras.io/guides/transfer_learning/)
5. [Label Smoothing in Inception — CVPR](https://openaccess.thecvf.com/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)
6. [Focal Loss — ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
7. [Grad-CAM — ICCV](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)
8. [Attention-based Deep Multiple Instance Learning — PMLR](https://proceedings.mlr.press/v80/ilse18a.html)
9. [Data handling and leakage in medical imaging — Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/10.1148/ryai.210290)
10. [Calibration of neural networks — ICML/PMLR](https://proceedings.mlr.press/v70/guo17a.html)
11. [Hidden stratification in medical imaging — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7665161/)
12. [External validation and out-of-distribution skin lesion AI — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9295694/)
