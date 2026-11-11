# MediFlow 실제 촬영 기반 도메인 적응 구현과 실행 상태

> 2026-11-11 정리 상태: **분석·검토 또는 과정 기록, 후속 학습 상태는 최신 로드맵 참조**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

2026-11-08 · 로드맵 3단계 분석 및 4·6단계 비교 실험 준비

**실제 촬영 124장의 분포 분석과 현재 모델의 예측·특징 추출을 실행했다.** 전체 Source ZIP은 Drive에서 위치를 확인했지만 연결 도구의 파일 크기 한도로 가져오지 못했다. 따라서 이번에 측정한 Source 쪽 통계는 로컬 예시의 참고값이며, Source Train 전체의 분포나 새 학습 결과로 쓰지 않는다. 전체 ZIP의 해시와 분할을 확인한 뒤 분석·변환 보정·학습·최종 평가를 수행하는 코드를 함께 구현했다.

## 1. 현재 코드와 데이터 계약

| 분야 | 현재 기준 모델 | 클래스 · 입력 | 기존 학습 자료 |
|---|---|---|---|
| 웹캠 얼굴 `web_skin` | PMG·EfficientNet-B0 v2 | 건선, 아토피, 여드름, 정상, 주사 · 256×256 | 원본 Train 3,600장과 기존 증강 3,600장, Val 500장, Test 400장 |
| 현미경 피부 `skin` | EfficientNet-B0 v1 | 패키지의 10종 순서 · 224×224 · 정상 클래스 없음 | 정제 원본 Train 7,249장과 파생 증강 7,249장, Val 1,000장, Test 700장 |
| 현미경 두피 `hair` | EfficientNet-B1 v2 | 모낭사이홍반, 미세각질, 비듬, 탈모, 피지과다 · 384×384 · 정상 클래스 없음 | 정제 원본 Train 10,032장과 기존 증강 5,015장, Val 1,252장, Test 1,252장 |

이 표의 수량은 기존 보고 기록이다. 실제 새 실행에서 ZIP을 다시 열어 개수와 클래스 순서를 확인한다. Hair는 현재 `CANDIDATE_INDEX.json`의 **5클래스** 후보를 명시적으로 유지한다. 별도 Hair 6클래스는 데이터와 평가 과제가 달라 이번 실행의 직접 대조군에 섞지 않는다. 모델은 RGB `float32` 0–255를 받고 내부에서 재조정한다. 크기 변환은 TensorFlow bilinear이며, PMG의 네 logit 출력을 합한 뒤 softmax를 적용한다. 학습 때 PMG는 기존 코드의 8·4·2 jigsaw 단계와 fusion 업데이트 순서를 재사용한다. [현재 후보와 해시](../../../results/CANDIDATE_INDEX.json) · [입출력 계약](../../../results/MODEL_USAGE.md).

Source 파일은 Drive의 [datasets 폴더](https://drive.google.com/drive/folders/1BUbZbCLmQ_IeRo5WIsHZXo7hDFMxqrSC)에서 `web_skin_datasets`, `skin_datasets`, `hair_datasets`로 확인했다. 각각 약 13.9GB, 10.5GB, 3.0GB의 압축 파일이다. 5종 Hair ZIP을 받아보려 했지만 도구가 **268,435,456바이트 한도**를 넘는다는 오류(HTTP 413)를 반환했다. 서버 메타데이터만으로는 SHA-256을 대조하지 못했다. [학습 데이터·실험 설정](../../../configs/domain_adaptation.json)에 후보별 기대 해시와 확인한 Drive URL을 기록했다. Colab에서는 실제 파일을 탐색해 하나로 특정한 뒤 해시를 계산한다.

Target 원본은 프로젝트의 `camera_capture/` 아래 분야·참여자·세션별로 있다. 모든 기록의 질환 라벨은 `unverified`다. P02 피부의 마지막 5장은 별도 정정 기록을 적용한다. 네 사람 모두 과거 분석에 이미 사용했으므로 **전체를 calibration(촬영 특성 조사) 자료로 취급**한다. 사진을 무작위로 나눠 독립 Target Test를 만들지 않는다. 추후 독립 평가에는 신규 참여자와 촬영 세션의 자료가 필요하다. [촬영 품질 점검](CAMERA_PILOT_IMAGE_QA_20261108.md).

## 2. 이번에 실행한 분석

[분석 모듈](../../../src/mediflow_datasets/domain_adaptation_analysis.py)은 이미지별 평균 휘도와 밝기 분위수, 표준편차, 대비, RGB 분위수, Lab 색, 밝고 어두운 영역의 비율, 영역 안의 조명 차이, 좌우 조명, 얼굴 상자의 크기와 위치, 미세 경계량을 계산한다. 두피는 어두운 가는 경계가 차지하는 비율도 계산한다. 이것은 모발·그림자·질감이 섞인 **가림의 보조 지표**이며 검증된 모발 분할 면적은 아니다. 실제 배율, 센서 노이즈, 카메라 자체의 초점 성능은 이 사진들만으로 분리해 측정하지 못했다.

영역은 웹캠의 양쪽 볼과 현미경 중앙부다. 모델에 넣는 크기로 먼저 변환했다. 영역 위치와 육안 확인 이력은 앞선 비교 보고서에 있다. 이미지별 값으로 분포·분산·분위수를 계산하고, Target은 참여자와 조명 조건별로 표시했다. 반복 사진이 많다고 참여자 수가 늘지는 않는다. 예시 통계에는 얼굴 영역을 찾지 못한 한 장을 제외해 19장을 썼다. 특징 추출은 예시 20장과 Target 124장 전체에 실행했다.

| 분야 | 예시 영역의 밝기 중앙값 | Target 8개 참여자·조건 그룹의 밝기 중앙값 범위 | 질감과 색 해석 |
|---|---:|---:|---|
| 얼굴 | 약 180 | 약 82~139 | 실제 얼굴이 더 작고 어둡다. 같은 사람의 조명 조건 사이 대비와 좌우 명암 차이가 반복된다. 색 변화 방향은 일정하지 않다. |
| 피부 | 약 83 | 약 152~158 | 노란 기와 대비가 크게 다르지만, 병변과 팔 피부의 내용 차이가 크다. 이 차이 전체를 카메라 색 보정량으로 쓰기 어렵다. |
| 두피 | 약 155 | 약 162~188 | 어두운 모발 경계가 실제 사진에 더 많다. 일부 두피 표면은 흐리다. 미세 경계량 증가를 두피 초점 개선으로 해석할 수 없다. |

### 현재 모델의 특징 분포

PMG의 fusion 특징, Skin·Hair의 global average pooling 특징을 추출했다. 같은 분야의 예시에서 정한 중심·스케일과 RBF 대역폭으로 MMD²와 CORAL 공분산 거리를 계산했다. 이미지별 값과 별도로 네 참여자의 평균 특징으로 다시 계산해 반복 사진의 영향을 드러냈다. PCA는 같은 모델의 특징을 공통 좌표로 투영한 탐색용 그림이다. 분야마다 특징과 차원이 달라 아래 수치를 분야 간 성능 순위로 비교하지 않는다.

| 분야 | 예시 / 실제 특징 수 | MMD² · 이미지별 | CORAL · 이미지별 | MMD² · 참여자 평균 |
|---|---:|---:|---:|---:|
| 얼굴 | 5 / 42 | 0.221917986869812 | 0.10632432252168655 | 0.23554754257202148 |
| 피부 | 10 / 42 | 0.5486253499984741 | 1.1546434164047241 | 0.74176025390625 |
| 두피 | 5 / 40 | 0.6891716718673706 | 2.8558993339538574 | 1.1091959476470947 |

이 값에는 질환·인물·촬영 구도 차이가 모두 섞여 있다. Source 예시 수가 작고 분할 소속도 불명확해 통계적 일반화를 주장하지 않는다. 실제 사진의 기존 모델 예측도 모두 기록했으며, Target Accuracy와 Macro-F1은 정답 부재로 `null`이다. 예측 점수 상승이나 특징 거리 감소를 분류 개선으로 판정하지 않는다.

분야별 로컬 결과 폴더는 다음과 같다. 각 폴더에는 `analysis.json`, 이미지별 CSV, 분위수 그림, PCA, MMD·CORAL, Target 예측과 특징 배열, 모델 및 코드 버전 기록이 있다. 촬영 자료와 같은 Git 제외 폴더다.

- [얼굴 분석 결과](../../../camera_capture/_review/analysis/initial_model/web_skin/analysis.json) · [분포 그림](../../../camera_capture/_review/analysis/initial_model/web_skin/metric_distributions.svg) · [PCA](../../../camera_capture/_review/analysis/initial_model/web_skin/feature_pca.svg)
- [피부 분석 결과](../../../camera_capture/_review/analysis/initial_model/skin/analysis.json) · [분포 그림](../../../camera_capture/_review/analysis/initial_model/skin/metric_distributions.svg) · [PCA](../../../camera_capture/_review/analysis/initial_model/skin/feature_pca.svg)
- [두피 분석 결과](../../../camera_capture/_review/analysis/initial_model/hair/analysis.json) · [분포 그림](../../../camera_capture/_review/analysis/initial_model/hair/metric_distributions.svg) · [PCA](../../../camera_capture/_review/analysis/initial_model/hair/feature_pca.svg)

## 3. 선택한 방법과 보류한 방법

**현재 실행할 후보는 Target 관찰에 근거한 복합 증강과 Source 라벨을 사용한 추가 학습**이다. 이 방법의 이름은 Target-guided Augmentation으로 기록한다. 실제 Target의 정답이나 잠재 정답을 만들어 학습하지 않는다. 기존 임의 강도의 `camera_augmentation.py`는 과거 미리보기로 남기고, 새 실험은 [측정값에서 보정한 변환](../../../src/mediflow_datasets/domain_adaptation_experiment.py)을 사용한다.

| 방법 | 현재 적용 판단 |
|---|---|
| Target-guided Augmentation | 구현. 전체 Source Train을 읽은 뒤 밝기 수준, 반복되는 대비·좌우 조명 변화, 확인 가능한 얼굴 크기 차이로 범위를 계산한다. 참여자 수가 적어 측정 차이의 절반을 쓰고 상한을 둔 초안을 만든다. Train 표본의 특징 보존을 확인한 뒤 고정한다. |
| MMD·CORAL 특징 정렬 학습 | 이번 데이터로는 보류. 지표 계산은 구현했다. 라벨 없는 Target을 쓸 수 있는 방법이지만, 현재는 4명이고 클래스 구성도 미확인이다. Skin·Hair-5에는 정상 클래스가 없어 건강 여부 미확인 참가자와 질환 Source 전체를 맞추면 분류 단서가 사라질 수 있다. |
| 라벨 있는 Target 미세조정 | 검수한 질환 라벨과 독립적인 사람·촬영 세션이 확보되면 별도 실험으로 추가한다. 현재의 일반 참여자 사진에 임의 라벨을 붙이지 않는다. |
| 기타 UDA·의사 라벨 학습 | 작은 Target과 클래스 불일치 위험을 고려해 이번 학습에서는 보류한다. 새 라벨·클래스 범위·독립 평가가 마련되면 다시 검토한다. |

MMD 기반 학습은 특징의 평균 임베딩을 맞추고, CORAL은 특징 공분산을 맞추는 방법이다. 두 방법은 기법 존재와 이번 데이터에 대한 타당성을 구분해 판단했다. [Deep Adaptation Networks](https://proceedings.mlr.press/v37/long15.html) · [Deep CORAL](https://arxiv.org/abs/1607.01719).

### 변환 범위를 계산하는 규칙

- 웹캠 밝기는 Source Train의 볼 밝기 중앙값과 네 참여자·두 조건의 볼 밝기 차이를 비교한다. 차이 분포의 25~75% 구간에 0.5를 곱하고, 절대 픽셀 이동을 최대 32로 제한한다. 이것은 과적합을 줄이기 위해 정한 설계 상한이며 측정된 장비 상수가 아니다.
- 웹캠의 얼굴 너비·높이 비율도 Source Train에서 검출률이 70% 이상일 때만 비율 차이를 완화해 적용한다. 다른 경우 크기 변환을 넣지 않는다.
- 대비와 좌우 조명은 같은 사람의 기준·변경에서 네 사람 모두 변화 방향이 같을 때만 넣는다. 대비는 0.90~1.15, 좌우 픽셀 차이는 절대 24 이내로 제한한다.
- 피부·두피의 Source와 Target은 관찰 내용이 달라, 전체 픽셀·색 통계를 맞추지 않는다. 같은 사람의 조건 변화에서 근거를 얻고, 색·흐림·노이즈·모발 면적처럼 질환 특징과 구분되지 않는 항목은 보류한다. 두피의 전체 대비도 모발 가림이 섞여 보정하지 않는다.

이 규칙은 밝기·대비·색·질감·구도를 모두 **분석**하되, 근거가 있는 항목만 **적용**하기 위한 것이다. Source 전체 분포를 읽지 못한 현재 상태에서는 `measured_profile.json`을 확정하지 않는다.

## 4. 학습·선정·평가 절차

| 조건 | 동일하게 유지하는 부분 | 바꾸는 부분 |
|---|---|---|
| Baseline | 현재 선정 모델과 입력 계약 | 저장본 그대로 평가 |
| Source-only | 같은 모델, 증강 포함 기존 Train, 분할, 난수·학습 설정 | 추가 학습만 수행 |
| Brightness-only | Source-only와 동일 | 같은 측정 프로필의 밝기만 입력 일부에 적용 |
| Target-guided Augmentation | Source-only와 동일 | 측정 근거를 통과한 변환 묶음을 입력 일부에 함께 적용 |
| Feature alignment DA | 동일 Source와 독립 Target 분리 필요 | 현재 조건에서는 미실행 상태로 명시 |

초기 설정은 seed 42, 추가 학습 3 epoch, batch 16, Adam 학습률 `1e-5`, 변환 적용 비율 0.5다. backbone 후반 30개 층을 조정하고 BatchNorm은 고정한다. 모든 추가 학습군에 같은 설정을 적용한다. PMG는 기존 jigsaw 8·4·2와 fusion 순서를 유지한다. 이 설정은 구현을 위한 초기 연구 조건이며 최적값으로 검증되지 않았다.

Epoch 선택은 원본·변환 Validation Macro-F1 중 더 낮은 값이 가장 큰 저장본으로 한다. 기존 후보보다 원본 Macro-F1 하락 0.01 이내, 클래스별 Recall 하락 0.05 이내이고, 변환 Validation Macro-F1 개선이 0.005 이상일 때 채택 후보로 올린다. 임계값은 첫 실행의 명시적 설계 선택이다. 결과를 보고 임계값을 바꾸면 새 실험으로 기록해야 한다.

Train과 Validation만으로 모든 군과 최종 후보를 고정한다. `finalize` 명령은 고정 파일·모델·변환 해시를 확인하고 기존 Source Test 및 같은 사진의 변환 복사본에서 각각 Accuracy, Macro-F1, 클래스별 Recall과 혼동행렬을 계산한다. Test 파일의 복사본은 독립 표본으로 합산하지 않는다. 기존 Test가 과거 선정 과정에서 평가된 자료라는 이력도 남긴다.

Source ZIP에서는 Original·Augmented의 Val/Test 일치와 Train 원본 포함 여부를 검사한다. 파일·픽셀 완전 중복, 라벨 충돌, 원본·사람·병변·세션의 교차 분할을 확인한다. dHash 해밍 거리 3 이하의 교차 분할 후보는 사람이 확인할 목록으로 남기고 미해결이면 학습을 막는다. 새 파생본은 런타임에서 원본의 분할을 유지한다. 원천 ID가 없을 때는 사람·병변·세션 독립성을 입증하지 못했다는 한계를 명시적으로 수락해야 한다. dHash는 모든 유사 사진과 재촬영 누수를 찾는 방법은 아니다.

새 Target Test는 별도 JSON에 파일, 분야, 참여자, 세션, 원본 ID를 기록한다. 라벨을 주려면 클래스명과 `label_evidence`가 필요하다. 기존 4명의 calibration 사진이나 같은 참여자를 Test로 넣으면 오류가 난다. 정답 없는 평가 자료의 정확도는 `null`로 유지한다.

## 5. 실행 방법과 준비물

재사용 모듈은 다음과 같다.

- [분석·학습·평가 CLI](../../../src/mediflow_datasets/domain_adaptation.py)
- [ZIP·Target 역할·누수 검사](../../../src/mediflow_datasets/domain_adaptation_data.py)
- [이미지 측정·분포 그림·변환 보정](../../../src/mediflow_datasets/domain_adaptation_analysis.py)
- [모델 특징·MMD/CORAL·세 학습군](../../../src/mediflow_datasets/domain_adaptation_experiment.py)
- [설정 파일](../../../configs/domain_adaptation.json) · [Colab 생성기](../../../scripts/build_domain_adaptation_colab.py)

PC에서는 다음처럼 예시 참고 분석을 다시 실행할 수 있다. 출력 폴더는 항상 새 경로를 지정한다.

```powershell
.venv\Scripts\python.exe -B -m mediflow_datasets.domain_adaptation analyze --domain web_skin --output camera_capture/_review/new_analysis/web_skin
```

전체 Source가 있는 환경에서는 실제 확인한 ZIP 경로를 `--source-zip`에 추가한다. `analyze`는 전체 ZIP 해시와 분할을 검증하고 Source Train 전체 측정값과 변환 초안을 만든다. `preview`로 클래스별 Train의 변환 전후를 확인한다. 이 검토 결과를 프로필의 `train_visual_review=approved`, `review_note`에 기록한 뒤 `train`을 실행한다. Source의 인물·세션 정보가 없으면 `--acknowledge-incomplete-identities`를 명시해야 한다. 독립 누수를 확인했다는 의미로 사용하지 않는다.

[Colab 노트북](../../../output/domain_adaptation_20261108_final/Domain_Adaptation_Colab.ipynb)과 [실행 ZIP](../../../output/domain_adaptation_20261108_final/mediflow_da_runtime.zip)을 만들었다. 실행 ZIP은 현재 코드, 세 선정 모델과 메타데이터, Target 측정값과 기존 모델 특징을 포함한다. 참여자 얼굴·피부·두피 원본은 포함하지 않는다. 노트북을 Colab에 열고 실행 ZIP을 선택한 뒤 Drive를 연결한다. 실제 ZIP을 찾아 한 파일로 특정하고 해시를 검사한다. 대용량 Source를 로컬 PC로 내려받지 않고 Drive에서 직접 읽는다.

Colab에서 학습 결과 ZIP을 받은 뒤 새 폴더에 보존하고, PC에서 `compare-target --domain ... --run 실제_결과_폴더 --output 새_출력_폴더`를 실행하면 같은 실제 촬영 사진의 모델별 예측과 특징 변화를 계산한다. Colab의 최종 Source 평가 배열이 함께 있으면 같은 기준 좌표의 MMD·CORAL도 비교한다. 학습·Test 평가가 끝나기 전에는 해당 수치가 결과에 나타나지 않는다.

## 6. 기존 성능과 이번 실행 결과의 구분

| 분야 | 기존 보고 Source Test Accuracy | 기존 보고 Source Test Macro-F1 | 이번 전체 Source Test 재평가 | 새 학습군 성능 |
|---|---:|---:|---|---|
| Skin | 0.9871428571428571 | 0.9871314132317626 | 실행 전 | 실행 전 |
| Web Skin | 0.915 | 0.9141049081029712 | 실행 전 | 실행 전 |
| Hair 5클래스 | 0.8003194888178914 | 0.8002222282821301 | 실행 전 | 실행 전 |

기존 수치는 현재 후보 색인의 보존값이다. 이번에는 사진 분포와 기존 모델의 예측·특징을 실제로 계산했다. 작은 합성 데이터·임의 가중치 모델로 학습부터 최종 평가까지의 코드 경로를 별도로 검사했다. 이것을 피부·두피 연구 학습의 성능 측정으로 보고하지 않는다.

검증 결과는 `ruff check src tests` 통과, 전체 자동 테스트 **133개 통과·3개 건너뜀·실패 0개**다. 3개는 기존 공통 노트북 테스트에서 선택 라이브러리 `matplotlib`이 없어 건너뛴 항목이다. 새 테스트는 체크포인트·데이터 해시, 클래스 순서, 파생본·인물 누수, Test를 제외한 보정·학습, 정답 없는 정확도, 특징 거리, Validation 고정 후 Test 평가와 재실행 보호를 확인했다. 최종 실행 ZIP을 새 임시 폴더에 풀어 코드·모델 해시, 세 분야의 측정 자료 로딩과 노트북 문법을 검증했다. Colab 런타임에서 직접 실행한 결과는 아직 없다.

남은 실행 조건은 전체 Source ZIP을 읽을 수 있는 Colab 런타임, 각 분할의 추가 검증과 Train 변환 검토다. 환자 라벨이 있는 Target Test가 확보되기 전까지는 실제 장비의 질환 정확도 개선을 확정할 수 없다. LLM·VLM은 카메라 적응 이후에 진행한다.
