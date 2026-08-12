# mediflow-datasets

피부질환 데이터 전처리, 증강, 학습, 결과 분석을 정리한 저장소입니다.

## 현재까지 진행 사항

### 1. 피부질환 데이터 확보

- 10개 클래스 데이터셋 확보
- 클래스별 원본 이미지 수집 완료
- 최종 데이터 구성
  - Train: 클래스당 730장
  - Validation: 클래스당 100장
  - Test: 클래스당 70장

- 데이터 출처: AI Hub 피부질환 데이터셋
- 참고 링크: https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%94%BC%EB%B6%80&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71864

### 2. 피부 데이터 전처리

- 질환별 폴더 구조로 데이터 정리
- Train / Validation / Test 분리
- 학습용 `processed` 폴더 구성

```text
processed/
├── original/
│   ├── train/
│   ├── val/
│   └── test/
└── augmented/
    ├── train/
    ├── val/
    └── test/
```

### 3. 피부 데이터 증강

- 원본 데이터와 증강 데이터를 별도로 구성
- 한 이미지에 여러 증강을 중복 적용하지 않고, 랜덤하게 하나의 증강 기법을 적용
- 실제 촬영 환경 변화를 고려한 증강 수행
- Validation과 Test는 원본 데이터만 사용

### 4. 데이터 저장 및 관리

- 전처리 및 증강 완료 데이터는 `processed` 폴더로 통합
- `processed` 폴더를 `processed.zip`으로 압축
- Google Drive의 `skin_dataset` 폴더에 저장
- Colab 학습 시 ZIP을 로컬로 복사 후 압축 해제하여 사용

### 5. 피부질환 분류 모델 학습

- 사용 모델: EfficientNet-B0
- ImageNet Pretrained Model 사용
- Input Size: `224 x 224`
- Batch Size: `32`
- Epoch: `15`
- Early Stopping: 사용하지 않음
- 학습 데이터셋
  - Original 데이터 기반 EfficientNet-B0
  - Augmented 데이터 기반 EfficientNet-B0

### 6. 학습 결과 분석

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Test Accuracy
- Confusion Matrix
- Classification Report
- Accuracy / Loss 그래프 자동 저장

### 7. Original / Augmented 모델 비교

- Original 모델과 Augmented 모델의 Test 성능을 비교
- Augmented 데이터를 사용한 EfficientNet-B0의 성능이 더 높게 나타남
- 현재 피부질환 분류 단계에서는 Augmented EfficientNet-B0를 최종 분류 모델로 사용

## 저장소 구성

- `preprocess.py`: 피부 데이터 전처리 및 증강 코드
- `skin_disease_efficientnet_15epoch_drive.ipynb`: EfficientNet 학습 코드
- `upload_staging/`: GitHub 업로드용 임시 정리 폴더

## 참고

이 저장소는 피부질환 분류 프로젝트의 데이터 준비, 학습, 결과 정리를 중심으로 구성됩니다.
