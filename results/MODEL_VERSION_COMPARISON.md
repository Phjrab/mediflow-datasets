# MediFlow 모델 버전·학습 방법 종합표

기준일: 2026-09-23. 이 문서는 실제로 선정·패키징된 Skin, Web Skin, Hair 모델의 버전,
학습 방법, 성능과 교체 이유를 한 번에 확인하는 기준표다. 모든 수치는 저장된 후보 결과 파일의
값을 그대로 사용했다.

## 1. 현재 사용할 모델

| 도메인 | 현재 버전 | 모델 | 입력 | 클래스 | 모델 경로 |
|---|---|---|---:|---:|---|
| Skin | v1 | EfficientNet-B0 | 224 | 10 | `skin/selected_models/v1/skin_model.keras` |
| Web Skin | v2 | PMG·EfficientNet-B0 | 256 | 5 | `web_skin/selected_models/v2/web_skin_model.keras` |
| Hair | v2 | EfficientNet-B1 | 384 | 5 | `hair/selected_models/v2/hair_model.keras` |

## 2. 전체 버전 비교

| 도메인·버전 | 당시 위치 | Backbone·방법 | Loss | 학습 단계 | Validation Accuracy | Test Accuracy | Test Macro F1 | 현재 상태 |
|---|---|---|---|---|---:|---:|---:|---|
| Skin v1 | 최초·현재 선정 | EfficientNet-B0·Augmented Train | CE | Head-only 15 | 0.982 | 0.9871428571 | 0.9871314132 | 현재 |
| Web Skin v1 | 논문 강화 전 최종 선정 | EfficientNet-B0 | CE | Head-only 15 + Partial FT 10 | 0.796 | 0.8825 | 0.8813822928 | v2로 교체 |
| Web Skin v2 | 논문 강화 후 선정 | PMG·EfficientNet-B0 | CE | Head-only 15 + Partial FT 10 | 0.85 | 0.915 | 0.9141049081 | 현재 |
| Hair v1 | 논문 강화 전 최종 선정 | EfficientNet-B1·256 | CE + Label Smoothing 0.05 | Head-only 15 + Partial FT 총 15 | 0.7771565318 | 0.7883386581 | 0.7885764577 | v2로 교체 |
| Hair v2 | 논문 강화 후 선정 | EfficientNet-B1·384·Adam | CE + Label Smoothing 0.05 | Head-only 15 + Partial FT 15 | 0.7963258786 | 0.8003194888 | 0.8002222283 | 현재 |

`CE`는 Categorical Cross Entropy, `Partial FT`는 Backbone 전체가 아니라 마지막 일부 층만
낮은 학습률로 다시 학습하는 부분 미세조정이다. 현재 EfficientNet 모델은 ImageNet 사전학습
가중치에서 시작했다.

## 3. Skin v1

| 항목 | 내용 |
|---|---|
| 후보 ID | `public_candidate_v1_b0_224_ce_augmented_20260909_075056` |
| 데이터 | 분할 간 완전 동일 중복을 제거한 Clean 데이터 |
| 비교 가설 | Train에 저장 증강본을 포함하면 Original Train보다 Validation 성능이 좋아지는가 |
| 고정 조건 | EfficientNet-B0, 224, CE, Seed 42, Head-only 15 Epoch |
| 비교 결과 | Original Val Accuracy 0.973 → Augmented 0.982 |
| 선정 이유 | 같은 Validation에서 Augmented가 더 높아 승자로 고정 |
| 최종 결과 | Test Accuracy 0.9871428571, Macro F1 0.9871314132 |

Skin의 Original과 Augmented는 v1·v2가 아니다. 하나의 v1 후보를 선택하기 위한 비교 조건이다.
승자로 고정된 Augmented 모델만 Test 평가와 패키징을 진행했다.

## 4. Web Skin v1 → v2

| 항목 | v1 | v2 |
|---|---|---|
| 선정 시점 | 논문 기반 강화 실험 전 | 논문 기반 강화 실험 후 |
| 구조 | EfficientNet-B0 분류기 | PMG·EfficientNet-B0 |
| 입력 | 256×256 | 256×256 |
| Loss | CE | CE |
| 1단계 | Head-only 15 Epoch | Head-only 15 Epoch |
| 2단계 | 마지막 일부 층 Partial FT 10 Epoch | 마지막 30개 비-BN 층 Partial FT 10 Epoch |
| Validation Accuracy | 0.796 | 0.85 |
| Validation Macro F1 | 0.7915394648 | 0.8473279632 |
| Test Accuracy | 0.8825 | 0.915 |
| Test Macro F1 | 0.8813822928 | 0.9141049081 |

### v1 선정 이유

논문 강화 전 여섯 학습 설정을 같은 Validation에서 비교했다. B0·256·CE가 최고 Validation
Accuracy를 기록했고, 동률에서는 더 단순하고 먼저 검증된 설정을 유지한다는 규칙에 따라 당시
최종 후보로 선정했다. 이후 고정 Test 평가와 패키징까지 완료했다.

### v2 선정 이유

PMG는 얼굴 전체의 전역 특징과 작은 피부 병변의 국소 특징을 여러 세밀도에서 함께 학습한다.
v1과 같은 256 입력을 유지하면서 Validation과 Test가 모두 개선됐다. PMG·B1·384는 Validation
성능이 더 높았지만 모델 크기와 계산량이 증가해, 성능과 배포 비용의 균형을 고려하여 B0·256을
v2로 선정했다.

### 변화량

| 지표 | v1 → v2 변화 |
|---|---:|
| Validation Accuracy | +0.054 |
| Test Accuracy | +0.0325 |
| Test Macro F1 | +0.0327226153 |

Web Skin v2 모델은 네 개의 PMG logit 출력을 반환한다. 네 출력을 합산한 뒤 softmax를 한 번
적용해야 하며, 첫 출력만 사용하면 저장된 성능을 재현할 수 없다.

## 5. Hair v1 → v2

| 항목 | v1 | v2 |
|---|---|---|
| 선정 시점 | 논문 기반 강화 실험 전 | 논문 기반 강화 실험 후 |
| 구조 | EfficientNet-B1 | EfficientNet-B1 |
| 입력 | 256×256 | 384×384 |
| Loss | CE + Label Smoothing 0.05 | CE + Label Smoothing 0.05 |
| Optimizer | Adam | Adam |
| 1단계 | Head-only 15 Epoch | Head-only 15 Epoch |
| 2단계 | Partial FT 10 Epoch + 추가 5 Epoch | Partial FT 15 Epoch |
| Validation Accuracy | 0.7771565318 | 0.7963258786 |
| Test Accuracy | 0.7883386581 | 0.8003194888 |
| Test Macro F1 | 0.7885764577 | 0.8002222283 |

### v1 선정 이유

논문 강화 전 해상도, Loss, Backbone과 미세조정 조건을 비교했다. B1·256·Label Smoothing 0.05의
부분 미세조정을 10 Epoch에서 총 15 Epoch로 연장했을 때 가장 높은 Validation Accuracy를 기록해
당시 최종 후보로 선정했다. 이 모델도 Test 평가와 패키징까지 완료한 정식 후보였다.

### v2 선정 이유

두피 클래스는 각질, 피지, 모낭 주변 변화처럼 작은 질감 차이가 중요하다. Backbone과 Label
Smoothing은 유지하고 입력 해상도만 256에서 384로 높이는 multi-resolution 가설을 검증했다.
SupCon, DINOv2, EfficientNetV2-S, SAM과 단순 앙상블을 비교한 결과 B1·384·Adam이 최종 후보로
가장 적합했다.

### 변화량

| 지표 | v1 → v2 변화 |
|---|---:|
| Validation Accuracy | +0.0191693468 |
| Test Accuracy | +0.0119808307 |
| Test Macro F1 | +0.0116457706 |

## 6. 학습 용어 정리

| 용어 | 이 프로젝트에서 한 일 | 목적 |
|---|---|---|
| ImageNet 전이학습 | ImageNet 가중치로 시작 | 적은 의료 이미지에서 처음부터 학습하는 부담 감소 |
| Head-only | Backbone을 고정하고 새 분류기만 학습 | 새 클래스에 먼저 안정적으로 적응 |
| Partial fine-tuning | 마지막 일부 Backbone 층만 낮은 학습률로 추가 학습 | 일반 이미지 특징을 피부·두피 특징에 맞게 조정 |
| Label Smoothing 0.05 | 정답 1.0에 과도하게 몰리지 않도록 Loss 조정 | Hair의 비슷한 클래스에서 과신 완화 |
| PMG | 여러 세밀도의 특징과 조각 학습 사용 | Web Skin의 얼굴 전역·국소 병변 특징 동시 학습 |
| 384 해상도 | Hair 입력을 256에서 384로 확대 | 작은 두피 질감과 모낭 특징 보존 |

## 7. 발표용 핵심 문장

- Skin은 Clean 데이터에서 Original과 Augmented를 비교해 Augmented 모델을 v1으로 선정했다.
- Web Skin v1은 일반 CE와 부분 미세조정으로 선정·패키징한 논문 강화 전 최종 모델이다.
- Hair v1은 Label Smoothing 0.05와 부분 미세조정을 적용해 선정·패키징한 논문 강화 전 최종 모델이다.
- Web Skin은 PMG를 적용한 v2에서 Test Accuracy가 0.8825에서 0.915로 향상됐다.
- Hair는 Label Smoothing과 B1을 유지하면서 입력 해상도를 384로 높인 v2에서 Test Accuracy가
  0.7883386581에서 0.8003194888로 향상됐다.
- 현재 사용할 모델은 Skin v1, Web Skin v2, Hair v2다.

## 8. 근거 파일

- 현재 모델 경로와 SHA-256: `CANDIDATE_INDEX.json`
- 버전별 구조화 기록: `<domain>/selected_models/vN/selection.json`
- 버전별 설명: `<domain>/selected_models/vN/MODEL_INFO.md`
- 전체 이력 CSV: `SELECTED_MODEL_HISTORY_20260923.csv`
- 현재 모델 CSV: `CURRENT_SELECTED_MODELS.csv`
- 원본 패키지·평가 결과: `<domain>/candidates/`
