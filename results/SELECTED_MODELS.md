# MediFlow 선정 모델

실제로 선정·패키징된 모델 버전만 기록한다. Skin은 v1 하나이고, Web Skin과 Hair는 v1에서 v2로 교체됐다.
학습 단계, Loss, 미세조정 방법과 성능 변화는 [모델 버전·학습 방법 종합표](MODEL_VERSION_COMPARISON.md)를 따른다.
아래 Hair v1/v2는 **5-class** 과제다. 6-class 선정본은 아래 별도 표와
[성능·실행 비용 종합 선정](FINAL_MODEL_SELECTION_20260928.md)에 기록한다.

| 도메인 | 현재 버전 | 모델 | 입력 | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|---:|
| Skin | [v1](skin/selected_models/v1/MODEL_INFO.md) | EfficientNet-B0 | 224 | 0.9871428571428571 | 0.9871314132317626 |
| Web Skin | [v2](web_skin/selected_models/v2/MODEL_INFO.md) | PMG with EfficientNet-B0 | 256 | 0.915 | 0.9141049081029712 |
| Hair | [v2](hair/selected_models/v2/MODEL_INFO.md) | EfficientNet-B1 | 384 | 0.8003194888178914 | 0.8002222282821301 |

## 버전 구성

| 도메인 | v1 | v2 |
|---|---|---|
| Skin | Clean 데이터에서 Augmented가 선택된 현재 후보 | 없음 |
| Web Skin | 논문 강화 전 최종 선정 모델: EfficientNet-B0 | 논문 강화 후 현재 모델: PMG·EfficientNet-B0 |
| Hair | 논문 강화 전 최종 선정 모델: B1·256 | 논문 강화 후 현재 모델: B1·384 |

Skin의 Original/Augmented는 별도 버전이 아니라 v1을 고르기 위한 비교 실험이다.

## 별도 Hair 6-class 선정

| 역할 | 모델 | 입력 | Validation Macro F1 | Test Macro F1 | Colab End-to-end p50 |
|---|---|---:|---:|---:|---:|
| [성능 우선 v1](hair/selected_models/6class_performance_v1/MODEL_INFO.md) | B1·원본·LS 0.05 | 384 | 0.8144942091318823 | 0.8091500773460419 | 409.83297050024703 ms |
| [경량 대안 v1](hair/selected_models/6class_light_v1/MODEL_INFO.md) | B0·증강·CE | 256 | 0.7819142733310591 | 0.8021613486071133 | 288.73898249958074 ms |

6-class 성능 우선 후보는 동일한 6-class Validation Macro F1 선두다. 경량 후보는 실행 비용
비교용 대안이다. 5-class와 6-class의 정확도·F1은 서로 다른 과제라 직접 비교하지 않는다.
기존 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)의 `hair` 경로는 5-class v2로 유지한다.
6-class를 사용할 때는 [별도 선정 색인](PUBLIC_MODEL_SELECTION_20260928.json)의
`hair_6class_primary` 또는 `hair_6class_light`를 명시적으로 선택한다.

## 폴더 원칙

- `selected_models`: 실제로 한 번 이상 선정·패키징된 모델만 저장
- `candidates`: 원본 ZIP, 전체 보고서, 재현 근거
- `experiments`: 선정 전 비교 실험
- `CANDIDATE_INDEX.json`: 프로그램이 읽는 현재 모델 경로

버전 번호는 학습 Epoch가 아니라 실제 후보 모델의 세대다.
