# Skin 선정 모델

Skin에서 실제로 선정·패키징한 모델 버전은 **v1 하나**다.

| 버전 | 상태 | 선정 조건 | 모델 | 입력 | Test Accuracy | Test Macro F1 |
|---|---|---|---|---:|---:|---:|
| v1 | 현재 | Clean 분할에서 Augmented Train 선택 | EfficientNet-B0 | 224 | 0.9871428571428571 | 0.9871314132317626 |

Original과 Augmented는 v1·v2가 아니라 같은 모델 후보를 고르기 위한 학습 데이터 비교 실험이다. Validation에서 Augmented가 선택됐고, 그 모델을 고정 Test로 평가해 v1 후보로 패키징했다.

- 새 코드 연결: `selected_models/v1` 사용
- 현재 버전 표식: `CURRENT_VERSION.txt`
- 모델 무결성 확인: 모델 옆 `.sha256` 파일 사용
- 상세 선정 이유: `v1/MODEL_INFO.md` 사용
- Original/Augmented 비교 근거: 기존 `experiments/` 사용
- 원본 후보 ZIP과 전체 보고서: 기존 `candidates/` 사용
