# Hair MedSigLIP-448 Linear Probe 결과

기준일: 2026-09-24  
상태: **Validation 분석 완료, 현재 Hair v2 유지**

## 1. 결론

MedSigLIP-448 encoder를 고정하고 Linear 분류기 하나만 학습한 결과, Original Validation
1,252장에서 Accuracy `0.7699680511182109`, Macro F1 `0.7687534259685547`을 기록했다.
현재 Hair v2인 EfficientNet-B1·384·Label Smoothing 0.05·Adam보다 Accuracy는
`0.0263578274760383`, Macro F1은 `0.02696125504295` 낮았다. 다섯 클래스의 F1도 모두
낮으므로 후보로 선정하지 않았다. Test 평가와 후보 패키징은 실행하지 않았다.

이 결과는 MedSigLIP 자체가 두피 분류에 부적합하다는 뜻이 아니다. 이번에 검증한 범위는
**frozen MedSigLIP embedding과 단일 Linear head**이며, encoder fine-tuning은 별도 가설이다.

## 2. 고정 조건과 변경 변수

| 구분 | 실제 실행 조건 |
|---|---|
| 데이터 | Hair clean ZIP SHA-256 `2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156` |
| 학습 | Augmented Train 15,047장 |
| 선정 | Original Validation 1,252장 |
| Test | 평가하지 않음 |
| 클래스 순서 | 모낭사이홍반, 미세각질, 비듬, 탈모, 피지과다 |
| seed | 42 |
| 변경 변수 | EfficientNet-B1 대신 frozen `google/medsiglip-448` embedding 사용 |
| 분류기 | 단일 Linear layer, AdamW, 최대 50 epoch |
| 입력 | MedSigLIP 공식 processor, 448×448 |
| 선정 기준 | Validation Macro F1 우선, 같으면 Accuracy |

실험 ID는 `hair_medsiglip_448_frozen_linear_seed_42`, protocol은
`hair_medsiglip_448_linear_probe_v1`이다. 최적 epoch는 50이며 NVIDIA L4에서 총
`864.7777227419999`초가 기록됐다. 실행 환경은 Python 3.13.15, NumPy 2.1.3,
PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 4.53.2다.

## 3. 현재 Hair v2와 전체 성능 비교

| 모델 | Validation Accuracy | Validation Macro F1 | 현재 후보 대비 Accuracy | 현재 후보 대비 Macro F1 |
|---|---:|---:|---:|---:|
| 현재 Hair v2 · B1·384·LS 0.05·Adam | 0.7963258785942492 | 0.7957146810115047 | 0 | 0 |
| MedSigLIP-448 · Frozen Linear | 0.7699680511182109 | 0.7687534259685547 | -0.0263578274760383 | -0.02696125504295 |

학습 손실은 1.5431에서 0.6982로 안정적으로 감소했고 Validation Macro F1도 0.5046에서
0.7688까지 대체로 상승했다. 학습이 붕괴한 결과가 아니라, 안정적으로 학습된 frozen embedding이
현재 Hair v2의 분리 성능에 도달하지 못한 결과다. 최적점이 마지막 50 epoch였으므로 더 긴 학습을
검토할 수는 있지만, 이는 별도 Validation 튜닝 실험이며 현재 결과를 바꾸어 해석하지 않는다.

## 4. 클래스별 F1

| 클래스 | Hair v2 F1 | MedSigLIP F1 | 변화 |
|---|---:|---:|---:|
| 모낭사이홍반 | 0.8378378378378377 | 0.8228571428571427 | -0.0149806949806950 |
| 미세각질 | 0.7400000000000001 | 0.7028112449799198 | -0.0371887550200803 |
| 비듬 | 0.7479338842975207 | 0.7253218884120172 | -0.0226119958855035 |
| 탈모 | 0.8353413654618472 | 0.8065173116089613 | -0.0288240538528859 |
| 피지과다 | 0.8174603174603174 | 0.7862595419847328 | -0.0312007754755846 |

가장 큰 하락은 미세각질이며 `-0.0371887550200803`이다. MedSigLIP의 클래스별 정답 수는
모낭사이홍반 216장, 미세각질 175장, 비듬 169장, 탈모 198장, 피지과다 206장이다. Hair v2는
각각 217장, 185장, 181장, 208장, 206장이므로 피지과다 정답 수만 같고 나머지는 감소했다.

## 5. 혼동 분석

미세각질과 비듬의 양방향 혼동은 MedSigLIP에서 `30 + 37 = 67장`, Hair v2에서
`36 + 35 = 71장`으로 4장 줄었다. 그러나 미세각질→탈모는 13장에서 19장으로, 미세각질→피지과다는
10장에서 17장으로, 비듬→피지과다는 18장에서 25장으로 증가했다. 탈모→모낭사이홍반도 19장에서
23장으로 늘었다. 특정 혼동 한 쌍의 감소가 전체 클래스 경계 개선으로 이어지지 않았고 모든 클래스
F1이 하락했다.

## 6. 선정 판단과 범위

- 현재 Hair v2를 그대로 유지한다.
- MedSigLIP Linear head는 `selected_models`에 넣지 않는다.
- Validation에서 후보가 되지 않았으므로 Test를 열지 않았다.
- 448 입력과 약 0.9B 규모 encoder의 다운로드·추론 비용까지 고려하면, 더 낮은 Validation 결과로
  현재 B1·384 후보를 교체할 근거가 없다.
- 사람·촬영 세션 단위 누수, 실제 USB 현미경 장비 성능, 정상·범위 밖 입력 처리는 이 실험으로
  검증되지 않았다.
- 출력 점수를 실제 질환 확률로 해석하지 않는다.

## 7. 보존 위치

- 재현 폴더:
  `results/hair/experiments/hair_medsiglip_linear_20260924_081708_9479fbb4`
- 원본 보고서 ZIP:
  `results/hair/archives/hair_medsiglip_linear_20260924_081708_9479fbb4_reports_72875297.zip`
- 원본 보고서 ZIP SHA-256:
  `f6c5e6b6fd865ade115663672b5801743f46561a2a8f37da8893334082988ecf`
- 주요 그림: `medsiglip_training_curves.png`,
  `medsiglip_validation_confusion_matrix.png`, `medsiglip_vs_hair_v2_validation.png`

완료 JSON의 모델 head, 지표, 이력, 예측 CSV와 환경 파일 SHA-256은 모두 재검증했다.
