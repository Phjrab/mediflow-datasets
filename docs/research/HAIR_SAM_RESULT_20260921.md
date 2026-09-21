# Hair B1·384 Adam 대 SAM 선별 결과

## 실험 목적

seed 42 Validation에서 가장 높았던 EfficientNet-B1·384에 Sharpness-Aware
Minimization(SAM)을 적용해 일반화 성능이 더 좋아지는지 확인했다. 기존 B1·384의 동일한
Stage 1 체크포인트에서 시작했으며, 데이터·분할·입력 크기·Loss·미세조정 범위와 epoch를
고정하고 optimizer update만 바꿨다.

## 재현 조건

| 항목 | 값 |
|---|---|
| 데이터 ZIP SHA-256 | `2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156` |
| 평가 분할 | original Validation 1,252장 |
| Backbone | EfficientNet-B1 |
| 입력 | 384×384 RGB |
| Loss | Label Smoothing 0.05 |
| seed | 42 |
| 시작 모델 | 기존 B1·384 Stage 1 최적 체크포인트 |
| 시작 모델 SHA-256 | `0e064976688f8c0f08869d2e6a31b4ebff3992720e21c8bb838bf5ba921e622e` |
| 미세조정 | 후반 30개 레이어 범위, BatchNormalization 고정, 15 epoch |
| 변경 변수 | Adam update 대 SAM, `rho=0.05`, Adam base optimizer |
| Test 사용 | 없음 |

독립 데이터 감사는 이번 실행에서 다시 수행하지 않았다. 기존에 확인한 ZIP SHA-256만
대조했으며 사람·병변·촬영 세션 누수와 perceptual near duplicate는 미검증 상태다.

## 결과

| 방법 | Validation Accuracy | Validation Macro F1 | 정답 수 |
|---|---:|---:|---:|
| 기존 Adam | 0.7963258786 | **0.7957146810** | 997/1,252 |
| SAM | 0.7963258786 | 0.7954748921 | 997/1,252 |
| SAM−Adam | 0.0000000000 | -0.0002397889 | 0 |

Accuracy는 완전히 같았고 Macro F1은 SAM이 약 `0.00024` 낮았다. 차이가 매우 작지만,
계산량이 더 큰 SAM을 채택할 성능 근거는 없다.

### 클래스별 F1

| 클래스 | Adam | SAM | SAM−Adam |
|---|---:|---:|---:|
| 모낭사이홍반 | 0.8378378378 | 0.8317580340 | -0.0060798038 |
| 미세각질 | 0.7400000000 | 0.7387755102 | -0.0012244898 |
| 비듬 | 0.7479338843 | 0.7546391753 | +0.0067052910 |
| 탈모 | 0.8353413655 | 0.8387096774 | +0.0033683120 |
| 피지과다 | 0.8174603175 | 0.8134920635 | -0.0039682540 |

SAM은 비듬과 탈모 F1을 올렸지만 모낭사이홍반, 미세각질과 피지과다 F1이 내려갔다.
`미세각질→비듬`과 `비듬→미세각질` 오류 합계는 Adam 71건에서 SAM 68건으로 3건 줄었다.
이 부분 개선만으로는 전체 Macro F1 감소와 추가 계산 비용을 상쇄하지 못했다.

SAM 학습곡선은 14번째 epoch에서 최고 Validation Accuracy를 기록했고 15번째에는 소폭
낮아졌다. Validation loss는 15 epoch 동안 계속 감소했으므로 심한 발산은 없었다. 실행은
정상적으로 끝났으며 저장된 1,252개 예측으로 Accuracy를 다시 계산해 같은 값을 확인했다.

## 판단과 다음 단계

SAM은 이번 `rho=0.05`, seed 42 조건에서 채택하지 않는다. rho를 반복 조정하면 동일
Validation에 과적합될 위험이 있으므로 추가 튜닝도 현재 우선순위로 두지 않는다. Hair의
선두 단일 조건은 **EfficientNet-B1·384, Label Smoothing 0.05, 기존 Adam 부분
미세조정**으로 유지한다.

B1·384 Adam을 최종 후보로 고정하고, 다음에는 이 후보만 Test에서 한 번 평가한다.

## 보존 자료

- 로컬 결과: `results/hair/experiments/sam_screen_20260921_145602_a5e8c406/`
- 원본 보고서 ZIP:
  `results/hair/experiments/sam_screen_20260921_145602_a5e8c406_results_988b45f9.zip`
- ZIP SHA-256:
  `e0fe4809a9f44c6c5ff4ef1e7a6746c6a8924e9bca8d1e1eccdcf45c65cb17b8`
- SAM 학습 시간: `954.794213444`초
- SAM 모델 파라미터 수: `6,581,644`

보고서 ZIP은 `.keras` 모델을 제외한 결과 ZIP이다. SAM 모델은 Drive의 원 실행 폴더에
남아 있지만, 현재 판단에서는 최종 후보로 채택하지 않는다.
