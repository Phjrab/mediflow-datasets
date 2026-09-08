# Web Skin 자동 실험 실행 안내

2026-09-08. 새 자동 실행 노트북 준비. 실제 데이터 전체 학습은 아직 수행하지 않았다.
기존 단일 부분 미세조정 노트북은 보존한다.

## 이미 끝난 데이터 검사

dataset_audit_20260907_174038_67c0377d.zip에서 동일 파일/픽셀의 분할 중복,
라벨 충돌, 같은 분할 중복, 읽기 오류가 발견되지 않았다.
Original/Augmented 평가셋의 내용·라벨·수량이 일치한다.
사람·병변·세션 및 변형된 증강본 출처의 겹침은 미검증이다.

사용자의 지시에 따라 중복 검사를 다시 실행하지 않는다.
입력 ZIP이 검증된 파일과 같은지만 다음 SHA-256으로 확인한다.

f8908af3d54e521ad14c37a44b569d33fe92be3b8b9b66a8d80faf4ba964072d

원본 ZIP은 자율설계2 폴더에 있어도 된다. 자동 검색이 모호하면 경로를 설정한다.
학습은 Augmented Train 7,200장(원본 3,600 + 추가 증강 3,600),
Validation 500장, Test 400장을 사용한다.

## 실행할 노트북

[web_skin_all_experiments_colab.ipynb](../../notebooks/web_skin_all_experiments_colab.ipynb)

1. Colab에서 노트북을 열고 GPU 런타임을 선택한다.
2. 첫 실행은 DATA_ZIP_OVERRIDE와 RESUME_SUITE_DIR을 비워둔다.
3. 런타임 → 모두 실행. 설치 후 버전 재시작 안내가 나면 재시작 후 다시 실행한다.
4. 마지막에 표시된 reports ZIP을 우선 공유한다.

기존 모델 ZIP은 필요 없다. 비교 조건을 맞추기 위해 ImageNet 가중치에서 새로 시작한다.
기존 저장 모델을 이어 학습하는 단일 노트북과 시작점이 다르다.
5개 모델에 동일한 2단계 학습량을 적용하고 1개를 추가 연장한다.

## 실험

| 순서 | 설정 | 비교 목적 |
|---|---|---|
| 1 | B0 / 224 / CE | 새 2단계 기준선 |
| 2 | B0 / 256 / CE | 1 대비 해상도 |
| 3 | B0 / 256 / LS 0.05 | 2 대비 Loss |
| 4 | B0 / 256 / Focal gamma 1.5, alpha 1.0 | 2 대비 Loss |
| 5 | B1 / 256 / LS 0.05 | 3 대비 Backbone |
| 6 | 5의 Stage 2 마지막 모델에서 5 Epoch 연장 | 학습량 |

공통: Seed 42, Batch 32, Dropout 0.3, 동일 데이터 및 클래스 순서.
Stage 1: Backbone 동결, Head 15 Epoch, Adam LR 1e-4.
Stage 2: 마지막 30개 계층 중 BN 제외 계층 미세조정, 새 Adam LR 1e-5, 10 Epoch.
CE는 Cross Entropy, LS는 Label Smoothing이다.
최대 130 Epoch. 연장 실험은 새 모델 30 Epoch를 처음부터 돌리는 것이 아니라 5 Epoch만 추가한다.
실제 시간은 Colab GPU와 Drive 속도에 따라 달라지며 여기서 측정하지 않았다.
증강 기법은 이번 실험에서 변경하지 않는다.
Stage 1→2 비교에는 부분 계층 해제뿐 아니라 학습률과 추가 학습량 변화도 포함된다.

실험 1~5는 각자의 Stage 1/Stage 2 최고 체크포인트를 Validation Accuracy로 선택한다.
실험 6은 Stage 2의 마지막(10번째) 모델과 optimizer 상태를 복원하고 학습한다.
연장 성능이 부모 실험 최고보다 좋지 않으면 부모 최고 모델을 유지한다.

## 후보 선정과 평가

- 모든 후보를 Validation Accuracy로 비교하고 같은 값이면 앞선 모델을 유지한다.
- Validation Macro F1, 클래스별 F1, 혼동행렬도 함께 제공한다.
- 후보를 JSON에 고정한 뒤 그 모델만 Test에서 평가한다.
- Test를 모든 모델에 반복 평가해 순위를 고르지 않는다.
- 기존 Augmented 결과는 보고값으로만 표시하며 이번에 재측정했다고 표현하지 않는다.
- 기존 Validation 보고값보다 새 후보가 낮으면 교체 권장하지 않는다는 메시지를 출력한다.
- 클래스: 건선, 아토피, 여드름, 정상, 주사. 범위 밖 입력 거부와 실제 웹캠 검증은 없다.
- 단일 Seed의 작은 차이를 확정적인 개선으로 해석하지 않는다.

## 저장되는 비교 자료

- all_training_curves.png: 6개 실험의 Accuracy/Loss를 3행 × 4열로 비교.
- validation_performance_dashboard.png: Validation Accuracy/Macro F1 및 클래스별 F1.
- all_validation_confusion_matrices.png: 6개 모델의 Validation 혼동행렬.
- 각 실험 training_curves.png 및 validation_errors.png: 고확신 오답 사진 최대 8장.
- final_test_confusion_matrix.png 및 final_test_errors.png: 최종 후보 Test 결과.
- experiment_comparison.csv, all_validation_results.json: 원래 정밀도의 수치.
- 각 실험의 stage1/stage2 best/last.keras, 학습 로그, 예측 목록, 설정.
- model_card.json: 후보 모델 위치, 해시, 입력/출력, 성능과 한계.

그림에서는 C0~C4와 class_mapping.json을 사용한다.
Focal, CE, LS의 Loss 계산식이 다르므로 절대 Loss 크기로 모델을 비교하지 않는다.
training_seconds_this_trial에서 연장 실험은 추가 5 Epoch 수행 시간만 의미한다.
파라미터 수와 모델 파일 크기는 기록하지만 정식 추론 속도/메모리 벤치마크는 별도다.

## 저장 위치와 재개

MyDrive/mediflow_experiments/web_skin/suite_실행ID/

- reports ZIP: 모델을 제외한 그래프·사진·보고서.
- full_models ZIP: 모델 포함 전체 결과.
- 각 실험의 모델·Epoch 로그는 Drive에 바로 저장한다.
- Colab이 중단되면 RESUME_SUITE_DIR에 기존 suite 폴더 경로를 입력한다.
- 완료된 실험은 설정 서명과 산출물 해시 검증 후 건너뛴다.
- 완료 표시가 없는 실험은 이전 attempt를 보존하고 새 attempt에서 처음부터 실행한다.
- Epoch 중간 재개를 보장하지 않는다. 다른 실행이 같은 suite를 동시에 수정하면 안 된다.
- 최종 Test 완료 기록이 있으면 무결성 확인 후 기존 결과를 사용한다.
- 기존 결과가 손상됐거나 설정/코드가 달라지면 재개를 거부한다.

## 코드 위치

- 재사용 실행 모듈: src/mediflow_datasets/experiment_suite.py.
- 노트북에는 모듈 원문을 포함하므로 Colab에 저장소나 별도 .py를 올릴 필요 없다.
- 모듈을 수정할 때에는 노트북의 내장 코드도 같이 갱신해야 한다.
- 테스트는 내장 코드와 모듈의 정확한 일치, 선택 정책, 실제 작은 모델의 저장/연장,
  완료 실험 재사용, 모델 변조 탐지를 검증한다.

이 자동 실험은 ROADMAP 4~6의 공개 데이터 실험과 7의 후보 준비에 해당한다.
실제 장비 검증과 최종 시스템 모델 확정까지 완료하는 실행은 아니다.

## 로컬 확인 결과

- `ruff check src tests`: 통과.
- 전체 `pytest`: 28개 통과. Keras/NumPy 호환 관련 경고 66건이 있었으며 실패는 없었다.
- 작은 가상 데이터와 작은 모델로 학습, 체크포인트 저장, 추가 학습, 완료 실험 재사용 및 손상 감지를 확인했다.
- 노트북 코드 문법과 내장 실행 코드의 원본 일치를 확인했다.
- 실제 Web Skin 전체 학습과 학습 결과 그래프 생성·육안 검수는 아직 실행하지 않았다. Colab 실행 후 확인해야 한다.
