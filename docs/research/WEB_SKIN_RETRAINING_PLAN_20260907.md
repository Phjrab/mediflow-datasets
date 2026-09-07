# Web Skin 재학습 시작 기록

2026-09-07. 상태: 노트북 준비 완료, Colab 데이터 검사·학습 실행 전.
실행 위치는 Colab, 데이터와 결과 보관은 Google Drive다.
실제 웹캠 검증 전 공개 데이터 실험으로 진행한다.

## 현재 단계

ROADMAP 1의 데이터 감사·기준 모델 재현에서 시작해 문제가 없을 때 4의 부분 미세조정으로 진행한다.
실제 장비 데이터 확보(2)와 Domain Gap 분석(3)은 이번 노트북에서 완료되지 않는다.

## 기존 결과 — 새 측정값이 아님

| 모델 | Validation Accuracy | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| Original | 0.6679999828338623 | 0.7850000262260437 | 0.7833125866587888 |
| Augmented | 0.6880000233650208 | 0.8199999928474426 | 0.8195818185502844 |

출처: results/web_skin/original/results.json 및 results/web_skin/augmented/results.json.
정상 클래스 성능에 비해 아토피·여드름의 성능이 낮다.
Augmented 아토피 Recall은 0.6875, 여드름 Precision은 0.6914893617021277이다.
Validation과 Test의 차이가 크지만 난이도·분포·누수 중 무엇이 원인인지는 확인 전이다.

클래스 순서: 건선 → 아토피 → 여드름 → 정상 → 주사.
얼굴 정면 전체 이미지 분류이며 현미경 Skin 10-class, Hair 5-class와 별개다.
정상 클래스는 포함하지만 범위 밖 이미지 거부 기능은 없다.

## 첫 실험

기존 Augmented 모델을 선택한 이유는 Validation Accuracy가 더 높기 때문이다.
Hair의 B1·256·LS 설정을 Web Skin에 그대로 적용하지 않는다.

- 가설: B0 후반부 일부를 낮은 학습률로 미세조정하면 기존 동결 모델의 Validation 성능이 개선된다.
- 고정: 기존 데이터 분할, 클래스 순서, Augmented Train, 원본 Validation/Test, B0, 224×224, Batch 32, Seed 42, Cross Entropy, Dropout 0.3.
- 변경: 마지막 30개 계층 중 Batch Normalization을 제외한 계층을 학습 가능하게 한다.
- 동반 변경: 새 Adam, 학습률 1e-5, 추가 10 Epoch. optimizer 상태 연속 학습이 아니며 학습시간 증가 효과와 완전히 분리된 실험도 아니다.
- 새 모델과 기존 모델을 같은 Validation으로 비교해 높은 쪽을 선택한다. 동률은 기존 모델 유지.
- Test는 선택된 모델에 한해 한 번 예측한다. 기존 Test 보고값은 새 측정과 구분한다.

## 데이터 감사

파일 SHA-256, RGB 픽셀 해시로 Original/Augmented의 모든 분할을 검사한다.
분할 간 중복, 라벨 충돌, 이미지 읽기 오류, 클래스 불일치,
Original/Augmented Validation/Test의 라벨·개수·내용 불일치를 발견하면 보고서를 저장하고 학습을 중단한다.
기존 폴더를 삭제하거나 자동 정제·재분할하지 않는다.

사람/병변/세션과 이미지의 연결, 회전·재압축된 유사 사진, 증강 파생 관계는 해시만으로 검증할 수 없다.
JSON/CSV 목록을 남겨 후속 확인 대상으로 삼는다.
기존 학습 시점의 데이터 해시가 없으므로 Validation 재현만으로 완전한 동일 분할을 입증하지 않는다.
중복 발견 후 새 분할로 재구성할 경우 기존 모델은 새 평가 사진을 과거에 보았을 수 있으므로
기존 모델에서 계속 학습하지 않고 ImageNet부터 새 기준선을 만드는 방향으로 다시 설계한다.

## 실행 방법

노트북: notebooks/web_skin_efficientnetB0_partial_finetuning_colab.ipynb.
Colab에 열고 GPU 런타임으로 위에서 아래로 실행한다.

- 데이터: Drive의 web_skin_processed ZIP 자동 검색.
- 기존 모델: web_skin_dataset_results ZIP/폴더 또는 web_skin_training_results ZIP/폴더 자동 검색.
- 여러 후보가 있으면 DATA_ZIP_OVERRIDE, BASELINE_OVERRIDE를 직접 지정한다.
- 기존 증강 모델 해시: 3b05c59500c272600438026b758842728b21d2ad2ce376cb435f436d4167310f.
- 데이터 검사 통과 후 기존 Validation이 허용 오차 1e-5 내에 재현되어야 학습한다.
- 저장: MyDrive/mediflow_experiments/web_skin/partial_finetune_224_실행ID/.
- 실행ID별 새 폴더를 사용하며 기존 실험을 덮어쓰지 않는다.

## 저장 결과

- 감사: audit_summary.json, image_inventory.csv, audit_issues.csv, dataset_counts.csv.
- 재현·설정: baseline_validation.json, training_config.json, code_snapshot.py.
- 모델: finetuned_best.keras, selected_model.keras.
- 학습: training_log.csv, training_history.json, training_curves.png.
- 평가: selection.json, evaluation.json, classification_report.csv, confusion_matrix.csv, test_predictions.csv, performance_dashboard.png.
- 보관: artifact_manifest.json, 위 파일을 묶은 결과 ZIP.

Colab 학습 완료 또는 감사 중단 후 생성된 결과 ZIP을 확인해 다음 단계를 정한다.
이번에는 기존 파일 수정·재학습 실행·Drive 업로드·커밋·푸시를 수행하지 않았다.

## 로컬 확인 결과

- 새 노트북의 코드 셀 8개 문법 검사 통과(pip 설치 매직은 별도 구문으로 제외).
- 가상 이미지 데이터로 정상 통과, 분할 중복 차단, 라벨 충돌 차단, 평가셋 개수 불일치 차단을 확인.
- 실제 기존 Web Skin 모델을 로드해 입력/출력, 내부 Rescaling, 후반부 30개 범위 및 BN 동결, compile 확인.
- 마지막 30개 Layer 중 학습 가능한 Layer는 23개. Layer 수와 가중치 텐서 수는 다른 개념이다.
- ruff check src tests 통과.
- pytest는 처음 기존 임시폴더 접근 권한 문제로 22개 통과/1개 준비 오류. 새 임시폴더 및 캐시 비활성화로 재실행해 23개 모두 통과.
- 실제 Drive 데이터 및 GPU 학습은 Colab 실행 후 확인해야 한다.
