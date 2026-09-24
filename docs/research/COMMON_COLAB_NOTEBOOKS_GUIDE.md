# 공통 Colab 노트북 사용 안내

작성: 2026-09-09. 공개 데이터 실험의 검증·재사용 도구이며 실제 장비 검증 완료를 의미하지 않습니다.
ROADMAP 1 및 4~7단계의 반복 실험을 지원합니다. 실제 장비 데이터 확보·Domain Gap 분석은 별도입니다.

## 평소 사용할 파일 3개

| 파일 | 목적 | 학습 여부 |
|---|---|---|
| `01_common_dataset_audit_colab.ipynb` | 데이터 구조·중복·손상·분할 검사 | 없음 |
| `02_common_original_vs_augmented_colab.ipynb` | 원본/저장된 증강본, 같은 조건 비교 | B0 224 두 모델 |
| `03_common_six_experiments_colab.ipynb` | 6개 실험·그래프·선정·패키징 | 5개 신규 + B1 연장 |

각 노트북은 코드가 내장되어 있어 혼자 업로드해도 실행됩니다. Python 파일이나 저장소 연결은 필요 없습니다.
로컬 기존 노트북은 삭제하거나 이동하지 않았습니다. 2026-09-09 Drive에서는 로컬 코드 사본
17개와 SHA-256 목록을 `archive/legacy_notebooks_20260909.zip`으로 보관한 뒤 기존 개별
노트북 14개를 삭제했습니다. 과거 실행의 모델·지표·그래프는 별도의 결과 폴더에 유지됩니다.

## 처음 실행할 때

1. 필요한 공통 노트북을 Drive의 notebooks 폴더에 업로드하고 Colab으로 엽니다.
2. 맨 위 설정의 DOMAIN을 `hair`, `web_skin`, `skin` 중 하나로 선택합니다.
3. PROJECT_ROOT는 `/content/drive/MyDrive/mediflow_Project`가 기본입니다.
4. 데이터 ZIP을 `datasets` 안에 둡니다. 하위 폴더도 검색합니다. 확장자가 없는 ZIP도 찾습니다.
   파일 이름은 대상 이름으로 시작해야 자동 검색됩니다. 여러 개이거나 다른 이름이면 DATA_ZIP에 전체 경로를 입력합니다.
5. 새 데이터에는 ①을 실행하고 결과 경로를 ②/③의 `AUDIT_DIR`에 넣는 것이 기본입니다.
   별도 검증을 생략하기로 한 경우에는 `AUDIT_DIR`을 비우고 확인된 ZIP 해시를
   `EXPECTED_DATA_SHA256`에 입력합니다.
6. 학습 노트북은 Colab GPU를 선택하고 모두 실행합니다. 설치 후 버전 확인이 실패하면 세션을 다시 시작합니다.
7. 끝나면 `2_results`에 저장된 결과 ZIP과 `selected_models`의 후보 ZIP을 확인합니다.

예전 검증 보고서는 이 공통 버전의 audit_manifest가 없어 바로 연결되지 않습니다.
현재 데이터에 공통 ①을 한 번 실행한 후, 동일 ZIP에는 그 보고서를 계속 재사용할 수 있습니다.
사진 내용을 다시 조사하는 대신 ZIP 내용 식별값과 보고서의 일치 여부를 확인합니다.
`EXPECTED_DATA_SHA256` 방식은 ZIP 파일이 확인된 파일과 같은지만 검사하며 독립 감사를 대신하지 않습니다.

## 대상별 설정

클래스 순서는 기존 `results/<domain>/original`의 JSON에서 읽어 생성 시 포함했습니다.
알파벳 정렬이나 폴더 탐색 순서로 바꾸지 않습니다.

| 대상 | 촬영 대상/장비 | 출력 수 |
|---|---|---:|
| hair | 두피 / USB 현미경 | 5 |
| web_skin | 얼굴 피부 / 웹캠 | 5 |
| skin | 피부 병변 / USB 현미경 | 10 |

새 클래스를 추가하려면 DOMAIN만 바꾸는 작업이 아닙니다. 클래스 계약과 데이터 검증을 수정한 새 버전이 필요합니다.
점수는 보정된 정확 확률이 아니며 범위 밖 입력을 자동 거부하지 않습니다.

## 학습 실험의 의미

②는 B0 / 224 / CE / Dropout 0.3 / Adam 1e-4 / 분류층만 15회 학습을 공통 적용합니다.
변경 변수는 원본 학습 사진과 저장된 증강 학습 사진입니다. 추가 온라인 증강은 없습니다.
Original의 val/test를 두 실험에 공통 사용하며, Augmented의 val/test도 같은 파일인지 ①에서 확인합니다.
증강본은 수가 많아 같은 epoch에서도 optimizer 업데이트 수와 실행 시간이 증가합니다.
따라서 동일 epoch 예산의 비교이지 동일 연산량 비교는 아닙니다.
기존 Skin 노트북은 Adam 1e-3를 사용했지만 새 공통 기본값은 1e-4입니다.
기존 결과의 정확 재현이 아닌 새 공통 프로토콜의 원본/증강 비교입니다.

③은 다음 비교 기준을 고정합니다.

| 실험 | 비교 대상 | 변경 |
|---|---|---|
| B0 224 CE | 새로운 baseline | 분류층 15 + 미세조정 10 |
| B0 256 CE | B0 224 CE | 입력 크기 |
| B0 256 LS 0.05 | B0 256 CE | Loss |
| B0 256 Focal 1.5 | B0 256 CE | Loss |
| B1 256 LS 0.05 | B0 256 LS 0.05 | Backbone |
| B1 추가 5회 | B1 256 LS 0.05 | 학습 시간 연장 |

고정: 분할, 데이터 종류, seed 42, batch 32, dropout 0.3, 학습률 1e-4/1e-5,
후반 30개 레이어(BatchNormalization 제외) 미세조정 정책, 추가 온라인 증강 없음.
B1의 후반 30개 레이어는 B0와 파라미터 수가 다릅니다. 학습 레이어 목록과 전체 파라미터 수를 기록합니다.
5회 연장은 직전 B1의 마지막 모델과 optimizer를 복원하며, 이전 우수 모델도 선택 후보로 유지합니다.
기존 Hair/Web Skin 모델을 초기 가중치로 쓰지 않습니다. 새 실험은 ImageNet에서 시작합니다.

모델 선정은 Validation Accuracy, 동점이면 먼저 실행한 실험/단계를 유지합니다.
Test는 선정 모델에만 사용합니다. ②에서도 두 모델의 비교는 Validation 기준입니다.
이전 Test 결과를 보며 반복 설정 변경하면 독립 최종 평가 의미가 약해지므로 새 연구는 별도 평가 계획이 필요합니다.

## 저장 내용과 재개

```text
mediflow_Project/
  datasets/...
  2_results/<domain>/audit_<시각_ID>/
  2_results/<domain>/<comparison 또는 suite>_<시각_ID>/
  2_results/<domain>/selected_models/<실행명_ID>/
```

기존 `2_results` 구조를 유지하며 새 실행도 대상별 하위 폴더에 저장합니다.
학습 기록에는 소스 사본·소스 해시·생성 당시 commit·설정·데이터 해시·이미지 분할 목록·클래스 순서·
환경·seed·단계별 최고/마지막 모델·CSV·JSON이 포함됩니다.

끊기면 같은 설정으로 RESUME_DIR에 기존 실행 폴더를 입력합니다.
완료 실험은 파일 해시 검사 후 재사용합니다. 중단된 실험은 새 attempt 폴더에서 처음부터 학습합니다.
epoch 중간부터 자동 재개하는 기능은 아닙니다. 이전 중단 파일도 보존합니다.
코드/환경/설정/데이터가 바뀌면 재개를 거부하므로 RESUME_DIR을 비우고 새 실험으로 시작합니다.

학습 곡선은 `all_training_curves.png`와 PDF, 성능은 `validation_performance_dashboard.png`,
혼동행렬은 `all_validation_confusion_matrices.png`, 비교 수치는 `experiment_comparison.csv`입니다.
그래프의 C0, C1 등은 class_names.json의 배열 순서입니다.
Loss 정의가 다른 CE/LS/Focal의 손실값을 직접 성능 순위로 비교하지 않습니다.
각 실험의 오분류 예시와 선정 모델의 Test 오분류 사진도 저장합니다.

결과 ZIP은 모델을 제외한 로컬 보관용입니다. 모든 실험 모델은 실행 폴더에 남고,
선정 모델은 `selected_models` 아래의 별도 후보 ZIP으로 묶습니다.
후보 ZIP에는 model.keras, 클래스 순서, 전처리, 모델 카드, 검증/테스트 결과, 소스와 해시 목록이 있습니다.
같은 완료 폴더에서 패키징을 다시 실행하면 새 패키지를 만듭니다. 완료 Test 결과는 검증 후 재사용합니다.

## 검증 범위와 현재 제한

파일 및 픽셀이 같은 중복, 라벨 충돌, 손상, 평가 분할 불일치 등 발견 시 학습을 차단합니다.
원본/증강본 각각 train/val/test와 클래스 폴더가 있어야 합니다.
사람·병변·촬영 세션 및 변형된 증강 출처는 대응 정보 미검증으로 보고서에 기록됩니다.
해시 검사만으로 이들 누수가 없다고 판단하지 않습니다. 유사 이미지 탐지 및 라벨의 임상적 정확성도 미검증입니다.
데이터 정제·삭제·자동 재분할 기능은 포함하지 않습니다.

새 코드의 5/10-class 학습·추가 학습·재사용·입출력·ZIP 경로·노트북 구문 검사를 제공합니다.
2026-09-09 로컬 검사: `ruff check src tests` 통과, 전체 `pytest` 48개 통과·3개 건너뜀.
건너뛴 항목은 두 학습 모드의 그림/패키징 전체 통합 검사와 이미지 중복 검사입니다.
로컬에는 matplotlib/pandas가 없고 패키지 설치 네트워크가 제한되어 그림 생성 및 전체 통합 검사는 미실행입니다.
실제 Colab GPU와 사용자 데이터로 새 학습을 실행한 결과는 아직 없습니다.
전체 통합 검사는 training 추가 의존성을 설치한 환경에서 `pytest tests/test_common_notebooks.py`로 실행합니다.

## 개발 시 수정 위치

공통 실행 코드는 `src/mediflow_datasets/common_engine.py`, `common_audit.py`, `common_workflow.py`입니다.
설명과 설정 셀은 `scripts/build_common_notebooks.py`에서 관리합니다.
수정 후 생성기를 실행하여 3개 노트북을 갱신하고 `ruff check src tests`와 `pytest`를 실행합니다.
노트북 내장 코드와 소스의 일치 여부도 검사합니다.
