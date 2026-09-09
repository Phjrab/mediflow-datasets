# Web Skin 공개 데이터 후보 v1 패키징 안내

ROADMAP 7의 공개 데이터 후보 보관 단계다. 실제 장비 검증과 시스템 모델 교체는 완료되지 않았다.

## 실행 파일

`web_skin_public_candidate_packaging_colab.ipynb` 원본 코드는
[기존 노트북 백업 ZIP](../../../notebooks/legacy_notebooks_20260909.zip)에 보관되어 있다.

1. Colab에서 노트북을 연다. GPU는 필요 없다.
2. `런타임 → 모두 실행` 후 Drive 접근을 허용한다.
3. 설치 버전 재시작 안내가 나오면 런타임을 재시작한 뒤 다시 모두 실행한다.
4. 마지막에 출력된 `public_candidate_v1_b0_256_ce_...zip`을 다운로드해 전달한다.

기존 모델 파일을 따로 다운로드하거나 업로드할 필요 없다. 학습, 이미지 중복 검사, Test 재평가를 다시 실행하지 않는다.

## 고정된 원본

Drive 원본 suite:
`MyDrive/mediflow_experiments/web_skin/suite_20260908_014452_72768a42`

선정 가중치:
`b0_256_ce/attempt_8977c45d04bc/stage2_best.keras`

모델 SHA-256:
`d4c834a7fd47480c1c45ddbdc09257942b38b2408d5b28effc34ee68c4fad97b`

EfficientNet-B0 / 256 / CE. Head 15 Epoch, 부분 미세조정 10 Epoch 중 최고 8번째 체크포인트다.
Validation Accuracy 0.796, Test Accuracy 0.8825, Test Macro F1 0.8813822927981046을 기존 보고값 그대로 보존한다.

원본 보고서 27개와 모델의 식별값을 검토 당시 값으로 고정했다. 다른 실험이나 변경된 결과를 조용히 패키징하지 않고 중단한다. source suite 폴더를 옮겼다면 SOURCE_SUITE만 변경한다.

## Hair와 동일하게 묶는 항목

- 선정 모델 한 개: `web_skin_model.keras`
- 클래스 순서: `class_names.json` (건선, 아토피, 여드름, 정상, 주사)
- 입력 처리: `preprocessing.json` (RGB float32 0~255, 내부 정규화, TensorFlow bilinear resize)
- 후보 설명: `MODEL_CARD.md`
- 후보 식별 및 수치·파일 해시: `package_manifest.json`
- 모델 입출력·가상 입력 실행 점검: `model_contract_check.json`
- 패키징 정책과 실행 코드: `packaging_policy.json`, `packaging_source.py`
- 원본 비교표·학습곡선·혼동행렬·평가 지표·예측 목록·학습 설정: `source_reports/`

source_reports의 원래 모델 카드 경로는 원본 suite 내부 위치를 보존한다. 패키지에서 실제로 사용할 경로는 최상위 package_manifest.json의 model_file이다.

Hair는 정상 클래스가 없었지만 Web Skin에는 정상 클래스가 있다. 범위 밖 입력 거부와 점수 보정은 구현되지 않았으며 점수를 정답 확률로 해석하지 않는다. 실제 웹캠 성능과 사람·병변·촬영 세션 누수 한계도 함께 보존한다.

## 저장과 검증

Drive 저장 위치: `MyDrive/mediflow_models/web_skin/`

새 후보 폴더, 동일 이름 ZIP, `.zip.sha256` 파일을 만든다. 이름에 실행 시각과 임의 식별자가 있어 기존 패키지를 보존한다.

- 검토된 원본 보고서와 모델의 해시 확인
- 모델 로드, 입력 256×256×3, 출력 5개, B0 파라미터 수 및 내부 Rescaling 확인
- 가상 입력에서 유한한 5개 점수와 합 확인 (성능 평가 아님)
- 원본 파일을 재저장하지 않고 바이트 그대로 복사
- 로컬 ZIP의 CRC와 내부 파일 해시 확인
- Drive 복사본의 파일별 해시 및 ZIP 전체 해시 확인

원본 데이터나 실험 산출물은 변경하지 않는다. 복사 실패 시 완료로 간주하지 않으며, 원본을 유지한 채 새 실행으로 다시 시도한다.

## 현재 완료 범위

패키징 실행 모듈과 Colab 노트북을 새로 만들었다. 실제 선정 모델은 Drive에 있으므로 이 로컬 작업에서 실제 후보 ZIP을 만들거나 Drive에 저장한 것은 아니다.

로컬 테스트는 임의 초기화한 실제 EfficientNet-B0 구조로 모델 입출력, 패키지 생성, ZIP 복원, 복사 무결성, 덮어쓰기 거부, 잘못된 모델/보고서 거부를 검사한다. 테스트 패키지는 실제 학습 후보가 아니다.

검증 결과: `ruff check src tests` 통과, 전체 `pytest` 39개 통과. Keras/NumPy 관련 경고 402건이 있었으나 실패는 없었다. 실제 Drive 연동과 학습 후보 가중치 검증은 Colab 실행 시 수행한다.
