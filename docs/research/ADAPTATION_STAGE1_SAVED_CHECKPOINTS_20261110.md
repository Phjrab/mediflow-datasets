# 1차 조명 적응 학습 v3

사용자의 결정으로 여섯 실험 모두 저장된 가중치에서 바로 시작한다. 분야별 기본 방식은 초기
원본 데이터 학습본으로 정했다. 선정 방식은 해당 분야의 기존 선정 파일을 사용한다.

| 분야 | 기본 출발 모델 | 선정 출발 모델 |
|---|---|---|
| 피부 | [초기 B0·224](../../results/skin/1_training/original/best_model.keras) | [v1 B0·224](../../results/skin/selected_models/v1/skin_model.keras) |
| 얼굴 | [초기 B0·224](../../results/web_skin/1_training/original/best_model.keras) | [v2 PMG+B0·256](../../results/web_skin/selected_models/v2/web_skin_model.keras) |
| 두피 | [초기 B0·224](../../results/hair/1_training/original/best_model.keras) | [5클래스 v2 B1·384](../../results/hair/selected_models/v2/hair_model.keras) |

준비 학습은 0 epoch, 1차 조명 적응은 6개 모델에 5 epoch씩 총 30 epoch다. 1차의 비교 대상은
저장 모델별 추가 학습 결과이며 기존 학습 이력·입력 크기·구조 차이가 포함된다. 같은 분야의
두 모델에는 같은 원본 Train, 측정한 조명 변환, 검증 자료, seed 42와 학습률 0.00001을 쓴다.
모델별 클래스 순서·입력 크기·출력 방식 및 RGB 0–255 계약을 보존한다.
초기 얼굴 B0·224를 위해 같은 중앙 자르기 정책의 실제 촬영 입력을 준비하고 암호화했다.

원본 사진 목록의 누수 검사와 피부·얼굴의 기존 제외 목록은 적용한다. 초기 피부·두피는 과거
학습 분할과 현재 분할이 다르며, 피부·얼굴 선정 모델에도 유사 사진에 노출된 이력이 남을 수
있다. 이에 따라 기존 모델이 학습한 자료와 현재 검증 자료의 독립성은 미확인으로 기록한다.
1차 점수는 **후속 후보를 고르기 위한 탐색 비교**로 사용한다. 독립 검증은 과거 학습에 쓰이지
않은 별도 자료가 확보된 뒤 수행한다. [초기 모델 점검](EVALUATION_REPRODUCTION_AUDIT_20260927.md) ·
[피부·얼굴 분할 점검](SKIN_WEB_SKIN_RESOLUTION_20261109.md)

1. [공통 ZIP v3](../../camera_capture/_review/stage1_release_v3_20261110/MediFlow_Adaptation_Common_v3.zip)를
   Drive의 `내 드라이브/mediflow_Project` 바로 아래에 올린다.
2. [1차 저장 모델 노트북](../../notebooks/camera_adaptation/MediFlow_Adaptation_Stage1_Saved_20261110.ipynb)을
   Colab에 열고 GPU를 선택한 뒤 **모두 실행**한다.
3. 화면에서 `Epoch 1/5`, 배치·검증 진행을 확인한다. 재실행할 때는 같은 v3로 실행해 마지막
   완료 epoch의 모델·옵티마이저를 복원한다. 결과는 Drive의 `MediFlow_adaptation_runs/<v3 해시>/stage1`이다.

각 실행의 `initial.keras`에 저장본 파일을 그대로 복사한다. `selection.json`은 2·3차에 사용할
분야별 후보의 `initial.keras` 경로와 해시를 기록한다. 같은 v3의 공통 ZIP을 다음 단계에서도
사용하며 암호화에 기존 3차 복호화 키를 재사용했다. 전체 학습은 사용자 Colab에서 실행한다.

로컬에서 ZIP에 포함된 여섯 가중치의 해시가 원본과 같음을 확인했다. 여섯 모델 모두 불러온 뒤
합성 입력 한 배치로 학습 갱신과 검증 출력의 형태를 점검했다. `ruff`를 통과했고 `pytest`는
174개 통과, 3개 건너뜀(`matplotlib` 미설치)이다. 이 점검은 실제 데이터로 수행한 1차 학습
결과와 구분한다.
