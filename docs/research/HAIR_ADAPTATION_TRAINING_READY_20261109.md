# 두피 촬영 밝기 적응 비교 학습 준비 — 2026-11-09

> 2026-11-10 현재 상태: 이 문서는 당시 준비 기록이다. 사용자가 학습 구성을 새로 정하기로 했으며
> 이전 큰 실행 ZIP은 코드·설정 이력을 보존한 뒤 정리했다.
> 현재 보관 위치는 [정리 기록](OUTPUT_REORGANIZATION_20261110.md)을 따른다.

## 현재 단계

로드맵 3·6단계의 촬영 환경 분석·변환 점검에서 4단계 추가 학습 준비로 진행했다.
**두피 Hair 5클래스의 비교 학습용 파일을 완성했다. 실제 질환 모델의 추가 학습과 최종
성능 평가는 아직 Colab에서 실행하지 않았다.** 얼굴·피부는 미해결 검토 상태를 유지한다.
이전 결과는 `DOMAIN_ADAPTATION_PREFLIGHT_RESULT_REVIEW_20261109.md`에 보존되어 있다.

## 이번에 완료한 검토

두피의 고정 Source Train 표본 120장, 클래스당 24장의 원본/변환 비교 사진을 15페이지
전체 확인했다. 미리보기 해상도에서 모발·각질 경계의 큰 소실이나 인공 테두리·기하 변형은
관찰하지 못했다. 이는 영상 품질에 대한 제한적 검토이며 임상 라벨이나 분류 성능의 검증은 아니다.

같은 표본의 v3 연산 수치 결과는 이전 기록과 같다. 새 경계값 비율의 최댓값은 `0.0`,
새 경계 근처 비율의 평균은 `0.0017556084526909725`, 최댓값은 `0.015521014178240741`이다.
0이나 255를 새로 만들지 않는 것은 연산의 설계 특성이다. 그 사실만으로 정보 보존이나
실제 장비 성능 향상을 주장하지 않는다. 경계 근처 압축과 최종 학습 효과는 별도 평가 대상이다.

이번에 승인한 범위는 고정 미리보기 120장의 `bounded_additive_v3` 밝기 연산을 사용하는
소규모 비교 실험이다. 과거 `profile_v2.json`과 검토 대기 결과는 수정하지 않았다.
새 실행 ZIP에 별도의 승인 프로필과 15페이지의 검토 근거·해시를 포함했다.

| 분야 | 현재 상태 | 근거와 미해결 사항 |
|---|---|---|
| hair | 비교 학습 준비 완료 | 현 검사에서 유사 사진 후보 0, 변환 표본 120장 전체 검토. 사람·병변·세션과 일부 증강 원본 정보는 부재 |
| web_skin | 학습 보류 | 후보 159묶음: Train/Test 149, Train/Validation 10. 얼굴 영역 검출 실패 552장의 추가 검토 필요 |
| skin | 학습 보류 | 후보 1618묶음: Train/Test 512, Train/Validation 1049, Validation/Test 57. 높은 유사 복제 우려 두 쌍과 출발 모델의 과거 학습 이력 검토 필요 |

이 수량은 중복 확정 건수가 아니다. 얼굴·피부의 후보를 자동 정상 처리하지 않았고,
피부 제외 제안도 아직 적용하지 않았다. 특히 Validation/Test 사이의 유사 후보도 있어,
피부 Train 두 장만 제외하면 모든 분할 문제가 해결된다고 설명하지 않는다.

## 비교 실험에서 고정한 것과 바꾸는 것

분야는 Hair 기존 5클래스이며 순서는 기존 메타데이터대로 다음과 같다.
`모낭사이홍반`, `미세각질`, `비듬`, `탈모`, `피지과다`. 정상 클래스는 없다.
Hair 6클래스 모델로 바꾸지 않는다. 실제 촬영 자료에 정상 또는 질환 정답을 임의로 붙이지 않는다.

| 항목 | 고정 설정 |
|---|---|
| 출발 모델 | 기존 EfficientNet-B1 384 · Label Smoothing 0.05 · Adam 선정 모델 |
| 모델 SHA256 | `9faa33b8f79f45ef6e7be415473e2ee5225f6195fff04d098828733db23ed3c2` |
| Source ZIP SHA256 | `2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156` |
| 분할 | 기존 original Train 10032 / Validation 1252 / Test 1252 유지 |
| 실제 학습 목록 | 기존 augmented Train 15047장, 두 비교군 공통 |
| 입력 | 기존 전체 이미지 리사이즈 384, float32 RGB 0–255, 외부 `/255` 없음 |
| 학습 | seed 42, 각 군 3 epoch, batch 16, Adam learning rate 0.00001 |
| 학습 가능 범위 | backbone 후반 30개 층, BatchNorm 고정, 기존 분류 head |
| 선정 기준 | 원본·변화 Validation 지표와 미리 정한 성능 유지 조건 |

세 모델의 역할은 다음과 같다.

1. `baseline`: 기존 저장 모델, 추가 학습 없음.
2. `source_only`: 기존 증강 데이터로만 추가 학습하는 대조군.
3. `brightness_only`: 같은 학습에 측정된 밝기 변환을 확률 0.5로 추가.

비교 학습 두 군 사이에서 바꾸는 변수는 촬영 밝기 변환 적용 여부 하나다.
현재 밝기 가산 범위는 `[-0.6363735198974609, 12.4506196975708]`이고,
가산 제안값을 v3 연산으로 경계에 부드럽게 접근시킨다. 이 범위가 실제 최종 픽셀 변화량과
완전히 같다는 뜻은 아니다. 대비·색·흐림·노이즈·기하 변환은 새로 추가하지 않는다.
`target_guided`는 두피에서 `brightness_only`와 같으므로 한 번만 학습하고 별칭을 기록한다.

원본 Source Test 사진을 보존한다. 학습 전 전체 분할 이미지를 중복·파생 누수 검사에
사용하지만, Test의 분류 결과는 변환 강도나 모델 선택에 사용하지 않는다.
원본·변화 Validation으로 선택한 기록을 해시로 고정한 다음에만 최종 Test를 평가한다.
기존 설정의 원본 성능 유지 허용값과 변화 성능 향상 기준도 실행 전에 설정 파일에 고정했다.

## 새 파일과 사용 순서

정식 노트북은 `notebooks/camera_adaptation/Hair_Adaptation_Training_Colab_20261109.ipynb`다.
2026-11-10 정리 갱신: 이 별도 학습 계획은 보관용이다. 다음 학습 구성은 미정이다.
이전 배포용 큰 ZIP은 정리했고 노트북 사본·코드·설정은
`camera_capture/_review/training_preparation_archive_20261110/code_and_configuration.zip`의
`standalone/hair_adaptation_training_ready_20261109/` 및 대응 `runtime/`에 보존한다.

- 노트북: `Hair_Adaptation_Training_Colab.ipynb`
- 실행 ZIP: `mediflow_hair_training_runtime.zip`
- 같은 내용의 Python 코드: `Hair_Adaptation_Training_Colab.py`
- 고정 실험 계획: `training_plan.json`
- ZIP·노트북 검증: `bundle_verification.json`, `release_verification.json`

사용자는 정식 노트북을 Colab에서 열고 GPU를 선택한 뒤 **런타임 → 모두 실행**한다.
파일 선택 창에서 이번 실행 ZIP 하나를 선택하고 기존 데이터가 있는 Drive 계정으로 연결한다.
DOMAIN, REVIEW_NOTE, 경로를 수정할 필요는 없다. 이전 분석용 ZIP을 선택하면 해시 검사로 거절한다.

다음 순서로 실행된다.

1. 실행 ZIP과 환경 확인. 호환 환경이 없으면 Colab 기본 환경과 분리해 준비한다.
2. Drive에서 해시가 맞는 기존 두피 ZIP을 찾아 실행 공간에 복사한다. 원본은 보존한다.
3. 클래스·분할·중복·증강 일관성·Source/Target 겹침을 다시 검사한다.
   알려지지 않은 사람·병변·세션·일부 증강 원본 정보는 확인 불가로 기록한다.
4. 고정한 두 학습 군을 각각 실행하고 Validation으로 선택을 고정한다.
5. 기존 모델과 두 새 모델을 원본 Source Test 및 별도 촬영 변화 Test에서 평가한다.
6. 모델·선정 기록·클래스별 지표·특징 배열·로그·보고서를 ZIP 하나로 내려받는다.

결과는 Drive의 새 `mediflow_hair_adaptation_<실행ID>` 폴더와 같은 이름의 ZIP에 보관한다.
완료된 모델 성능은 `hair/RESULTS.md`, 상세 Recall·Confusion Matrix는
`hair/training/final_evaluation/comparison.json`, 실행 상태는 `summary.json`에서 확인한다.
실패하면 성공으로 표시하지 않고 로그와 부분 결과를 포함한 ZIP을 내보내도록 구성했다.
필요하면 마지막 다운로드 버튼을 다시 누를 수 있다. GPU가 없으면 학습을 시작하지 않는다.

## 코드와 검증 결과

재사용 실행 코드는 `src/mediflow_datasets/domain_adaptation_training.py`,
패키지 생성기는 `scripts/build_hair_adaptation_training.py`다.
`src/mediflow_datasets/domain_adaptation.py`의 실행 기록에 새 코드와 Pillow/OpenCV 버전도
포함했다. 새 검사는 `tests/test_domain_adaptation_training.py`에 있다.

- `ruff check src tests`와 새 생성기 검사: 통과.
- 전체 pytest: **157개 통과, 3개 건너뜀, 실패 0개**.
  건너뛴 3개는 기존 노트북 검사의 선택 라이브러리 matplotlib 부재다.
- 추가한 검사 8개: 설정·클래스 순서·근거 파일 변경 거절, 미검토 분야 거절,
  밝기 외 변수 혼입 거절, 학습 실패 뒤 Test 평가 중단 및 실행 순서 확인.
- ZIP CRC·73개 파일 목록·내부 해시·검토 기록·노트북 문법 검사 통과.
- 촬영 원본 해시가 기존 calibration 기록과 일치. 촬영 사진 원본은 실행 ZIP에 포함하지 않았다.
- 기존 선정 모델 세 개의 해시는 모두 기존 색인과 일치했다.

최종 실행 ZIP은 48050129바이트이며 SHA256은
`e06a2defcab44294248080a7d97f8202cae8d00f3648deaa8a84140e8f6bba2b`다.
기준 Git commit은 `ffd71eca68ff1e21c145ed56158f4cfa77f7f037`이고 이번 코드는 미커밋 상태다.
변경 파일 자체와 해시를 실행 ZIP에 포함했다. Git commit이나 push는 하지 않았다.
위 검사는 코드와 실행 파일 검사이며 실제 질환 모델 학습 완료를 뜻하지 않는다.

## 다음 단계와 사용자가 할 일

사용자가 할 일은 **새 노트북에서 비교 학습 실행 후 결과 ZIP 다운로드**다.
같은 사전 분석을 다시 실행하거나 원본 데이터셋 ZIP을 PC에서 다시 올릴 필요는 없다.

결과 ZIP을 확보하면 저장된 Validation 선택과 Test 결과를 대조하고, 새 모델로 PC에 보관한
실제 두피 40장의 예측 변화와 Source/Target 특징 거리 MMD·CORAL을 계산한다.
이를 위해 실제 촬영 원본을 Colab에 다시 올릴 필요는 없다.
실제 촬영 질환 정답은 여전히 없으므로 Target Accuracy·Macro-F1은 계산하지 않는다.
확신도나 특징 거리 변화만으로 질환 분류 성능 개선을 주장하지 않는다.

얼굴·피부는 별도로 후보 검토와 데이터·기준 모델 이력을 해결한 뒤 학습을 준비한다.
MMD/CORAL 특징 정렬 학습은 현재 네 명의 미확인 클래스 구성 때문에 보류 상태다.
LLM·VLM은 촬영 환경 적응 작업 이후로 유지한다.
