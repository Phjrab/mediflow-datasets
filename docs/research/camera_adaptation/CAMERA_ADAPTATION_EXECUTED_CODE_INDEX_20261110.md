# 촬영 환경 적응: 실제 실행 코드와 결과 대응표

> 2026-11-12 보관 안내: 이 문서의 과거 실행 ZIP 링크는 복원 안내로 연결한다. 중복 모델을 포함한 ZIP은 작은 원본 구성과 유지한 모델로 정확히 복원할 수 있다. 아래 실험 수치와 당시 실행 이력은 보존했다.

> 2026-11-11 정리 상태: **분석·검토 또는 과정 기록, 후속 학습 상태는 최신 로드맵 참조**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

2026-11-10 작성. 사용자가 제시한 Colab 노트북 6개를 로컬 실행본과 회수한 결과에 대조했다.
보고서에서 완료된 분석, 중간 학습 기록, 실행 준비를 구분하기 위한 색인이다.
같은 날 사용자가 다음 학습을 새로 정하기로 하여, 사용하지 않을 준비본의 큰 ZIP은 정리하고
코드·설정은 [이력 ZIP](../../../camera_capture/_review/history/training_preparation/code_and_configuration.zip)에 보존한다.
범위는 이번 카메라 적응 작업이며, 이전 공개 데이터 모델 학습의 전체 노트북 목록은 아니다.
로드맵 3단계 Domain Gap 분석과 4단계 Partial Fine-tuning의 실행 이력 정리에 해당한다.

## 1. 사진에 있는 여섯 노트북의 상태

| 노트북 이름 | 실제 확인된 범위 | 보고서에서의 분류 |
|---|---|---|
| `Domain_Adaptation_Colab.ipynb` | 얼굴 원본 Train 3,600장·실제 촬영 42장의 분석과 변환 미리보기 | 분석 결과 있음. 학습·최종 평가 완료로 쓰지 않는다. |
| `Hair_Skin_Analysis_Colab.ipynb` | 두피 원본 Train 10,032장·촬영 40장, 피부 원본 Train 7,249장·촬영 42장 분석 | 두 분야 분석 완료 기록과 당시 코드 사본 있음. |
| `Adaptation_Preflight_Colab.ipynb` | 세 분야 입력 정책·유사 후보·변환 QA·특징 거리 점검 | 사전 점검 결과 있음. 새 모델 학습 결과는 아니다. |
| `MediFlow_9_Experiments_Colab_20261109.ipynb` | 사용자가 제공한 두피 `lighting` 조건의 1·2회차 학습 및 Validation 기록 | 부분 학습. 중단·폐기 후 재학습 의사가 확인된 시행착오 기록이다. |
| `Hair_Adaptation_Training_Colab_20261109.ipynb` | 별도 두피 학습 실행본과 실행 ZIP 준비 | 이 별도 노트북으로 만든 완결 학습 결과는 회수·확인되지 않았다. |
| `Skin_WebSkin_Resolution_Release_20261109.ipynb` | 피부·얼굴의 제외 목록을 반영한 학습 실행본과 ZIP 준비 | 이 별도 노트북으로 만든 완결 학습 결과는 회수·확인되지 않았다. |

실행 결과가 확인되지 않았다는 표현은 사용자가 노트북을 한 번도 열지 않았다는 뜻이 아니다.
로컬 노트북의 출력 칸이 비어 있다는 사실도 미실행의 증거로 쓰지 않았다.
학습 실행 ZIP의 제작 당시 `training_executed: false`도 이후 Colab 실행 상태를 나타내지 않는다.

## 2. 분석 결과가 있는 실행본 세 개

### A. 얼굴 초기 분석

- [로컬 노트북](../../../output/domain_adaptation_20261108_final/Domain_Adaptation_Colab.ipynb)
- [함께 사용한 실행 ZIP](../../../camera_capture/_review/history/project_cleanup_20261112/README.md)
- [회수한 얼굴 분석 결과](../../../camera_capture/_review/domain_adaptation_colab_20261109/web_skin/analysis/analysis.json)
- [분포 그래프](../../../camera_capture/_review/domain_adaptation_colab_20261109/web_skin/analysis/metric_distributions.svg)
- [PCA 그림](../../../camera_capture/_review/domain_adaptation_colab_20261109/web_skin/analysis/feature_pca.svg)
- [학습 사진 변환 미리보기](../../../camera_capture/_review/domain_adaptation_colab_20261109/web_skin/train_preview/source_train_before_after.png)

밝기·대비·색·미세 질감·얼굴 구도 등의 측정과 데이터 분할 점검 결과를 확보했다.
회수된 간단 검토 ZIP에는 7개 파일이 있으며 `reproduction.json`, 실행 당시 코드 사본,
`feature_diagnostics.json`은 없다. 따라서 이 초기 얼굴 실행의 코드 버전을 바이트 단위로
입증하거나 누락된 MMD·CORAL 수치를 추정하지 않는다. 세 분야 특징 거리 수치는 뒤의
사전 점검 결과를 사용한다.

노트북 자체에는 분석 이후 검토 승인·학습·최종 평가 칸도 들어 있다. 실제 확인된 실행 범위는
분석·미리보기이며, 당시 검토 승인 칸의 오류와 미해결 유사 후보가 있었다.
노트북에 학습 코드가 있다는 이유로 학습을 완료했다고 기록하지 않는다.

### B. 두피·피부 일괄 분석

- [로컬 노트북](../../../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.ipynb)
- [같은 실행 코드의 Python 내보내기](../../../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.py)
- [함께 사용한 실행 ZIP](../../../camera_capture/_review/history/project_cleanup_20261112/README.md)
- [두피 분석 결과](../../../camera_capture/_review/domain_adaptation_colab_20261109/hair/analysis)
- [피부 분석 결과](../../../camera_capture/_review/domain_adaptation_colab_20261109/skin/analysis)

두 분야 모두 `completed.json`의 `mode: analysis`, `status: completed`를 확인했다.
`batch_status.json`은 `analysis_completed_review_pending`, `training_executed: false`다.
결과에는 통계 JSON·CSV, 분포 그래프, Source/Target 특징과 PCA, 예측 기록,
MMD·CORAL 진단, 누수 점검, 변환 미리보기, 환경 및 코드 사본이 있다.

분포 측정은 각 분야의 원본 Train 전체에 수행했다. 특징 분포 비교는 Source 전체의
모든 특징을 추출한 비교가 아니라, 클래스당 Train 8장의 표본과 실제 촬영 자료를 사용했다.
Source 특징 표본은 두피 40장, 피부 80장이다.

두피·피부 각각 실행 당시 기록된 코드 9개의 해시가 결과 폴더의 `analysis/source_code/`
사본 및 원래 실행 ZIP의 코드와 일치했다. 원래 실행 코드를 인용할 때 이 사본을 사용한다.

### C. 세 분야 사전 점검

- [로컬 노트북](../../../output/adaptation_preflight_v2_20261109/Adaptation_Preflight_Colab.ipynb)
- [같은 실행 코드의 Python 내보내기](../../../output/adaptation_preflight_v2_20261109/Adaptation_Preflight_Colab.py)
- [함께 사용한 실행 ZIP](../../../camera_capture/_review/history/project_cleanup_20261112/README.md)
- [세 분야 회수 결과](../../../camera_capture/_review/preflight_v2_fbe37d44)
- [결과 요약](../../../camera_capture/_review/preflight_v2_fbe37d44/summary.json)
- [기존 상세 검토 보고서](DOMAIN_ADAPTATION_PREFLIGHT_RESULT_REVIEW_20261109.md)

원래 Source ZIP과 Train 목록을 확인하고, 이전에 계산한 전체 Train 통계의 캐시를 재사용했다.
이번 실행에서 전체 Train 통계를 다시 처음부터 계산했다고 서술하지 않는다.
세 분야의 입력 정책, 클래스 순서, 유사 사진 후보, 변환 전후 QA, 픽셀 경계 잘림,
얼굴 영역 검출 실패 목록과 특징 분포 진단을 추가로 확보했다.

변환 QA 표본은 얼굴 120장·두피 120장·피부 240장으로 총 480장이다.
유사 후보의 원시 쌍 수는 얼굴 484, 두피 0, 피부 7,770이며 확정 중복 건수와 구분한다.
세 분야 모두 `training_executed: false`, `source_test_accuracy_evaluated: false`다.

각 분야 `reproduction.json`에 기록된 코드 10개가 결과의 `source_code/` 사본 및 당시 실행
ZIP과 일치했다. 세 분야 전체 30개 코드 기록을 확인했다.
이때의 `profile_v2.json`은 당시 점검한 설정이다. 이후의 보완 프로필이나 통합 학습본의
확장 설정과 섞어서 그때부터 사용한 것으로 표현하지 않는다.

## 3. 보고서에 인용할 핵심 Python 코드

노트북은 Colab 실행 순서를 제공한다. 실제 분석 계산은 실행 ZIP 안의 Python 모듈이 담당한다.
현재 `src/`는 후속 수정이 있으므로 과거 결과를 설명할 때 실행 당시 사본과 해시를 기준으로 한다.

| 설명할 내용 | 당시 코드를 확인할 위치 |
|---|---|
| 초기 분석 실행 순서와 결과 저장 | `camera_capture/_review/domain_adaptation_colab_20261109/hair/analysis/source_code/domain_adaptation.py` |
| 밝기·대비·Lab 색·미세 질감·영역 통계 | `camera_capture/_review/preflight_v2_fbe37d44/hair/source_code/domain_adaptation_analysis.py` |
| 이미지 중복·분할 정보 점검 | 같은 폴더의 `domain_adaptation_data.py` |
| 모델 특징 추출·MMD·CORAL 진단 | 같은 폴더의 `domain_adaptation_experiment.py` |
| 유사 후보 묶음·변환 QA·얼굴 검출 실패 점검 | 같은 폴더의 `domain_adaptation_preflight.py` |

원래 생성 도구는 각각 `scripts/build_domain_adaptation_colab.py`,
`scripts/build_domain_adaptation_batch_colab.py`, `scripts/build_adaptation_preflight.py`다.
생성기는 실행본을 만들기 위한 코드이며 계산 결과를 만든 실행 모듈과 역할이 다르다.
생성기만 보고서 부록에 넣으면 실제 분석 로직이 빠질 수 있다.

## 4. 중단한 통합 학습 기록

- [통합 실행 노트북](../../../notebooks/camera_adaptation/archive/MediFlow_9_Experiments_Colab_20261109.ipynb)
- [당시 통합 실행 코드·설정 이력](../../../camera_capture/_review/history/training_preparation/code_and_configuration.zip)
- [사용자 제공 원문](../../../camera_capture/_review/history/executed_code/user_supplied_history_original.txt)
- [원문에서 추출한 1·2회차 JSON](../../../camera_capture/_review/history/executed_code/user_supplied_hair_lighting_history.json)

사용자가 붙여넣은 두피 `lighting/attempt_…/history.json`에서 epoch 1과 2를 확인했다.
각 회차에 가중치 갱신 1,881회, 원본 Validation 1,252장과 확장 변환 Validation 1,252장의
지표가 있다. `mmd_sum`, `coral_sum`, `alignment_steps`, `target_images_used`는 모두 0이다.
이는 조명 중심 조건의 기록과 일치한다. 정렬 조건까지 학습했다는 근거로 사용하지 않는다.

사용자는 실행 중단과 이번 학습 결과 삭제 후 재학습 의사를 밝혔다. 이후 Drive의 실제
종료·삭제 상태를 이 작업에서 확인하지는 않았다. 학습된 `.keras` 체크포인트도 로컬로 회수하지
못했으며, 붙여넣은 숫자 기록만 보고서용 시행착오 근거로 보관했다. 이 기록은 재학습을
이어가는 체크포인트가 아니고 최종 모델 성능이나 아홉 실험 완료 결과도 아니다.

확장 변환 코드는 `camera_extended.py`, 정렬 손실 코드는 `domain_alignment.py`,
학습·평가는 `domain_nine.py`이며 보관 이력의 `runtime/adaptation_9experiments_20261109/`
아래에 당시 코드 버전이 보존돼 있다. 대용량 실행 ZIP 컨테이너 자체는 정리했다.
준비된 코드·로컬 시험 통과와 실제 질환 데이터에서 해당 조건의 학습 완료를 구분한다.
속도 최적화한 새 실행본은 이 색인 작성 중에 만들지 않았다.

## 5. 결과가 확인되지 않은 두 학습 준비본

- 별도 두피 학습본 (삭제한 이전 준비 자료)
- 피부·얼굴 학습본 (삭제한 이전 준비 자료)

두피 실행본은 이력 ZIP의 `standalone/hair_adaptation_training_ready_20261109/` 내보내기와,
피부·얼굴 실행본은 `standalone/skin_web_resolution_release_20261109/` 내보내기와
각각 바이트가 일치한다. 이 별도 노트북들의 완결 학습
결과가 회수되지 않아 보고서의 완료 실험으로 분류하지 않았다.
재현 준비 또는 실험 설계의 부록으로 보존할 수 있다.

## 6. 보고서에 사용할 수 있는 설명

> 웹캠 얼굴 피부, USB 현미경 피부, USB 현미경 두피를 독립된 분류 과제로 구분하여
> 기존 원본 Train과 4명에게서 수집한 실제 촬영 사진 124장의 영상 특성을 비교하였다.
> 밝기·대비·색·미세 질감·영역 구성의 분포를 분석하고, 기존 모델 특징의 MMD·CORAL과
> PCA를 이용해 도메인 차이를 탐색하였다. 세 분야의 변환 전후 Train 표본을 생성하고
> 유사 사진 후보와 입력 정책, 픽셀 경계 잘림 등을 사전 점검하였다.
> 실제 촬영 자료에는 확인된 질환 정답이 없어 실제 장비 정확도를 계산하지 않았다.

추가로 중단 이력이 필요하면 “통합 적응 학습을 시도하여 두피 조명 조건의 2회차까지
학습·검증 기록을 확보했으나, 실행 시간 문제로 중단하고 재학습하기로 했다”고 구분해 쓴다.
중간 Validation 개선을 실제 장비 분류 성능 향상으로 표현하지 않는다.

Source의 사람·병변·세션 단위 독립성을 확인하지 못했고, Target 전원은 이미 분석에 관여한
자료다. Target 독립 Test나 실제 질환 성능 검증으로 설명하지 않는다.

## 7. 이번 대조에서 확인한 사항과 보존 방법

- 노트북 6개의 경로·해시·연결된 실행 ZIP·결과 근거를 색인에 기록했다.
- Python 코드 칸의 문법을 확인했다. 기존 노트북의 `%pip` 칸은 일반 Python 문법 검사에서 제외했다.
  Colab 실행이나 재학습은 수행하지 않았다.
- 두피·피부 분석 18개와 세 분야 사전 점검 30개, 총 48개 코드 기록의 당시 해시를 대조했다.
- 사용자 제공 중간 학습 기록은 원문과 JSON으로 새 폴더에 보관했다.
- 색인 최초 작성 시 기존 코드·노트북·결과·연구 문서는 수정하거나 이동하지 않았다.
  이후 같은 날 별도 정리 작업에서 준비본과 그림의 보관 위치를 정리하고 이 문서의 경로를 갱신했다.
- 재사용 Python 런타임 코드를 변경하지 않았으므로 전체 학습 테스트를 다시 실행하지 않았다.

[파일 해시와 근거의 기계 판독 색인](../../../camera_capture/_review/history/executed_code/code_index.json).
Drive의 현재 노트북을 내려받아 바이트가 동일한지까지 확인한 것은 아니다.
부록에는 로컬 내보내기 노트북과 연결된 실행 ZIP 또는 당시 `source_code/` 사본을 함께 제시하면
현재 코드 수정과 당시 결과의 관계를 설명할 수 있다.
프로젝트에 풀어 보관한 분석 결과 폴더는 유지한다. 다운로드용 중복 ZIP을 정리해도
실행 당시 코드 사본·수치·그림이 사라지지 않도록 이 보관 경로를 기준으로 관리한다.
