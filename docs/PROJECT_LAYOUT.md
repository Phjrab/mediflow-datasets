# 현재 프로젝트 폴더 안내 · 2026-11-12

현재 상태는 실제 촬영 환경 적응 **21회 학습·81개 Source Test 비교·4개 모델 패키징 완료**다.
[전체 과정](research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md)과
[조절값·장비 비교](research/camera_adaptation/ADAPTATION_NUMERIC_COMPARISON_20261112.md)를 먼저 읽는다.

## 자주 찾는 자료

| 찾을 것 | 위치 |
| --- | --- |
| 실제 촬영 환경용 모델 | [기존 장비 3개 패키지](../results/camera_adaptation_packages_20261111/README.md), [스마트폰 4차](../results/smartphone_adaptation_package_20261112/README.md) |
| 기존 공개 데이터 선정 모델 7개 | [모델 경로 안내](../results/SELECTED_MODEL_PATHS.md) |
| 21회 학습 이력·학습곡선·평가 | [결과 폴더 안내](../results/README.md) |
| 실제 조절값·분포 거리·성능 비교 | [수치 자료 폴더](../presentation/report_assets/adaptation_numeric_review_20261112/README.md) |
| 발표자료·보고서 그림 | [발표자료실](../presentation/README.md) |
| 실제 실행한 코드·노트북 | [실행 대조표](../notebooks/EXECUTION_INDEX.md), [카메라 노트북](../notebooks/camera_adaptation/README.md) |
| 연구 배경·현재 실행 순서 | [연구 배경](research/PROJECT_BACKGROUND.md), [로드맵](ROADMAP.md) |
| 정리·삭제·복원 기록 | [최신 정리 기록](research/maintenance/PROJECT_STORAGE_CLEANUP_20261112.md) |

## 폴더별 역할

| 폴더 | 역할·보존 이유 |
| --- | --- |
| `results/` | 고유 모델·학습 이력·평가 JSON/CSV·예측·패키지. 사용 후보와 과거 비교 모델을 구분한다. |
| `presentation/` | 발표자료·보고서 그림·수치표. 기존 자료를 유지하고 새 엑셀도 여기에 통합했다. |
| `docs/` | 현재 안내와 연구 기록. 날짜가 있는 과거 문서의 미실행 표현은 당시 시점 기록이다. |
| `notebooks/` | 실행한 Colab 노트북과 보관용 준비·중단 기록. 완료 상태는 안내에서 확인한다. |
| `output/` | 실행 준비물·분석 노트북·최신 실행 ZIP. 학습 결과는 `results/`에서 찾는다. |
| `camera_capture/` | 촬영 원본·분석 입력·개인 자료·복원용 보관본. 로컬 전용이며 Git 제외다. |
| `data_examples/` | 세 분야별 고정 샘플 이미지. 촬영 원본이나 전체 학습 데이터셋이 아니다. |
| `src/mediflow_datasets/`, `tests/` | 재사용 Python 코드와 검사. |
| `scripts/`, `configs/` | 실행·생성 도구와 고정 실험 설정. 과거 설정도 재현 근거로 유지한다. |
| `.venv/` | 현재 실행에 필요한 Python·TensorFlow 환경. |
| `.git/` | 커밋 이력·복구·원격 연결 정보. 모델을 지워도 이력 용량까지 바로 줄어들지는 않는다. |
| `.tmp/` | 재생성 가능한 검사·작업 캐시. 이번 정리에서 중복 라이브러리와 오래된 검사 산출물을 제거했다. |

`output/`과 `outputs/`에 흩어졌던 새 수치 엑셀은 `presentation/report_assets/adaptation_numeric_review_20261112/`로 옮겼다.
이동 전후 내용은 같으며 비어 있던 `outputs/`는 제거했다. 기존 결과·원본·노트북 폴더는 이동하지 않았다.

현재 패키지는 공개 데이터 후보 색인을 자동 교체하지 않는다. 실제 시스템에서 사용할 모델과 입력 장비·부위를 함께 정한다.
실제 장비 질환 정답 평가, 실기기 속도·메모리, UI·LLM/VLM·장비 통합은 별도 후속 과제다.
