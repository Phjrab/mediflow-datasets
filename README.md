# MediFlow 피부·두피 이미지 분류

USB 현미경 피부 `skin`, 웹캠 얼굴 `web_skin`, USB 현미경 두피 `hair`를 서로 다른 분류기로 관리하는 연구 프로젝트입니다.
스마트폰 얼굴 모델은 `web_skin`의 별도 촬영 환경 후보입니다.

## 현재 상태 · 2026-11-12

실제 촬영 환경 적응의 **21회 학습·81개 Source Test 비교·연구용 패키지 4개**를 완료했습니다.
기존 장비 사진 124장과 스마트폰 사진 52장은 질환 정답이 없는 촬영 환경 조사 자료입니다.
실제 장비 질환 정확도·실기기 속도·UI/LLM/VLM/장비 통합은 후속 과제입니다.

| 지금 찾을 것 | 안내 |
| --- | --- |
| 수집부터 최종 평가까지 전체 과정 | [어댑테이션 완료 정리](docs/research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md) |
| 무엇을 얼마나 조절했는지·실제 장비와 얼마나 가까운지 | [수치 비교 보고서·엑셀](docs/research/camera_adaptation/ADAPTATION_NUMERIC_COMPARISON_20261112.md) |
| 현재 폴더와 자료 위치 | [프로젝트 폴더 안내](docs/PROJECT_LAYOUT.md) |
| 완료·남은 작업 | [로드맵](docs/ROADMAP.md) |
| 모든 연구 문서·실제 사용 코드 | [문서 안내](docs/README.md), [문서 색인](docs/DOCUMENT_INDEX.md), [실행 대조표](notebooks/EXECUTION_INDEX.md) |
| 발표자료·보고서 그림 | [발표자료실](presentation/README.md), [보고서 자료](presentation/report_assets/README.md) |
| 이번 용량 정리·복원 방법 | [정리 기록](docs/research/maintenance/PROJECT_STORAGE_CLEANUP_20261112.md) |

## 실제 촬영 환경 후보

| 입력 | Validation에서 선정한 모델 | 사용 자료 |
| --- | --- | --- |
| 현미경 피부 | 4차 EfficientNet-B0·224·10클래스 | [기존 장비 패키지](results/camera_adaptation_packages_20261111/README.md) |
| 웹캠 얼굴 | 4차 PMG·B0·256·5클래스 | [기존 장비 패키지](results/camera_adaptation_packages_20261111/README.md) |
| 현미경 두피 | 2차 EfficientNet-B1·384·5클래스 | [기존 장비 패키지](results/camera_adaptation_packages_20261111/README.md) |
| 스마트폰 얼굴 | 4차 PMG·B0·256·5클래스 | [스마트폰 패키지](results/smartphone_adaptation_package_20261112/README.md) |

위 후보와 기존 공개 데이터 선정 모델을 구분합니다. 기존 공식 연결 경로인
[CANDIDATE_INDEX.json](results/CANDIDATE_INDEX.json)은 변경하지 않았습니다.
기존 선정 모델 7개는 [모델 경로 안내](results/SELECTED_MODEL_PATHS.md),
원본 결과와 학습곡선은 [results 안내](results/README.md)에서 찾습니다.

## 기존 연구와 보존 기준

9월 공개 데이터 연구는 [상세 보고서](docs/FINAL_DETAILED_REPORT_20260928.md),
프로젝트의 연구 방향은 [PROJECT_BACKGROUND.md](docs/research/PROJECT_BACKGROUND.md)를 따릅니다.
Hair 6클래스는 별도 연구이며 이번 두피 적응 5클래스와 섞지 않습니다.

모델·원본 결과·실행 당시 코드·학습곡선·발표자료를 보존합니다.
`camera_capture/`, 복호화 키, 실행 ZIP, 실제 참여자 사진이 포함된 비교 그림은 로컬 전용입니다.
과거 날짜의 준비 문서는 당시 이력이며 최신 상태는 위 완료 정리와 로드맵을 따릅니다.
이 결과는 연구 및 스크리닝 보조용이며 실제 질환 진단 성능을 검증한 결과가 아닙니다.

## 설치

Python 3.10–3.13 환경에서 저장소 루트를 기준으로 실행합니다. 저장된 모델은 TensorFlow
2.20.0 및 Keras 3.13.2에서 생성되었으므로 동일한 버전을 고정합니다.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 현재 후보 확인

기존 통합 경로의 세 후보 모델 파일, 입력 크기, 클래스 순서와 Web Skin PMG의 네 출력 결합 방법은
[`MODEL_USAGE.md`](results/MODEL_USAGE.md)를 따릅니다. 보관된 샘플 이미지로 세 후보의 추론
계약을 검사하려면 새 출력 폴더를 지정해 다음을 실행합니다.

```bash
python -m mediflow_datasets.candidate_reproduction --output candidate_check
```

이는 샘플 입력에 대한 기능 확인이며 새로운 Test 성능 측정은 아닙니다.

## 초기 학습 모델의 단일 이미지 추론 — 과거 연구용

```bash
mediflow-infer web_skin "data_examples/web_skin/정상_000002.png"
```

설치형 명령 대신 다음과 같이 실행해도 됩니다.

```bash
python -m mediflow_datasets.cli skin "data_examples/skin/광선각화증_0001.png"
python -m mediflow_datasets.cli hair "data_examples/hair/비듬_0006.jpg"
```

출력은 예측 클래스, 최고 점수, 전체 클래스별 점수를 포함한 JSON입니다. 현재 저장된
EfficientNet 후보 안에 `Rescaling(1/255)`이 있으므로 RGB 픽셀을 0–255 `float32`로
전달하며 별도의 `/255.0` 정규화를 하지 않습니다.

위 CLI는 기존 `results/*/1_training` 모델을 읽는 과거 연구용 실행 경로입니다. 현재 선정
후보의 성능이나 동작을 확인하는 명령으로 해석하지 않습니다.

## 데이터 전처리

세 스크립트는 목적과 데이터 구조가 서로 다릅니다.

```bash
# USB 현미경 피부 10-class
python scripts/preprocess_skin.py --source <원본폴더> --output <출력폴더>

# 웹캠 얼굴 피부 5-class
python scripts/preprocess_web_skin.py --source <원본폴더> --output <출력폴더>

# USB 현미경 두피 5-class
python scripts/preprocess_hair.py --source <원본폴더> --output <출력폴더>
```

출력 폴더가 이미 있으면 기본적으로 오류가 발생합니다. 기존 결과를 삭제하고 다시 만들
의도가 확실한 경우에만 명령 끝에 `--overwrite`를 추가하세요. 원본 폴더는 삭제하지 않습니다.

## 검사와 테스트

```bash
ruff check src tests
pytest
```

테스트는 클래스 순서, `.keras` 내부 출력 차원, EfficientNet 내부 정규화, 추론 입력 범위와
실제 세 모델 로드를 확인합니다. TensorFlow가 설치되지 않은 환경에서는 모델 로드 테스트만
건너뛰며, GitHub Actions에서는 고정된 TensorFlow/Keras를 설치해 전체 테스트를 실행합니다.

## 기존 평가 재현

정제 전 원래 Test 데이터셋의 클래스 폴더가 있을 때에만 original/augmented 모델의 지표를
다시 계산할 수 있습니다. 현재 보관 자료에는 이 사진이 없어 원래 평가의 추론 재실행은
완료하지 못했습니다. 가능한 범위와 이미 확인한 결과는
[`평가 재현 점검`](docs/research/public_data/EVALUATION_REPRODUCTION_AUDIT_20260927.md)에 기록했습니다.

```bash
mediflow-evaluate hair <hair-test-폴더> --variant original --output reports/hair-original.json
mediflow-evaluate hair <hair-test-폴더> --variant augmented --output reports/hair-augmented.json
```

출력에는 Accuracy, Macro F1, 클래스별 Precision/Recall/F1 및 Confusion Matrix가
포함됩니다. 출력 JSON이 이미 존재하면 `--overwrite` 없이는 덮어쓰지 않습니다.

전체 연구 진행 순서와 단계별 완료 기준은 [`docs/ROADMAP.md`](docs/ROADMAP.md)를
참고하세요. 34개 저장 모델의 Colab 실행 시간 측정과 그 한계는
[측정 해석](docs/research/public_data/RUNTIME_BENCHMARK_REVIEW_20260928.md)에 있습니다.

1차 프로젝트 정리와 발표용 요약은
[`docs/presentations/1차.md`](docs/presentations/1차.md)에 기록되어 있습니다.

과거 작업 양식은 [보관 ZIP](docs/zip/README.md)의 `docs/archive/setup/TASK_PROMPT_TEMPLATE.md`에
남아 있습니다. 현재 작업 규칙은 `AGENTS.md`를 따릅니다.

## 저장소 구조

```text
.
├── src/mediflow_datasets/   # 추론·분석·적응 학습 코드
├── tests/                   # 메타데이터·전처리·모델 로드 테스트
├── data_examples/           # 모델별 샘플 이미지
├── results/                 # 기존 모델과 평가 결과
├── notebooks/               # 기존 학습 노트북
│   └── camera_adaptation/   # completed / archive / 안내
├── configs/                 # 고정된 실험 설정
├── camera_capture/          # 실제 촬영 원본·분석 입력 (로컬 전용)
├── output/                  # 분석·실행 준비본·최신 실행 ZIP
├── presentation/            # 발표 자료·그림·수치 엑셀
├── scripts/                 # 카메라·데이터 전처리 스크립트
├── docs/                    # 연구 배경·로드맵·발표 자료
├── .github/workflows/ci.yml
├── AGENTS.md
└── pyproject.toml
```
