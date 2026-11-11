# MediFlow 피부·두피 이미지 분류

피부·두피 연구 결과를 보존하면서 세 전문 이미지 분류 모델을 재현 가능하게 관리하고,
데이터셋 전처리를 실행하기 위한 Python 프로젝트입니다. 이 결과는 의료 진단이 아닌 연구 및
스크리닝 보조 목적으로 사용해야 합니다.

## 현재 작업 상태 · 2026-11-12

**실제 촬영 환경 적응의 계획한 학습·Source Test·패키징을 완료했습니다.**
[촬영부터 최종 결과까지 전체 정리](docs/research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md)를 먼저 확인하세요.
기존 장비 15회·스마트폰 6회, 총 21회 학습과 81개 비교 기록을 보관했습니다.
[완료 실험의 정확한 조절값](presentation/report_assets/adaptation_completion_20261112/README.md)은 17개 프로필·21회 기록입니다.

스마트폰 얼굴 52장 기반 web_skin 적응 **실제 5회 학습·결과 회수**를 완료했습니다.
[결과 분석](docs/research/camera_adaptation/SMARTPHONE_ADAPTATION_RESULT_REVIEW_20261112.md)과
[모델·원본 결과](results/smartphone_web_skin_adaptation_20261112/README.md)를 보관했습니다.
Validation 기준은 4차를 선택했지만 장비 유사성 최적을 입증한 결과는 아닙니다. 새 Source Test는 미실행입니다.

후속 [휴대폰 근접 후보 추가 1회 결과](docs/research/camera_adaptation/SMARTPHONE_JOINT_RESULT_REVIEW_20261112.md)도 회수했습니다.
5 epoch 완료·최적 3 epoch이며, 기존 실험을 합쳐 휴대폰 실제 학습은 총 6회입니다.
새 변환에서 추가 5차는 421/499, 적응 전은 424/499였고 공통 Validation 후보는 여전히 4차입니다.
[추가 모델·원본 기록](results/smartphone_joint_adaptation_20261112/README.md)을 보관했습니다.
이후 [스마트폰 최종 Test 30개 비교](docs/research/camera_adaptation/SMARTPHONE_FINAL_TEST_RESULT_REVIEW_20261112.md)를 완료했고
Validation에서 미리 선정한 [스마트폰 4차 사용 패키지](results/smartphone_adaptation_package_20261112/README.md)를 만들었습니다.
4차 원본은 367/400, 강한 변환은 355/400이며 실제 휴대폰 질환 성능은 정답 자료가 없어 미측정입니다.

처음에는 [지금까지 한 작업·결과·자료 위치](docs/PROJECT_SUMMARY_20261111.md)를 확인하세요.

[피부 4차·얼굴 4차·두피 2차 사용 ZIP](results/camera_adaptation_packages_20261111/README.md)을
독립 패키징하고 실제 로드·원본 예측 일치·압축 해제본 단독 실행을 확인했습니다.

실제 카메라 환경 적응을 먼저 진행하고 LLM·VLM 작업은 이후로 미뤘습니다.
1차 조명 적응 6회, 2차 색·미세 대비·흐림 확장 3회, 3차 MMD·CORAL 정렬 3회와
[최종 Source Test 분석](docs/research/camera_adaptation/ADAPTATION_FINAL_TEST_RESULT_REVIEW_20261110.md)을 완료했습니다.
실제 촬영 4명·124장은 질환 정답이 없는 촬영 특성 조사 자료이며 독립 Target Test가 아닙니다.

**조정 전 4차 학습을 완료했고 재설정 4차는 실행하지 않기로 했습니다.**
[4차 결과](docs/research/camera_adaptation/ADAPTATION_STAGE4_BEFORE_REFIT_RESULT_REVIEW_20261111.md)를 유지하고,
4차 추가 Test 15개와 기본~3차 결과 재사용 36개를 포함한 [51개 비교 결과](docs/research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md)를 확인했습니다.
[Test 노트북](notebooks/camera_adaptation/completed/MediFlow_Adaptation_Stage4_Test_20261111.ipynb)은 완료 실험의 재현용입니다.
[평가 계획·파일 위치](docs/research/camera_adaptation/ADAPTATION_STAGE4_FINAL_TEST_20261111.md)와 [로드맵](docs/ROADMAP.md)을 따릅니다.

코드·설정·노트북·보고서·수치 결과·학습 완료 모델을 Git에 보관합니다.
`camera_capture/`, 복호화 키, 실행 ZIP, 실제 참여자 사진이 들어간 비교 그림은 로컬 전용입니다.
해당 자료를 가리키는 문서 링크는 로컬 작업 공간에서만 열립니다.
카메라 노트북 실행에는 기존 Drive의 공통 ZIP·원본 데이터 ZIP·이전 단계 결과가 필요합니다.

정리된 [카메라 노트북 안내](notebooks/camera_adaptation/README.md)에서 현재 4차·완료 실험·보관용 파일을 구분합니다.
보고서용 그림은 [자료 안내](presentation/report_assets/README.md), 이번 정리 내역은 [완료 후 폴더 정리 기록](docs/research/maintenance/PROJECT_WRAPUP_CLEANUP_20261111.md)에 있습니다.

[전체 문서 대조표](docs/DOCUMENT_INDEX.md) · [실제 사용 코드·미실행 준비 대조표](notebooks/EXECUTION_INDEX.md) · [촬영 분석 폴더 안내](camera_capture/_review/README.md).

## 기존 연구 안내

처음 보는 팀원은 [최종 상세 보고서](docs/FINAL_DETAILED_REPORT_20260928.md)부터 읽으면 됩니다.
2026-10-26 당시 점검은 [LLM·VLM 착수 전 종합 점검](docs/research/integration/PRE_LLM_VLM_PROJECT_AUDIT_20261026.md),
9월 연구 마감 당시의 현황은 [2026-09-28 상태](docs/PROJECT_STATUS_20260928.md), 선정 결과는
[2026-09-28 모델 선정](results/FINAL_MODEL_SELECTION_20260928.md)에 있습니다.
실제 선정 모델 7개의 경로와 사용 상태는 [모델 경로 안내](results/SELECTED_MODEL_PATHS.md)에 모았습니다.

2026-10-26부터 LLM 결과 설명과 VLM 독립 질환 분류 실험을 준비합니다. 선정 모델 7개는 파일·ZIP·
클래스·전처리를 대조하고 실제 샘플 추론을 확인했습니다. 먼저 최신 후보용 공통 추론과
결과 형식을 준비하며, 기존 초기 모델용 CLI와 구분합니다. LLM·VLM 런타임은 아직 구현 전입니다.
팀원의 별도 MedVLM 저장소에는 최신 개발 브랜치의 의료 QA/VQA 학습·평가가 있으므로,
[두 저장소 분석과 연결 방향](docs/research/integration/TEAM_LLM_VLM_REPOSITORY_REVIEW_20261026.md)을 참고합니다.

## 모델

| 도메인 | 입력 환경 | 클래스 수 | 현재 공개 데이터 후보 |
|---|---|---:|---|
| `skin` | USB 현미경 피부 병변 | 10 | EfficientNet-B0 · 224 · CE · Augmented |
| `web_skin` | 웹캠 얼굴 피부 | 5 | PMG · EfficientNet-B0 · 256 · CE |
| `hair` | USB 현미경 두피 | 5 | EfficientNet-B1 · 384 · LS 0.05 · Adam |

위 표는 기존 세 모델 통합 색인인 [CANDIDATE_INDEX.json](results/CANDIDATE_INDEX.json)의 경로입니다.
Hair 6클래스는 별도의 성능 우선·경량 후보가 있으며, 기존 5클래스 경로를 자동으로 바꾸지
않았습니다. 다섯 선정본의 경로·해시·지표는
[PUBLIC_MODEL_SELECTION_20260928.json](results/PUBLIC_MODEL_SELECTION_20260928.json)에 있습니다.

기존 자료는 용도에 따라 정리되어 있습니다. 샘플 이미지는 [`data_examples/`](data_examples),
저장 모델과 평가 결과는 [`results/`](results), 학습 노트북은 [`notebooks/`](notebooks)에
보존되어 있습니다. 세부 연구 배경은
[`docs/research/PROJECT_BACKGROUND.md`](docs/research/PROJECT_BACKGROUND.md)를 참고하세요.

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
├── output/                  # 실행한 분석 3폴더와 pdf
├── presentation/            # 발표 자료·보고서 그림
├── scripts/                 # 카메라·데이터 전처리 스크립트
├── docs/                    # 연구 배경·로드맵·발표 자료
├── .github/workflows/ci.yml
├── AGENTS.md
└── pyproject.toml
```
