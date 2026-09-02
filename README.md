# MediFlow 피부·두피 이미지 분류

피부·두피 연구 결과를 보존하면서 세 EfficientNet-B0 모델을 재현 가능하게 추론하고,
데이터셋 전처리를 실행하기 위한 Python 프로젝트입니다. 이 결과는 의료 진단이 아닌 연구 및
스크리닝 보조 목적으로 사용해야 합니다.

## 모델

| CLI 이름 | 입력 환경 | 클래스 수 | 기본 모델 |
|---|---|---:|---|
| `skin` | USB 현미경 피부 병변 | 10 | augmented |
| `web_skin` | 웹캠 얼굴 피부 | 5 | augmented |
| `hair` | USB 현미경 두피 | 5 | augmented |

기존 자료는 용도에 따라 정리되어 있습니다. 샘플 이미지는 [`data_examples/`](data_examples/),
저장 모델과 평가 결과는 [`results/`](results/), 학습 노트북은 [`notebooks/`](notebooks/)에
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

## 단일 이미지 추론

```bash
mediflow-infer web_skin "data_examples/web_skin/정상_000002.png"
```

설치형 명령 대신 다음과 같이 실행해도 됩니다.

```bash
python -m mediflow_datasets.cli skin "data_examples/skin/광선각화증_0001.png"
python -m mediflow_datasets.cli hair "data_examples/hair/비듬_0006.jpg"
```

출력은 예측 클래스, 최고 확률, 전체 클래스별 확률을 포함한 JSON입니다. 저장된
EfficientNetB0 안에 `Rescaling(1/255)`이 있으므로 CLI는 RGB 픽셀을 0–255 `float32`로
전달하며 별도의 `/255.0` 정규화를 하지 않습니다.

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

원래 Test 데이터셋의 클래스 폴더를 지정하면 original/augmented 모델의 지표를 다시
계산할 수 있습니다.

```bash
mediflow-evaluate hair <hair-test-폴더> --variant original --output reports/hair-original.json
mediflow-evaluate hair <hair-test-폴더> --variant augmented --output reports/hair-augmented.json
```

출력에는 Accuracy, Macro F1, 클래스별 Precision/Recall/F1 및 Confusion Matrix가
포함됩니다. 출력 JSON이 이미 존재하면 `--overwrite` 없이는 덮어쓰지 않습니다.

전체 연구 진행 순서와 단계별 완료 기준은 [`docs/ROADMAP.md`](docs/ROADMAP.md)를
참고하세요.

1차 프로젝트 정리와 발표용 요약은
[`docs/presentations/1차.md`](docs/presentations/1차.md)에 기록되어 있습니다.

Codex에서 새 작업을 시작할 때는 [`docs/TASK_PROMPT_TEMPLATE.md`](docs/TASK_PROMPT_TEMPLATE.md)를
복사해 사용하면 진행 상황, 완료 내용, 다음 단계와 사용자 작업을 같은 형식으로 확인할 수
있습니다. 저장소의 상시 작업 규칙은 `AGENTS.md`에 기록되어 있습니다.

## 저장소 구조

```text
.
├── src/mediflow_datasets/   # 추론 패키지와 CLI
├── tests/                   # 메타데이터·전처리·모델 로드 테스트
├── data_examples/           # 모델별 샘플 이미지
├── results/                 # 기존 모델과 평가 결과
├── notebooks/               # 기존 학습 노트북
├── scripts/                 # 카메라·데이터 전처리 스크립트
├── docs/                    # 연구 배경·로드맵·발표 자료
├── .github/workflows/ci.yml
├── AGENTS.md
└── pyproject.toml
```
