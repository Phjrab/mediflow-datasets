# 팀원 전달용 후보 모델 재현 기준

작성일: 2026-09-14. 목적은 같은 모델과 같은 사진에서 같은 입력·점수가 나오는지 확인하는 것이다.
정확도 재평가, 정상 판별 검증, 실제 장비 성능 검증을 대신하지 않는다.

## 전달할 자료

| 자료 | 위치 |
|---|---|
| 세 후보 ZIP과 해시 | `results/<domain>/candidates/`의 후보 ZIP·SHA-256 |
| 후보 경로·모델 해시 목록 | `results/CANDIDATE_INDEX.json` |
| 정정한 사용 안내 | `results/MODEL_USAGE.md` |
| 이번 기준 결과·입력 배열 | `results/reproducibility_v1_20260914/` 전체 |
| 실행 코드 | `src/mediflow_datasets/candidate_reproduction.py` |
| 환경 정보 | `pyproject.toml`, 기준 결과의 `environment` |
| 원본 예시 사진 | 아래 여섯 파일 |

원본 예시 파일을 재저장하거나 메신저의 사진 압축 기능을 거치지 않는다. ZIP/파일 전송으로
같은 바이트를 유지한다. 이번 작업에서 사진을 외부에 업로드하거나 팀원에게 전송하지 않았다.

| 모델 | 예시 사진: 저장소 기준 경로 |
|---|---|
| Hair | `data_examples/hair/비듬_0006.jpg` |
| Hair | `data_examples/hair/미세각질_0012.jpg` |
| Web Skin | `data_examples/web_skin/정상_000002.png` |
| Web Skin | `data_examples/web_skin/여드름_000034.png` |
| Skin | `data_examples/skin/광선각화증_0001.png` |
| Skin | `data_examples/skin/보웬병_0001.png` |

파일명은 예시를 식별하기 위한 것이며 이번에 전문가가 확인한 정답 라벨이 아니다.
표본 두 장씩으로 모델의 정확도나 일반화 성능을 판단하지 않는다.

## 실행 방법

이 저장소 구조를 유지하고 후보 ZIP을 `CANDIDATE_INDEX.json`에 적힌 위치에 압축 해제한다.
별도 Python 가상환경에서 프로젝트 루트의 다음 명령으로 필요한 패키지를 설치한다.
Jetson에 호환성이 확인되지 않은 TensorFlow를 바로 설치하라는 뜻은 아니다. 먼저 호환되는
PC 환경에서 재현하고, Jetson 실행 환경과 변환은 담당 팀원이 별도 검증한다.

```bash
python -m pip install -e .
python -m mediflow_datasets.candidate_reproduction --output results/reproduction_team_run1 --reference results/reproducibility_v1_20260914
```

출력은 반드시 새 폴더를 지정한다. 이미 있는 폴더는 오류로 중단한다. 기준 결과를 다시 만들거나
덮어쓰면 비교 기준이 사라지므로 `--reference`를 생략하지 않는다.

## 기준 결과의 구성

- `reference.json`: 실제 측정한 여섯 예시의 전체 점수, 예측 인덱스, 클래스 순서,
  원본 사진·모델 해시, 환경, 코드 해시, Git 커밋과 변경 상태.
- `hair_0.npy` 등 여섯 배열: 디코딩·resize를 끝낸 모델 입력의 실제 float32 값.
  이미지 파일이 아니라 모델에 전달한 숫자 배열이며 `allow_pickle=False`로 읽는다.
- `reference_source.py.txt`: 실행에 사용한 코드 사본. 원본 모듈은 `src/`에 있다.

표시용 반올림 점수 대신 JSON의 전체 소수값을 비교한다. 입력값은 RGB 0~255이며 정상 여부를
나타내는 값이 아니다. 모델의 softmax 결과는 이미 출력에 포함되어 있어 다시 softmax하지 않는다.

## 통과 기준

| 검사 | 기준 |
|---|---|
| 검사 사례·모델 식별 | 동일 사례, 동일 후보 ID·모델 SHA-256 |
| 원본 파일 | 경로와 파일 SHA-256 일치 |
| 클래스 | 같은 개수·문자열·순서, 정상 클래스 포함 여부 일치 |
| 입력 | 같은 shape, float32 RGB, 배열 원소 절대오차 ≤ 0.0001 |
| 출력 | 유한한 0~1 점수, 합이 1에 가까움, 전체 배열 `atol=1e-5`, `rtol=1e-4` |
| 최상위 결과 | 예측 인덱스 일치 |

출력 배열의 비교식은 `|새 값-기준 값| ≤ 1e-5 + 1e-4 × |기준 값|`이다.
이 값은 초기 **실행 동일성 확인을 위한 공학적 기준**이며 의료 판정 임계값이 아니다.
CPU/GPU·라이브러리 차이로 실패하면 차이 원인을 기록하고 검토한다. 통과시키기 위해 자동으로
허용 오차를 늘리거나 기준 결과를 새 값으로 교체하지 않는다. ONNX/TensorRT의 FP16·INT8 변환은
이 여섯 사례 외에 별도 고정 평가 세트로 성능 변화를 확인해야 한다.

## 실패했을 때 확인 순서

1. 모델·사진 해시가 다름: 전달 파일과 후보 버전 확인.
2. 입력 배열이 다름: EXIF 자동 회전, BGR/RGB, Pillow resize, crop/padding, 외부 `/255`,
   JPEG 재압축 여부 확인.
3. 입력은 같고 점수만 다름: TensorFlow/Keras 버전, `training=False`, 실행 장치·수치 정밀도 확인.
4. 점수 배열은 같은데 질환명이 다름: 클래스 순서·이름 매핑 확인.

오류·낮은 점수·범위 밖 입력을 정상으로 변환하지 않는다. Hair·Skin의 정상 여부 판단은
`not_supported`다. Web Skin의 정상 클래스도 실제 환자에서 모든 질환이 없음을 보장하지 않는다.

## 이번 확인과 남은 확인

이번 기준은 로컬 Windows CPU, TensorFlow 2.20.0 / Keras 3.13.2에서 실제 세 후보를 읽어 만들었다.
완료된 실행의 상세 환경과 비교 결과는 각 실행 폴더 JSON을 기준으로 한다.
JPEG·PNG와 EXIF 방향 정보가 있는 테스트 파일을 대상으로 새 전처리와 학습 당시 디렉터리
로더의 배열을 비교하는 자동 테스트를 추가했다.

팀원 환경·키오스크·Jetson에서의 재현은 아직 수행하지 않았다. 단계 1 준비 이후 팀원이 같은
명령으로 비교하고, 통과한 입력·출력 계약을 웹과 LLM에 연결하는 것이 다음 단계다.

로컬에서 독립적으로 두 번째 실행을 수행했고 여섯 사례 모두 비교를 통과했다.
결과: `results/reproducibility_check_20260914/reference.json`의 `comparison.passed=true`.
이는 기존 Test Accuracy를 재측정하거나 질환 판단의 타당성을 검증한 결과가 아니다.

`ruff check src tests`와 새 재현 테스트 4개는 통과했다. 테스트의 pytest 캐시 저장 권한 경고는
검사 성공 여부에 영향을 주지 않았다. 전체 `pytest`는 31 passed / 25 failed / 3 skipped였다.
실패 항목은 기존 `models.py`가 참조하는 `results/<domain>/original|augmented` 경로와
정리 후 없는 이전 노트북을 찾는 테스트들이다. 현재 파일은 `1_training` 등에 보관되어 있으며,
이번 작업에서는 과거 경로의 실행 코드·테스트나 노트북을 복원/변경하지 않았다.
따라서 **새 후보 재현은 통과했지만 저장소 전체 테스트는 통과하지 않았다.**
