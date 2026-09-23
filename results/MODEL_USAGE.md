# MediFlow 후보 모델 사용법

이 문서는 Hair, Web Skin, Skin 공개 데이터 후보 모델을 프로그램이나 Jetson에서 사용할 팀원을
위한 최소 사용 안내다. 실제 후보 목록과 SHA-256은 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)을
기준으로 확인한다.

## 1. 사용할 모델

| 대상 | 모델 파일 | 입력 | 출력 |
|---|---|---:|---:|
| Hair | `hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4/hair_model.keras` | 384×384 RGB | 5 |
| Web Skin | `web_skin/candidates/public_candidate_v2_pmg_b0_256_ce_20260922_235840_093d10de/web_skin_pmg_model.keras` | 256×256 RGB | 4 logits×5 → 5 |
| Skin | `skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056/model.keras` | 224×224 RGB | 10 |

팀원에게는 `CANDIDATE_INDEX.json`에 지정된 각 도메인의 후보 ZIP을 전달하면 된다. ZIP 안에는 모델,
클래스 순서, 전처리 정보와 평가 기록이 들어 있다. 같은 이름의 `.zip.sha256`은 전달 중 파일이
바뀌거나 손상되지 않았는지 확인할 때 사용한다.

## 2. 모델 선택 기준

| 촬영 장비 | 촬영 부위 | 사용할 모델 |
|---|---|---|
| 웹캠 | 얼굴 전체 | `web_skin` |
| USB 현미경 | 두피 | `hair` |
| USB 현미경 | 피부 병변 | `skin` |

USB 현미경이라는 정보만으로 Hair와 Skin을 선택하면 안 된다. 촬영 부위가 두피인지 피부
병변인지 함께 확인한다.

## 3. 클래스 순서

배열의 위치가 모델 출력 인덱스와 연결되므로 순서를 바꾸면 안 된다.

### Hair

```text
0 모낭사이홍반
1 미세각질
2 비듬
3 탈모
4 피지과다
```

### Web Skin

```text
0 건선
1 아토피
2 여드름
3 정상
4 주사
```

### Skin

```text
0 광선각화증
1 기저세포암
2 보웬병
3 사마귀
4 지루각화증
5 편평세포암
6 표피낭종
7 피부섬유종
8 혈관종
9 흑색점
```

배포 코드에서는 직접 작성한 목록보다 후보 폴더의 `class_names.json`을 읽는 것이 안전하다.

## 4. 입력 전처리 — 2026-09-14 정정

기존 안내의 **EXIF 자동 회전·Pillow resize 예제는 사용하지 않는다.** 재현 기준은 학습에
사용한 TensorFlow/Keras 이미지 로더다. EXIF는 사진에 들어 있는 방향 정보다.

- TensorFlow `tf.io.decode_image(..., channels=3, expand_animations=False)`로 읽는다.
- EXIF 방향으로 자동 회전하지 않는다.
- `tf.image.resize(..., method="bilinear", antialias=False)`로 크기를 맞춘다.
- Hair는 384×384, Web Skin은 256×256, Skin은 224×224다. 비율 유지 crop/padding은 넣지 않는다.
- 입력은 `(1, 높이, 너비, 3)`의 RGB `float32`, 픽셀 0~255다.
- 외부 `/255.0`, ImageNet 평균·표준편차 정규화를 적용하지 않는다.
- 모델 호출은 `training=False`, 로드는 `compile=False`를 사용한다.

Hair와 Skin은 단일 softmax 출력을 그대로 사용한다. Web Skin PMG 모델은 네 개의 logit
출력을 반환하므로 후보 ZIP의 `inference.py`처럼 네 출력을 합산한 뒤 softmax를 한 번
적용한다. 첫 출력만 사용하거나 branch마다 softmax를 적용하면 재현 결과와 달라진다.

근거: 각 현재 후보의 `preprocessing.json`. Hair v2 패키지는 입력 크기, resize, 픽셀 범위와
내부 정규화를 명시한다. 기존 Hair v1 후보 ZIP은 과거 실험 기록으로 보존했다.

실시간 OpenCV 프레임은 BGR이므로 RGB로 한 번 변환해야 한다. 이미 RGB인 입력을 다시
뒤집지 않는다. 브라우저 회전·크롭·JPEG 재압축도 입력을 바꾸므로 최초 비교에서는 원본
파일을 그대로 사용한다. 실제 촬영의 방향 보정 정책은 재현 확인 후 별도로 검증한다.

## 5. Python 실행과 기준 결과

재사용 가능한 실행 코드는 `src/mediflow_datasets/candidate_reproduction.py`에 있다.
프로젝트 루트에서 아래 예제를 실행한다. 모델 경로는 현재 작업 폴더가 아니라 코드가 위치한
저장소를 기준으로 찾는다. 기존 `mediflow-infer`는 과거 모델 경로를 쓰므로 이 후보 검증에
사용하지 않는다.

```python
import json
import numpy as np
import tensorflow as tf
from mediflow_datasets.candidate_reproduction import ROOT, preprocess

index = json.loads((ROOT / "results/CANDIDATE_INDEX.json").read_text(encoding="utf-8"))
spec = index["candidates"]["hair"]
model_path = ROOT / "results" / spec["model"]
classes = json.loads(model_path.with_name("class_names.json").read_text(encoding="utf-8"))
model = tf.keras.models.load_model(model_path, compile=False)
inputs = preprocess(ROOT / "data_examples/hair/비듬_0006.jpg", spec["input_size"])
outputs = model(inputs, training=False)
if spec.get("output_adapter") == "sum_logits_softmax":
    outputs = tf.nn.softmax(tf.add_n(outputs), axis=-1)
scores = np.asarray(outputs)[0]
assert scores.shape == (len(classes),)
print({"candidate": spec["candidate_id"], "classes": classes, "scores": scores.tolist()})
```

위는 최소 예제다. 실제 전달 검증에는 모델 해시·입력 크기·클래스 수·출력값 검사까지 포함한
아래 명령을 사용한다. 여러 요청을 처리하는 서비스에서는 모델을 한 번 로드해 재사용한다.

```bash
python -m mediflow_datasets.candidate_reproduction --output results/reproduction_team_run1
```

출력 폴더는 매번 새 이름을 쓴다. 기존 결과는 덮어쓰지 않는다. 비교 성공 시
모델 해시, 입력 shape, 클래스 수와 softmax 출력을 검사한 새 `reference.json`이 생성된다.
입력 파일이나 모델이 없거나 손상되면 실행을 중단하며 정상 결과를 대신 만들지 않는다.
자세한 전달 목록은 [재현 기준](REPRODUCTION_GUIDE.md)을 따른다.

## 6. SHA-256 확인

ZIP이 있는 폴더에서 확인한다.

### Linux / Jetson

```bash
sha256sum -c public_candidate_vN_....zip.sha256
```

### Windows PowerShell

```powershell
Get-FileHash -Algorithm SHA256 public_candidate_vN_....zip
```

계산된 값은 `.zip.sha256` 또는 `CANDIDATE_INDEX.json`의 `package_sha256`과 같아야 한다.

## 7. 결과 해석과 오류 처리

- 가장 큰 출력값의 클래스를 `predicted_class`로 사용한다.
- softmax `score`는 정답일 확률로 보정된 값이 아니다.
- Hair와 Skin에는 정상 클래스가 없다. 낮은 점수를 정상으로 바꾸지 않는다.
- 현재 정상 여부 판단은 지원하지 않으며, 불확실한 결과는 추가 확인 대상으로 설명한다.
- 낮은 점수 거부 임계값도 아직 검증하지 않았다. 임의로 50% 등을 기준으로 붙이지 않는다.
- Web Skin에는 정상 클래스가 있지만 범위 밖 이미지 거부 기능은 없다.
- 파일이 없거나 이미지 디코딩에 실패하면 예측하지 말고 오류를 반환한다.
- 장비와 촬영 부위 조합이 맞지 않으면 모델을 임의로 선택하지 않는다.
- 실제 장비 촬영 데이터 성능은 아직 검증되지 않았다.

학습 당시 확인한 환경은 TensorFlow 2.20.0, Keras 3.13.2다. Jetson에서 ONNX 또는 TensorRT로
변환할 경우에도 RGB 순서, 입력 크기, 0~255 픽셀 범위, 내부 Rescaling과 클래스 순서를 동일하게
유지해야 한다.

LLM에는 domain, candidate_id, 클래스 순서와 점수, normal_class_included,
calibration_status를 함께 전달한다. Hair·Skin은 normal_class_included=false,
normal_assessment=not_supported로 전달하며 점수와 의료적 위험도를 구분한다.
