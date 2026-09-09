# MediFlow 후보 모델 사용법

이 문서는 Hair, Web Skin, Skin 공개 데이터 후보 모델을 프로그램이나 Jetson에서 사용할 팀원을
위한 최소 사용 안내다. 실제 후보 목록과 SHA-256은 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)을
기준으로 확인한다.

## 1. 사용할 모델

| 대상 | 모델 파일 | 입력 | 출력 |
|---|---|---:|---:|
| Hair | `hair/candidates/public_candidate_v1_b1_256_ls005_20260907_120834/hair_model.keras` | 256×256 RGB | 5 |
| Web Skin | `web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/web_skin_model.keras` | 256×256 RGB | 5 |
| Skin | `skin/candidates/public_candidate_v1_b0_224_ce_augmented_20260909_075056/model.keras` | 224×224 RGB | 10 |

팀원에게는 각 `candidates` 폴더의 `public_candidate_v1_....zip`을 전달하면 된다. ZIP 안에는 모델,
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

## 4. 입력 전처리

- 이미지를 EXIF 방향에 맞게 회전한다.
- RGB 3채널로 변환한다.
- Hair와 Web Skin은 256×256, Skin은 224×224로 bilinear resize한다.
- 자료형은 `float32`, 픽셀 범위는 0~255로 유지한다.
- 외부에서 `/255.0`을 적용하지 않는다.
- 현재 모델에는 `Rescaling(1/255)`이 포함되어 있다. 외부 정규화를 추가하면 두 번 정규화된다.
- Crop이나 비율 유지 padding은 학습 전처리에 없으므로 임의로 추가하지 않는다.

## 5. Python 최소 실행 예제

아래 예제는 압축을 푼 후보 폴더를 `results` 아래에서 실행하는 경우다.

```python
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


RESULTS = Path("results")

MODELS = {
    "hair": {
        "model": RESULTS / (
            "hair/candidates/"
            "public_candidate_v1_b1_256_ls005_20260907_120834/"
            "hair_model.keras"
        ),
        "classes": RESULTS / (
            "hair/candidates/"
            "public_candidate_v1_b1_256_ls005_20260907_120834/"
            "class_names.json"
        ),
        "size": (256, 256),
    },
    "web_skin": {
        "model": RESULTS / (
            "web_skin/candidates/"
            "public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/"
            "web_skin_model.keras"
        ),
        "classes": RESULTS / (
            "web_skin/candidates/"
            "public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/"
            "class_names.json"
        ),
        "size": (256, 256),
    },
    "skin": {
        "model": RESULTS / (
            "skin/candidates/"
            "public_candidate_v1_b0_224_ce_augmented_20260909_075056/"
            "model.keras"
        ),
        "classes": RESULTS / (
            "skin/candidates/"
            "public_candidate_v1_b0_224_ce_augmented_20260909_075056/"
            "class_names.json"
        ),
        "size": (224, 224),
    },
}


def predict(image_path: str, domain: str) -> dict:
    spec = MODELS[domain]
    class_names = json.loads(spec["classes"].read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(spec["model"], compile=False)

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image = image.resize(spec["size"], Image.Resampling.BILINEAR)
        inputs = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

    outputs = np.asarray(model.predict(inputs, verbose=0)[0])
    if len(outputs) != len(class_names):
        raise ValueError("모델 출력 수와 class_names.json이 다릅니다.")

    index = int(np.argmax(outputs))
    return {
        "domain": domain,
        "predicted_class": class_names[index],
        "score": float(outputs[index]),
        "scores": {
            name: float(value)
            for name, value in zip(class_names, outputs, strict=True)
        },
    }


print(predict("sample.jpg", "hair"))
```

여러 사진을 처리할 때는 매번 `load_model`을 호출하지 말고 프로그램 시작 시 모델을 한 번 불러와
메모리에 보관한다.

## 6. SHA-256 확인

ZIP이 있는 폴더에서 확인한다.

### Linux / Jetson

```bash
sha256sum -c public_candidate_v1_....zip.sha256
```

### Windows PowerShell

```powershell
Get-FileHash -Algorithm SHA256 public_candidate_v1_....zip
```

계산된 값은 `.zip.sha256` 또는 `CANDIDATE_INDEX.json`의 `package_sha256`과 같아야 한다.

## 7. 결과 해석과 오류 처리

- 가장 큰 출력값의 클래스를 `predicted_class`로 사용한다.
- softmax `score`는 정답일 확률로 보정된 값이 아니다.
- Hair와 Skin에는 정상 클래스가 없다.
- Web Skin에는 정상 클래스가 있지만 범위 밖 이미지 거부 기능은 없다.
- 파일이 없거나 이미지 디코딩에 실패하면 예측하지 말고 오류를 반환한다.
- 장비와 촬영 부위 조합이 맞지 않으면 모델을 임의로 선택하지 않는다.
- 실제 장비 촬영 데이터 성능은 아직 검증되지 않았다.

학습 당시 확인한 환경은 TensorFlow 2.20.0, Keras 3.13.2다. Jetson에서 ONNX 또는 TensorRT로
변환할 경우에도 RGB 순서, 입력 크기, 0~255 픽셀 범위, 내부 Rescaling과 클래스 순서를 동일하게
유지해야 한다.
