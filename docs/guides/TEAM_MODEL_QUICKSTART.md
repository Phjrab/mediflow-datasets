# 팀 모델 빠른 사용 안내

현재 통합 대상은 `results/CANDIDATE_INDEX.json`에 지정된 세 공개 데이터 후보다. 과거
`original/`, `augmented/` 모델보다 이 인덱스의 후보를 우선 사용한다.

## 후보 모델

| 대상 | 입력 | 현재 모델 |
|---|---:|---|
| Hair | 384×384 RGB | `results/hair/selected_models/v2/hair_model.keras` |
| Web Skin | 256×256 RGB | `results/web_skin/selected_models/v2/web_skin_model.keras` |
| Skin | 224×224 RGB | `results/skin/selected_models/v1/skin_model.keras` |

각 버전의 선정 이유는 해당 `selected_models/vN/MODEL_INFO.md`에 있다. 후보 ZIP에는 모델,
`class_names.json`, `preprocessing.json`, 평가 결과와 SHA-256 manifest가 들어 있다. 정확한 현재
경로와 해시는 `results/CANDIDATE_INDEX.json`을 기준으로 읽는다.

## 모델 선택

| 장비 | 촬영 부위 | 모델 |
|---|---|---|
| 웹캠 | 얼굴 전체 | Web Skin |
| USB 현미경 | 두피 | Hair |
| USB 현미경 | 확대 피부 병변 | Skin |

장비가 같아도 두피와 피부 병변 모델은 서로 바꾸어 사용하지 않는다.

## 공통 전처리

1. 이미지를 RGB 3채널로 디코딩한다.
2. 후보별 크기에 맞게 TensorFlow bilinear 방식으로 resize한다.
3. `float32` 픽셀 0–255를 그대로 모델에 전달한다.
4. 모델 내부에 `Rescaling(1/255)`이 있으므로 외부 `/255`를 적용하지 않는다.
5. 추가 ImageNet 정규화, crop과 padding을 적용하지 않는다.
6. 출력 인덱스는 각 후보의 `class_names.json` 순서로 해석한다.

Hair와 Skin은 모델의 단일 softmax 출력을 그대로 사용합니다. Web Skin PMG 모델은 네 개의
logit 출력을 반환하므로 패키지의 `inference.py`처럼 네 출력을 합산한 뒤 softmax를 한 번
적용해야 합니다. 첫 출력만 사용하거나 각 출력에 softmax를 따로 적용하면 안 됩니다.

세부 Python 예제와 오류 처리는
[`results/MODEL_USAGE.md`](../../results/MODEL_USAGE.md)를 따른다.

## 결과 해석

- softmax 점수는 정답일 확률로 보정된 값이 아니다.
- 범위 밖 입력을 자동으로 거부하는 기능은 검증되지 않았다.
- 현재 성능은 공개 데이터 기준이며 실제 장비 환자 데이터 성능을 의미하지 않는다.
