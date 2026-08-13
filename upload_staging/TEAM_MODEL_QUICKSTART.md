# Team Model Quickstart

이 문서는 팀원이 현재 저장소에서 학습 완료된 모델을 바로 확인하고 추론에 사용할 수 있도록 만든 빠른 시작 가이드입니다.

## 1) 프로젝트 폴더 구조 요약

```text
upload_staging/
├─ README.md
├─ code/
│  ├─ camera.py
│  ├─ preprocess.py
│  ├─ preprocess1.py
│  └─ preprocess2.py
├─ skin_results/
│  ├─ results_original/
│  │  ├─ best_model.keras
│  │  ├─ classification_report.txt
│  │  └─ results.json
│  └─ results_augmented/
│     ├─ best_model.keras
│     ├─ classification_report.txt
│     └─ results.json
├─ web_skin_results/
│  └─ web_skin_training_results/
│     ├─ original/
│     │  ├─ best_model.keras
│     │  ├─ classification_report.csv
│     │  └─ results.json
│     ├─ augmented/
│     │  ├─ best_model.keras
│     │  ├─ classification_report.csv
│     │  └─ results.json
│     └─ original_vs_augmented.csv
└─ hair_results/
   └─ hair_model_results/
	  ├─ original/
	  │  ├─ best_model.keras
	  │  ├─ class_names.json
	  │  ├─ classification_report.csv
	  │  └─ training_config.json
	  ├─ augmented/
	  │  ├─ best_model.keras
	  │  ├─ class_names.json
	  │  ├─ classification_report.csv
	  │  └─ training_config.json
	  └─ final_summary.csv
```

## 2) 어떤 모델을 바로 쓰면 되는가

- Skin Microscopy 10-class: `skin_results/results_augmented/best_model.keras`
- Web Skin 5-class: `web_skin_results/web_skin_training_results/augmented/best_model.keras`
- Hair Microscopy 5-class: `hair_results/hair_model_results/augmented/best_model.keras`

기본적으로 성능이 더 좋은 Augmented 모델을 기본 배포 후보로 사용하면 됩니다.

## 3) 팀원용 최소 추론 코드 (TensorFlow/Keras)

아래 코드는 단일 이미지 1장을 읽어서 예측 클래스와 확률을 출력합니다.

```python
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def load_class_names(model_type: str):
	if model_type == "hair":
		p = Path("hair_results/hair_model_results/augmented/class_names.json")
		data = json.loads(p.read_text(encoding="utf-8"))
		return [data[str(i)] for i in range(len(data))]

	if model_type == "web_skin":
		# results.json의 classes 순서를 사용
		p = Path("web_skin_results/web_skin_training_results/augmented/results.json")
		data = json.loads(p.read_text(encoding="utf-8"))
		return data["classes"]

	if model_type == "skin":
		p = Path("skin_results/results_augmented/results.json")
		data = json.loads(p.read_text(encoding="utf-8"))
		return data["classes"]

	raise ValueError("model_type must be one of: hair, web_skin, skin")


def load_model_path(model_type: str):
	mapping = {
		"hair": "hair_results/hair_model_results/augmented/best_model.keras",
		"web_skin": "web_skin_results/web_skin_training_results/augmented/best_model.keras",
		"skin": "skin_results/results_augmented/best_model.keras",
	}
	return mapping[model_type]


def predict_one(img_path: str, model_type: str):
	classes = load_class_names(model_type)
	model = tf.keras.models.load_model(load_model_path(model_type))

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img).astype("float32") / 255.0
	x = np.expand_dims(x, axis=0)

	probs = model.predict(x, verbose=0)[0]
	idx = int(np.argmax(probs))

	print("model_type:", model_type)
	print("pred_class:", classes[idx])
	print("confidence:", float(probs[idx]))


if __name__ == "__main__":
	# 예시
	predict_one("sample.jpg", "web_skin")
```

## 4) 빠른 실행 순서

1. 가상환경 생성/활성화
2. `pip install tensorflow pillow numpy`
3. 위 예제 코드 파일 저장 (예: `quick_infer.py`)
4. `sample.jpg` 준비
5. `python quick_infer.py`

## 5) 모델 선택 기준

- 웹캠 얼굴 사진: Web Skin 모델 사용
- USB 현미경 피부 확대 사진: Skin Microscopy 모델 사용
- USB 현미경 두피 확대 사진: Hair Microscopy 모델 사용

입력 장비/촬영 거리/피사체가 다르면 모델도 반드시 분리해서 사용해야 성능 저하를 줄일 수 있습니다.

## 6) 팀원 체크리스트

- 입력 이미지가 모델 목적과 맞는지 확인
- 리사이즈 224x224 및 `x/255.0` 정규화 적용
- 클래스 순서를 JSON/`results.json` 기준으로 고정
- 성능 비교는 Original vs Augmented 같은 평가셋에서만 수행

