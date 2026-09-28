# 선정 모델 경로와 사용 상태

이 문서는 **로컬 `results/`에 실제로 있는 선정 모델 7개**를 찾기 위한 안내다. 학습 실험 전체 목록이 아니다. 모델 경로는 저장소 루트 기준이며, 클릭하면 실제 파일이 열린다. 자세한 선정 이유와 원래 평가 수치는 각 `MODEL_INFO.md`와 [버전 비교](MODEL_VERSION_COMPARISON.md)를 따른다.

| 대상·버전 | 상태 | 모델 파일 | 입력 | 클래스 | 선정 이유·설정 |
|---|---|---|---:|---:|---|
| Skin v1 | 현재 통합 후보 | [skin_model.keras](skin/selected_models/v1/skin_model.keras) | 224×224 | 10 | [설명](skin/selected_models/v1/MODEL_INFO.md) |
| Web Skin v1 | 논문 강화 전 선정본, 현재는 v2 사용 | [web_skin_model.keras](web_skin/selected_models/v1/web_skin_model.keras) | 256×256 | 5 | [설명](web_skin/selected_models/v1/MODEL_INFO.md) |
| Web Skin v2 | 현재 통합 후보 | [web_skin_model.keras](web_skin/selected_models/v2/web_skin_model.keras) | 256×256 | 5 | [설명](web_skin/selected_models/v2/MODEL_INFO.md) |
| Hair 5클래스 v1 | 논문 강화 전 선정본, 현재는 v2 사용 | [hair_model.keras](hair/selected_models/v1/hair_model.keras) | 256×256 | 5 | [설명](hair/selected_models/v1/MODEL_INFO.md) |
| Hair 5클래스 v2 | 현재 통합 후보 | [hair_model.keras](hair/selected_models/v2/hair_model.keras) | 384×384 | 5 | [설명](hair/selected_models/v2/MODEL_INFO.md) |
| Hair 6클래스 성능 우선 v1 | 별도 연구 후보 | [hair_model.keras](hair/selected_models/6class_performance_v1/hair_model.keras) | 384×384 | 6 | [설명](hair/selected_models/6class_performance_v1/MODEL_INFO.md) |
| Hair 6클래스 경량 v1 | 별도 경량 대안 | [hair_model.keras](hair/selected_models/6class_light_v1/hair_model.keras) | 256×256 | 6 | [설명](hair/selected_models/6class_light_v1/MODEL_INFO.md) |

## 어떤 모델을 연결할까

- **기존 세 분류기 통합:** [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)에 지정된 Skin v1, Web Skin v2, Hair **5클래스** v2를 사용한다. 이 파일은 현재 통합 경로의 기계 판독 색인이다.
- **Hair 6클래스 연구:** [PUBLIC_MODEL_SELECTION_20260928.json](PUBLIC_MODEL_SELECTION_20260928.json)의 `hair_6class_primary` 또는 `hair_6class_light`를 명시적으로 선택한다. 기존 Hair 5클래스 경로를 자동으로 교체하지 않는다.
- **과거 선정본 확인:** Web Skin v1과 Hair 5클래스 v1은 당시 정식 선정·패키징된 모델로 보존한다. 새 통합의 기본 경로는 아니다.

## 모델 폴더를 전달할 때

`.keras` 하나만 복사하지 말고 같은 폴더의 `class_names.json`(출력 순서), `preprocessing.json`(입력 계약), `selection.json`(선정 기록), `MODEL_INFO.md`(선정 이유), `.keras.sha256`(파일 확인값)을 함께 전달한다. Web Skin v2는 네 출력 결합 규칙이 있으므로 [inference.py](web_skin/selected_models/v2/inference.py)도 필요하다. 입력·출력 처리와 오류 대응은 [모델 사용법](MODEL_USAGE.md)을 따른다.

세 도메인은 촬영 장비와 부위를 함께 보고 구분한다. 웹캠 얼굴은 Web Skin, USB 현미경 두피는 Hair, USB 현미경 피부 병변은 Skin이다. 모든 수치는 공개 데이터 평가 결과이며 실제 장비 성능을 뜻하지 않는다. 원본 후보 ZIP과 상세 실험 자료는 각 도메인의 `candidates/`, `experiments/`, `1_training/`에 그대로 보존한다.
