# 발표·공유용 선정 모델 7개

이곳은 `results/<도메인>/selected_models/`에서 **원본을 보존하고 복사한 사본**이다. 각 폴더의 `.keras`, `class_names.json`, `preprocessing.json`, `selection.json`, `MODEL_INFO.md`, SHA-256 파일을 함께 전달해야 입력 크기·출력 순서가 유지된다. 복사 후 모델 본체와 기계 판독 메타데이터의 SHA-256을 원본과 대조했다. Hair 6종의 `MODEL_INFO.md` 복사본에 있는 원본 링크만 이 폴더 위치에 맞게 수정했다. [파일별 출처·모델 해시](모델_목록.csv).

| 폴더 | 과제·역할 | 구조 | 현재 발표에 사용할 Test Macro F1 |
|---|---|---|---:|
| [skin_10class_v1](skin_10class_v1/MODEL_INFO.md) | Skin 10종 현미경, 현재 통합 후보 | B0·224·증강 | 0.9871314132 |
| [web_skin_5class_v1](web_skin_5class_v1/MODEL_INFO.md) | Web Skin 5종, 논문 강화 전 선정 이력 | B0·256·CE | 0.8813822928 |
| [web_skin_5class_v2](web_skin_5class_v2/MODEL_INFO.md) | Web Skin 5종, 현재 통합 후보 | PMG B0·256 | 0.9141049081 |
| [hair_5class_v1](hair_5class_v1/MODEL_INFO.md) | Hair 5종, 논문 강화 전 선정 이력 | B1·256·LS 0.05 | 0.7885764577 |
| [hair_5class_v2](hair_5class_v2/MODEL_INFO.md) | Hair 5종, 현재 통합 후보 | B1·384·LS 0.05·Adam | 0.8002222283 |
| [hair_6class_performance_v1](hair_6class_performance_v1/MODEL_INFO.md) | Hair 6종, 성능 우선 별도 후보 | B1·384·원본·LS 0.05 | 0.8091500773 |
| [hair_6class_light_v1](hair_6class_light_v1/MODEL_INFO.md) | Hair 6종, 실행 비용 대안 | B0·256·증강·CE | 0.8021613486 |

**버전 주의:** Hair 5종 `v1`/`v2`와 Hair 6종 `performance_v1`/`light_v1`은 분류 문제가 다르다. Hair 6종을 5종 모델의 가중치를 이어 학습한 `v3`으로 부르면 안 된다. `양호`는 Hair 6종 출력 인덱스 5다. Web Skin 5종은 별도 출력에 `정상`이 포함된다. 정확한 순서는 각 `class_names.json`을 따르고 인덱스를 임의로 정렬하지 않는다.

**입력:** 저장된 EfficientNet 후보는 각 `preprocessing.json`의 크기로 직접 resize한 RGB `float32` 0–255 픽셀을 받는다. 모델 안에 `Rescaling(1/255)`이 있으므로 외부에서 `/255`를 추가하지 않는다. PMG v2는 `preprocessing.json`에 기록된 **네 logits 합산 후 5-way softmax** 규칙을 적용해야 한다. 저장된 출력 점수를 임상적 정답 확률로 주장하지 않는다. 기기 종류와 촬영 부위로 Skin/ Web Skin/ Hair를 먼저 구분해야 한다.

복사본을 바로 실행할 때는 `.keras`만 건네지 말고 메타데이터와 [원래 모델 사용 안내](../../results/MODEL_USAGE.md), [선정 색인](../../results/PUBLIC_MODEL_SELECTION_20260928.json)을 함께 본다. `hair_6class_*`는 별도 공개 데이터 연구 후보이며 현재 5종 통합 버전을 자동으로 교체하지 않았다. MedSigLIP은 비교 실험이었고 최종 선정 모델이 아니므로 이 폴더에 포함하지 않았다. 초기 34개 모든 실험 체크포인트도 선정 모델 목록과는 별개다.
