# MediFlow 공개 데이터 모델 선정 — 성능과 실행 비용 종합 (2026-09-28)

## 선정 결과

이 문서의 **최종 선정**은 현재 보유한 공개 데이터와 Colab 측정에 대한 후보 고정이다. 실제 카메라·USB 현미경 데이터 또는 최종 하드웨어에서 검증한 운영 모델 확정을 뜻하지 않는다. 기계가 읽는 경로와 원본 수치는 [PUBLIC_MODEL_SELECTION_20260928.json](PUBLIC_MODEL_SELECTION_20260928.json)에 기록했다.

| 분류 과제 | 선정 역할 | 모델·입력 | Validation Macro F1 | Test Macro F1 | 이번 End-to-end p50 | 저장 모델 크기 |
|---|---|---|---:|---:|---:|---:|
| Skin 10-class | 유지 | [B0·224·증강 v1](skin/selected_models/v1/MODEL_INFO.md) | 0.9820008885656606 | 0.9871314132317626 | 282.2758660004183 ms | 17,198,406 bytes |
| Web Skin 5-class | 유지 | [PMG B0·256 v2](web_skin/selected_models/v2/MODEL_INFO.md) | 0.8473279632397033 | 0.9141049081029712 | 304.2439384998943 ms | 36,377,179 bytes |
| Hair 5-class | 별도 과제의 기존 선정본 | [B1·384 v2](hair/selected_models/v2/MODEL_INFO.md) | 0.7957146810115047 | 0.8002222282821301 | 407.29406550008207 ms | 47,219,220 bytes |
| Hair 6-class | **성능 우선 선정본** | [B1·384·원본 v1](hair/selected_models/6class_performance_v1/MODEL_INFO.md) | 0.8144942091318823 | 0.8091500773460419 | 409.83297050024703 ms | 47,234,592 bytes |
| Hair 6-class | 경량 대안 | [B0·256·증강 v1](hair/selected_models/6class_light_v1/MODEL_INFO.md) | 0.7819142733310591 | 0.8021613486071133 | 288.73898249958074 ms | 29,005,042 bytes |

Accuracy와 p95를 포함한 모든 원시 수치는 [34개 측정 기록](runtime_benchmark_20260928_051423/benchmark.csv)과 각 후보 패키지의 평가 JSON에 있다. Hair 5-class와 6-class는 출력 클래스와 Test 세트가 다른 과제이므로 두 행의 정확도·F1을 개선 전후 수치처럼 빼거나 순위를 매기지 않는다. Skin, Web Skin, Hair 사이의 정확도도 서로 다른 과제의 점수다.

## 선정 판단

**Skin:** 같은 정제 데이터의 원본 모델보다 증강 모델의 Validation Accuracy와 Macro F1이 높고, 이번 측정에서 두 모델의 지연 시간은 거의 같았다. 기존 B0·224·증강 후보를 유지한다. 초기 데이터 모델과 정제 후 모델의 정확도는 분할이 달라 직접 비교하지 않는다.

**Web Skin:** PMG B0·256은 일반 B0·256·CE보다 Validation Accuracy가 `0.054`, Macro F1이 `0.0557884984830505` 높으면서 이번 지연 시간 증가는 `21.4067715005513 ms`였다. PMG B1·384는 B0·256 PMG보다 Validation Accuracy가 `0.016` 더 높지만 지연 시간은 `115.6614559999980 ms` 더 길었다. 분류 성능과 실행 비용을 함께 고려한 기존 PMG B0·256 선정을 유지한다. PMG B1·384는 Validation 성능 선두 연구 조건으로 보존한다.

**Hair 5-class:** B1·384 Adam은 B1·256 기준선보다 Validation Accuracy가 `0.0207667731629393` 높았고, 이번 지연 시간은 `8.20166799940127 ms` 길었다. 같은 B1·384 SAM은 Accuracy가 같고 Macro F1은 낮으며 지연 시간은 더 길었다. 기존 v2를 유지한다. 이 판단은 5-class 과제에만 적용한다.

**Hair 6-class:** 네 조건의 동일한 Validation에서 B1·384·원본이 Macro F1 `0.8144942091318823`으로 선두였으므로 **분류 성능 우선 후보**로 선정한다. B0·256·증강은 두 B0 조건 중 Validation Macro F1이 더 높아 **경량 대안**으로 둔다. 두 조건의 같은 Test에서 Macro F1은 각각 `0.8091500773460419`, `0.8021613486071133`이다. 이번 실행 지연 시간은 각각 `409.83297050024703 ms`, `288.73898249958074 ms`다. 따라서 정확도 우선과 실행 비용 우선이 서로 다른 선택을 이끈다. 두 후보는 구조·해상도·Loss·학습 이미지 종류가 함께 달라 차이를 한 요인에 귀속할 수 없다. Test는 Validation으로 후보를 고정한 뒤 평가했으며, 이번 실행 시간 결과만을 이유로 Test에 맞춰 새 모델을 선정하지 않았다.

MedSigLIP은 기본 인코더와 분류층을 모두 포함해 시간을 측정했지만, 이번 Validation에서는 각각 해당 과제의 선정 모델보다 낮았고 로드된 가중치 크기가 컸다. 짧은 Colab 지연 시간만으로 후보를 교체하지 않는다. [측정 해석 문서](../docs/research/RUNTIME_BENCHMARK_REVIEW_20260928.md)에 그래프별 해석과 측정 한계를 적었다.

## 파일 사용 기준

- 팀 통합의 **기존 3개 기본 경로**는 [CANDIDATE_INDEX.json](CANDIDATE_INDEX.json)에 그대로 둔다. 그중 `hair`는 **5-class** 모델이다. 이 파일을 6-class 모델 경로로 바꾸면 기존 5-class 출력 계약과 재현 코드가 달라지므로 자동 교체하지 않았다.
- Hair 6-class를 사용하는 작업은 [성능 우선](hair/selected_models/6class_performance_v1/) 또는 [경량](hair/selected_models/6class_light_v1/) 폴더 중 하나를 **명시적으로 선택**한다. 두 폴더 모두 `hair_model.keras`, `class_names.json`, `preprocessing.json`, `selection.json`, 모델 SHA-256 파일을 갖는다.
- 두 6-class 모델의 원본 배포 ZIP과 전체 평가 기록은 [hair/candidates](hair/candidates/)에 그대로 보존한다. 기존 5-class v1/v2 모델과 연구 실험 결과도 이동·삭제하지 않았다.
- 입력은 각 폴더의 `preprocessing.json`, 출력 순서는 `class_names.json`을 따른다. EfficientNet 모델의 내부 `Rescaling(1/255)` 때문에 외부 `/255`를 적용하지 않는다. Web Skin PMG는 네 logit 출력을 합산한 뒤 softmax를 적용한다. 자세한 사용 계약은 [MODEL_USAGE.md](MODEL_USAGE.md)를 따른다.

## 남은 검증

이번 지연 시간은 제품명이 기록되지 않은 Colab GPU에서 합성 이미지 한 장으로 측정됐다. 실제 기기에서의 지연 시간, 최대 메모리, 전력, 카메라 입력 처리와 현장 사진 정확도는 아직 확인되지 않았다. Hair 6-class 데이터는 기존 사진의 사람·촬영 세션 단위 겹침과 신규 `양호` 원천 라벨의 독립 검증이 남아 있다. 따라서 Hair 6-class 성능 우선 후보를 **연구상 1순위**로 두되, 운영 모델로 확정했다는 표현은 쓰지 않는다.
