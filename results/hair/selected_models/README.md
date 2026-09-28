# Hair 선정 모델

기존 팀 통합 경로의 Hair **5-class** 버전은 **v2**이다. Hair **6-class**는 별도 과제로 보존하며 자동 교체하지 않는다. 각 버전 폴더에는 실제 모델과 클래스 순서, 전처리 계약, 선정 이유가 들어 있다.

| 버전 | 상태 | 모델 | 입력 | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|---:|
| v1 | 논문 강화 전 최종 선정 | EfficientNet-B1 | 256 | 0.7883386581469649 | 0.7885764577048346 |
| v2 | 현재 | EfficientNet-B1 | 384 | 0.8003194888178914 | 0.8002222282821301 |

| 별도 6-class 폴더 | 역할 | 모델 | 입력 | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|---:|
| [6class_performance_v1](6class_performance_v1/MODEL_INFO.md) | 성능 우선 | EfficientNet-B1 | 384 | 0.7915708812260537 | 0.8091500773460419 |
| [6class_light_v1](6class_light_v1/MODEL_INFO.md) | 경량 대안 | EfficientNet-B0 | 256 | 0.7869731800766283 | 0.8021613486071133 |

- 새 코드 연결: `selected_models/v2` 사용
- 5-class 현재 버전 표식: `CURRENT_VERSION.txt`; 6-class 선정은 [종합 선정 목록](../../PUBLIC_MODEL_SELECTION_20260928.json)을 사용
- 6-class 성능 우선 폴더 표식: `CURRENT_6CLASS_VERSION.txt`; 경량 대안은 별도로 선택
- 모델 무결성 확인: 모델 옆 `.sha256` 파일 사용
- 과거 선정 근거 확인: 각 버전의 `MODEL_INFO.md` 사용
- 원본 ZIP·전체 보고서·해시 검증: 기존 `candidates/` 사용
- `candidates/`는 재현 보관소이므로 삭제하지 않는다.
