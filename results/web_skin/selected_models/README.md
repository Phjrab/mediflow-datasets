# Web Skin 선정 모델

현재 팀 통합에 사용할 버전은 **v2**이다. 각 버전 폴더에는 실제 모델과 클래스 순서, 전처리 계약, 선정 이유가 들어 있다.

| 버전 | 상태 | 모델 | 입력 | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|---:|
| v1 | 논문 강화 전 최종 선정 | EfficientNet-B0 | 256 | 0.8825 | 0.8813822927981046 |
| v2 | 현재 | PMG with EfficientNet-B0 | 256 | 0.915 | 0.9141049081029712 |

- 새 코드 연결: `selected_models/v2` 사용
- 현재 버전 표식: `CURRENT_VERSION.txt`
- 모델 무결성 확인: 모델 옆 `.sha256` 파일 사용
- 과거 선정 근거 확인: 각 버전의 `MODEL_INFO.md` 사용
- 원본 ZIP·전체 보고서·해시 검증: 기존 `candidates/` 사용
- `candidates/`는 재현 보관소이므로 삭제하지 않는다.
