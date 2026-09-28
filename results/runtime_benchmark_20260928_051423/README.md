# 저장 모델 34개 실행 시간 측정

2026-09-28 Colab 실행 결과를 로컬에 보존한 폴더다. Skin 4, Web Skin 12, Hair 5-class 14, Hair 6-class 4개 조건이 모두 `measured`로 기록됐다. **기존 저장 모델을 불러와 측정한 결과이며 재학습 결과가 아니다.**

| 파일 | 내용 |
|---|---|
| `runtime_benchmark_20260928_051423-20260928T053201Z-1-001.zip` | 사용자가 전달한 원본 ZIP. SHA-256 `253a3a0990cc5452efb6380461033503d544f1e9f3d52f75443a5c70c7cdc56a` |
| `benchmark.json`, `benchmark.csv` | 34개 조건의 상태, 기존 Validation 지표, 이번 실행 시간 측정값 |
| `environment.json` | 기록된 측정 환경과 반복 횟수. GPU 제품명은 포함되지 않음 |
| `skin_*`, `web_skin_*`, `hair5_*`, `hair6_*` PNG | 과제별 dashboard, accuracy–latency trade-off, 지표 개요. 총 12장 |

발표·보고서용 해석과 비교 시 주의점은 [검토 문서](../../docs/research/RUNTIME_BENCHMARK_REVIEW_20260928.md)를 본다. 특히 초기 모델과 정제 후 모델, Hair 5-class와 6-class의 Validation Accuracy를 같은 과제로 취급하면 안 된다. 지연 시간은 Colab 환경에서 측정됐으며 실제 장비 속도가 아니다.
