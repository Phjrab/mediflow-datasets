# 4차 추가 Source Test 완료 · 2026-11-11

평가 계획 `dda96d8dc716`의 조정 전 4차 신규 15개 평가와 이전 기본~3차 결과 36개를 확인했다. 재설정 4차는 포함하지 않는다. [결과 분석](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_TEST_RESULT_REVIEW_20261111.md) · [원본 비교표](test_comparison.csv) · [비교 그래프](plots/test_comparison.png).

| 분야 | 원본 | 기존 조명 | 기존 확장 | 4차 조명 | 4차 확장 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 피부 | 699/700 | 699/700 | 699/700 | 699/700 | 699/700 |
| 얼굴 | 368/400 | 363/400 | 362/400 | 367/400 | 354/400 |
| 두피 | 995/1252 | 996/1252 | 985/1252 | 990/1252 | 990/1252 |

피부는 기존 조건에서 기본~4차가 같았고, 얼굴은 원본·기존 조명에서 기본보다 정답 수가 늘었다. 얼굴 기존 확장은 3차보다 1장 적었고 두피는 기본보다 기존 세 조건 모두 낮았다. 새 4차 변환은 이전 모델 기록이 없으므로 기존 대비 개선 폭을 계산하지 않는다.

분야별 JSON·NPZ·test_rows는 이번 4차의 실제 추론 결과다. NPZ는 `allow_pickle=False`로 읽으며 같은 행의 점수·정답·경로를 대응시킨다. `test_comparison.csv`는 신규와 재사용 51행, `class_metrics.csv`는 340행이다. 재사용한 이전 JSON 5개는 중복 보관하지 않고 [경로·해시 참조](import_verification.json)에 연결했다. `evaluation_plan.json`, `code`, `logs`와 실제 평가 그래프는 원본 바이트로 보존했다. 학습 모델은 [완료한 4차 결과](../camera_adaptation_stage4_before_refit_20261111/README.md)에 있다.

ZIP 무결성, 계획·코드·입력 순서, 예측 배열 15개, 지표 51개와 CSV 전체를 검증했다. [기계 판독 요약](analysis_summary.json)을 보존했다. 원본 Test 사진을 로컬에서 다시 추론하지는 않았다. 실제 장비 질환 정답이 없어 실제 장비 Accuracy는 비어 있다. Validation의 후속 후보와 공식 배포 후보 색인은 변경하지 않았다.

Ruff 통과, pytest 200개 통과·3개 건너뜀. [최종 검증 기록](verification_summary.json).
