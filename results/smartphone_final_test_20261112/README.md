# 스마트폰 얼굴 · 최종 Source Test 완료 결과

원본 400장, 적응 전·1~5차 6모델 × 5조건 = 30개 비교를 완료했다.
Validation 선정 4차를 유지했고 Test로 모델·강도를 다시 고르지 않았다.

[분석 보고서](../../docs/research/camera_adaptation/SMARTPHONE_FINAL_TEST_RESULT_REVIEW_20261112.md).

| 찾을 것 | 파일 |
| --- | --- |
| 전체 원본 지표 | [test_results.json](test_results.json) |
| 고정 평가 계획 | [evaluation_plan.json](evaluation_plan.json) |
| 완료·실행 환경 | [summary.json](summary.json) |
| 비교표·클래스 지표 | [비교 CSV](test_comparison.csv), [클래스 CSV](class_metrics.csv) |
| 개별 예측 12,000행 | [predictions.csv](predictions.csv) |
| 모델 간 정오답 변경 검산 | [paired_prediction_changes.json](paired_prediction_changes.json) |
| 점수 배열·조건별 지표 | [scores](scores), [metrics](metrics) |
| 그래프·모델별 혼동행렬 | [plots](plots) |
| 실행 코드·고정 분할 | [reproduction](reproduction) |
| 수신·재계산·보존 검사 | [receipt_verification.json](receipt_verification.json) |

4차 원본 367/400·강한 변환 355/400, 적응 전은 각각 366/400·327/400이다.
이는 가상 변환을 적용한 기존 Source Test이며 실제 휴대폰 질환 성능은 정답이 없어 미측정이다.
모델·사진·복호화 키를 이 결과 폴더에 복사하지 않았다. 원본 ZIP은 Downloads에 그대로 보존했다.
