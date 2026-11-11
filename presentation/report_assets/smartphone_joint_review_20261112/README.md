# 휴대폰 공동 유사성 검토

**후속 상태:** 사용자가 7항목 근접 후보의 탐색 학습을 승인해 추가 5차와 최종 Test까지 완료했다.
[전체 완료 정리](../../../docs/research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md)를 따른다.
아래 ‘미학습·미적용’은 검토 당시 기록이다. 모든 유사성 보호 기준 미통과 판정은 그대로 유지한다.

[해석·전체 수치](../../../docs/research/camera_adaptation/SMARTPHONE_JOINT_SIMILARITY_REVIEW_20261112.md)를 확인한다. 현재 학습 설정에 미적용인 검토 자료다.

196개 고유 후보를 작은 표본으로 점검하고 12개를 전체 Train 표본 118장·세 시드로 재생했다.
8개 중 7개 지표를 함께 가깝게 한 후보는 있었지만 모든 항목·개인별 기준을 함께 통과한 후보는 없었다.

- `review.json`: 고정 검토 기준·후보별 판정. `measurements.json`: 시드별 실제 측정.
- `distribution_summary.json`: 분위수·분산·설정용 참여자 간 차이.
- `findings.json`: 7항목 개선 후보와 악화 최소화 후보를 구분한 결과 해석.
- `broad_candidate_review_only.json`: 조명·약한 색 조절의 7항목 후보. 미채택·미학습·실행 설정 미적용.
- `review_candidate.json`: 가장 나빠진 항목의 증가를 최소화한 별도 후보. 현재 1차와 동일하며 최종 채택본이 아니다.
- `joint_findings_comparison.png`: 원본·현재 1/2/4차와 7항목 후보의 거리 비교.
- `joint_distance_comparison.png`, `joint_metric_distributions.png`: 대조군과 보수적 후보의 거리·분포.
- `verification.json`, `code/`: 원본 보존·측정 재계산·테스트·실행 환경과 분석 코드 기록.

이 자료는 학습 결과가 아니며 실행 ZIP·기존 모델·후보 색인을 바꾸지 않았다. 실제 촬영 참여자를 포함한
비교 사진은 `camera_capture/_review/smartphone_web_skin_20261111/joint_review_20261112`에 로컬 전용으로 보관한다.
