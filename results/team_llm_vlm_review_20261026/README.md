# 팀원 LLM VLM 저장소 검토 근거

2026-10-26 공개 GitHub의 고정 커밋과 로컬 CPU 검사에 대한 기록이다.
분석은 [저장소 검토 보고서](../../docs/research/integration/TEAM_LLM_VLM_REPOSITORY_REVIEW_20261026.md),
후속 작업은 [실행 가이드](../../docs/research/integration/LLM_VLM_PROJECT_INTEGRATION_GUIDE_20261026.md)를 따른다.

- `repository_provenance.json`: 기본·개발 브랜치 SHA, 브랜치 차이와 고정 커밋의 원격 CI 조회.
- `source_inventory.json`: 검토한 텍스트 코드 snapshot의 경로·내용 해시. 원격 Git blob 해시와 로컬 파일 해시는 다른 종류다.
- `reported_metric_crosscheck.json`: 팀원의 공개 집계 CSV 정답 건수와 점수 대조. 원본 예측에서 정확도를 재계산한 것이 아니다.
- `team_reported_paired_metrics.csv`, `team_published_artifact_verification.json`: 고정 커밋에서 받은 팀원 공개 집계의 사본. 이번에 새 GPU 추론을 수행한 기록이 아니다.
- `medvlm_review_results.json`: 최신 계약·평가 검사와 약한 응답 검증 재현.
- `medvlm_active_training_review_results.json`: 최신 학습·평가 재현성 코드 검토와 선택 검사 결과.
- `forge_training_review_results.json`: Forge 학습·데이터 검사와 이미지 입력/행 분할 문제 재현.
- `forge_system_review_results.json`: Forge 인증 저장 검사와 export template 누락 재현.

동일한 테스트가 여러 subset에 포함될 수 있어 통과 수를 합산하지 않는다. Windows에서는
POSIX 파일 권한·`fcntl`·symlink 제한, 샌드박스에서는 HTTP loopback 연결 제한이 있었다.
각 제한과 원격 Ubuntu CI 결과를 구분한다. 이번에는 모델·학습 이미지·가중치를 다운로드하거나
GPU 학습·원격 서버 변경·Git push·PR 작성을 수행하지 않았다.
