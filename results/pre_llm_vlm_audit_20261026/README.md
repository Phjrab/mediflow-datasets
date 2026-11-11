# LLM과 VLM 착수 전 로컬 점검 근거

실행일: 2026-10-26. [점검 보고서](../../docs/research/integration/PRE_LLM_VLM_PROJECT_AUDIT_20261026.md)의 근거다.

- `artifact_integrity.json`: 일곱 선정 모델의 실제 SHA-256, 원본 ZIP·발표 사본·클래스·전처리,
  Hair 6종 두 후보의 저장 예측 검증, 34개 측정 상태, 노트북과 주요 문서·발표 목록 확인.
- `saved_evaluation_recheck.json`: 초기 6개 보고서 및 현재 세 후보의 저장 예측·지표 대조.
- `selected_model_smoke.json`: TensorFlow 2.20.0/Keras 3.13.2, 로컬 CPU에서 선정 모델
  7개 × 예시 사진 2장의 실제 추론 기록. 새 Test 성능 측정이나 임상 검증이 아니다.
- `check_summary.json`, `pytest.log`, `pytest.xml`: 정적 검사·의존성 검사와 자동 테스트 결과.
  최종 128개 통과, 3개 건너뜀, 실패·오류 0개. `matplotlib` 미설치로 공통 노트북 검사 3개를 건너뛰었다.
- `audit_source.py.txt`: 이번 점검에 사용한 보조 코드의 원문 사본. 설치형 추론 명령이 아니며,
  같은 출력 폴더가 이미 있으면 덮어쓰지 않는다. SHA-256은 `artifact_integrity.json`에 기록했다.

이 폴더에는 원본 이미지·모델 사본을 추가하지 않았다. 기존 모델·평가 기록은 원래 위치에 보존한다.
