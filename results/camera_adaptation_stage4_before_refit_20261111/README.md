# 조정 전 4차 학습 결과 · 2026-11-11

설정 `492ee01afb45`의 세 분야 5 epoch 완료 결과다. 현재 재설정본 `e6bc85bc0a7d`와 구분한다.

[결과 분석](../../docs/research/camera_adaptation/ADAPTATION_STAGE4_BEFORE_REFIT_RESULT_REVIEW_20261111.md) · [원본 비교표](validation_comparison.csv) · [학습곡선](plots/learning_curves.png).

| 모델 | 실제 저장 파일 | 최적 epoch | 클래스 수 | 공통 Validation 선택 |
| --- | --- | ---: | ---: | --- |
| skin | [skin/best.keras](skin/best.keras) | 3 | 10 | 조정 전 4차 |
| web_skin | [web_skin/best.keras](web_skin/best.keras) | 3 | 5 | 조정 전 4차 |
| hair | [hair/best.keras](hair/best.keras) | 4 | 5 | 2차 |

피부·얼굴은 이번 모델이 선택됐고 두피는 기존 2차를 유지했다. 세 모델 파일은 실제 학습의 비교 근거로 모두 보존한다. 공식 후보 색인을 바꾸지 않았다.

분야별 `complete.json`은 3모델×4조건의 실제 지표·혼동행렬, `run_configuration.json`은 모델·클래스 순서·설정·환경, `expanded/history.json`은 5개 회차 기록이다. `study.json`의 바깥 객체는 실행 환경·설정 해시이고 내부 `study`가 실제 고정 설정이다. `code`는 실행 당시 코드 사본이다.

분할 목록은 중복 저장하지 않았다. 해당 분야의 `results/camera_adaptation_stage1_20261110/<분야>/source_manifest.json`을 사용하며 [원본과 같은 파일의 경로·해시](import_verification.json)에 연결돼 있다. ZIP에 모든 회차의 옵티마이저 체크포인트가 포함된 것은 아니므로 이 폴더만으로 임의 회차의 재학습을 이어갈 수 있다고 표현하지 않는다.

Source Validation 결과이며 Test·실제 장비 질환 Accuracy는 이번에 측정하지 않았다. 현재 저장 모델의 입력은 내부 Rescaling을 이용하는 RGB float32 0~255다. Web Skin은 네 PMG logit 합산 후 softmax를 사용한다. Skin·Hair 5클래스에는 정상 클래스가 없고 Web Skin에는 정상 클래스가 있다. 실제 참여자의 질환 정답은 미확인이다.

[가져오기 검증](import_verification.json) · [합성 입력 모델 로드 검사](model_load_verification.json) · [기계 판독 요약](analysis_summary.json).

프로젝트 검사: ruff 통과, pytest 195개 통과·3개 건너뜀(matplotlib 미설치). [최종 검증 기록](verification_summary.json).
