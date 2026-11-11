# 휴대폰 얼굴 적응 결과 · 2026-11-12

후속 [최종 Source Test](../smartphone_final_test_20261112/README.md)와
[스마트폰 4차 패키징](../smartphone_adaptation_package_20261112/README.md)을 완료했다.
아래는 이전 1~4차 학습 수신 당시 기록이며 원본 결과는 유지했다.

사용자 Colab의 실제 5회 학습 결과를 보관했다. [전체 결과 분석](../../docs/research/camera_adaptation/SMARTPHONE_ADAPTATION_RESULT_REVIEW_20261112.md)을 먼저 확인한다.

- 1차 초기 B0·224와 선정 PMG·B0·256, 2·3·4차 각 5 epoch. 이후 출발 모델은 선정 PMG다.
- `summary.json`, `selection.json`: 완료 상태와 Source Validation 기준 4차 선택.
- `validation_comparison.json`: 적응 전~4차 × 원본·조명·약한 확장·강한 확장 20개 결과.
- `feature_and_prediction_diagnostics.json`: 실제 휴대폰 42장과 P05 10장의 특징 거리·예측 변화. 질환 정답 없음.
- 각 실행 폴더: `best.keras`, 최적 epoch, 전체 이력·실행 설정·학습곡선. 1차 두 모델도 비교를 위해 보존.
- `validation_comparison.png`, `validation_confusion_matrices.png`: 전달된 원본 그래프.
- `reproduction/`: 실제 실행 코드 72개·study·강도 계획·이전 분포 검토·runtime 파일 해시 목록.
- `receipt_verification.json`, `model_load_verification.json`: 수신·수치 대조와 실제 모델 로드 검사.

4차는 강한 변환 정답 370→416/499, 원본 424→419/499였다. 이 선택은 Validation 성능 기준이며 장비 유사성 최적이라는 뜻은 아니다.
새 Source Test와 실제 휴대폰 질환 Accuracy는 없다. 공식 후보와 기존 웹캠·현미경 패키지는 유지한다.

원본 다운로드 ZIP의 SHA-256은 `945099999a5dd5d9661e8e7b64d6a15ce6e6823350db3ab682e2a9f48adc6433`다.
ZIP 66개 항목을 바이트 그대로 보관하고 원본 ZIP·사진·키·중간 optimizer 가중치는 중복 저장하지 않았다.
모델은 RGB float32 0–255 입력이며 외부 `/255`를 추가하지 않는다. PMG의 네 logit을 합산한 뒤 softmax를 적용한다.
`reproduction`은 실행 코드 보관본이다. 전체 재현에는 기존 전용 실행 ZIP·Source 데이터·개인 키가 필요하다.
각 결과 폴더의 수치는 수정·반올림하지 않았다. 단독 배포 패키지는 이번 작업에서 만들지 않았다.
