# 어댑테이션 조절값 · 2026-11-11

**시점 기록:** 이 폴더의 휴대폰 수치는 11월 11일 계획이다. 11월 12일 사용자 요청으로
[질감·흐림 강도](../../../docs/research/camera_adaptation/SMARTPHONE_APPEARANCE_STRENGTH_20261112.md)를 추가했다.
이전 검사 기록은 작성 시점의 원본 대조 결과이며, 현재 설정의 재검증으로 해석하지 않는다.

[해설·전체표](../../../docs/research/camera_adaptation/ADAPTATION_EXPERIMENT_PARAMETER_LEDGER_20261111.md)에서 단위와 해석을 먼저 확인한다.

- `MediFlow_Adaptation_Parameters_20261111.xlsx`: 발표·보고서용 단계별 설정, 학습 기록, 숫자 원본·출처.
- `ADAPTATION_EXPERIMENT_PARAMETERS_20261111.json`: 전체 정밀도, 16개 설정·20개 완료/예정 기록·58개 출처 해시.
- `ADAPTATION_EXPERIMENT_PARAMETERS_20261111.csv`: 변수 하한·상한 184행. Excel에서 바로 열 수 있다.
- `ADAPTATION_TRAINING_RUNS_20261111.csv`: 완료 15회·휴대폰 예정 5회, 출발 모델·최적 epoch·체크포인트 해시.
- `verification.json`: 원본 대조·파일 보존·워크북 내용 및 그림 확인 결과.

원본 이미지·모델·결과 수치는 바꾸지 않았다. 현재 휴대폰은 학습 전이며 MMD·CORAL 계수와 최적 epoch는 미계산이다. 설정값의 간단한 화면 표시는 전체 정밀도를 생략할 수 있으므로 정확한 인용은 JSON·CSV를 사용한다. 성능 수치는 재작성하지 않았다.
