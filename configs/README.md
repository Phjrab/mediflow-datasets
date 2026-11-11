# 설정 파일 안내

기존 실행 기록이 참조하는 JSON 경로와 바이트는 유지한다. 준비된 파일이 있다는 사실과 실제 실행 완료를 구분한다.

| 파일 | 역할·상태 |
| --- | --- |
| [smartphone_web_skin_20261111.json](smartphone_web_skin_20261111.json) | 휴대폰 얼굴 실제 5회 완료 설정. 2·3차 약한 질감·흐림 / 4차 강한 변화 / P05 분리. [결과](../results/smartphone_web_skin_adaptation_20261112/README.md) |
| [domain_adaptation.json](domain_adaptation.json) | 초기 Source/Target 분석 도구 설정, 결과·당시 코드와 함께 확인 |
| [nine_experiments.json](nine_experiments.json) | 이전 9개 통합 실행안. 일부 실행 후 중단한 이력이며 현재 학습 지시 아님 |
| [camera_strength_20261110.json](camera_strength_20261110.json) | 이전 확대 후보의 분석·생성 기준. 완료한 4차 설정과 구분 |
| [camera_revised_20261111.json](camera_revised_20261111.json) | 실제 학습 완료한 조정 전 4차, 설정 492ee01afb45 |
| [adaptation_final_test_20261110.json](adaptation_final_test_20261110.json) | 기본~3차의 완료한 Source Test 고정 계획 |
| [adaptation_stage4_test_20261111.json](adaptation_stage4_test_20261111.json) | 완료한 4차 추가 Test 15개·이전 36개 재사용 계획, dda96d8dc716 |
| [archive](archive/README.md) | 실행하지 않기로 한 재설정 4차 e6bc85bc0a7d |

[프로젝트 정리](../docs/PROJECT_SUMMARY_20261111.md) · [완료 결과](../results/README.md).
