# Web Skin 공개 데이터 후보 패키지 검증 완료

2026-09-08. ROADMAP 7 중 공개 데이터 후보 패키지의 수령·로컬 보관·모델 실행 검증을 완료했다. 실제 웹캠 검증과 시스템 모델 교체는 수행하지 않았다.

## 보관 위치

- 원본 ZIP 복사본: [public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e.zip](../../../results/web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e.zip)
- 압축 해제본: `results/web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e/`
- 모델: 위 폴더의 `web_skin_model.keras`
- 로컬 확인 기록: `results/web_skin/candidates/public_candidate_v1_b0_256_ce_20260908_081603_3f1ce76e_local_verification.json`

다운로드 원본, 기존 original/augmented 모델 및 학습 결과는 수정하지 않았다. 패키지 원본에 검증 결과를 덧붙이지 않고 별도 기록을 작성했다.

## 확인 결과

| 검사 | 결과 |
|---|---|
| ZIP 크기 | 27168528 bytes |
| ZIP 내 파일 | 35개, CRC 오류 없음 |
| manifest 무결성 목록 | 34개 파일 SHA-256 모두 일치 |
| 기존 실험 보고서 대조 | 27개 모두 바이트/해시 일치 |
| 모델 해시 | 검토 당시 선정 체크포인트와 일치 |
| 클래스 순서 | 건선, 아토피, 여드름, 정상, 주사 |
| 실제 모델 로드 | 성공, EfficientNet-B0 |
| 입력 | 256×256×3, RGB float32 0~255 |
| 내부 정규화 | Rescaling(1/255) 확인, 외부 /255 금지 |
| 출력 | 5개, 가상 입력에서 유한한 0~1 점수와 합 확인 |
| 파라미터 수 | 4055976 |
| 로컬 ZIP 복사 | 원본 SHA-256 일치 |

가상 입력 검사는 실행 가능 여부만 확인한다. 실제 사진 재추론, Test 성능 재평가, 재학습은 하지 않았다. 저장된 성능은 이전 Colab 평가의 수치를 그대로 보존한 것이다.

## 후보 식별

- EfficientNet-B0 / 256 / Categorical Crossentropy.
- Stage 1 15 Epoch, Stage 2 10 Epoch 중 최고 8번째 체크포인트.
- Validation Accuracy: 0.796.
- Test Accuracy: 0.8825 (400장 중 353장 정답).
- Test Macro F1: 0.8813822927981046.
- 상태: `public_data_candidate_v1_not_device_validated`.

ZIP SHA-256:
`980f92301efdbd0f65971a157620114688ab4368bd4511e112038d8b6dbc1694`

모델 SHA-256:
`d4c834a7fd47480c1c45ddbdc09257942b38b2408d5b28effc34ee68c4fad97b`

## 다음 단계

공개 데이터 후보를 다시 만들거나 학습할 필요는 없다. 추후 통합 시 이 패키지의 모델 파일과 class_names.json, preprocessing.json, package_manifest.json을 함께 사용한다. 기존 추론 코드가 자동으로 이 후보를 선택하도록 변경한 것은 아니다.

실제 웹캠 데이터가 확보되면 별도 검증한다. 사람·병변·촬영 세션 단위 겹침, 범위 밖 입력 거부, 점수 보정의 미검증 한계는 유지한다. 현재 패키지 검증·보관을 위해 사용자가 추가로 할 일은 없다.
