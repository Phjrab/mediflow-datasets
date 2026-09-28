# Hair selected model v1

- 상태: **논문 강화 전 최종 선정 모델 — 이후 v2로 교체**
- 원본 후보 ID: `public_candidate_v1_b1_256_ls005_20260907_120834`
- 구조: EfficientNet-B1
- 입력: 256×256 RGB float32, 0–255
- Loss: Categorical Crossentropy with Label Smoothing 0.05
- Validation Accuracy: `0.7771565318107605`
- Test Accuracy: `0.7883386581469649`
- Test Macro F1: `0.7885764577048346`
- 출력 처리: `direct_softmax`
- 모델 SHA-256: `0842822f6b6299ee7a0cb6b6925681d54705e7e26b3682dee0f1c49f8e72057b`

## 선정 이유

논문 기반 강화 실험을 시작하기 전, 해상도·Loss·Backbone·미세조정 조건을 비교했다. B1·256·Label Smoothing의 미세조정 연장 결과가 가장 높은 Validation Accuracy를 기록해 당시 최종 Hair 후보로 선정한 뒤 Test 평가와 패키징까지 완료했다.

## 버전 설명

논문 강화 전 최종 선정 모델이다. 이후 B1·384가 Validation과 고정 Test 성능을 높여 v2로 교체됐지만, v1도 당시 정식 선정·패키징된 후보로 보존한다.

## 사용 파일

- 모델: `hair_model.keras`
- 클래스 순서: `class_names.json`
- 전처리 계약: `preprocessing.json`
- 구조화된 기록: `selection.json`

전체 원본 보고서와 배포 ZIP은 `results/hair/candidates/public_candidate_v1_b1_256_ls005_20260907_120834`에서 보존한다.
