# Hair selected model v2

- 상태: **현재 사용 모델**
- 원본 후보 ID: `public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4`
- 구조: EfficientNet-B1
- 입력: 384×384 RGB float32, 0–255
- Loss: Categorical Crossentropy with Label Smoothing 0.05
- Validation Accuracy: `0.7963258785942492`
- Test Accuracy: `0.8003194888178914`
- Test Macro F1: `0.8002222282821301`
- 출력 처리: `direct_softmax`
- 모델 SHA-256: `9faa33b8f79f45ef6e7be415473e2ee5225f6195fff04d098828733db23ed3c2`

## 선정 이유

논문 기반 비교에서 고해상도 B1·384가 SupCon, DINOv2, EfficientNetV2-S, SAM과 앙상블보다 최종 후보로 적합했고, 작은 각질·피지·모낭 특징을 더 잘 보존해 현재 후보로 선정했다.

## 버전 설명

현재 버전이다. 실제 USB 현미경 환자 데이터에서는 아직 검증하지 않았다.

## 사용 파일

- 모델: `hair_model.keras`
- 클래스 순서: `class_names.json`
- 전처리 계약: `preprocessing.json`
- 구조화된 기록: `selection.json`

전체 원본 보고서와 배포 ZIP은 `results/hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4`에서 보존한다.
