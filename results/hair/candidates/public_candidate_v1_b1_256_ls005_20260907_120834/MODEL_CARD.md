# MediFlow Hair public-data candidate v1

## 상태
실제 USB 현미경 데이터로 검증하기 전의 공개 데이터 후보입니다. 의료 진단 모델로 확정된 상태가 아닙니다.

## 모델
- EfficientNet-B1, ImageNet 사전학습
- 입력: 256×256 RGB float32, 픽셀 0–255
- 모델 내부 Rescaling(1/255), 외부 정규화 금지
- Label Smoothing 0.05
- Stage 1 15 Epoch, Stage 2 총 15 Epoch
- 클래스 순서: ['모낭사이홍반', '미세각질', '비듬', '탈모', '피지과다']

## 공개 데이터 결과
- Validation Accuracy: 0.777157
- Test Accuracy: 0.788339
- Macro F1: 0.788576

## 한계
실제 장비 검증, Domain Gap 분석과 범위 밖 입력 처리는 완료되지 않았습니다.
