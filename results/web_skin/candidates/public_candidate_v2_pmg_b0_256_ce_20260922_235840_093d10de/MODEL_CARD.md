# MediFlow Web Skin 공개 데이터 후보 v2

PMG / EfficientNet-B0 / 256 / Cross Entropy / Adam.

클래스 순서: ['건선', '아토피', '여드름', '정상', '주사']

Validation Accuracy: 0.85

Test Accuracy: 0.915

Test Macro F1: 0.9141049081029712

입력은 얼굴 피부 RGB float32 0~255를 256×256으로 resize합니다. 외부 /255는 금지합니다. 모델의 네 logit 출력을 직접 사용하지 말고 inference.py처럼 합산 후 softmax를 적용해야 합니다.

실제 웹캠 환자 검증과 범위 밖 입력 거부 기능은 포함되지 않습니다.
