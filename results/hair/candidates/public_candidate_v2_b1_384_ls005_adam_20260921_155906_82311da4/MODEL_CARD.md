# MediFlow Hair 공개 데이터 후보 v2

EfficientNet-B1 / 384 / Label Smoothing 0.05 / Adam.

클래스 순서: ['모낭사이홍반', '미세각질', '비듬', '탈모', '피지과다']

Validation Accuracy: 0.7963258785942492

Test Accuracy: 0.8003194888178914

Test Macro F1: 0.8002222282821301

입력은 두피 RGB float32 0~255를 384×384로 resize합니다. 외부 /255는 금지합니다.
범위 밖 입력 거부 기능과 실제 USB 현미경 환자 검증은 포함되지 않습니다.
