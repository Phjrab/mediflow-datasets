# Hair MedSigLIP-448 Linear Probe 실험 계획

기준일: 2026-09-24  
상태: **Colab 실행·Validation 분석 완료, 기준 모델보다 낮아 미채택**

실측 결과와 최종 판단은
[`HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md`](HAIR_MEDSIGLIP_LINEAR_RESULT_20260924.md)에
기록했다.

## 1. 실험 목적

현재 Hair 배포 후보 v2는 `EfficientNet-B1·384·Label Smoothing 0.05·Adam`이다. 고정
Validation 1,252장에서 Accuracy `0.7963258785942492`, Macro F1
`0.7957146810115047`을 기록했고, 후보 고정 후 Test에서 Accuracy `0.8003194888178914`,
Macro F1 `0.8002222282821301`을 기록했다. 이 모델과 결과는 수정하지 않는다.

이번 실험은 **의료 이미지로 사전학습된 MedSigLIP-448의 특징이 두피 5개 클래스를 더 잘
분리하는가**라는 한 가지 가설을 확인했다. Validation Accuracy `0.7699680511182109`,
Macro F1 `0.7687534259685547`로 현재 Hair v2보다 낮았으므로 Test와 패키징은 진행하지 않았다.

## 2. 근거와 적용 범위

Google의 공식 모델 카드에 따르면 MedSigLIP은 피부과 영상을 포함한 의료 이미지·텍스트 쌍으로
사전학습됐으며, 고정 embedding 위에 분류기를 학습하는 Linear Probe를 공식 평가 방법으로
제시한다.

하지만 공개 설명의 피부과 영상은 임상 사진과 피부경 사진 중심이며 MediFlow Hair의 USB
현미경 두피 영상과 같은 촬영 방식이라고 볼 근거는 없다. Web Skin 실험에서도 frozen Linear
Probe가 기존 PMG v2보다 낮았다. 따라서 Hair 성능을 예상하지 않고 자체 Validation에서만
전이 가능성을 검증한다.

- 모델 카드: https://huggingface.co/google/medsiglip-448
- 공식 시작 안내: https://developers.google.com/health-ai-developer-foundations/medsiglip/get-started
- 공식 fine-tuning 예제:
  https://github.com/google-health/medsiglip/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
- MedGemma Technical Report: https://arxiv.org/abs/2507.05201

## 3. 실험 설계

| 구분 | 고정 또는 변경 내용 |
|---|---|
| 연구 가설 | MedSigLIP 의료 embedding이 Hair 클래스의 선형 분리 가능성을 높인다 |
| 데이터 | 기존 Hair clean ZIP SHA-256 `2ac726...e156` |
| 학습 | Augmented Train 15,047장 |
| 선정 | Original Validation 1,252장 |
| Test | 읽거나 평가하지 않음 |
| 클래스 순서 | 모낭사이홍반, 미세각질, 비듬, 탈모, 피지과다 |
| seed | 42 |
| 변경 변수 | ImageNet EfficientNet-B1 대신 frozen MedSigLIP-448 embedding 사용 |
| 분류기 | Linear 층 하나, AdamW, 50 epoch |
| 필요한 동반 변경 | MedSigLIP 공식 processor와 448×448 입력 |
| 선정 기준 | Validation Macro F1 우선, 정확히 같으면 Accuracy |

기존 Hair v2를 다시 학습하지 않고 저장된 동일 Validation 지표를 기준값으로 사용한다.
MedSigLIP encoder 전체는 고정하므로 결과는 full fine-tuning 성능이 아니라 frozen embedding의
Linear Probe 성능이다.

## 4. 실행 시간과 재개

처리 대상은 Train과 Validation을 합쳐 16,299장으로 Web Skin의 7,700장보다 약 2.1배 많다.
Web Skin L4 실행이 약 10분 41초였지만 파일 I/O와 GPU 상황이 달라 Hair 시간을 정확히 두 배로
단정하지 않는다.

embedding은 128장 단위로 Drive 실행 폴더에 저장한다. Colab 연결이 끊기면 같은 결과 폴더를
`RESUME_DIR`에 입력하여 검증된 shard를 재사용한다. 토큰 값은 결과에 저장하지 않는다.

## 5. 산출물과 판단

- `medsiglip_linear_head.pt`
- `validation_metrics.json`, `validation_predictions.csv`
- `linear_history.json`
- 학습곡선, 혼동행렬, 기존 Hair v2 비교 그림
- 실행 설정, 환경과 소스
- embedding cache를 제외한 보고서 ZIP

새 모델이 Validation 기준을 넘더라도 현재 v2를 즉시 교체하지 않는다. 별도 최종 노트북에서
고정 후보 하나만 Test 평가하고, 448 입력·0.9B 기반 모델 다운로드와 추론 비용까지 비교한다.
기준을 넘지 못하면 실패 결과를 보존하고 현재 Hair v2를 유지한다.

## 6. 제한

- 사람·촬영 세션 단위 누수는 현재 메타데이터로 확인하지 못했다.
- 공개 데이터가 MedSigLIP 사전학습 자료와 관련됐는지 완전히 확인할 수 없다.
- 공개 Validation은 실제 USB 현미경 장비 검증을 대신하지 않는다.
- 점수는 실제 질환 확률로 해석하지 않는다.
