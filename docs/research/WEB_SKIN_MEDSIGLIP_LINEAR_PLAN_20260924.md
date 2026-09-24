# Web Skin MedSigLIP-448 Linear Probe 실험 계획

기준일: 2026-09-24  
상태: **Validation 실행 완료, 미채택**  
결과: [`WEB_SKIN_MEDSIGLIP_LINEAR_RESULT_20260924.md`](WEB_SKIN_MEDSIGLIP_LINEAR_RESULT_20260924.md)

## 1. 실험 목적

현재 Web Skin 배포 후보 v2는 `PMG·EfficientNet-B0·256·CE`이다. 고정 Validation
500장에서 Accuracy `0.85`, Macro F1 `0.8473279632397033`을 기록했고, 후보 고정 후 Test
400장에서 Accuracy `0.915`, Macro F1 `0.9141049081029712`을 기록했다. 이 모델과 결과는
수정하지 않는다.

이번 실험은 **의료 이미지로 사전학습된 MedSigLIP-448의 특징이 기존 얼굴 피부 5개 클래스를
더 잘 분리하는가**라는 한 가지 가설을 확인한다. 아직 새 성능은 측정되지 않았으므로 성능 향상을
완료된 사실로 표현하지 않는다.

## 2. 근거

Google의 공식 모델 카드에 따르면 MedSigLIP은 의료 이미지와 텍스트를 같은 embedding 공간에
배치하도록 학습된 SigLIP 계열 모델이다. 피부과 이미지를 포함한 여러 의료 영상으로 사전학습됐고,
448×448 입력을 사용한다. Google은 데이터 효율적인 분류, zero-shot 분류와 검색을 권장 활용으로
제시한다.

공식 공개 평가에서 Dermatology Skin Conditions 1,612개·79개 클래스의 Linear Probe AUC는
`0.881`로 보고됐다. 이 수치는 MediFlow의 Accuracy나 Macro F1과 데이터·지표가 다르므로 직접
비교하거나 MediFlow의 예상 성능으로 사용하지 않는다. 다만 **고정 embedding 위에 작은 분류기를
학습하는 Linear Probe 자체가 공식 평가된 방법**이라는 실험 근거로 사용한다.

- 모델 카드: https://huggingface.co/google/medsiglip-448
- 공식 시작 안내: https://developers.google.com/health-ai-developer-foundations/medsiglip/get-started
- 공식 fine-tuning 예제:
  https://github.com/google-health/medsiglip/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
- MedGemma Technical Report: https://arxiv.org/abs/2507.05201

## 3. 실험 설계

| 구분 | 고정 또는 변경 내용 |
|---|---|
| 연구 가설 | MedSigLIP 의료 embedding이 Web Skin 클래스의 선형 분리 가능성을 높인다 |
| 데이터 | 기존 clean ZIP SHA-256 `f8908af3...964072d` |
| 학습 | Augmented Train 7,200장 |
| 선정 | Original Validation 500장 |
| Test | 읽거나 평가하지 않음 |
| 클래스 순서 | 건선, 아토피, 여드름, 정상, 주사 |
| seed | 42 |
| 변경 변수 | PMG·ImageNet EfficientNet 대신 frozen MedSigLIP-448 embedding 사용 |
| 분류기 | Linear 층 하나, AdamW, 50 epoch |
| 필요한 동반 변경 | MedSigLIP 공식 processor와 448×448 입력 |
| 선정 기준 | Validation Macro F1 우선, 정확히 같으면 Accuracy |

MedSigLIP의 400M vision encoder는 고정한다. 전체 미세조정은 첫 실험에서 하지 않는다. 따라서
측정 결과는 “MedSigLIP을 완전히 학습한 성능”이 아니라 **고정된 의료 특징의 Linear Probe
성능**이다.

## 4. 재개와 산출물

7,700장의 embedding은 128장 단위로 Drive 실행 폴더에 저장한다. Colab 연결이 끊기면 같은
결과 폴더를 `RESUME_DIR`에 입력하여 검증된 shard를 재사용한다. 토큰은 모델 다운로드에만
사용하고 결과에 기록하지 않는다.

실행 후 다음 파일이 생성된다.

- `medsiglip_linear_head.pt`: 작은 Linear 분류기와 입력 계약
- `validation_metrics.json`, `validation_predictions.csv`
- `linear_history.json`
- `medsiglip_training_curves.png`
- `medsiglip_validation_confusion_matrix.png`
- `medsiglip_vs_pmg_validation.png`
- `medsiglip_validation_summary.json`
- 모델·지표·소스가 포함되고 embedding cache는 제외된 보고서 ZIP

## 5. 판단과 후속 단계

새 모델이 사전 선언한 Validation 선정 기준을 통과해도 현재 v2를 즉시 교체하지 않는다. 그때
별도의 최종 노트북을 만들어 고정된 MedSigLIP encoder와 Linear head 하나만 Test에서 한 번
평가하고, 추론 비용·모델 크기·Hugging Face 이용 조건까지 PMG v2와 비교한다. 기준을 넘지 못하면
실패 결과도 보존하고 현재 v2를 유지한다.

## 6. 제한

- 사람·병변·촬영 세션 단위 누수는 현재 메타데이터로 확인하지 못했다.
- 공개 데이터가 MedSigLIP 사전학습 자료와 직간접적으로 겹쳤는지 완전히 확인할 수 없다.
- 현재 Validation은 실제 웹캠 장비 검증을 대신하지 않는다.
- 점수는 실제 질환 확률이나 임상 진단 확률로 해석하지 않는다.
