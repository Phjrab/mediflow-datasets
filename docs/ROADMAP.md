# MediFlow 실행 로드맵

이 문서는 `docs/research/PROJECT_BACKGROUND.md`의 연구 방향을 실제 개발 단계와 완료 기준으로
구체화한다. 한 번에 여러 변수를 바꾸지 않고 각 단계의 입력, 설정, 결과를 기록한다.

공개 데이터 기반 논문 실험(8단계)과 Hair 6클래스 데이터·후보 실험(9단계 일부)을 완료했다.
9단계의 원천 라벨·사람 단위 검증은 남아 있다. 12단계의 저장 모델 34개 Colab 실행 시간
측정과 공개 데이터 후보별 종합 선정은 완료했으며, GPU 제품명·최대 메모리·실제 장비 측정은
남아 있다. 다음 연구는 11단계 세 분류기별 모델 구조 비교와 12단계 장비 측정이다. 10단계 시스템 통합은
보류한다. Skin, Web Skin, Hair 최종 후보와 사용 계약은
`results/CANDIDATE_INDEX.json`, 전체 과정과 논문 근거는
`docs/research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md`를 기준으로 한다. 실제 장비 사진을
이용한 2·3단계와 시스템 통합 10단계는 별도 범위로 남겨둔다.

## 2026-09-14 이후 우선 방향

2026-09-28 방향 갱신: 실제 장비 연결과 LLM/VLM 통합은 현재 작업에서 보류한다. 기존 공개 데이터
후보 모델은 그대로 보존한다. Hair 정상 데이터의 출처·촬영 방식·라벨을 검증한 뒤 별도의 6-class
데이터 버전을 만들고, 세 도메인에서 각각 모델 구조를 비교한다. 마지막으로 선정 후보의 실행 비용을
같은 측정 환경에서 비교한다. 앞서 완료한 논문 기반 학습법 실험의 설계 기록은
[`PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md`](research/PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md)에 보존한다.

## 1. 현재 모델·평가 재현

- [x] TensorFlow 2.20.0 / Keras 3.13.2 환경 고정
- [x] original/augmented 모델 6개 로드 경로 정의
- [x] 모델 입력, 출력 차원, 클래스 순서, 내부 정규화 자동 검증
- [x] 단일 이미지 추론 CLI 검증
- [x] 초기 Original/Augmented 모델 6개 로드와 기존 클래스별 보고서·요약의 내부 일치성 확인
- [x] 현재 선정 후보 3개의 저장 예측에서 Accuracy·Macro F1·혼동행렬 재계산 및 모델 해시 확인
- [ ] 정제 전 원래 Test 사진으로 초기 모델 6개 추론 재실행 — 원본 데이터셋 미보관으로 수행 불가

2026-09-27 점검 기록: [저장 모델·평가 기록 재현 점검](research/EVALUATION_REPRODUCTION_AUDIT_20260927.md).
초기 평가의 수치와 저장 결과는 대조했지만, Test 사진 자체가 없어 원래 추론 결과를
새로 생성하지 못했다. 정제 후 Test로 대체하면 같은 평가가 아니므로 재현 완료로 표시하지 않는다.

완료 기준: 동일 Test 데이터에서 Accuracy, Macro F1, 클래스별 지표와 Confusion Matrix를
재계산하고 기존 결과와 차이가 있으면 원인을 설명한다.

## 2. 실제 장비 소규모 검증 데이터 확보

- 사람, 부위, 촬영 세션 단위의 익명 표본 ID 부여
- 웹캠 얼굴, 현미경 피부, 현미경 두피를 별도 프로토콜로 촬영
- 장비, 배율, 조명, 거리, 해상도, 촬영일 조건 기록
- 학습과 튜닝에 사용하지 않는 최초 검증 세트로 보존

완료 기준: 모델별 클래스와 촬영 조건을 갖춘 검증 세트 및 메타데이터 표가 존재한다.

## 3. Domain Gap 분석

- 공개 Test와 실제 장비 데이터의 Accuracy, Macro F1, 클래스별 Recall 비교
- Confusion Matrix와 낮은 신뢰도 및 오분류 사례 검토
- 조명, 색감, 초점, 배율, 배경별 성능 분해

완료 기준: 모델별 주요 실패 유형과 다음 실험의 우선 가설을 문서화한다.

## 4. Partial Fine-tuning

- 기존 baseline을 변경하지 않고 새 실험으로 수행
- Head-only baseline 후 EfficientNet 후반부 일부만 해제
- 낮은 학습률과 동일 데이터 분할 사용
- 한 실험에서는 unfreeze 범위 또는 학습률 하나만 변경

완료 기준: baseline 대비 동일 Test와 실제 장비 검증 세트의 변화를 비교한다.

## 5. 입력 해상도 비교

- 우선 224와 256 비교
- 데이터 분할, seed, backbone, loss 등 다른 조건 고정
- 정확도뿐 아니라 추론 시간과 메모리 사용량 기록

완료 기준: 실제 장비 검증 성능과 실행 비용을 함께 고려해 해상도를 선택한다.

## 6. 증강·Loss·Backbone 실험

- Domain Gap 분석으로 확인된 실제 변화에 맞춰 증강 조정
- 클래스 불균형보다 분리 가능성 문제를 먼저 검토
- 필요할 때만 Focal Loss와 다른 backbone을 각각 독립 실험

완료 기준: 변경 이유와 효과가 baseline 대비 표로 비교 가능하다.

## 7. 최종 모델 선정

- 공개 Test와 실제 장비 검증 결과를 모두 비교
- Macro F1, 클래스별 Recall, 안정성, 추론 비용을 함께 평가
- 선택 모델과 클래스 순서, 전처리, 버전을 고정

완료 기준: 모델 카드와 재현 가능한 평가 JSON을 남긴다.

## 8. 논문 기반 모델 강화

- 현재 `skin` 10-class, `web_skin` 5-class, `hair` 5-class와 Clean 분할 유지
- [x] Hair 반복 기준선 실행 노트북 준비, 현재 계획에서는 사용하지 않음
- [x] Hair 통합 실행 노트북 준비 후 계산 비용 때문에 실행 보류
- [x] Hair 기준선 대 SupCon seed 42 단일 비교 노트북 준비
- [x] Hair SupCon seed 42 단독 선별 실험
- [x] Hair 기준선·SupCon 반복 검증 노트북 준비, 현재 계획에서는 사용하지 않음
- [x] Hair 최종 후보 고정: B1 384·Label Smoothing 0.05·Adam
- [x] Hair의 유사 클래스에 Supervised Contrastive Learning 1차 비교
- [x] DINOv2·EfficientNetV2·B1 384 단일 seed 통합 선별 노트북 준비
- [x] DINOv2와 EfficientNetV2의 seed 42 선별 결과 생성
- [x] B1 256·384의 seed 42 입력 해상도 비교 결과 생성
- [x] 저장된 SupCon·B1 384의 오답 보완성과 1:1 ensemble 필요성 판단
  - 1:1 ensemble Macro F1 `0.7942285284`로 B1 384 단일 모델 `0.7957146810`보다 낮아 미채택
- [x] 최선 단일 모델 B1 384용 SAM optimizer 비교 노트북 준비
- [x] B1 384 Adam 대 SAM seed 42 Validation 비교 실행
  - Accuracy 동일, SAM Macro F1이 `0.0002397889` 낮아 미채택
- [x] 고정된 B1 384 Adam 최종 Test·패키징 노트북 준비
- [x] 고정된 B1 384 Adam Hair Test 최종 1회 평가와 후보 v2 패키징
  - Test Accuracy `0.8003194888`, Macro F1 `0.8002222283`
- [x] Web Skin 기준선 대비 WS-DAN·PMG·MixStyle 통합 노트북 준비
- [x] Web Skin 논문 기반 세 방법을 Colab에서 실행하고 Validation 결과 생성
- [x] 세 방법 중 Validation Macro F1 선두 후보 PMG 고정
  - Validation Accuracy `0.85`, Macro F1 `0.8473279632397033`
  - 기존 B0·256·CE 대비 Accuracy `+0.054`, Macro F1 `+0.0557884984830505`
- [x] PMG 선두 후보의 B1·384 scale 추가 실험 노트북 준비
- [x] PMG·B1·384를 Validation에서 실행하고 기존 PMG·B0·256과 비교
  - B1·384 Accuracy `0.866`, Macro F1 `0.8643216544321994`
  - B0·256 대비 Accuracy `+0.016`, Macro F1 `+0.0169936911924961`
  - B1·384는 공개 Validation 성능 선두 연구 후보로 보존
- [x] 입력 비용과 모델 크기를 함께 고려해 Web Skin 배포 후보를 PMG·B0·256으로 고정
- [x] 고정된 PMG·B0·256 최종 Test·패키징 노트북 준비
- [x] 고정된 PMG·B0·256 Test 1회 평가와 후보 v2 패키징
  - Test Accuracy `0.915`, Macro F1 `0.9141049081029712`
- [x] Web Skin MedSigLIP-448 frozen Linear Probe 선별 노트북 구현
  - 동일 clean 데이터·seed 42에서 의료 embedding 한 변수만 검증
  - 128장 단위 embedding cache와 중단 후 재개 지원
  - 기존 PMG v2는 보존하고 Test는 실행하지 않음
- [x] Web Skin MedSigLIP-448 Linear Probe Colab 실행과 Validation 결과 분석
  - Accuracy `0.824`, Macro F1 `0.8195774289`
  - PMG v2보다 Macro F1 `0.0277505344` 낮아 미채택, Test 미실행
- [x] Hair MedSigLIP-448 frozen Linear Probe 선별 노트북 구현
  - Hair clean 데이터·seed 42·기존 v2 지표를 고정
  - 128장 단위 embedding cache와 중단 후 재개 지원
  - 기존 Hair v2는 보존하고 Test는 실행하지 않음
- [x] Hair MedSigLIP-448 Linear Probe Colab 실행과 Validation 결과 분석
  - Accuracy `0.7699680511`, Macro F1 `0.7687534260`
  - Hair v2보다 Macro F1 `0.0269612550` 낮고 5개 클래스 F1이 모두 낮아 미채택
  - Test와 후보 패키징 미실행
- [x] 세 도메인의 전체 실험을 선택 이유·방법 원리·실제 적용·결과·채택 여부 기준으로 통합 기록
- 후보 선정 전 Test를 열지 않고 Validation Macro F1 사용

완료 기준: 각 실험의 논문 근거, 고정 조건과 단일 변경 변수가 기록되고, 최종 후보 하나만
고정 Test에서 평가된다.

## 9. Hair ‘양호’ 클래스 데이터셋 v2 — 공개 데이터 실험 완료, 원천 검증 후속

2026-09-28 실행: 기존 5클래스 Clean v1 분할을 보존하고 `양호` 원본 534장을 더한
6클래스 데이터 ZIP을 별도로 만들었다. B0·256/B1·384의 원본·증강 네 조건을 비교해
B1·384·원본을 Validation 기준 성능 우선 후보로 선정·Test 평가·패키징했다.
B0·256·증강도 경량 비교 후보로 고정해 별도 Test 평가·패키징했다.
[5클래스·6클래스 종합 분석](research/HAIR_5CLASS_6CLASS_EXPERIMENT_ANALYSIS_20260928.md)에
데이터·학습 조건·동일 Test 비교와 남은 한계를 기록했다. 아래 체크되지 않은 항목은
후속 검증 과제이며, 이번 공개 데이터 성능으로 실제 장비 성능을 주장하지 않는다.

세부 기준: [Hair ‘양호’ 후보 검증·v2 구성 계획](research/HAIR_NORMAL_V2_DATA_PLAN_20260928.md).
사용자가 확인한 후보 사진의 여섯 증상값은 모두 `0`이다. 처음 약 2,400장에서 파일명이
겹친 항목의 사진도 직접 확인해 약 530장을 남겼다. 정리 전 원본은 백업으로 보관 중이다.
업로드된 ZIP의 1차 점검에서 JPG 534장·파일명 기준 비식별 ID 105개·파일/픽셀 완전 중복
0건을 확인했다. [점검 보고서](research/HAIR_GOOD_UPLOAD_AUDIT_20260928.md)를 참고한다.
기존 데이터와의 중복·사람 단위 겹침 및 파일별 라벨 재확인을 마쳐야 `양호` 수와 분할을 확정한다.

- [ ] Hair 정상 사진의 출처, 촬영 장비·배율, 라벨 근거와 사용 가능 여부 확인
      (`docs/presentations/DATA_SOURCE_CLASS_TABLE.md`에는 원천 `양호` 811장으로 기록돼 있으나,
      해당 원본 사진의 현 보관 여부와 학습 적합성은 아직 확인하지 않음)
- [ ] 정상과 기존 5종 질환의 촬영 조건·개인/세션 중복 여부 점검
- [ ] 사람·촬영 세션 단위 분할 및 완전 동일/유사·증강 유래 이미지 검사; 확인 불가 항목도 기록
- [ ] 기존 Hair Clean v1과 5-class 후보를 보존하고 별도 6-class 데이터·클래스 계약 생성
- [ ] 같은 6-class 분할에서 기준 모델을 학습하고 Validation의 클래스별 지표·혼동행렬 확인

완료 기준: 정상 라벨 근거, 출처, 클래스 수량, split manifest, 증강 lineage와 ZIP SHA-256이
보존된 Hair 6-class 데이터 및 기준 평가가 존재한다. 기존 Hair 5-class 수치와 6-class 수치는
과제가 달라 단순 정확도 차이로 개선 효과를 주장하지 않는다. Skin은 적합한 정상 원천을 확보하지
못했으므로 10-class를 유지하고, 정상 클래스 추가 일정은 잡지 않는다. Web Skin은 기존 정상
클래스를 유지한다.

## 11. 세 도메인 모델 구조 비교 — 계획

- [ ] Skin 10-class, Web Skin 5-class, Hair 6-class를 서로 독립적인 실험으로 설계
- [ ] 각 도메인에서 데이터 버전·분할·클래스 순서·평가 방식은 고정하고 모델 구조를 비교
- [ ] 구조별 입력 크기나 권장 전처리 등 불가피한 동반 변경은 별도 기록하여 효과를 혼동하지 않음
- [ ] 같은 Validation에서 Accuracy·Macro F1·클래스별 Recall/F1·혼동행렬을 비교
- [ ] 현재 선정 후보와 새 구조의 비교는 같은 클래스 과제 안에서만 수행; Hair 5-class 후보는
      Hair 6-class 후보의 직접 대조군으로 사용하지 않음
- [ ] 후보를 Validation으로 고정한 뒤에만 해당 데이터 버전의 Test를 최종 평가

완료 기준: 세 도메인 각각에 대해 구조별 학습 설정·결과·선정 이유와 미채택 이유가 남는다.
모델 구조 비교가 이미 끝난 과거 실험과 새 실험의 데이터 버전을 섞지 않는다.

## 12. 실행 비용·하드웨어 측정 — Colab 측정 완료, 장비 측정 남음

2026-09-28 결과: 저장 모델 34개(Skin 4, Web Skin 12, Hair 5-class 14, Hair 6-class 4)의
실행 시간을 동일 Colab 런타임에서 측정했다. [원시 기록·그림](../results/runtime_benchmark_20260928_051423/README.md),
[측정 해석](research/RUNTIME_BENCHMARK_REVIEW_20260928.md),
[공개 데이터 모델 종합 선정](../results/FINAL_MODEL_SELECTION_20260928.md)을 보존한다.

- [ ] 측정할 장비와 소프트웨어 버전, 전력/실행 모드, 입력 크기 및 추론 경로 기록
- [x] 동일 Colab GPU에서 배치 1, 워밍업 후 반복 실행으로 전처리 포함 지연 시간의 중앙값·상위 구간 측정
- [ ] 모델 파일 크기, 실행 메모리와 장비 자원 사용량을 함께 기록
- [x] 정확도와 지연 시간의 상충 관계를 과제별로 시각화; 과거 학습 시간의 측정 환경 차이를 명시
- [x] 현재 공개 데이터 평가와 실제 장비 사진 성능은 별개로 표시

완료 기준: 세 도메인의 후보를 동일한 측정 조건에서 비교한 표·그래프와 원시 측정 기록이 있다.
측정 전에는 실제 장비 속도나 키오스크 적합성을 확정해서 표현하지 않는다.

## 10. Router와 시스템 통합 — 보류

- 안구, 웹캠 피부, 현미경 피부, 현미경 두피 입력을 명시적으로 구분
- 자동 라우팅 전에는 사용자가 입력 종류를 선택하는 안전한 방식부터 구현
- 전문 모델 결과를 공통 JSON 스키마로 변환
- 의료 진단이 아닌 연구 및 스크리닝 보조라는 한계를 UI와 결과에 표시

완료 기준: 입력 종류별 올바른 모델 호출, 오류 처리, 통합 테스트가 동작한다.
