# MediFlow 실행 로드맵

이 문서는 `docs/research/PROJECT_BACKGROUND.md`의 연구 방향을 실제 개발 단계와 완료 기준으로
구체화한다. 한 번에 여러 변수를 바꾸지 않고 각 단계의 입력, 설정, 결과를 기록한다.

현재 공개 데이터 연구는 8단계까지 완료했다. Skin, Web Skin, Hair 최종 후보와 사용 계약은
`results/CANDIDATE_INDEX.json`, 전체 과정과 논문 근거는
`docs/research/MEDIFLOW_PROJECT_COMPREHENSIVE_FINAL_20260923.md`를 기준으로 한다. 2·3·9·10단계는
프로젝트 범위가 다시 열릴 때 진행하는 후속 과제다.

## 2026-09-14 이후 우선 방향

프로젝트 소유자의 범위 변경에 따라 실제 장비 연결과 LLM/VLM 통합은 현재 작업에서 보류한다.
기존 공개 데이터 후보 모델과 클래스 구성을 기준선으로 보존하고, 우선 논문 기반 학습법을
검증한다. 세부 실험은
[`PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md`](research/PAPER_BASED_MODEL_ENHANCEMENT_PLAN.md)를 따른다.

## 1. 현재 모델·평가 재현

- [x] TensorFlow 2.20.0 / Keras 3.13.2 환경 고정
- [x] original/augmented 모델 6개 로드 경로 정의
- [x] 모델 입력, 출력 차원, 클래스 순서, 내부 정규화 자동 검증
- [x] 단일 이미지 추론 CLI 검증
- [ ] 원래 Test 데이터 위치 연결
- [ ] 세 모델 original/augmented 평가 JSON 재생성
- [ ] 저장된 기존 결과와 허용 오차 내 일치 여부 기록

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

## 9. 정상 클래스 데이터셋 v2 — 후속 보류

- Hair 원천 JSON에서 `양호`와 증상별 0단계 정의 확인
- Skin 정상피부의 촬영 장비·배율·출처 확인
- 사람·병변·촬영 세션 단위 분할과 exact/near duplicate 검사
- 기존 Clean v1과 후보 모델은 수정하지 않고 별도 v2 생성

완료 기준: 정상 라벨의 근거, 출처, 클래스 수량, split manifest, 증강 lineage와 ZIP SHA-256이
보존된 `skin` 11-class 및 `hair` 6-class 데이터가 존재한다.

## 10. Router와 시스템 통합 — 보류

- 안구, 웹캠 피부, 현미경 피부, 현미경 두피 입력을 명시적으로 구분
- 자동 라우팅 전에는 사용자가 입력 종류를 선택하는 안전한 방식부터 구현
- 전문 모델 결과를 공통 JSON 스키마로 변환
- 의료 진단이 아닌 연구 및 스크리닝 보조라는 한계를 UI와 결과에 표시

완료 기준: 입력 종류별 올바른 모델 호출, 오류 처리, 통합 테스트가 동작한다.
