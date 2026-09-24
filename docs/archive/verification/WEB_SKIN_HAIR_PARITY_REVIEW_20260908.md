# Web Skin 자동 학습 코드와 Hair 실험 대조 검토

검토일: 2026-09-08. ROADMAP 4~6 공개 데이터 성능 개선 실험의 실행 전 검토다.

## 판단

Hair의 정리 데이터 224/256, Label Smoothing, Focal Loss, B1 실험과 Web Skin 자동 실행 코드의 핵심 학습 방식이 일치한다. 검토 범위에서 학습을 막는 오류는 발견하지 않았다. 같은 방법을 적용했다는 뜻이며 성능 향상이나 Colab 전체 실행 성공을 미리 보장하는 결과는 아니다.

기존 노트북은 변경하지 않았다. 사용할 검토본:
`web_skin_all_experiments_reviewed_colab.ipynb` 원본 코드는
[기존 노트북 백업 ZIP](../../../notebooks/legacy_notebooks_20260909.zip)에 보관되어 있다.

## 비교

| 항목 | Hair 정리 데이터 실험 | Web Skin 검토 결과 |
|---|---|---|
| 학습 출발점 | ImageNet 사전학습 B0/B1 | 동일. 기존 Hair/Web Skin 질환 모델을 불러오는 방식이 아님 |
| 학습 데이터 | 정리된 Augmented Train | 검사 완료된 Web Skin Augmented Train 7,200장 |
| 평가 데이터 | 원본 Validation/Test | 기존 감사에서 원본과 일치한 Validation 500장/Test 400장 |
| 클래스 | 모낭사이홍반·미세각질·비듬·탈모·피지과다 | 건선·아토피·여드름·정상·주사. 기존 Web Skin 결과 JSON 순서와 일치 |
| 이미지 입력 | RGB float32 0~255, 내부 정규화 | 동일. 외부 /255 없음, 224/256 bilinear resize |
| 1단계 | 특징 추출부 고정, 15 Epoch, Adam 1e-4 | 동일 |
| 2단계 | 1단계 최고 모델 로드, 마지막 30개 계층 중 BN 제외, 10 Epoch, 새 Adam 1e-5 | 동일 |
| 분류부 | GlobalAveragePooling, Dropout 0.3, 5개 softmax | 동일 |
| 고정 조건 | Batch 32, Seed 42 | 동일. 단일 Seed 실험이며 반복 실험의 변동성 검증은 아님 |
| Loss 비교 | CE, LS 0.05, Focal alpha 1.0/gamma 1.5 | 동일 |
| Backbone 비교 | B0 256 LS 대비 B1 256 LS | 동일. 마지막 30개 계층의 실제 파라미터 수는 두 구조에서 다름 |
| 클래스 가중치 | 정리 후 클래스 개수에 따른 balanced 가중치 | Web Skin은 클래스당 1,440장으로 같아 생략. 각 클래스에 1을 주는 것과 동일 |
| 증강 변경 | 저장된 증강 데이터 사용 | 동일. 온라인 증강이나 새로운 증강 기법을 추가한 실험은 아님 |
| 단계별 모델 선정 | Validation Accuracy 최고, 동점이면 앞선 단계 | 동일 |
| 추가 5 Epoch | stage2_finetune_best.keras에서 optimizer 복원 | stage2_last.keras에서 optimizer 복원. 마지막 10번째 모델부터 실제 추가 5회라는 정책 |
| 연장 실패 시 | 부모 최고 모델 유지 | 동일 |
| Test 사용 | 각 실험 후보 고정 후 평가 | 전체 6개 중 Validation으로 최종 후보를 고정한 뒤 그 후보만 평가 |
| 결과 | 학습곡선·분류 결과·모델 | 각 모델 Validation 지표/오답 사진, 전체 3×4 학습곡선, 성능 대시보드, 혼동행렬, 최종 후보 Test, 모델/보고서 ZIP |

Hair의 최고 체크포인트와 마지막 체크포인트가 다르면 연장 학습 경로는 달라진다. 따라서 두 노트북의 연장을 무조건 동일하다고 표현하면 안 된다. Web Skin의 정책은 원래 자동 실험 설계대로 유지하고 검토본에 설명을 추가했다.

## 확인한 문제와 처리

기존 비교표의 `training_seconds_this_trial`은 실제로 모델 생성·학습·저장·Validation 예측을 모두 포함한 시간이었다. 검토본에서는 `elapsed_seconds_this_trial`로 표시한다. 실행 모듈의 기존 `training_seconds` 기록 필드는 보존하고 그 의미를 노트북에 설명했다. 순수 학습 시간이나 추론 속도 측정값으로 쓰면 안 된다.

검토본에는 `notebook_revision`을 설정 서명에 추가했다. 이전 노트북으로 이미 생성한 suite를 이 검토본으로 재개하면 설정 불일치로 중단된다. 기존 실행을 재개할 때는 원래 노트북을 사용하고, 검토본 첫 실행은 RESUME_SUITE_DIR을 비운다.

## 확인 방법과 한계

- `ruff check src tests`: 통과. 전체 `pytest`: **36개 통과**, Keras/NumPy 호환 관련 경고 86건, 실패 없음.
- 노트북과 재사용 모듈의 내장 코드 일치 및 코드 셀 문법을 검사했다.
- 실제 EfficientNet-B0 224, B1 256 구조에서 짧은 학습을 실행했다. 테스트는 인터넷 다운로드 없이 임의 초기화 가중치를 사용한다. 실제 코드가 ImageNet 가중치를 요청하는지 별도로 확인한다.
- 후반부 학습 가능 가중치가 실제로 변경되고 BN의 이동 평균/분산이 변경되지 않는지 검사했다.
- 실제 디렉터리 이미지 로더로 Web Skin 클래스 순서, 예측 파일 순서, float32 픽셀 범위와 크기를 검사했다.
- CE/LS/Focal 세 Loss로 학습한 작은 모델의 저장·복원, optimizer 상태를 확인했다.
- 기존 테스트는 단계별 학습·저장·연장·완료 실험 재사용·변조 감지·Validation 전용 모델 선정을 확인한다.
- 데이터 중복 검사는 재실행하지 않았다. 현재 로컬에서는 이전에 전달된 감사 ZIP을 찾지 못했으며, 기존 확인 기록과 고정된 데이터 SHA-256을 사용한다. Colab은 동일 ZIP인지 확인한 뒤 읽는다.
- 실제 Web Skin 전체 학습, Colab GPU/Drive 연동, ImageNet 가중치 다운로드는 이번 로컬 검토에서 실행하지 않았다.
- 그래프 코드는 문법과 흐름을 검토했지만 실제 렌더링 검증은 하지 못했다. 로컬에 matplotlib이 없고 설치 네트워크 접근이 제한되어 있다. Colab 전체 실행 후 생성된 PNG를 확인해야 한다.
- 사람·병변·세션 단위 데이터 겹침과 실제 웹캠 성능 미검증 한계는 유지된다.

수치 파일의 실제 정밀도는 그대로 저장한다. 서로 다른 CE/LS/Focal의 절대 Loss 수치로 모델 순위를 매기지 않는다. 모델 간 성능 비교는 동일 Validation의 Accuracy/F1을 사용한다.
