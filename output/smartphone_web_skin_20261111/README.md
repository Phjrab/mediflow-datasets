# 스마트폰 얼굴 적응 · 실행 안내

**실행 완료 · 2026-11-12:** 이 준비본으로 실제 5회 학습한 결과를
[프로젝트 results](../../results/smartphone_web_skin_adaptation_20261112/README.md)에 회수했다.
[결과 분석](../../docs/research/camera_adaptation/SMARTPHONE_ADAPTATION_RESULT_REVIEW_20261112.md)을 먼저 확인한다.
아래 순서는 완료 실험을 재현하기 위한 안내다.

**web_skin 전용이며 실제 학습은 5회다.** 2026-11-12 사용자 요청으로 2·3차에는 약한 질감·흐림,
4차에는 더 강한 질감·흐림을 추가했다. [새 강도와 고정 조건](../../docs/research/camera_adaptation/SMARTPHONE_APPEARANCE_STRENGTH_20261112.md)을 따른다.
ZIP·노트북을 이미 올렸다면 두 파일을 최신본으로 교체한다. 복호화 키는 그대로다.
이 폴더는 실행 준비본이며 실제 Colab 학습 결과는 위 results에 있다. 새 Source Test·휴대폰 질환 정확도 평가는 아직 수행하지 않았다.

## 사용자가 할 일

1. [스마트폰 전용 실행 ZIP](MediFlow_Smartphone_WebSkin_Runtime_20261111.zip)을
   Drive의 `내 드라이브/mediflow_Project` 폴더에 업로드한다.
2. [전용 노트북](../../notebooks/camera_adaptation/MediFlow_Smartphone_WebSkin_Stages_20261111.ipynb)을 Colab에서 연다.
3. GPU 런타임을 선택하고 **모두 실행**한다.
4. 가려진 입력란이 나오면 [스마트폰 전용 키 파일](MediFlow_Smartphone_Target_Key_20261111.txt)의 한 줄을 붙여넣는다.

코드와 경로를 수정할 필요는 없다. 기존 Drive의 `mediflow_Project/datasets`에 있는
web_skin 데이터 ZIP을 이름과 SHA-256으로 확인해 재사용한다.
이전 웹캠용 공통 ZIP이나 단계별 결과 ZIP을 추가로 올리지 않는다.
사진 입력은 전용 ZIP 안에 암호화해 넣었다. 원본 휴대폰 사진을 다시 업로드할 필요가 없다.
ZIP과 키는 개인 자료로 보관하고 키를 코드 셀·Git·공유 문서에 붙이지 않는다.

## 한 번에 실행하는 순서

| 구분 | 적용 내용 | 실제 학습 |
| --- | --- | --- |
| 1차 초기 모델 | 기존 초기 B0·224 저장본 + 휴대폰 조명 변환 | 5 epoch |
| 1차 선정 모델 | 기존 PMG·B0·256 저장본 + 같은 조명 변환 | 5 epoch |
| 2차 | 같은 학습 전 저장본 + 1차 조명·Lab a 증가 + 질감 0.95~1.05·흐림 σ 0~0.5px | 5 epoch |
| 3차 | 같은 학습 전 저장본 + 2차 변환 + MMD·CORAL 학습 손실 | 5 epoch |
| 4차 | 같은 학습 전 저장본 + 기존 4차 조명 + 질감 0.85~1.15·흐림 σ 0~1.0px, 색 이동 없음 | 5 epoch |

GPU 하나에서 순서대로 실행한다. 3·4차를 앞 단계의 학습된 가중치에서 이어서 시작하지 않는다.
흐림 값은 256px 입력 기준이다. 224px 모델이면 최대 0.4375px·0.875px로 조정한다.
질감·흐림은 사용자 요청 탐색 설정이며 이전 분포 근접성 검토가 채택한 값은 아니다.
매 단계의 epoch·배치 진행률·손실·남은 학습 예상 시간·Validation 결과·Drive 저장 상태가 출력된다.
첫 배치에서는 TensorFlow 계산 준비로 시간이 더 걸릴 수 있다. PMG는 배치당 네 번 가중치를 갱신한다.
실제 소요 시간은 GPU와 저장소 상태에 따라 달라지며 실행 전 예상 완료 시간을 보장하지 않는다.

## 저장되는 결과

Drive의 `내 드라이브/MediFlow_smartphone_runs/<실행 ZIP 해시 앞 12자리>/`에 저장한다.

- 각 실제 학습의 `best.keras`, 완료 기록, epoch별 이력, 원본·변환 Validation 지표, 학습곡선.
- 2차의 별도 학습 모델과 색·약한 질감·흐림, 4차의 더 강한 질감·흐림 설정 및 원본 근거.
- 기본·1·2·3·4차의 공통 Validation Accuracy·Macro-F1·클래스별 지표·혼동행렬과 비교 그래프.
- MMD·CORAL 학습 손실과 학습 전후 특징 거리·휴대폰 예측 변화. 점수를 정답 확률로 해석하지 않는다.
- 최종 선정 후 처음 확인하는 P05의 예측 변화와 특징 거리. 질환 정답이 없어 정확도를 계산하지 않는다.
- `MediFlow_Smartphone_WebSkin_Results_20261111.zip`을 Drive에 만들고 마지막 셀에서 다운로드한다.

학습이 끊기면 **같은 노트북을 다시 모두 실행하고 같은 키를 입력**한다.
완료한 학습은 건너뛰고 진행 중 학습은 마지막 저장 epoch에서 복원한다.
데이터를 Colab 로컬 SSD에 한 번 복사하고 해상도별 입력을 재사용한다.
결과 ZIP에는 원본 데이터·참여자 사진·키·중간 optimizer 스냅샷을 넣지 않는다.

## 분석·검사 자료

[설계와 실측 근거](../../docs/research/camera_adaptation/SMARTPHONE_WEB_SKIN_ADAPTATION_20261111.md) ·
[고정 실험 설정](../../configs/smartphone_web_skin_20261111.json) ·
[준비 기록](preparation.json) · [실제 저장 모델 로드 확인](model_smoke_verification.json) ·
[최종 준비 검증](verification.json).

최신 코드 검사 수와 파일 보존 결과는 `verification.json`을 기준으로 확인한다.
학습 전후 모델·메타데이터·ZIP의 연결 검증은 [파일 대조 기록](artifact_verification.json)에 있다.

[2차 재검토·웹캠 비교](../../docs/research/camera_adaptation/SMARTPHONE_STAGE2_COMMON_NEARER_20261111.md) ·
[재검토 원본 수치](stage2_review.json) · [웹캠·휴대폰 측정값](webcam_smartphone_comparison.json).
이전 재검토는 당시 측정 기록이다. 최신 추가 강도는 [고정 계획](appearance_strength_plan_20261112.json)과
[재생 진단](appearance_strength_replay_20261112.json)에 별도로 보존했다.

촬영 52장은 질환 정답 없는 환경 조사 자료다. 사진만으로 정상·질환 라벨을 붙이지 않았다.
현재 입력 장비의 질환 분류 정확도나 스마트폰 UI에서의 진단 성능을 검증한 결과가 아니다.
