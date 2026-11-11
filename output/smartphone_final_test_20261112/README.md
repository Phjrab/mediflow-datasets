# 스마트폰 얼굴 모델 · 고정 최종 Source Test

**2026-11-12 사용자 Colab 실행·400장·30개 비교·결과 검산을 완료했다.**
[최종 Test 결과](../../docs/research/camera_adaptation/SMARTPHONE_FINAL_TEST_RESULT_REVIEW_20261112.md)와
[스마트폰 패키지](../../results/smartphone_adaptation_package_20261112/README.md)를 따른다.
아래는 완료 실험의 재현 안내이며 준비 JSON의 미실행 표시는 당시 원본 기록으로 보존했다.

1. [전용 노트북](../../notebooks/camera_adaptation/MediFlow_Smartphone_Final_Test_20261112.ipynb)을 Colab에서 연다.
2. GPU 런타임을 선택하고 **모두 실행**한다. 새 실행 ZIP 업로드와 복호화 키 입력은 없다.
3. 끝나면 `MediFlow_Smartphone_Final_Test_Results_20261112.zip`이 자동 다운로드된다. 이 파일을 전달하면 결과를 대조한다.

## 이번 비교

| 항목 | 고정 내용 |
| --- | --- |
| 분야 | web_skin, 스마트폰 환경 적응 모델 |
| 모델 6개 | 적응 전 선정 PMG·1·2·3·4·추가 5차 |
| Test | 기존 원본 Test 400장, 원본 파일과 분할 유지 |
| 조건 5개 | 원본·1차 조명·2차 약한 확장·4차 강한 확장·새 공동 근접 변환 |
| 총 비교 | 6모델 × 5조건 = 30개, 30번 학습하는 것이 아님 |
| 사전 고정 후보 | Validation으로 선택한 4차, Test에서 다시 선정하지 않음 |
| 입력 | 중앙 정사각형 256px RGB float32 0–255, 외부 `/255` 없음 |
| 클래스 순서 | 건선·아토피·여드름·정상·주사 |
| 평가 변환 | 기존 범위와 seed 1707 고정, 각 변환 조건에서는 100% 적용 |

학습 때 변환 적용 확률 0.5와 Test 변환 조건의 100% 적용을 구분한다.
Test 조건은 강도를 더 높인 것이 아니라 원본과 변환을 각각 분리해 확인하기 위한 것이다.
가상 변환을 적용한 Source Test이며 실제 휴대폰 질환 사진의 정확도는 아니다.
이번에는 실제 참여자 사진·P05를 열지 않으며 MMD·CORAL 진단도 반복하지 않는다.

## 기존 Drive 입력 재사용

```text
내 드라이브/mediflow_Project/MediFlow_Smartphone_WebSkin_Runtime_20261111.zip
내 드라이브/mediflow_Project/datasets/  ← 기존 web_skin 원본 ZIP을 이름·해시로 탐색
내 드라이브/MediFlow_smartphone_runs/edc193fee3c2/
내 드라이브/MediFlow_smartphone_runs/joint_a3d4036b9f90/
```

이전에 학습에 사용한 실제 경로를 재사용한다. 파일이 없거나 해시가 다르면 평가 전에 중단한다.
키·참여자 사진·초기 B0 모델을 Test 런타임에 추출하지 않는다.
노트북에 새 평가 코드·고정 계획만 포함되어 있어 큰 ZIP을 새로 올리지 않는다.
입력 사진 캐시는 Colab 로컬 SSD에 만들며 사용자 PC에 복사하는 것이 아니다.

## 진행 표시·중단 후 재실행

입력 400장 준비 → 모델 1/6 → 조건명 → 배치 1/50 등 실제 진행을 표시한다.
모델 하나씩 로드하고 같은 입력을 재사용해 GPU 메모리와 반복 이미지 처리를 줄였다.
조건 하나가 끝날 때 예측 점수와 지표를 저장한다. 중단되면 같은 노트북을 모두 실행한다.
완료한 조건은 실행 계획·모델·점수 해시와 지표를 확인한 뒤 재사용한다.
중간에 끊긴 조건은 처음부터 다시 계산하며 손상된 결과는 조용히 사용하지 않고 오류로 알린다.

## 결과 위치와 포함 내용

```text
내 드라이브/MediFlow_smartphone_runs/test_ad2338eaa932/
  MediFlow_Smartphone_Final_Test_Results_20261112.zip
```

결과에는 비교 JSON/CSV, 클래스별 Precision·Recall·F1, 개별 이미지 예측과 5개 점수,
30개 혼동행렬을 담은 6개 그림, 전체 비교 그래프, 고정 분할·실행 코드·환경·모델 해시가 들어간다.
모델 파일·사진·복호화 키는 결과 ZIP에 넣지 않는다. 학습곡선은 새 학습을 하지 않아 새로 생성하지 않는다.
실제 Test 수치는 Colab 완료 후에만 보고한다.

Source의 사람·병변·세션 정보가 없고 과거 모델의 Test 노출도 미확인이다.
완전한 독립성이나 실제 스마트폰 질환 성능을 입증한 결과로 표현하지 않는다.

[고정 평가 계획](evaluation_plan.json) · [실행 준비 기록](preparation.json) ·
[모델·합성 입력 확인](model_smoke_verification.json) · [실행 전 검사](verification.json) ·
[평가 설계](../../docs/research/camera_adaptation/SMARTPHONE_FINAL_TEST_20261112.md).
