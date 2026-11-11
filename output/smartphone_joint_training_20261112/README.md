# 휴대폰 실측 근접 후보 · 추가 1회 학습

**2026-11-12 실행 완료:** 사용자가 수행한 5 epoch 결과를 회수했고 최적 3 epoch를 확인했다.
[새 모델·원본 결과](../../results/smartphone_joint_adaptation_20261112/README.md)와
[기존 모델 비교 분석](../../docs/research/camera_adaptation/SMARTPHONE_JOINT_RESULT_REVIEW_20261112.md)을 따른다.
아래는 완료 실험의 재현 안내다. `preparation.json`·`verification.json`은 실행 전 준비 시점의 원본 기록으로 보존했다.

사용자가 승인한 7항목 근접 후보 `f10926b662a4a43c`를 기존 선정 PMG·B0·256의 적응 전 저장본에서
독립적으로 5 epoch 학습했다.

## 재현할 때 실행 순서

1. [새 노트북](../../notebooks/camera_adaptation/MediFlow_Smartphone_Joint_Training_20261112.ipynb)을 Colab에서 연다.
2. GPU 런타임을 선택하고 **모두 실행**한다.
3. 가려진 입력란에는 기존 스마트폰 전용 키 파일의 한 줄을 입력한다. 키를 새로 만들지 않았다.

**새로운 큰 실행 ZIP을 올리지 않는다.** 이미 올린 다음 파일을 그대로 사용한다.

```text
내 드라이브/mediflow_Project/MediFlow_Smartphone_WebSkin_Runtime_20261111.zip
```

이전 1~4차 결과는 다음 폴더에 있어야 한다. 새 코드가 읽기만 하며 결과를 옮기거나 수정하지 않는다.

```text
내 드라이브/MediFlow_smartphone_runs/edc193fee3c2/
```

기존 공통 ZIP 해시 `edc193fee3c25254ce947b1a9613ad8ec2cc4d0f90f3c4f283eff5d415ea3ad9`와
이전 모델·원본 지표 해시를 확인한다. 노트북에 새 실행 코드·고정 설정이 들어 있다.
`joint_code_patch_20261112.zip`은 로컬 재현용 보조 사본이고 사용자가 따로 업로드할 파일이 아니다.
이전 노트북 대신 위 **추가 1회 전용 노트북**을 실행해야 한다.

## 학습과 결과

- 밝기 −19.624129908304198~−10.181201425385177, 대비 1~1.1808786167229048.
- 좌우 조명차 −0.42238616943359375~0, Lab a 0~1.0117187500000009, Lab b −1.2126953125000002~0.
- 미세 질감 배율 1·추가 흐림 σ 0. MMD·CORAL을 학습 손실에 추가하지 않는다.
- 원래 선정 PMG 저장본에서 시작한다. 기존 4차 학습 결과를 이어 학습하지 않는다.
- 기존 분할·클래스 순서·seed 42·Adam 학습률 0.00001·batch 8·변환 확률 0.5·5 epoch를 유지한다.
- 실제 epoch·배치·Loss·Validation·Drive 저장 로그, 학습곡선·공통 비교 그림·혼동행렬이 나온다.
- 중단되면 같은 노트북을 다시 실행한다. 마지막 저장 epoch에서 재개하고 완료한 학습은 반복하지 않는다.
- 같은 세션의 Source ZIP과 입력 캐시가 일치하면 재사용한다. 나머지는 Colab 로컬 SSD에 준비한다. 사용자 PC SSD가 아니다.

기존 20개 Source Validation 결과를 바꾸지 않고 재사용한다. 이전 다섯 모델에 새 변환 조건 5개,
새 모델에 공통 다섯 조건 5개를 추가 측정해 **6모델×5조건=30개 비교**를 만든다.
학습 epoch는 원본과 새 후보 변환 중 낮은 Macro-F1, 원본 Macro-F1, 이전 epoch 순서로 고른다.
최종 비교 후보는 공통 다섯 조건의 최저 Macro-F1·원본 Macro-F1·앞 모델 순서다. 장비 유사성 선정 기준이 아니다.

설정용 실제 휴대폰 42장의 MMD·CORAL 및 예측 변화는 학습 후 진단으로 기록한다.
기존 P05는 이미 확인한 자료이므로 이번 런타임에 넣지 않는다. P05와 Source Test를 재평가하지 않는다.
질환 정답이 없으므로 Target Accuracy는 없다. 공식 후보 변경·배포 패키징도 하지 않는다.

새 Drive 결과 위치는 다음과 같다.

```text
내 드라이브/MediFlow_smartphone_runs/joint_a3d4036b9f90/
```

끝나면 `MediFlow_Smartphone_Joint_Results_20261112.zip`을 Drive에 저장하고 자동 다운로드한다.
이 ZIP에는 새 최적 모델 1개·학습/검증 기록·비교 그림·실행 코드/설정이 들어간다.
기존 네 모델·사진·키·중간 optimizer 가중치는 중복 포함하지 않는다.

## 이 후보의 의미

영상 통계 8개 중 7개가 원본보다 가까워졌지만 조명 불균일·개인차 보호 기준은 미통과했다.
이번 실행은 사용자가 승인한 탐색 학습이며 장비 동등성이나 실제 질환 정확도 개선을 보장하지 않는다.
기존 학습·자료와 별도 저장한다. [검토 근거](../../docs/research/camera_adaptation/SMARTPHONE_JOINT_SIMILARITY_REVIEW_20261112.md)와
[학습 계약](../../docs/research/camera_adaptation/SMARTPHONE_JOINT_TRAINING_20261112.md)을 따른다.

[고정 설정](study.json) · [준비 기록](preparation.json) · [검사 기록](verification.json) ·
[PMG 실제 로드·합성 배치 확인](model_smoke_verification.json).
