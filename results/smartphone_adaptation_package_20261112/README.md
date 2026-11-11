# 스마트폰 얼굴 적응 · 최종 연구용 패키지

Validation에서 미리 선정한 **4차 PMG·EfficientNet-B0·256**을 패키징했다.
기존 웹캠 모델과 별도이며 공식 공개 데이터 후보 색인은 유지했다.

- Git에서 받은 뒤 사용할 폴더: [package](package/README.md)
- 로컬 전달용 ZIP: [web_skin_smartphone_adaptation_stage4_20261112.zip](web_skin_smartphone_adaptation_stage4_20261112.zip)
- 모델·ZIP 해시·단독 실행 검사: [package_index.json](package_index.json)
- 실제 전처리·원본 예측 일치 검사: [contract_verification.json](package/contract_verification.json)
- 전체 검사·원본 보존·결과 검산: [completion_verification.json](completion_verification.json)
- 전체 과정: [최종 정리](../../docs/research/camera_adaptation/ADAPTATION_COMPLETION_20261112.md)

모델 SHA-256은 `f08958c745811dc34d8e876e742f911abc1f6938babdc64bd82a284d7e052c9e`다.
정상 포함 5클래스는 건선·아토피·여드름·정상·주사 순서다.
기본 휴대폰 입력은 EXIF 방향과 ICC→sRGB 처리 뒤 중앙 정사각형 256px RGB float32 0–255다.
외부 `/255`는 없고 실제 사진에 학습 때의 조명·색·흐림을 다시 적용하지 않는다.

원본 Source Test 367/400·강한 변환 355/400이며 실제 휴대폰 질환 Accuracy는 정답이 없어 미측정이다.
모델·메타데이터·추론 코드·원본 보고서를 함께 제공했다. 사진·키·원본 데이터셋은 포함하지 않는다.
ZIP은 기존 `.gitignore`에 따라 로컬 전용이며, Git에는 압축을 푼 사용 폴더와 검증 기록을 보관한다.
