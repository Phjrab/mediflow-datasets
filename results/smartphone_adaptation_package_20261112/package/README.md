# MediFlow 스마트폰 얼굴 · 4차 연구용 패키지

Validation에서 미리 선정한 PMG·EfficientNet-B0·256 저장 모델이다. 기존 웹캠 패키지와 별도다.
스마트폰으로 촬영한 얼굴 입력을 위해 EXIF 방향 보정·ICC→sRGB·중앙 자르기를 포함한다.
정상 포함 5클래스: **건선·아토피·여드름·정상·주사**. 실제 질환 정답이 있는 휴대폰 평가 결과는 없다.

## 실행

Python 3.10~3.13 환경에서 `python -m pip install -r requirements.txt` 후
`python inference.py --image "휴대폰사진.jpg"`를 실행한다. 사진은 실제 파일 경로로 바꾼다.
여러 장은 `python example_usage.py "사진1.jpg" "사진2.jpg"`로 모델을 한 번만 로드한다.
JPEG·PNG 입력을 권장한다. 현재 Pillow에서 읽지 못하는 HEIC는 JPEG/PNG로 준비해야 한다.

스마트폰 기본 경로는 EXIF 방향을 반영하고 내장 ICC 색상 프로필을 sRGB로 변환한다.
프로필이 없으면 sRGB로 가정하며 손상된 프로필은 오류다. 중앙 정사각형 256px,
RGB float32 0–255, TensorFlow bilinear·antialias=False다. 외부 `/255`를 추가하지 않는다.
학습 때의 조명·색·흐림 변환은 실제 사진에 다시 적용하지 않는다.

연구 재현용 Source Test 입력은 `python inference.py --image "원본데이터.png" --input-kind source`다.
이 경로는 고정 Test와 같이 TensorFlow로 읽으며 EXIF·ICC 보정을 하지 않는다.
아래 Source Test 수치는 이 `source` 경로의 기록이다.
기본 스마트폰 경로의 실제 질환 성능으로 쓰지 않는다.

PMG 네 logit을 더한 뒤 softmax를 적용하며 클래스별 점수는 정답 확률로 보정되지 않았다.
JSON에 모델·분야·예측 인덱스/이름·5개 점수·입력 종류를 기록한다.
스마트폰+얼굴인지 호출자가 확인해 선택한다. 품질·촬영 부위·범위 밖 입력은 자동 판정하지 않는다.
범위 밖 이미지도 기존 클래스 하나로 나올 수 있으며 질환 진단·치료 판단용으로 검증하지 않았다.
파일 없음·해독/ICC 실패·해시/클래스/모델 계약 불일치는 JSON 오류와 종료 코드 1로 알린다.

## 고정 모델의 Source Test 기록

| 조건 | Accuracy | Macro-F1 |
| --- | --- | --- |
| clean | 0.9175 | 0.9172235025222394 |
| phone_lighting | 0.9175 | 0.9172144021213157 |
| phone_expanded | 0.9175 | 0.9172211327056615 |
| phone_refit | 0.8875 | 0.8869611005630087 |
| phone_joint | 0.915 | 0.9147589669933156 |

모델·클래스·전처리·학습/검증/Test 원본·해시를 함께 보존했다. 사진·복호화 키는 포함하지 않는다.
