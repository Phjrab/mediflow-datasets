# Hair·Skin 일괄 분석과 결과 ZIP 다운로드

2026-11-09 · 로드맵 3단계 실제 촬영 환경 차이 분석

사용자가 세 분야의 분석을 먼저 모으기로 정했다. Web Skin은 다운로드된 검토 ZIP에서
Source Train 3,600장과 촬영 42장의 분석 결과를 확인했다. 이 노트북은 남은 Hair와 Skin의
분석 및 변환 미리보기를 차례로 실행한다. 모델 추가 학습과 최종 Test 평가는 실행하지 않는다.

## 실행 파일

- [Hair_Skin_Analysis_Colab.ipynb](../../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.ipynb)
- [같은 내용의 Colab용 Python 코드](../../output/hair_skin_batch_20261109/Hair_Skin_Analysis_Colab.py)
- [기존 실행 ZIP](../../output/domain_adaptation_20261108_final/mediflow_da_runtime.zip)
- [재사용 실행 모듈](../../src/mediflow_datasets/domain_adaptation_batch.py)
- [노트북 생성기](../../scripts/build_domain_adaptation_batch_colab.py)
- [자동 검사](../../tests/test_domain_adaptation_batch.py)

새 노트북에는 실행 코드 칸이 하나 있다. Colab에 열고 GPU를 선택한 뒤 **모두 실행**을 누른다.
파일 선택 창에서 기존 `mediflow_da_runtime.zip` 하나를 선택하고, 학습 데이터셋이 있는 Drive
계정으로 연결한다. 분야명·경로·검토 승인 문구를 사용자가 수정할 필요는 없다.

Hair는 현재 후보 색인의 **5클래스 B1·384** 모델, Skin은 **10클래스 B0·224** 모델을 유지한다.
Hair 6클래스 모델이나 다른 태스크를 자동으로 대체하지 않는다. 촬영 원본 대신 기존 ZIP에
포함된 Hair 40장·Skin 42장의 측정값과 기존 모델 특징을 사용한다.

## 수행 순서와 결과

1. 업로드한 실행 ZIP 전체 해시와 내부 파일 해시를 검사한다.
2. TensorFlow 2.20.0·Keras 3.13.2 및 필수 라이브러리와 계산 동작을 검사한다.
3. 호환 환경이면 재사용한다. 버전이나 TensorFlow 관련 의존성이 맞지 않으면 Colab의 기본
   환경을 바꾸지 않고 별도 환경을 만들고 필요한 패키지를 설치한다.
4. Drive에서 실제 `hair_datasets`를 찾고 기존 분석 CLI의 해시·분할 검사, Train 전체 측정,
   특징 비교 및 변환 미리보기를 실행한다. 이어서 `skin_datasets`에 같은 절차를 적용한다.
5. 각 분야의 분석 결과·유사 사진 목록·환경 및 코드 기록·변환 전후 이미지·로그를 ZIP에 담는다.
6. 두 ZIP을 Drive에 보관하고 PC 다운로드를 각각 요청한다.

Drive 저장 위치는 매 실행마다 생성되는
`내 드라이브/mediflow_da_hair_skin_<이번 실행 ID>/hair/`, `…/skin/`이다.
상위 폴더에 `mediflow_review_hair_<ID>.zip`, `mediflow_review_skin_<ID>.zip`이 보관된다.
정확한 경로는 실행 출력에 표시된다. 같은 노트북을 다시 실행해도 기존 폴더를 덮어쓰지 않는다.

브라우저가 두 번째 자동 다운로드를 차단하면 여러 파일 다운로드를 허용하거나 마지막에
표시된 분야별 다운로드 버튼을 사용한다. Drive 사본도 남는다. 한 분야가 실패하면 로그를
담은 `mediflow_failed_<분야>_<ID>.zip`으로 구분하고 다른 분야의 분석은 계속한다.
설치·파일 업로드·Drive 연결처럼 공통 준비 단계가 실패하면 그 원인을 먼저 해결해야 한다.

## 이전 오류와의 관계

이전 `REVIEW_NOTE` 오류는 학습 전에 검토를 요구하는 칸에서 발생했다. 새 노트북의 범위는
분석과 미리보기까지이므로 해당 학습 칸을 호출하지 않는다. 검토 내용을 임의로 승인하거나
유사 사진 목록을 일괄 해제한 것이 아니다. 모든 후보와 인물·세션 정보의 부재를 결과에 남긴다.
검토 후 추가 학습은 별도 단계다. 정답 없는 촬영 자료의 정확도는 계산하지 않는다.

TensorFlow 설치는 별도 환경을 권장하는 [공식 설치 안내](https://www.tensorflow.org/install/pip)를
참고했다. Linux 전용 환경에 `tensorflow[and-cuda]==2.20.0`을 설치하며 실제 GPU 인식 결과를
출력한다. Colab 기본 TensorFlow 2.21의 부속 패키지를 삭제하거나 버전을 바꾸지 않는다.
GPU가 인식되지 않으면 CPU로 분석하며 처리 시간이 늘어날 수 있다.

## 확인 범위

실제 Colab의 전체 Hair·Skin 분석을 대신 실행한 결과는 아니다. 로컬에서는 순차 실행,
분야별 파일 분리, 두 ZIP 내보내기, 다운로드 콜백, 오류 기록 후 다음 분야 진행, 기존 결과 보존,
ZIP 무결성·경로 검사, 환경 분리와 학습 미호출을 검사한다. 실행 검증 기록은
[verification.json](../../output/hair_skin_batch_20261109/verification.json)에 보관한다.
