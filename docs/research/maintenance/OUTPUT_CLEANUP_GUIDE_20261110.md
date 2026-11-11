# output 폴더 정리 안내 — 2026-11-10 최종 구성

> 2026-11-11 정리 상태: **안내·기준 문서, 자체가 학습 완료 근거는 아님**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

사용자가 이전 학습 준비본을 정리하고 다음 학습 구성은 새로 정하기로 했다.
초기의 “9개 폴더 보존” 제안은 이 최종 정리로 대체한다.

## 현재 output

- `domain_adaptation_20261108_final`: 얼굴 초기 분석 실행본·당시 실행 ZIP.
- `hair_skin_batch_20261109`: 두피·피부 분석 실행본.
- `adaptation_preflight_v2_20261109`: 세 분야 사전 점검 실행본·당시 실행 ZIP.
- `pdf`: 공부·발표 설명 PDF 3개.

실제 분석 결과와 당시 코드 사본은 프로젝트의 `camera_capture/_review/`에 보존한다.
[실행 코드·결과 대응표](../camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md)를 기준으로 보고서에 연결한다.

## 이동한 보고서 자료

- [데이터·초기 학습 그림](../../../presentation/report_assets/data_figures_20261026)
- [Validation 비교 그림·원본 수치](../../../presentation/report_assets/validation_comparisons_20261027)

두 폴더의 파일 22개는 이동 전후 바이트가 동일하다. 그림·CSV·설명과 ZIP까지 그대로 옮겼다.

## 정리한 학습 준비본

이전 준비 폴더 8개와 그 안의 대용량 실행 ZIP을 제거했다.
코드·노트북·설명·설정은 [보관 이력](../../../camera_capture/_review/history/training_preparation)에 남겼다.
ZIP 항목 704개를 대조했으며, 기존 파일과 동일한 모델·입력 배열 등 151개 항목은
프로젝트 내부의 동일 파일에 연결하고 나머지는 이력 ZIP에 보관했다.
중복 항목을 제외한 실제 대응 파일은 57개다.

원래 큰 ZIP 컨테이너 자체와 동일한 바이트를 보존한 것은 아니다.
개별 파일 내용·해시·현재 보관 위치를 보존했다. 이력 ZIP은 현재 학습 실행 ZIP이 아니다.
원본 모델, 원본 촬영 사진, 기존 분석 수치와 실험 결과를 제거하지 않았다.

당시 참고 입력은 [분리 보관한 JSON](../../../camera_capture/_review/training_reference_inputs_20261110)에 있다.
보관된 생성 도구는 이 자료를 읽도록 경로를 수정해 제거된 준비 ZIP에 의존하지 않게 했다.
이전 노트북은 보관용이며, 다음 학습의 모델·조건·실행 횟수는 미정이다.

전체 이동·삭제 대상, 해시와 실제 적용 기록은
[정리 결과 보고서](OUTPUT_REORGANIZATION_20261110.md)에 연결돼 있다.
