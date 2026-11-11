# 학습 전 output 구조 정리 결과 — 2026-11-10

> 2026-11-11 정리 상태: **분석·검토 또는 과정 기록, 후속 학습 상태는 최신 로드맵 참조**. [현재 작업 안내](../../README.md) · [전체 문서 대조표](../../DOCUMENT_INDEX.md). 아래 본문은 작성 시점 기록이며, 본문·원본 JSON의 이전 경로는 정리 대조표로 추적한다.

사용자가 분석 결과를 보고서에 사용하고 다음 학습 구성을 새로 정하기 위해 정리를 요청했다.
이전 9개 통합 학습과 별도 학습 준비본을 보관 이력으로 전환했다.
새 학습의 모델·분야 조합·실험 조건·실행 횟수는 아직 정하지 않았다.
이 작업에서 학습이나 속도 최적화를 실행하지 않았다.

## 1. 최종 구조

`output/`에는 아래 네 폴더만 남겼다.

| 폴더 | 용도 |
|---|---|
| [domain_adaptation_20261108_final](../../../output/domain_adaptation_20261108_final) | 얼굴 초기 분석 실행본과 원래 실행 ZIP |
| [hair_skin_batch_20261109](../../../output/hair_skin_batch_20261109) | 두피·피부 분석 실행본 |
| [adaptation_preflight_v2_20261109](../../../output/adaptation_preflight_v2_20261109) | 세 분야 사전 점검 실행본과 원래 실행 ZIP |
| [pdf](../../../output/pdf) | 공부·발표 설명 PDF |

이 네 폴더의 기존 파일 내용은 변경하지 않았다.
[실제 코드와 결과 대응표](../camera_adaptation/CAMERA_ADAPTATION_EXECUTED_CODE_INDEX_20261110.md)에서 분석 결과와 연결한다.

## 2. 이동한 그림·수치

| 이전 위치 | 새 위치 |
|---|---|
| `output/presentation_data_figures_20261026` | [presentation/report_assets/data_figures_20261026](../../../presentation/report_assets/data_figures_20261026) |
| `output/validation_comparisons_20261027` | [presentation/report_assets/validation_comparisons_20261027](../../../presentation/report_assets/validation_comparisons_20261027) |

15개와 7개, 합계 22개 파일을 그대로 이동하고 해시를 대조했다.
PNG·SVG·CSV·근거 JSON·설명·공유용 ZIP의 내용은 바꾸지 않았다.

## 3. 제거한 이전 준비 폴더

- `output/domain_adaptation_20261108`
- `output/hair_adaptation_training_20261109`
- `output/hair_adaptation_training_ready_20261109`
- `output/skin_web_adaptation_resolved_20261109`
- `output/skin_web_adaptation_ready_20261109`
- `output/skin_web_adaptation_final_20261109`
- `output/skin_web_resolution_release_20261109`
- `output/adaptation_9experiments_20261109`

이 폴더들은 실행 준비본이며 실제 Source 분석 결과 폴더 자체가 아니다.
삭제 전 파일 목록·해시를 기록하고, 원래 ZIP의 모든 파일 내용이 보관 이력 또는 프로젝트의
동일 파일에 남는지 확인했다. 중단한 학습의 원격 Drive 삭제 상태를 확인한 작업은 아니다.

## 4. 보고서용 이력과 참고 입력

- [코드·노트북·설정 보관 ZIP](../../../camera_capture/_review/history/training_preparation/code_and_configuration.zip)
- [원래 파일 목록과 ZIP 항목별 대응표](../../../camera_capture/_review/history/training_preparation/reorganization_plan.json)
- [실제 이동·삭제 적용 기록](../../../camera_capture/_review/history/training_preparation/application_result.json)
- [보관 이력 읽는 법](../../../camera_capture/_review/history/training_preparation/README.md)
- [참고 입력 JSON과 출처](../../../camera_capture/_review/training_reference_inputs_20261110/provenance.json)
- [사용자 제공 두피 1·2회차 중간 기록](../../../camera_capture/_review/history/executed_code)

준비본의 코드·설정은 `standalone/<이전 폴더명>/` 및 `runtime/<이전 폴더명>/`에 있다.
모델과 입력 배열 등의 중복 파일은 기존 파일의 경로·해시를 연결해 중복 저장을 줄였다.
원래 대용량 ZIP 컨테이너의 압축 메타데이터까지 같은 바이트로 보존한 것은 아니다.
보고서용 개별 파일 내용과 당시 해시를 보존했다.

실제 연구 분석 자료는 다음 위치에 그대로 있다.

- [전체 Source Train·실제 촬영 분석](../../../camera_capture/_review/domain_adaptation_colab_20261109)
- [세 분야 사전 점검과 당시 코드 사본](../../../camera_capture/_review/preflight_v2_fbe37d44)
- [촬영 원본과 검토 자료](../../../camera_capture)

## 5. 코드·문서 갱신

`scripts/build_domain_nine.py`는 과거 실행본 재현 도구로 보존한다.
제거된 두피·피부·얼굴의 큰 준비 ZIP 대신, 그 안에서 그대로 분리한 참고 JSON을 읽게 했다.
반복문이나 학습 알고리즘·변환 수치·클래스 순서·학습 설정은 바꾸지 않았다.
이 도구를 다음 학습의 확정 계획으로 해석하지 않는다.

로드맵, 실행 코드 색인, 과거 학습 안내의 보관 상태·경로,
발표 자료실 경로와 `notebooks/camera_adaptation/README.md`를 갱신했다.
문서 갱신 전 사본은 보관 이력의 `before_path_updates/`에 있다.
정리 후에도 날짜가 붙은 기존 학습 노트북 자체는 보관용으로 남아 있다.

## 6. 확인 결과

이동·보존에 대한 대조 항목과 테스트 결과는
[최종 검증 기록](../../../camera_capture/_review/history/training_preparation/final_verification.json)에 저장한다.
전체 테스트가 진행 중일 때는 완료로 단정하지 않고 해당 파일의 상태를 확인한다.

새 학습 전에 남은 일은 학습 구성 재설계, 실행 속도·진행 표시·중단과 이어받기 개선이다.
기존 분석 결과는 같은 입력 자료에 대한 근거로 재사용한다.
