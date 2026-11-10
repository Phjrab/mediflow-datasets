# MediFlow 데이터·초기 학습 발표 그림

3840 × 2160, 16:9 PNG 6장과 수정 가능한 SVG 6장입니다. PNG는 PowerPoint에 그림으로 삽입할 수 있습니다. SVG는 글꼴이 설치된 환경에서 글자를 수정할 수 있으며 Noto Sans KR 또는 맑은 고딕을 사용합니다.

1. [세 분류기의 초기 데이터 선정](01_initial_data_selection.png) / [SVG](01_initial_data_selection.svg)
2. [초기 분류 클래스와 선정 범위](02_initial_class_scope.png) / [SVG](02_initial_class_scope.svg)
3. [초기 원본 데이터의 학습·검증·평가 분할](03_initial_data_split.png) / [SVG](03_initial_data_split.svg)
4. [세 기본 모델의 초기 원본·증강 학습 결과](04_first_training_comparison.png) / [SVG](04_first_training_comparison.svg)
5. [중복 검사 후 세 데이터셋의 최종 구성](05_clean_dataset_composition.png) / [SVG](05_clean_dataset_composition.svg)
6. [Hair 양호 추가 후 6종 데이터 구성](06_hair_good_added_dataset.png) / [SVG](06_hair_good_added_dataset.svg)

## 발표 배치

개발·학습 과정에는 01·02·03·05·06을 해당 설명에 사용하고, 초기 실험 결과에는 04를 사용하세요. 전체 연구 순서로 설명할 때는 01·02·03·04·05·06입니다. 기존 그림을 덮어쓰지 않고 새로 제작했습니다.

## 출처

- [AI Hub 피부종양 이미지 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71864)
- [AI Hub 안면부 피부질환 이미지 합성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71863)
- [AI Hub 유형별 두피 이미지](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=216)

프로젝트 근거 문서와 원본 결과 JSON의 경로·SHA-256은 figure_sources_and_data.json에 있습니다. 공개 원천 규모는 당시 조사 기록이며 원천 전체 ZIP을 새로 계수하지 않았습니다. 2026-10-26 AI Hub 216 페이지에서는 101,027건을 확인했습니다. 71864·71863 페이지 본문은 웹 도구로 읽히지 않아 저장된 원천 표와 연구 기록을 사용했습니다.

## 해석

- 초기 Skin·Web Skin은 원천 클래스당 1,000장 중 900장을 첫 분할에 사용했습니다. 제외된 100장의 파일별 사유는 남아 있지 않습니다.
- Web Skin의 첫 구성은 Skin·Hair 선정 이후입니다. 같은 시점에 세 데이터가 완성됐다는 뜻이 아닙니다.
- 원본·증강 수량을 구분했습니다. 증강 장수는 독립 사람 수가 아닙니다.
- 정제 후 원본 총량은 Skin 8,949 / Web Skin 4,500 / Hair 5종 12,536입니다.
- Hair 양호 534장을 더한 6종은 별도 원본 13,070장입니다. 파일명 105개 ID의 실재 사람·세션 대응은 독립 확인되지 않았습니다.
- 완전 동일 사진 정리로 모든 사람·병변·세션 누수를 배제한 것은 아닙니다.
- 04의 초기 지표는 원본 결과 JSON 그대로이며, 새로 학습하거나 Test 재추론한 결과가 아닙니다. 정제 전후 점수는 동일 평가의 개선률로 빼지 않습니다.
