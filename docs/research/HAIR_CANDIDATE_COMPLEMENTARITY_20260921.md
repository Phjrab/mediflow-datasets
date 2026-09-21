# Hair 후보 오답 보완성과 ensemble 검토

## 목적

SupCon B1·256과 EfficientNet-B1·384가 서로 다른 사진을 맞힌다면 두 모델의 확률 평균
ensemble이 단일 모델보다 좋아질 수 있다. 두 실험에 공통으로 사용한 **seed 42의 동일한
Validation 1,252장** 예측을 이미지 경로 순서까지 대조해 이 가능성을 확인했다. Test는
열지 않았다.

## 비교 결과

| 후보 | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| SupCon B1·256 | 0.7915335463 | 0.7910880666 |
| B1·384 | **0.7963258786** | **0.7957146810** |
| 두 후보 1:1 확률 평균 | 0.7947284345 | 0.7942285284 |

두 단일 모델이 모두 맞힌 사진은 936장, SupCon만 맞힌 사진은 55장, B1·384만 맞힌 사진은
61장, 둘 다 틀린 사진은 200장이었다. 예측 클래스가 달랐던 사진은 131장이었으며, 두 모델
중 하나라도 맞으면 정답으로 보는 실제로 사용할 수 없는 상한(oracle)은 0.8402555911이었다.
McNemar 정확 양측 검정의 p값은 0.6426674254로, seed 42의 두 후보 정답률 차이를 통계적으로
뚜렷하다고 볼 근거도 확보하지 못했다.

핵심 혼동인 `미세각질↔비듬` 양방향 오류는 SupCon 69건, B1·384 71건, 확률 평균 68건이었다.
일부 보완성은 있지만, 1:1 평균의 전체 Accuracy와 Macro F1이 B1·384 단일 모델보다 낮았다.

## 판단

현재는 ensemble을 채택하지 않는다. 가중치를 Validation에 반복 조정하면 후보 선정 데이터에
과적합될 수 있고, 모델 두 개를 배포하는 비용도 생긴다. 선두 단일 모델인 B1·384에만 SAM을
적용해 optimizer 변경 효과를 먼저 확인한다. 이 판단은 seed 42 Validation 선별 결과이며,
최종 성능 또는 Test 성능에 대한 결론은 아니다.

## 재현 자료

- 요약: `results/hair/experiments/candidate_comparison_20260921/candidate_complementarity_summary.json`
- 이미지별 예측 비교: `candidate_predictions_comparison.csv`
- 발표용 그림: `candidate_complementarity_dashboard.svg`
- 비교 대상의 데이터 ZIP SHA-256:
  `2ac7260663cf69835ba50edb6ae8c7e7ac13be9c73b9f7ea24f60e0342e7e156`
- SupCon 결과: `supcon_compare_20260917_120447_38d75491`
- B1·384 결과: `paper_screen_20260917_141641_874f03f1`
