# Hair 실험 결과 색인

현재 최종 후보는 B1·384·Label Smoothing 0.05·Adam이다. Test Accuracy는
`0.8003194888178914`, Test Macro F1은 `0.8002222282821301`이다.

| 단계 | 결과 폴더/문서 | 핵심 결과 | 채택 |
|---|---|---|---|
| Clean 초기 6개 | `docs/research/HAIR_EXPERIMENT_SUMMARY_20260907.md` | B1·256 v1 후보 Test F1 0.7885764577 | 이전 후보 |
| SupCon | `supcon_compare_20260917_120447_38d75491/` | Validation F1 0.7910880666 | 보조 후보 |
| DINOv2·EfficientNetV2-S·B1 384 | `paper_screen_20260917_141641_874f03f1/` | B1·384 Validation F1 0.7957146810 | 선두 |
| 후보 보완성 | `candidate_comparison_20260921/` | 1:1 ensemble F1 0.7942285284 | 미채택 |
| SAM | `sam_screen_20260921_145602_a5e8c406/` | SAM F1 0.7954748921 | 미채택 |
| 최종 Test | `docs/research/HAIR_FINAL_CANDIDATE_V2_20260921.md` | Test F1 0.8002222283 | 최종 후보 |

최종 패키지는 `results/hair/candidates/public_candidate_v2_b1_384_ls005_adam_20260921_155906_82311da4.zip`이다.
