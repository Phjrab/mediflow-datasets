# Repository instructions

## Scope

This repository contains research artifacts and executable Python tooling for three distinct
image classifiers: `skin` (USB microscope skin), `web_skin` (webcam facial skin), and `hair`
(USB microscope scalp). Keep these domains and their class mappings separate.

## Safety and preservation

- Keep sample images in `data_examples/`, saved models and their original reports in `results/`,
  notebooks in `notebooks/`, preprocessing utilities in `scripts/`, and research documents in
  `docs/research/`. Do not move these artifacts again without an explicit user request.
- Never delete or modify existing images, `.keras` files, result JSON/CSV files, notebooks, or
  research documents unless the user explicitly requests it.
- Do not invent, round, or rewrite reported performance metrics.
- Dataset preprocessing must never delete an existing output directory unless the caller passes
  an explicit `--overwrite` option.
- Do not add external normalization before the saved EfficientNet models. They contain an
  internal `Rescaling(1/255)` layer and expect float32 RGB pixels in the 0–255 range.

## Development workflow

- Read `docs/research/PROJECT_BACKGROUND.md` before planning research or training work; it is the
  source of truth for the research rationale. Follow `docs/ROADMAP.md` for execution phases.
- Keep reusable runtime code under `src/mediflow_datasets/` and tests under `tests/`.
- Resolve model and metadata paths relative to the repository, never to a contributor's PC.
- Preserve class order from the result JSON files; array index is part of the model contract.
- Run `ruff check src tests` and `pytest` after Python changes.
- Model-loading tests require TensorFlow and may be slower than metadata/unit tests.

## Communication and progress reporting

The project owner is not expected to know Python, machine learning, Git, Colab, or infrastructure
terminology. Explain work in plain Korean and do not assume unstated technical knowledge.

For every multi-step task:

- Before changing files, briefly state the current project phase, the goal of this task, and the
  files or systems that are expected to change.
- During long work, report meaningful progress and explain technical terms the first time they
  appear.
- After implementation, always report these five items:
  1. `현재 단계`: where this work sits in `docs/ROADMAP.md`.
  2. `이번에 한 일`: concrete files, code, data, or configuration changed.
  3. `확인 결과`: tests, measurements, and whether they passed or failed.
  4. `다음 단계`: the recommended next action and why it comes next.
  5. `사용자가 할 일`: approvals, logins, device capture, or decisions required from the user;
     write `없음` when none are required.
- Separate completed work from proposed work. Never imply that a proposal, upload, Colab run,
  training run, commit, or push happened unless it was actually completed and verified.
- When an experiment changes, list the variables kept fixed and the single variable being tested
  so the user can understand what caused a result difference.
- When blocked, explain the cause, its impact, and the smallest action needed from the user.
