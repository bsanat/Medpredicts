# MedPredict ML

- Python package `src/medpredict_ml`; use the local `.venv` and pinned `pyproject.toml`.
- `python -m medpredict_ml download`, `prepare`, `train`, `evaluate`, `export`, `verify` form the reproducible pipeline.
- Training inputs: age, sex (male=0/female=1), BMI, total cholesterol SI, fasting plasma glucose SI. Preserve the feature order.
- One row per `SEQN`; strict one-to-one joins. Never impute the target or split the same participant across partitions.
- Fit preprocessing only on train. Choose configuration on validation. Freeze selection before opening the test set. Do not tune after test evaluation.
- Data source: official NHANES 2017–March 2020 public-use files, with NCHS conditions and attribution. Raw/processed rows stay ignored by Git.
- This estimates a current measured SBP, not a future value, diagnosis, or substitute for a blood pressure device.
- Keep ONNX preprocessing/inverse target transform embedded and verify Python/ONNX/Java parity. Artifacts and metadata must carry matching SHA-256/version.
- Run `python -m pytest --cov=medpredict_ml --cov-report=term-missing` and the actual pipeline. Synthetic fixtures are only for tests and runtime parity, never training/evaluation claims.

## New classification research

- The five-feature order, complete-case policy and SBP commands above describe the existing M2 experiment. Keep its artifacts and viewed test frozen. Do not overwrite them for a classification task.
- Read `.agents/skills/medpredict-ml-engineering/SKILL.md` for ML planning or implementation. New targets require their own task ID, cohort, label contract, split and output directory before training.
- The diagnostic roadmap is a proposal, not evidence of confirmed diagnoses, a released API or new metrics. A single elevated laboratory value and a questionnaire report are distinct from an adjudicated diagnosis.
- Preserve pinned dependencies and CPU-first execution. Do not introduce Lightning, cloud jobs, tracking uploads or paid APIs merely because an external skill suggests them.
- For documentation-only work, validate the changed documents/skill metadata; do not rerun training, unlock the test set or execute the full pipeline. For actual pipeline changes, use an isolated workspace and report the checks actually run.

