---
name: medpredict-ml-research
description: MedPredict ML research rules (M1 Diabetes, M2 systolic BP on NHANES, M3 glucose) — data splits, leakage prevention, baselines vs neural nets, metrics, uncertainty, ONNX export and parity, and honest reporting. Use for any work in ml-ai/, model artifacts, explainability, or the AI assistant.
---

# MedPredict ML research conventions

Source of truth: `documentation/09_План_тестирования_и_оценки.md`, `06_Спецификация_требований.md`, `ml-ai/README.md`. Read them before changing protocols.

## Modules (state as of 24 Oct 2026)
- **M1** sklearn Diabetes progression — offline Ridge(alpha=10) vs MLP done, final MAE 39.466 (target scale, not glucose, not probability). Not integrated.
- **M2** current systolic BP, NHANES 2017–Mar 2020, inputs: age, sex, BMI, total cholesterol (mmol/L), fasting plasma glucose (mmol/L). 3 318 participants, 2 324/497/497. MLP 5→32→16→1 weighted test MAE 11.225 vs Ridge 11.268 (no convincing NN advantage). SBP ≥150: all 42 underestimated, MAE 41.366. ONNX bundle exists only locally; published backend has no ONNX provider.
- **M3** future CGM glucose — CGMacros audited, **NO_GO**. Do not train without a valid longitudinal dataset.

## Hard rules
- Split by participant (M3: by subject *before* windowing, no future leakage). Fit imputation/scaling/encoding/feature selection/HPO only on train or CV folds.
- The M2 test set has already been viewed. Any new M2 claim needs **new independent data** or a pre-registered protocol; never tune on the old test.
- Always compare against the same-feature baseline (mean, Ridge, HistGradientBoosting) on the same rows. Report MAE, RMSE, signed bias, subgroup errors (range of true value, age, sex) with group sizes; paired bootstrap (1 000) 95% CI for metric differences.
- Fix seeds, search budget and primary metric before running. Never pick the best seed on test.
- Never show "±MAE" or an arbitrary confidence as an individual interval. Individual uncertainty (conformal, quantile, ensembles) must report measured coverage and width on held-out data.
- Feature importance / SHAP describe model behaviour, not causes of disease. Say so in UI and text.
- Keep units, target meaning, feature order and preprocessing identical from training → ONNX → Java → UI. Preprocessing inside the ONNX graph; record SHA-256, library versions, and Python/ONNX/Java numeric parity.
- No raw medical rows, participant IDs or private splits in Git. Store procedures and manifests.
- Research prototype: no diagnosis, treatment or medication advice; no clinical validation for Kazakhstan. Demo value 134.2 is not a model output.

## AI assistant (supervisor recommendation, MED-55)
- Separate from numeric prediction; never produces or changes a predicted number.
- Answers only from approved project documents (fields, units, app functions, model limits, PDF export); refuses diagnosis/treatment; cites the source section.
- Evaluate with a fixed question set (grounded accuracy, refusal correctness, RU/KK/EN) before release.

## Completion
Report what actually ran, versions, seeds, artifacts, metrics with CIs, negative results, and remaining limits. Status vocabulary: не запускалось / выполнено / воспроизведено / принято с ограничениями / отклонено.
