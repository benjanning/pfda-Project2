# Explain the errors before ranking the models

The lab is a browser view of the existing reproducible evaluation. It makes no new model-development claim: the three estimators, seed, split, features and training-only preprocessing remain unchanged. A fourth row is a simple contextual baseline. It always predicts the majority class of the training labels (250 benign, 148 malignant), without using holdout outcomes to choose a class.

## Counts and denominators

Malignant (1) is the positive label. The 171 repeatedly reused holdout rows contain 107 benign and 64 malignant labels. Matrix rows are actual labels; columns are predictions.

| Quantity | Definition | Meaning |
| --- | --- | --- |
| Precision | TP / (TP + FP) | Among malignant predictions, how many matched actual labels? |
| Recall | TP / (TP + FN) | Among actual malignant labels, how many were found? |
| F1 | 2TP / (2TP + FP + FN) | Combines precision and recall; does not include true negatives |
| Balanced accuracy | (TP / (TP + FN) + TN / (TN + FP)) / 2 | Gives each actual class equal weight |
| False-negative rate | FN / (TP + FN) | Fraction of actual malignant labels missed |
| False-positive rate | FP / (TN + FP) | Fraction of actual benign labels predicted malignant |
| Accuracy | (TP + TN) / all rows | Overall agreement; can hide class imbalance |

The baseline gets 107 / 171 correct (62.6%) yet misses all 64 malignant labels. Its recall and F1 are zero, balanced accuracy is 50%, and precision is undefined because there are no malignant predictions. The export uses JSON `null` for undefined precision and the UI displays N/A. An undefined ratio is never silently reported as a measured zero.

## One auditable route to the interface

`evaluate.py` retains the existing fixed models and writes the original predictions, comparisons, matrices and methodology. `build_explorer.py` checks the exact dataset hash, partition coverage/disjointness, saved row identities and source labels. It reconstructs each confusion matrix from saved predictions and reconciles its scores with the existing comparison CSV. It then computes the baseline and descriptive metrics, generates `docs/data.json`, and copies the source interface from `web/`.

The browser reads this JSON. Every metric, matrix and comparison-table row uses that same export. Selection only changes which saved record is displayed; there are no inputs for patient data, training, thresholds or model tuning. Dataset/prediction hashes and the source revision are visible and downloadable. Generated `docs/` is ignored in Git and built by the existing release workflow, avoiding checked-in revision values that become stale after a merge.

Eight Python regressions cover existing evaluation reproducibility, source/partition integrity, independent scikit-learn reconciliation of exported metrics, baseline derivation and corrupted-artifact rejection. Three browser journeys check every model and baseline at desktop/mobile widths, keyboard selection, actual rendered values, no page overflow and a visible load-failure state. Genuine UI images come from these journeys. The single source workflow verifies, captures, exports the versioned portfolio package and deploys Pages; the central portfolio still uses a separately reviewed update PR.

## Interpretation limits

Cross-validation means and fold standard deviations come from the existing five training-only folds. A fold SD is descriptive variability, not a confidence interval. No CV score is invented for the training-majority baseline. The holdout has been reused across models and portfolio iterations, so it is exploratory and cannot support unbiased model selection or an untouched final-test claim. No independent cohort, calibration, clinical validation or causal feature attribution is claimed. The baseline contextualizes imbalance; it does not improve any classifier.
