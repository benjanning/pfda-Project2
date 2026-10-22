# Paired saved-error evidence

## Question and result

Do approaches with similar aggregate scores make mistakes on the same examples? On the existing 171-row holdout, logistic regression and random forest each make ten errors: four shared, six logistic-only and six forest-only; both are correct on 155 rows. This describes the saved outputs only. No ranking, significance test or new unbiased estimate follows from it.

## Implementation

The existing exporter first checks dataset hash, exact training/holdout partition, source labels, saved predictions, confusion matrices and scores. It then exports only zero-based source row positions, actual binary labels and saved predictions for the three classifiers plus the training-majority baseline. Source identifiers and feature measurements are not exported in this view. B=0 and M=1 follow the supplied CSV's existing evaluation mapping.

All sixteen ordered pairs are partitioned into both wrong, only A wrong, only B wrong and both correct. The four groups preserve original holdout order and cover all 171 unique row positions. Same-approach comparisons validly have no discordant rows; swapping A/B swaps the two unique-error groups.

The frontend reads the verified export. Native selectors choose A, B and an evidence group. Four labelled counts keep the denominator visible. The table reports source position, actual label, predictions and outcomes: actual 1 predicted 0 is a false negative; actual 0 predicted 1 is a false positive. A visible empty state handles zero-row groups. The table scrolls within a labelled region on mobile; no colour interpretation is required.

## Verification and limits

Independent pandas masks from holdout-predictions.csv verify every exported group, source row, count, error marginal, self comparison and swap symmetry. Tests also assert that exporting leaves original evaluation outputs byte-for-byte unchanged. Browser tests independently read the original saved CSV and reconcile all sixteen pairs and four groups at desktop/mobile widths, with keyboard control and empty-state checks. Existing metric, provenance, reproducibility and loading-failure tests remain.

The existing split, classifiers, seeds, preprocessing, training-only cross-validation, dataset and prediction files are unchanged. The holdout has been repeatedly reused and remains exploratory. Overlap is not causal evidence, clinical validation, calibration, an ensemble recommendation or a new model-selection result. No patient inputs or diagnostic service are provided.

Run the existing reproduction commands in README.md. To verify the live static app after a successful release, set EXPLORER_BASE to its public demo URL and run npm run test:browser. The single existing source release/export and central review-PR process remain authoritative.
