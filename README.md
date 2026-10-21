# Breast cancer classification

Compare three classifiers, inspect their mistakes and understand the limits of a single public teaching dataset.

![Verified saved-results explorer](https://raw.githubusercontent.com/benjanning/pfda-Project2/portfolio-published/portfolio/export/assets/explorer.jpg)

**Python · pandas · scikit-learn · Matplotlib**

[Explore saved results](https://benjanning.github.io/pfda-Project2/) · [Visual case study](https://benjanning.github.io/projects/projects/machine-learning/) · [Portfolio](https://benjanning.github.io/projects/)

## Inspect the errors

The read-only results lab shows genuine confusion counts, malignant-class precision, recall and F1, balanced accuracy and their denominators. Selecting a model reads saved outputs; it never trains or accepts patient inputs. It also compares a baseline chosen from the **training** majority (250 benign / 148 malignant labels). That always-benign guess reaches 62.6% accuracy while missing all 64 malignant-labelled holdout examples.

Undefined precision is displayed as N/A when there are no malignant predictions; the baseline F1 is zero because it misses every positive label. A native keyboard-accessible selector, explicit matrix labels, comparison table and small-screen layout make the evidence inspectable without relying on colour.

[Metric definitions, export checks and limits](RESULTS-LAB.md)

## The investigation

The original coursework notebook explores the Wisconsin Diagnostic dataset. A separate, reproducible portfolio evaluation retains all 30 features, uses a deterministic stratified split and compares logistic regression, a decision tree and a random forest. Preprocessing is fitted inside each training fold.

The point is to explain the errors as well as the scores. Saved predictions, labelled confusion matrices, dataset hashes, exact partitions and dependency versions provide an audit trail. Identifier columns never enter the feature matrix.

![Confusion matrices on the same reused holdout](results/confusion-matrices.png)

Lower-left cells are malignant-labelled examples predicted benign. These errors must remain visible; a high overall accuracy does not make this a diagnostic system.

![Descriptive source feature distributions](results/feature-distributions.png)

## Reproduce the evidence

```bash
python -m pip install -r requirements.txt
python evaluate.py
python -m unittest discover -p "test_*.py" -v
python plot_portfolio.py
python build_explorer.py
python -m http.server 8091 --bind 127.0.0.1 --directory docs
```

The explorer uses generated `docs/data.json`; source UI lives in `web/`. For browser checks, run `npm ci`, install Chromium with `npx playwright install chromium`, then `npm run test:browser` (no separately running server needed).

Regression tests check source/partition integrity, reconcile predictions with metrics and matrices, and reproduce result files byte-for-byte. Results are in [model-comparison.csv](results/model-comparison.csv), [holdout-predictions.csv](results/holdout-predictions.csv), [confusion-matrices.json](results/confusion-matrices.json) and [methodology.json](results/methodology.json). `plot_evaluation.py` remains a compatibility entry point for the current figure generator.

Seed 40, 398 training examples, 171 held-out examples, five stratified training-only cross-validation folds. The chart's error bars show fold standard deviation, not confidence intervals. All three models are compared on the same split; the holdout has also been reused across portfolio iterations. It is exploratory, not an untouched final evaluation or an unbiased model-selection estimate. There is no independent cohort, clinical validation, calibration study or causal feature-importance claim.

## Source and original work

The [UCI Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) is the public source; the existing CSV is preserved. No clinical decision service is provided. The original `PFDA Project 2.ipynb` and its references remain separate and unchanged in this phase; its historical narrative may differ from later reproducible outputs.

The single source release workflow reruns calculations, regression tests, figures and desktop/mobile browser journeys, deploys the generated read-only explorer to GitHub Pages, exports commit-identified visuals and a computed result summary, and makes them available to the central portfolio's review-PR synchronisation. [Original documentation and references](COURSEWORK.md) · [Visual provenance](assets/PROVENANCE.md).
