"""Build a read-only results explorer; never fit or tune a classifier here."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

from evaluate import load_data

ROOT = Path(__file__).resolve().parent


def metrics(counts):
    """Positive label is malignant (1). Undefined precision is represented as null."""
    (tn, fp), (fn, tp) = counts
    if any(type(n) is not int or n < 0 for n in (tn, fp, fn, tp)):
        raise ValueError("Confusion counts must be nonnegative integers")
    if not (tn + fp and fn + tp):
        raise ValueError("Both actual classes are required")
    return dict(
        accuracy=(tn + tp) / (tn + fp + fn + tp),
        balanced_accuracy=((tp / (tp + fn)) + (tn / (tn + fp))) / 2,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn),
        false_positive_rate=fp / (tn + fp),
        false_negative_rate=fn / (tp + fn),
    )


def make_report(root=ROOT):
    result = root / "results"
    methodology = json.loads((result / "methodology.json").read_text("utf8"))
    matrices = json.loads((result / "confusion-matrices.json").read_text("utf8"))
    source = root / "Data/data.csv"
    if hashlib.sha256(source.read_bytes()).hexdigest() != methodology["dataset_sha256"]:
        raise ValueError("Source bytes differ from the evaluated dataset")
    _, labels = load_data(source)
    train = methodology["train_indices"]
    holdout = methodology["holdout_indices"]
    if (
        set(train) & set(holdout)
        or set(train + holdout) != set(labels.index)
        or len(set(train)) != len(train)
        or len(set(holdout)) != len(holdout)
    ):
        raise ValueError("Invalid evaluation partition")
    training_counts = Counter(int(labels.loc[i]) for i in train)
    # Deterministic tie rule, chosen solely from training labels.
    majority = min(training_counts, key=lambda label: (-training_counts[label], label))
    with (result / "holdout-predictions.csv").open(newline="", encoding="utf8") as file:
        predictions = list(csv.DictReader(file))
    indices = [int(row["row_index"]) for row in predictions]
    if len(indices) != len(set(indices)) or set(indices) != set(holdout):
        raise ValueError("Saved predictions do not match the evaluated holdout")
    for row in predictions:
        if int(row["actual"]) != int(labels.loc[int(row["row_index"])]):
            raise ValueError("Prediction labels do not match source rows")
    with (result / "model-comparison.csv").open(newline="", encoding="utf8") as file:
        comparison = {row["model"]: row for row in csv.DictReader(file)}
    models = []
    for name in [*matrices, "Training-majority baseline"]:
        counts = [[0, 0], [0, 0]]
        for row in predictions:
            actual = int(row["actual"])
            predicted = int(row[name]) if name in matrices else majority
            if predicted not in (0, 1):
                raise ValueError("Unexpected predicted label")
            counts[actual][predicted] += 1
        if name in matrices and counts != matrices[name]["counts"]:
            raise ValueError("Saved predictions and confusion matrix disagree")
        scores = metrics(counts)
        cv = None
        if name in matrices:
            saved = comparison[name]
            for field, exported in [
                ("holdout_accuracy", "accuracy"),
                ("holdout_balanced_accuracy", "balanced_accuracy"),
                ("malignant_recall", "recall"),
            ]:
                if abs(float(saved[field]) - scores[exported]) > 1e-12:
                    raise ValueError("Saved scores and predictions disagree")
            cv = dict(
                mean=float(saved["cv_balanced_accuracy"]),
                fold_sd=float(saved["cv_fold_sd"]),
            )
        models.append(dict(name=name, counts=counts, metrics=scores, cv=cv))
    # Source row positions only: no original identifiers or feature measurements.
    observations = [
        dict(
            row_index=int(row["row_index"]),
            actual=int(row["actual"]),
            predictions={
                model["name"]: int(row[model["name"]]) if model["cv"] else majority
                for model in models
            },
        )
        for row in predictions
    ]
    pairs = []
    for a in models:
        for b in models:
            groups = dict(
                both_wrong=[], only_a_wrong=[], only_b_wrong=[], both_correct=[]
            )
            for row in observations:
                wrong_a = row["predictions"][a["name"]] != row["actual"]
                wrong_b = row["predictions"][b["name"]] != row["actual"]
                group = (
                    "both_wrong"
                    if wrong_a and wrong_b
                    else (
                        "only_a_wrong"
                        if wrong_a
                        else "only_b_wrong" if wrong_b else "both_correct"
                    )
                )
                groups[group].append(row["row_index"])
            pairs.append(dict(a=a["name"], b=b["name"], groups=groups))
    return dict(
        schema_version=1,
        observations=observations,
        pairs=pairs,
        source_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        dataset_sha256=methodology["dataset_sha256"],
        predictions_sha256=hashlib.sha256(
            (result / "holdout-predictions.csv").read_bytes()
        ).hexdigest(),
        training_rows=len(train),
        training_classes={str(k): training_counts[k] for k in (0, 1)},
        holdout_rows=len(predictions),
        holdout_classes={
            str(k): sum(int(row["actual"]) == k for row in predictions) for k in (0, 1)
        },
        baseline_label=majority,
        models=models,
        note=methodology["note"],
    )


if __name__ == "__main__":
    report = make_report()
    destination = ROOT / "docs"
    destination.mkdir(exist_ok=True)
    for source in (ROOT / "web").iterdir():
        shutil.copyfile(source, destination / source.name)
    (destination / "data.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    baseline = report["models"][-1]
    errors = ", ".join(
        f'{model["name"]}: {model["counts"][1][0]} missed / {report["holdout_classes"]["1"]}'
        for model in report["models"][:-1]
    )
    summary = f'Saved predictions from the unchanged three-model evaluation reconcile with source labels and confusion counts. {errors}. The training-majority baseline scores {baseline["metrics"]["accuracy"]:.1%} accuracy but misses {baseline["counts"][1][0]} of {report["holdout_classes"]["1"]} malignant-labelled holdout examples. The explorer reports positive-class precision/recall/F1 and balanced accuracy without retraining. This repeatedly reused holdout is exploratory, with no independent or clinical validation.'
    paired = next(
        pair
        for pair in report["pairs"]
        if pair["a"] == "Logistic regression" and pair["b"] == "Random forest"
    )["groups"]
    summary += (
        f" Logistic regression and random forest each make "
        f'{len(paired["both_wrong"]) + len(paired["only_a_wrong"])} and '
        f'{len(paired["both_wrong"]) + len(paired["only_b_wrong"])} saved errors respectively: '
        f'{len(paired["both_wrong"])} shared errors, '
        f'{len(paired["only_a_wrong"])} logistic-only errors and '
        f'{len(paired["only_b_wrong"])} forest-only errors. '
        "Paired counts describe these same rows; they are not a significance test or a new performance estimate."
    )
    (ROOT / "results/explorer-summary.txt").write_text(summary + "\n", encoding="utf8")
    print("Built saved-results explorer: four models, no runtime training.")
