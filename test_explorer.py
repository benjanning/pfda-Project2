import json
from pathlib import Path
import tempfile
import shutil
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

from build_explorer import make_report, metrics, ROOT


class ExplorerTests(unittest.TestCase):
    def test_every_exported_metric_reconciles_with_saved_predictions(self):
        report = make_report()
        saved = pd.read_csv(ROOT / "results/holdout-predictions.csv")
        for model in report["models"]:
            predicted = (
                saved[model["name"]]
                if model["cv"]
                else np.full(len(saved), report["baseline_label"])
            )
            self.assertEqual(
                model["counts"],
                confusion_matrix(saved.actual, predicted, labels=[0, 1]).tolist(),
            )
            for name, expected in {
                "recall": recall_score(saved.actual, predicted),
                "f1": f1_score(saved.actual, predicted),
                "balanced_accuracy": balanced_accuracy_score(saved.actual, predicted),
            }.items():
                self.assertAlmostEqual(model["metrics"][name], expected)
            if any(predicted):
                self.assertAlmostEqual(
                    model["metrics"]["precision"],
                    precision_score(saved.actual, predicted),
                )
            else:
                self.assertIsNone(model["metrics"]["precision"])

    def test_baseline_uses_training_labels_only(self):
        report = make_report()
        source = pd.read_csv(ROOT / "Data/data.csv")
        method = json.loads((ROOT / "results/methodology.json").read_text("utf8"))
        counts = source.loc[method["train_indices"], "diagnosis"].value_counts()
        self.assertEqual(report["baseline_label"], {"B": 0, "M": 1}[counts.idxmax()])
        self.assertEqual(report["training_classes"], {"0": 250, "1": 148})
        baseline = report["models"][-1]
        self.assertEqual(baseline["counts"], [[107, 0], [64, 0]])
        self.assertIsNone(baseline["cv"])
        self.assertEqual(baseline["metrics"]["balanced_accuracy"], 0.5)
        self.assertEqual(baseline["metrics"]["recall"], 0)
        self.assertIn("reused", report["note"])

    def test_known_counts_and_undefined_precision(self):
        result = metrics([[104, 3], [7, 57]])
        self.assertAlmostEqual(result["precision"], 57 / 60)
        self.assertAlmostEqual(result["recall"], 57 / 64)
        self.assertAlmostEqual(result["f1"], 114 / 124)
        self.assertAlmostEqual(result["false_positive_rate"], 3 / 107)
        self.assertAlmostEqual(result["false_negative_rate"], 7 / 64)
        self.assertIsNone(metrics([[107, 0], [64, 0]])["precision"])
        self.assertEqual(metrics([[107, 0], [64, 0]])["f1"], 0)
        for counts in [[[1, -1], [1, 1]], [[1, 0], [0, 0]], [[True, 0], [1, 1]]]:
            with self.assertRaises(ValueError):
                metrics(counts)

    def test_every_paired_partition_matches_independent_source_masks(self):
        report = make_report()
        saved = pd.read_csv(ROOT / "results/holdout-predictions.csv")
        saved["Training-majority baseline"] = report["baseline_label"]
        self.assertEqual(len(report["pairs"]), 16)
        for pair in report["pairs"]:
            a = saved[pair["a"]].ne(saved.actual)
            b = saved[pair["b"]].ne(saved.actual)
            expected = {
                "both_wrong": saved.loc[a & b, "row_index"].tolist(),
                "only_a_wrong": saved.loc[a & ~b, "row_index"].tolist(),
                "only_b_wrong": saved.loc[~a & b, "row_index"].tolist(),
                "both_correct": saved.loc[~a & ~b, "row_index"].tolist(),
            }
            self.assertEqual(pair["groups"], expected)
            all_rows = [row for group in pair["groups"].values() for row in group]
            self.assertEqual(len(all_rows), 171)
            self.assertEqual(set(all_rows), set(saved.row_index))
            self.assertEqual(
                len(expected["both_wrong"]) + len(expected["only_a_wrong"]),
                int(a.sum()),
            )
            self.assertEqual(
                len(expected["both_wrong"]) + len(expected["only_b_wrong"]),
                int(b.sum()),
            )
            reverse = next(
                p
                for p in report["pairs"]
                if p["a"] == pair["b"] and p["b"] == pair["a"]
            )
            self.assertEqual(
                expected["only_a_wrong"], reverse["groups"]["only_b_wrong"]
            )
            if pair["a"] == pair["b"]:
                self.assertEqual(expected["only_a_wrong"], [])
                self.assertEqual(expected["only_b_wrong"], [])

    def test_anonymous_paired_rows_reconcile_and_source_outputs_remain_unchanged(self):
        files = [
            ROOT / "results" / name
            for name in [
                "holdout-predictions.csv",
                "methodology.json",
                "confusion-matrices.json",
                "model-comparison.csv",
            ]
        ]
        before = [path.read_bytes() for path in files]
        report = make_report()
        saved = pd.read_csv(files[0])
        for row, source in zip(report["observations"], saved.to_dict(orient="records")):
            self.assertEqual(set(row), {"row_index", "actual", "predictions"})
            self.assertEqual(row["row_index"], source["row_index"])
            self.assertEqual(row["actual"], source["actual"])
            for model in report["models"]:
                self.assertEqual(
                    row["predictions"][model["name"]],
                    source[model["name"]] if model["cv"] else report["baseline_label"],
                )
        self.assertEqual(before, [path.read_bytes() for path in files])

    def test_source_mismatch_fails_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "results", root / "results")
            (root / "Data").mkdir()
            (root / "Data/data.csv").write_text("changed source bytes", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "Source bytes differ"):
                make_report(root)

    def test_inconsistent_saved_artifacts_fail_before_export(self):
        for corruption in ("label", "partition", "matrix"):
            with self.subTest(
                corruption=corruption
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(ROOT / "results", root / "results")
                shutil.copytree(ROOT / "Data", root / "Data")
                if corruption == "matrix":
                    path = root / "results/confusion-matrices.json"
                    data = json.loads(path.read_text("utf8"))
                    data["Logistic regression"]["counts"][0][0] += 1
                    path.write_text(json.dumps(data), encoding="utf8")
                else:
                    path = root / "results/holdout-predictions.csv"
                    data = pd.read_csv(path)
                    if corruption == "label":
                        data.loc[0, "actual"] = 1 - data.loc[0, "actual"]
                    else:
                        data.loc[0, "row_index"] = data.loc[1, "row_index"]
                    data.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    make_report(root)


if __name__ == "__main__":
    unittest.main()
