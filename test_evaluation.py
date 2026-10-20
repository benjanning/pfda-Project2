import unittest,json,tempfile
from pathlib import Path
import pandas as pd,numpy as np
from evaluate import evaluate,load_data
ROOT=Path(__file__).parent
class EvaluationTests(unittest.TestCase):
    def test_source_integrity_and_partition(self):
        X,y=load_data(ROOT/'Data/data.csv');m=json.loads((ROOT/'results/methodology.json').read_text())
        self.assertEqual(X.shape,(569,30));self.assertEqual(set(y),{0,1})
        train=set(m['train_indices']);holdout=set(m['holdout_indices'])
        self.assertFalse(train&holdout);self.assertEqual(train|holdout,set(range(569)))
        self.assertEqual((len(train),len(holdout)),(398,171))
        self.assertNotIn('id',X);self.assertNotIn('diagnosis',X)
    def test_predictions_reconcile_with_metrics_and_matrices(self):
        predictions=pd.read_csv(ROOT/'results/holdout-predictions.csv');metrics=pd.read_csv(ROOT/'results/model-comparison.csv');matrices=json.loads((ROOT/'results/confusion-matrices.json').read_text())
        for row in metrics.itertuples():
            matrix=np.array(matrices[row.model]['counts']);actual=predictions.actual;pred=predictions[row.model]
            self.assertEqual(matrix.sum(),171);self.assertEqual(matrix[1,0],sum((actual==1)&(pred==0)))
            self.assertEqual(row.false_negatives,matrix[1,0]);self.assertAlmostEqual(row.malignant_recall,matrix[1,1]/matrix[1].sum())
            self.assertAlmostEqual(row.holdout_accuracy,(matrix[0,0]+matrix[1,1])/171)
    def test_results_are_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            evaluate(d)
            for name in ['model-comparison.csv','holdout-predictions.csv','confusion-matrices.json']:
                self.assertEqual((Path(d)/name).read_bytes(),(ROOT/'results'/name).read_bytes())
if __name__=='__main__':unittest.main()
