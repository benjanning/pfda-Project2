"""New reproducible evaluation alongside the original educational notebook.
All features are retained. Preprocessing is fitted inside each training fold.
Not clinical validation. Outputs must not be interpreted as diagnostic guidance.
"""
from pathlib import Path
import json
import hashlib,platform,sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

def load_data(path):
    data=pd.read_csv(path)
    y=data['diagnosis'].map({'B':0,'M':1})
    X=data.drop(columns=['diagnosis','id','Unnamed: 32'])
    if X.shape[1]!=30 or y.isna().any() or not np.isfinite(X.to_numpy()).all() or not data['id'].is_unique:
        raise ValueError('Expected 30 finite features, unique source identifiers and B/M labels.')
    return X,y

def evaluate(output=None):
    root = Path(__file__).resolve().parent
    source=root/'Data/data.csv'
    X,y=load_data(source)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40, stratify=y)
    assert set(X_train.index).isdisjoint(X_test.index)
    models = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=40)),
        'Decision tree': DecisionTreeClassifier(random_state=40),
        'Random forest': RandomForestClassifier(n_estimators=200, random_state=40, n_jobs=1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    rows = []
    predictions=pd.DataFrame({'row_index':X_test.index,'actual':y_test.to_numpy()})
    matrices={}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        predictions[name]=prediction
        matrices[name]=dict(labels=['Benign (0)','Malignant (1)'],counts=[[int(tn),int(fp)],[int(fn),int(tp)]])
        rows.append(dict(model=name, cv_balanced_accuracy=float(scores.mean()), cv_fold_sd=float(scores.std(ddof=1)),
                         holdout_accuracy=accuracy_score(y_test, prediction),
                         holdout_balanced_accuracy=balanced_accuracy_score(y_test, prediction),
                         malignant_recall=recall_score(y_test, prediction), false_negatives=int(fn)))
    out = Path(output) if output else root / 'results'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'model-comparison.csv', index=False)
    predictions.to_csv(out/'holdout-predictions.csv',index=False)
    summary='All 30 features; 398 training examples; 171 examples in a repeatedly reused exploratory holdout. False negatives among '+str(int(y_test.sum()))+' malignant-labelled holdout examples: '+', '.join(row['model']+' '+str(row['false_negatives']) for row in rows)+'. Training-only five-fold cross-validation keeps preprocessing inside each fold. Saved predictions reconcile with the metrics and confusion matrices; repeated runs reproduce the same result files. No independent or clinical validation.'
    (out/'portfolio-summary.txt').write_text(summary+'\n',encoding='utf-8')
    (out/'confusion-matrices.json').write_text(json.dumps(matrices,indent=2)+'\n',encoding='utf-8')
    (out / 'methodology.json').write_text(json.dumps(dict(seed=40, features=X.shape[1], training_rows=len(X_train), holdout_rows=len(X_test),
        training_classes=y_train.value_counts().to_dict(), holdout_classes=y_test.value_counts().to_dict(), folds=5,
        train_indices=X_train.index.tolist(),holdout_indices=X_test.index.tolist(),positive_label='Malignant (1)',preprocessing='StandardScaler fitted within each LogisticRegression training fold; all 30 features retained',
        dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),python=platform.python_version(),scikit_learn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,
        note='Exploratory single-dataset comparison; the holdout has been reused across models and portfolio iterations. It is not an unbiased model-selection estimate or an untouched final test. No independent or clinical validation.'), indent=2),encoding='utf-8')
    return pd.DataFrame(rows)

if __name__ == '__main__':
    print(evaluate().to_string(index=False))
