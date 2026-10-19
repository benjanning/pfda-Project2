"""New reproducible evaluation alongside the original educational notebook.
All features are retained. Preprocessing is fitted inside each training fold.
Not clinical validation. Outputs must not be interpreted as diagnostic guidance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score

def evaluate():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'Data/data.csv')
    y = data['diagnosis'].map({'B': 0, 'M': 1})
    X = data.drop(columns=['diagnosis', 'id', 'Unnamed: 32'])
    if y.isna().any() or X.isna().any().any():
        raise ValueError('Unexpected missing values or labels')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40, stratify=y)
    assert set(X_train.index).isdisjoint(X_test.index)
    models = {
        'Logistic regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=40)),
        'Decision tree': DecisionTreeClassifier(random_state=40),
        'Random forest': RandomForestClassifier(n_estimators=200, random_state=40, n_jobs=1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy')
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        rows.append(dict(model=name, cv_balanced_accuracy=float(scores.mean()), cv_fold_sd=float(scores.std(ddof=1)),
                         holdout_accuracy=accuracy_score(y_test, prediction),
                         holdout_balanced_accuracy=balanced_accuracy_score(y_test, prediction),
                         malignant_recall=recall_score(y_test, prediction), false_negatives=int(fn)))
    out = root / 'results'
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'model-comparison.csv', index=False)
    (out / 'methodology.json').write_text(json.dumps(dict(seed=40, features=X.shape[1], training_rows=len(X_train), holdout_rows=len(X_test),
        training_classes=y_train.value_counts().to_dict(), holdout_classes=y_test.value_counts().to_dict(), folds=5,
        note='Exploratory single-dataset comparison; holdout reuse across models is not an unbiased model-selection estimate. No independent or clinical validation.'), indent=2))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    print(evaluate().to_string(index=False))
