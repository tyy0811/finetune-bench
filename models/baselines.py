"""Non-DL baselines: TF-IDF + LogReg (B1), LightGBM tabular-only (B2)."""

import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import compute_metrics


def train_tfidf_logreg(
    train_narratives: list[str],
    train_labels: np.ndarray,
    test_narratives: list[str],
    test_labels: np.ndarray,
    class_names: list[str],
    max_features: int = 20_000,
) -> dict:
    """B1: TF-IDF + Logistic Regression baseline."""
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_train = tfidf.fit_transform(train_narratives)
    X_test = tfidf.transform(test_narratives)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    test_preds = clf.predict(X_test)
    test_metrics = compute_metrics(test_labels.tolist(), test_preds.tolist(), class_names)

    mlflow.set_experiment("finetune-bench")
    with mlflow.start_run(run_name="B1_tfidf_logreg"):
        mlflow.log_params({
            "variant": "B1",
            "model": "TF-IDF + LogisticRegression",
            "max_features": max_features,
        })
        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

    print(f"B1 Test Macro-F1: {test_metrics.macro_f1:.4f}")
    print(f"B1 Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"\n{test_metrics.report}")

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "variant": "B1",
    }


def train_lightgbm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """B2: LightGBM tabular-only baseline."""
    import lightgbm as lgb

    num_classes = len(class_names)
    train_data = lgb.Dataset(train_features, label=train_labels)
    val_data = lgb.Dataset(val_features, label=val_labels, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "class_weight": "balanced",
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
    )

    test_proba = model.predict(test_features)
    test_preds = np.argmax(test_proba, axis=1)

    test_metrics = compute_metrics(test_labels.tolist(), test_preds.tolist(), class_names)

    mlflow.set_experiment("finetune-bench")
    with mlflow.start_run(run_name="B2_lightgbm"):
        mlflow.log_params({
            "variant": "B2",
            "model": "LightGBM",
            "num_leaves": 31,
        })
        mlflow.log_metrics({
            "test_macro_f1": test_metrics.macro_f1,
            "test_accuracy": test_metrics.accuracy,
        })

    print(f"B2 Test Macro-F1: {test_metrics.macro_f1:.4f}")
    print(f"B2 Test Accuracy: {test_metrics.accuracy:.4f}")
    print(f"\n{test_metrics.report}")

    return {
        "test_macro_f1": test_metrics.macro_f1,
        "test_accuracy": test_metrics.accuracy,
        "per_class_f1": test_metrics.per_class_f1.tolist(),
        "variant": "B2",
    }
