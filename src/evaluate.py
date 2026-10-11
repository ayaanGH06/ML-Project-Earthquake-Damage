"""Shared evaluation helpers so all models are compared the same way."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)


def micro_f1(y_true, y_pred):
    """Competition metric. (For single-label multiclass, micro-F1 == accuracy.)"""
    return f1_score(y_true, y_pred, average="micro")


def evaluate_model(name, y_true, y_pred, save_dir="results"):
    """Print a report, save the confusion matrix figure and a metrics JSON.

    The JSON (results/metrics/<name>.json) holds micro-F1, macro-F1 and
    per-class precision / recall / F1 so the comparison notebook can read
    every model's results without retraining anything.
    """
    save_dir = Path(save_dir)
    (save_dir / "figures").mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics").mkdir(parents=True, exist_ok=True)

    score = micro_f1(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"{name}: micro-F1 = {score:.4f}")
    print(classification_report(y_true, y_pred))

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for c in sorted(np.unique(y_true)):
        r = report[str(c)]
        per_class[str(c)] = {
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1-score"]),
            "support": int(r["support"]),
        }

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"{name} - confusion matrix")
    plt.savefig(save_dir / "figures" / f"{name}_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()

    with open(save_dir / "metrics" / f"{name}.json", "w") as f:
        json.dump(
            {
                "model": name,
                "val_micro_f1": float(score),
                "val_macro_f1": float(macro),
                "per_class": per_class,
            },
            f,
            indent=2,
        )
    return score


def save_permutation_importance(name, model, X_val, y_val, n_rows=5000,
                                n_repeats=5, random_state=42, save_dir="results",
                                n_jobs=1):
    """Permutation importance of the raw input columns, saved to
    results/metrics/<name>_feature_importance.json.

    `model` must be a fitted pipeline that accepts the raw feature DataFrame.
    Importance = mean drop in validation micro-F1 when one column is shuffled.

    n_jobs=1 on purpose: with n_jobs=-1 joblib copies the fitted model to every
    worker process, which can exhaust shared memory / disk for a large random
    forest ("No space left on device"). The models themselves still predict in
    parallel internally.
    """
    from sklearn.inspection import permutation_importance

    n = min(n_rows, len(X_val))
    X_s = X_val.sample(n=n, random_state=random_state)
    y_s = y_val.loc[X_s.index]
    imp = permutation_importance(
        model, X_s, y_s, scoring="f1_micro", n_repeats=n_repeats,
        random_state=random_state, n_jobs=n_jobs,
    )
    result = {
        "model": name,
        "n_rows": int(n),
        "n_repeats": int(n_repeats),
        "importance_mean": dict(zip(X_s.columns, map(float, imp.importances_mean))),
        "importance_std": dict(zip(X_s.columns, map(float, imp.importances_std))),
    }
    out = Path(save_dir) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}_feature_importance.json", "w") as f:
        json.dump(result, f, indent=2)
    return result
