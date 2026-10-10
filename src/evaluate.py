"""Shared evaluation helpers so all models are compared the same way."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)


def micro_f1(y_true, y_pred):
    """Competition metric. (For single-label multiclass, micro-F1 == accuracy.)"""
    return f1_score(y_true, y_pred, average="micro")


def evaluate_model(name, y_true, y_pred, save_dir="results"):
    """Print a report, save the confusion matrix figure and metrics JSON."""
    save_dir = Path(save_dir)
    (save_dir / "figures").mkdir(parents=True, exist_ok=True)
    (save_dir / "metrics").mkdir(parents=True, exist_ok=True)

    score = micro_f1(y_true, y_pred)
    print(f"{name}: micro-F1 = {score:.4f}")
    print(classification_report(y_true, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"{name} - confusion matrix")
    plt.savefig(save_dir / "figures" / f"{name}_confusion.png", dpi=150, bbox_inches="tight")
    plt.show()

    with open(save_dir / "metrics" / f"{name}.json", "w") as f:
        json.dump({"model": name, "val_micro_f1": score}, f, indent=2)
    return score
