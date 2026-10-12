"""One-command demo: retrains the final model (gradient boosting), prints its
validation report, and prints the saved results of all four models.

    python main.py --data_dir data/raw          # retrain final model + show results table
    python main.py --results_only               # only show the saved results table (no data needed)
"""
import argparse
import json
from pathlib import Path

MODELS = [
    ("gradient_boosting", "Gradient boosting"),
    ("random_forest", "Random forest"),
    ("neural_network", "Neural network"),
    ("knn", "KNN"),
]
MAJORITY_BASELINE = 0.5689  # always predict grade 2


def print_saved_results(metrics_dir="results/metrics"):
    """Read the metrics JSON saved by each model notebook and print one table."""
    metrics_dir = Path(metrics_dir)
    print("\nSaved validation results (from results/metrics/)")
    print(f"{'Model':<20}{'micro-F1':>10}{'macro-F1':>10}   recall (grade 1 / 2 / 3)")
    for key, label in MODELS:
        path = metrics_dir / f"{key}.json"
        if not path.exists():
            print(f"{label:<20}  missing: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        r = [m["per_class"][c]["recall"] for c in ("1", "2", "3")]
        print(f"{label:<20}{m['val_micro_f1']:>10.4f}{m['val_macro_f1']:>10.4f}   "
              f"{r[0]:.2f} / {r[1]:.2f} / {r[2]:.2f}")
    print(f"{'Always grade 2':<20}{MAJORITY_BASELINE:>10.4f}")


def train_final_model(data_dir):
    """Retrain the final model with the configuration selected in notebook 05."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline

    from src.data import build_tree_preprocessor, load_raw, train_val_split
    from src.evaluate import micro_f1

    X, y, _ = load_raw(data_dir)
    X_tr, X_val, y_tr, y_val = train_val_split(X, y)

    model = Pipeline([
        ("preprocessor", build_tree_preprocessor(X_tr)),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=500,          # early stopping ends training at about 350
            max_leaf_nodes=63,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )),
    ])
    model.fit(X_tr, y_tr)
    pred = model.predict(X_val)

    print("Final model: gradient boosting (learning rate 0.1, 63 leaves)")
    print("Validation micro-F1:", round(micro_f1(y_val, pred), 4))
    print(classification_report(y_val, pred))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--metrics_dir", default="results/metrics")
    p.add_argument("--results_only", action="store_true",
                   help="skip retraining and only print the saved results table")
    args = p.parse_args()

    if not args.results_only:
        train_final_model(args.data_dir)
    print_saved_results(args.metrics_dir)
