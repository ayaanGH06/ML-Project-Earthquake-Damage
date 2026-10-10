"""One-command demo: trains a quick baseline and prints the validation score.

    python main.py --data_dir /path/to/folder/with/csvs
"""
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from src.data import build_tree_preprocessor, load_raw, train_val_split
from src.evaluate import micro_f1

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=None)
    args = p.parse_args()

    X, y, _ = load_raw(args.data_dir)
    X_tr, X_val, y_tr, y_val = train_val_split(X, y)
    model = make_pipeline(
        build_tree_preprocessor(X_tr),
        RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42),
    )
    model.fit(X_tr, y_tr)
    print("Validation micro-F1:", round(micro_f1(y_val, model.predict(X_val)), 4))
    # TODO (team): replace with the final best model + load saved results
