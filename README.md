# Predicting Earthquake Damage to Buildings (Nepal 2015)

UE24CS352A - Machine Learning mini-project.
Predict the damage grade (1 = low, 2 = medium, 3 = near-total destruction) of
buildings hit by the 2015 Gorkha earthquake. Metric: micro-averaged F1.

**Team:** Syed Ayaan Hasan (PES2UG24CS545) and Sriniket Deeduvanu (PES2UG24CS521)

## Dataset

The dataset is from the DrivenData competition **[Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/)**. The original competition provides the earthquake building-damage dataset used for this project.

For convenience, the required dataset files have been uploaded to Google Drive:

**[Download the dataset from Google Drive](https://drive.google.com/drive/folders/1SK97eOOp3-lE7OAG4u4h-36QI1lKBzQ6?usp=sharing)**

The folder contains:

- `train_values.csv` — training features
- `train_labels.csv` — training labels
- `test_values.csv` — test features

Download all three files and place them in:

```text
data/raw/
```

## Project structure
```
data/raw/          dataset CSVs (not committed)
notebooks/         one notebook per stage/model
src/data.py        shared loading, train/val split, preprocessing
src/evaluate.py    shared metrics + confusion matrix plots
results/           saved figures and metric JSONs
report/            one-page write-up (PDF)
presentation/      slide deck
main.py            retrains the final model and prints the saved results table
```

## Setup and run
**Local**
```bash
git clone https://github.com/ayaanGH06/ML-Project-Earthquake-Damage.git && cd ML-Project-Earthquake-Damage/
pip install -r requirements.txt
python main.py --data_dir data/raw     # retrain the final model, then print the results table
python main.py --results_only          # only print the saved results (no data needed)
```
**Google Colab:** open any notebook in `notebooks/`, edit the paths in the first
(setup) cell, and run all cells.

## Who did what
| Part | Owner |
|---|---|
| Preprocessing, EDA (`01`), KNN (`02`), Neural network (`03`) | Syed Ayaan Hasan |
| Random Forest (`04`), Gradient boosting (`05`), Comparison (`06`) | Sriniket Deeduvanu |

## Results
All models use the same stratified 80/20 split (seed 42: 208,480 training and 52,121 validation buildings).
Scores are on the validation set. The competition test set has no public labels and was not scored.

| Model | Selected setting | Validation micro-F1 | macro-F1 | Recall, grades 1 / 2 / 3 |
|---|---|---|---|---|
| **Gradient boosting** | learning rate 0.1, 63 leaves, stopped at 350 iterations | **0.7459** | **0.6929** | 0.52 / 0.85 / 0.63 |
| Random forest | 300 trees, minimum leaf size 2 | 0.7314 | 0.6645 | 0.45 / 0.86 / 0.59 |
| Neural network | hidden layers 256-128-64, L2 0.01 | 0.6972 | 0.6328 | 0.44 / 0.81 / 0.57 |
| KNN | k = 51, uniform weights | 0.6802 | 0.6066 | 0.40 / 0.82 / 0.52 |
| Baseline | always predict grade 2 | 0.5689 | - | - |

Gradient boosting is the final model. Location (`geo_level_1_id`) is the most important feature for both tree models.
Main limitations: hyperparameters were chosen on the same validation split that is reported, there is a single split,
and duplicate buildings may favour high-capacity models. See `report/` for the write-up and `presentation/` for the slides.
