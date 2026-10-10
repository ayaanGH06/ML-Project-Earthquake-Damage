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

The dataset files are **git-ignored** and should not be committed to the repository.

Alternatively, if the project supports it, you can specify a different dataset location using the `EQ_DATA_DIR` environment variable or the `--data_dir` command-line flag.

## Project structure
```
data/raw/          dataset CSVs (not committed)
notebooks/         one notebook per stage/model
src/data.py        shared loading, train/val split, preprocessing
src/evaluate.py    shared metrics + confusion matrix plots
results/           saved figures and metric JSONs
report/            one-page write-up (PDF)
presentation/      slide deck
main.py            quick end-to-end demo
```

## Setup and run
**Local**
```bash
git clone https://github.com/ayaanGH06/ML-Project-Earthquake-Damage.git && cd ML-Project-Earthquake-Damage/
pip install -r requirements.txt
python main.py --data_dir data/raw
```
**Google Colab:** open any notebook in `notebooks/`, edit the paths in the first
(setup) cell, and run all cells.

## Who did what
| Part | Owner |
|---|---|
| Preprocessing, EDA (`01`), KNN (`02`), Neural network (`03`) | Syed Ayaan Hasan |
| Random Forest (`04`), Gradient boosting (`05`), Comparison (`06`) | Sriniket Deeduvanu |

## Results
_To Be filled. Fill in the final comparison table here (model, validation micro-F1)._
