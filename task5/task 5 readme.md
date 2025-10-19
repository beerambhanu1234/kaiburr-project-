# Consumer Complaint Text Classification

This project trains text classifiers to categorize CFPB consumer complaints into four classes:

- 0: Credit reporting, repair, or other
- 1: Debt collection
- 2: Consumer Loan
- 3: Mortgage

Artifacts (EDA plots, metrics, model) are written to `datascience/outputs/`.

## 1) Setup

```bash
# From repo root
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# Or cmd
# .venv\Scripts\activate.bat

pip install -r datascience/requirements-ds.txt
```

## 2) Get Dataset

Download the Consumer Complaint Database (CSV):
- Catalog: https://catalog.data.gov/dataset/consumer-complaint-database
- Direct bulk CSV: https://files.consumerfinance.gov/ccdb/complaints.csv.zip

Unzip and place CSV at:
```
datascience/data/complaints.csv
```

(You can pass a different path with `--csv`.)

## 3) Run EDA, Train, Evaluate, Predict

```bash
python datascience/complaint_classification.py \
  --csv datascience/data/complaints.csv \
  --test_size 0.2
```

What happens:
- Cleans and filters to the 4 categories.
- EDA:
  - `outputs/label_distribution.png`
  - `outputs/text_length_distribution.png`
  - `outputs/eda_summary.json`
- Preprocessing: lowercasing, non-alphanumerics removal, stopword removal, lemmatization (NLTK).
- Features: TF-IDF (1–2 grams).
- Models compared: Logistic Regression, LinearSVC, MultinomialNB.
- Metrics:
  - `outputs/metrics.json`
  - Confusion matrices per model: `outputs/confusion_matrix_<model>.png`
- Best model persisted as: `outputs/best_model_<name>.joblib`
- Prediction demo printed to stdout.

## 4) Predict Only

```bash
python datascience/complaint_classification.py --csv datascience/data/complaints.csv --no_train --predict_text "Sample complaint text here"
```

## Notes

- If NLTK data is missing, it downloads `stopwords`, `wordnet`, and `punkt` automatically at first run.
- Adjust vectorizer or algorithms in `get_models()` and TF-IDF params in `train_and_evaluate()`.
