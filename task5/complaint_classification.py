import os
import sys
import re
import json
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from joblib import dump, load

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    _ = nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    _ = nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

CATEGORIES = {
    'Credit reporting, repair, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3,
}
INV_CATEGORIES = {v: k for k, v in CATEGORIES.items()}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", ' ', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def preprocess_texts(texts: List[str]) -> List[str]:
    sw = set(stopwords.words('english'))
    processed = []
    for t in texts:
        t = clean_text(t)
        tokens = [tok for tok in t.split(' ') if tok and tok not in sw]
        tokens = lemmatize_tokens(tokens)
        processed.append(' '.join(tokens))
    return processed


def load_data(csv_path: str) -> pd.DataFrame:
    # Expecting Consumer Complaint database CSV with columns like 'product', 'consumer_complaint_narrative' or 'complaint_what_happened'
    df = pd.read_csv(csv_path, low_memory=False)

    # Try to find text column
    text_col_candidates = [
        'consumer_complaint_narrative',  # older CFPB naming
        'complaint_what_happened',       # alternate naming
        'complaint_narrative',
        'narrative'
    ]
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Could not find complaint text column in CSV. Tried: {text_col_candidates}. Columns present: {list(df.columns)[:20]} ...")

    if 'product' not in df.columns:
        raise ValueError("CSV must include 'product' column for category mapping.")

    # Map products to our 4 categories
    df = df[df['product'].isin(CATEGORIES.keys())].copy()
    df['label'] = df['product'].map(CATEGORIES)
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    df = df.dropna(subset=['text', 'label'])

    # Remove very short texts
    df = df[df['text'].astype(str).str.len() > 20]

    return df


def eda(df: pd.DataFrame) -> None:
    # Label distribution plot
    counts = df['label'].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=[INV_CATEGORIES[i] for i in counts.index], y=counts.values, palette='Blues_d')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    plt.title('Label Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'))
    plt.close()

    # Text length distribution
    text_len = df['text'].str.len()
    plt.figure(figsize=(8, 4))
    sns.histplot(text_len, bins=50, color='#4e79a7')
    plt.title('Text Length Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'text_length_distribution.png'))
    plt.close()

    summary = {
        'num_rows': int(len(df)),
        'avg_text_len': float(text_len.mean()),
        'label_counts': counts.to_dict(),
    }
    with open(os.path.join(OUTPUT_DIR, 'eda_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def get_models() -> dict:
    return {
        'logreg': LogisticRegression(max_iter=200, n_jobs=None),
        'linearsvc': LinearSVC(),
        'mnb': MultinomialNB(),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[str, Pipeline, dict]:
    X = df['text'].tolist()
    y = df['label'].astype(int).values

    # Preprocess
    X_clean = preprocess_texts(X)

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=test_size, random_state=random_state, stratify=y)

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, model in get_models().items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)),
            ('clf', model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=[INV_CATEGORIES[i] for i in sorted(INV_CATEGORIES.keys())], digits=3)

        results[name] = {
            'accuracy': acc,
            'f1_weighted': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
        }

        # Save confusion matrix plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[INV_CATEGORIES[i] for i in range(4)], yticklabels=[INV_CATEGORIES[i] for i in range(4)])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png'))
        plt.close()

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipe

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Persist best model
    model_path = os.path.join(OUTPUT_DIR, f'best_model_{best_model_name}.joblib')
    dump(best_pipeline, model_path)

    return best_model_name, best_pipeline, results


def predict_texts(model: Pipeline, texts: List[str]) -> List[Tuple[str, int, str]]:
    processed = preprocess_texts(texts)
    preds = model.predict(processed)
    return [(t, int(y), INV_CATEGORIES[int(y)]) for t, y in zip(texts, preds)]


def main():
    parser = argparse.ArgumentParser(description='Consumer Complaint Text Classification')
    parser.add_argument('--csv', type=str, default=os.path.join(os.path.dirname(__file__), 'data', 'complaints.csv'), help='Path to Consumer Complaint CSV')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--no_train', action='store_true', help='Skip training; only run prediction using existing model')
    parser.add_argument('--predict_text', type=str, nargs='*', help='Texts to classify')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found at {args.csv}\nPlease download the Consumer Complaint Database CSV and place it there.\nDataset catalog: https://catalog.data.gov/dataset/consumer-complaint-database\nCFPB bulk download: https://files.consumerfinance.gov/ccdb/complaints.csv.zip", file=sys.stderr)
        sys.exit(1)

    print('Loading data...')
    df = load_data(args.csv)
    print(f"Loaded {len(df)} rows after filtering categories {list(CATEGORIES.keys())}.")

    print('Running EDA...')
    eda(df)
    print(f"EDA artifacts saved to {OUTPUT_DIR}")

    model: Pipeline
    if not args.no_train:
        print('Training and evaluating models...')
        best_name, model, results = train_and_evaluate(df, test_size=args.test_size)
        print(f"Best model: {best_name} | F1(weighted)={max(r['f1_weighted'] for r in results.values()):.4f}")
    else:
        # Load latest model file
        candidates = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('best_model_') and f.endswith('.joblib')]
        if not candidates:
            print('No trained model found. Run without --no_train first.')
            sys.exit(1)
        latest = sorted(candidates)[-1]
        model = load(os.path.join(OUTPUT_DIR, latest))
        print(f'Loaded model: {latest}')

    # Prediction demo
    demo_texts = args.predict_text or [
        "I received repeated calls about a debt that I do not owe.",
        "My credit report lists incorrect information and the agency will not fix it.",
        "The mortgage company increased my escrow payment without proper notice.",
        "I applied for a consumer loan and the terms were misrepresented."
    ]
    preds = predict_texts(model, demo_texts)
    print('\nPrediction samples:')
    for t, y, label in preds:
        print(f"- {label} ({y}): {t}")


if __name__ == '__main__':
    main()
