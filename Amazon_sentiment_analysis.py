import re
import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

warnings.filterwarnings("ignore")


def safe_read_csv(path: str, nrows=None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(p, encoding="utf-8", nrows=nrows)
        if df.shape[1] == 1:
            raise ValueError("Single-column parse; trying fallback")
        return df
    except Exception:
        try:
            df = pd.read_csv(p, header=None, engine='python', encoding='latin-1',
                             sep='","', names=['polarity', 'title', 'text'], nrows=nrows)
            df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
            return df
        except Exception:
            raw = p.read_text(encoding='latin-1', errors='ignore').splitlines()
            rows = [r.split(',') for r in raw if r.strip() != ""]
            df = pd.DataFrame(rows)
            return df


def normalize_label(val):
    """
    Updated mapping to match your files:
      1 -> negative (0)
      2 -> positive (1)
    Also keeps backwards compatibility for 0->0, 'neg'/'pos' strings, and some 1-5 scale heuristics.
    """
    if pd.isna(val):
        return None
    
    try:
        ival = int(float(val))
        if ival == 0:
            return 0
        if ival == 1:
           
            return 0
        if ival == 2:
            
            return 1
        if ival in (4, 5):
            return 1
        return None
    except Exception:
        s = str(val).strip().lower()
        if s in {'negative', 'neg', 'n', '0', 'false'}:
            return 0
        if s in {'positive', 'pos', 'p', '2', 'true'}:
            return 1
        s_clean = re.sub(r"[^\w]", "", s)
        if s_clean in {'negative', 'neg', 'n'}:
            return 0
        if s_clean in {'positive', 'pos', 'p'}:
            return 1
    return None


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(http|https)://\S+|www\.\S+|\S+@\S+", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    cols = list(df.columns)

    if 'polarity' in cols:
        pol_col = 'polarity'
    else:
        pol_col = cols[0] if len(cols) >= 1 else None

    text_col = None
    for candidate in ['text', 'review', 'tweet', 'content']:
        if candidate in cols:
            text_col = candidate
            break

    if text_col is None:
        if 'title' in cols and len(cols) >= 3:
            text_col = cols[2]
        elif len(cols) >= 2:
            text_col = cols[1]
        else:
            text_col = None

    title_col = 'title' if 'title' in cols else None

    if pol_col is None:
        raise RuntimeError("Unable to detect a label/polarity column in the CSV.")

    pol_series = df[pol_col] if pol_col in df.columns else pd.Series([None] * len(df))
    title_series = df[title_col] if title_col in df.columns else pd.Series([''] * len(df))
    text_series = df[text_col] if text_col in df.columns else pd.Series([''] * len(df))

    combined = title_series.fillna('').astype(str) + ' ' + text_series.fillna('').astype(str)
    prepared = pd.DataFrame({'polarity_raw': pol_series, 'full_text_raw': combined})

    prepared['label'] = prepared['polarity_raw'].apply(normalize_label)
    prepared = prepared.dropna(subset=['label']).copy()
    prepared['label'] = prepared['label'].astype(int)

    prepared['cleaned_review'] = prepared['full_text_raw'].apply(clean_text)
    prepared = prepared[prepared['cleaned_review'].str.strip() != '']

    return prepared[['cleaned_review', 'label']]


def build_pipeline(max_features=5000, ngram_range=(1, 2)):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
        ('clf', MultinomialNB())
    ])
    return pipeline


def safe_classification_report(y_true, y_pred):
    unique = sorted(set(y_true))
    if len(unique) == 0:
        print("ERROR: No labels found in y_true.")
        return
    if len(unique) == 1:
        present_label = int(unique[0])
        counts = {int(present_label): int((y_true == present_label).sum())}
        print("\nWARNING: Test set contains only one class after preprocessing.")
        print("Label counts in test set:", counts)
        print("Accuracy (trivial):", accuracy_score(y_true, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        return
    print(classification_report(y_true, y_pred, target_names=['negative', 'positive'], zero_division=0))


def get_safe_proba(pipeline, X):
    if not hasattr(pipeline, "predict_proba"):
        return None

    proba = pipeline.predict_proba(X)
    clf = pipeline.named_steps.get('clf', None)
    n = proba.shape[0]
    out = np.zeros((n, 2), dtype=float)  

    if clf is None or not hasattr(clf, "classes_"):
        if proba.shape[1] == 2:
            return proba
        if proba.shape[1] == 1:
            out[:, 1] = proba[:, 0]
            return out
        return proba

    classes = list(clf.classes_)
    if len(classes) == 2:
        idx0 = classes.index(0) if 0 in classes else None
        idx1 = classes.index(1) if 1 in classes else None
        if idx0 is not None:
            out[:, 0] = proba[:, idx0]
        if idx1 is not None:
            out[:, 1] = proba[:, idx1]
        return out

    present = classes[0]
    if present == 0:
        out[:, 0] = proba[:, 0]
    else:
        out[:, 1] = proba[:, 0]
    return out


def main(args):
    train_path = args.train
    test_path = args.test

    print(f"Loading training data from: {train_path}")
    train_raw = safe_read_csv(train_path)
    print("Raw train shape:", train_raw.shape)
    try:
        print(train_raw.head(2).to_string(index=False))
    except Exception:
        pass

    train_df = prepare_dataframe(train_raw)
    print(f"Training records after cleaning and label filter: {len(train_df)}")

    test_df = None
    if test_path and Path(test_path).exists():
        print(f"\nLoading test data from: {test_path}")
        test_raw = safe_read_csv(test_path)
        print("Raw test shape:", test_raw.shape)
        try:
            print(test_raw.head(2).to_string(index=False))
        except Exception:
            pass
        test_df = prepare_dataframe(test_raw)
        print(f"Test records after cleaning and label filter: {len(test_df)}")

    if test_df is None or len(test_df) == 0:
        if len(train_df) < 5:
            print("WARNING: Not enough data to create separate test set (train size < 5). Using train as test (results optimistic).")
            test_df = train_df.copy()
        else:
            print("No valid test data available after cleaning — performing train/test split (stratified).")
            train_df, test_df = train_test_split(train_df, test_size=args.test_size,
                                                 stratify=train_df['label'], random_state=42)
            print(f"Split -> train: {len(train_df)}, test: {len(test_df)}")

    print("\nLabel distribution (train):")
    print(train_df['label'].value_counts().to_string())
    print("\nLabel distribution (test):")
    print(test_df['label'].value_counts().to_string())

    X_train = train_df['cleaned_review'].values
    y_train = train_df['label'].values
    X_test = test_df['cleaned_review'].values
    y_test = test_df['label'].values

    if len(X_test) == 0:
        raise SystemExit("ERROR: test set is empty after preprocessing. Aborting.")

    pipeline = build_pipeline(max_features=args.max_features, ngram_range=(1, args.max_ngram))

    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    print("\nPredicting on test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy on test set: {acc:.4f}\n")

    safe_classification_report(y_test, y_pred)

    if args.save_model:
        joblib.dump(pipeline, args.save_model)
        print(f"\nSaved trained pipeline to: {args.save_model}")

    if args.cross_val:
        print("\nPerforming stratified cross-validation on training data...")
        skf_splits = min(5, max(2, len(y_train) // 2))
        skf = StratifiedKFold(n_splits=skf_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=args.n_jobs)
        print(f"Cross-val accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}, folds={len(scores)}")

    examples = args.examples or [
        "This product is absolutely amazing! I love it and would highly recommend it.",
        "The quality is terrible and it broke after only one use. I am very disappointed."
    ]
    print("\nSample predictions:")
    proba_samples = get_safe_proba(pipeline, examples) if hasattr(pipeline, "predict_proba") else None
    for i, ex in enumerate(examples):
        cleaned = clean_text(ex)
        pred = pipeline.predict([cleaned])[0]
        label_str = "positive" if int(pred) == 1 else "negative"
        if proba_samples is not None:
            p_neg, p_pos = proba_samples[i, 0], proba_samples[i, 1]
            print(f'Input: "{ex}" -> {label_str} (p_neg={p_neg:.3f}, p_pos={p_pos:.3f})')
        else:
            print(f'Input: "{ex}" -> {label_str}')

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust sentiment analysis training script.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to training CSV")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test CSV (optional)")
    parser.add_argument("--max_features", type=int, default=5000, help="Max TF-IDF features")
    parser.add_argument("--max_ngram", type=int, default=2, help="Max ngram size (1..N)")
    parser.add_argument("--save_model", type=str, default="sentiment_pipeline.joblib", help="File path to save trained pipeline")
    parser.add_argument("--cross_val", action="store_true", help="Perform stratified cross-validation on training data")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for cross-validation")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split size when test file not available")
    parser.add_argument("--examples", nargs="*", help="Example sentences to run through the final model")
    args = parser.parse_args()
    main(args)
