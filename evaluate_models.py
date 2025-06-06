import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer_utils import tokenize_codes, extract_features, extract_codebert_features
from data_loader import load_jsonl
from config import RANDOM_SEED
from model_trainer import train_models

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    human_data = load_jsonl("data/random_human_1680.jsonl", 0)
    ai_data = load_jsonl("data/random_ai_1680.jsonl", 1)
    all_data = human_data + ai_data
    random.shuffle(all_data)

    codes = [code for code, _ in all_data]
    labels = [label for _, label in all_data]

    metrics_dict = run_full_evaluation_from_data(codes, labels, method='tfidf')
    df = pd.DataFrame(metrics_dict).T
    print(df)
    return df

def run_full_evaluation(human_path, ai_path, method='tfidf', progress_callback=None):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    human_data = load_jsonl(human_path, 0)
    ai_data = load_jsonl(ai_path, 1)
    all_data = human_data + ai_data
    random.shuffle(all_data)

    codes = [code for code, _ in all_data]
    labels = [label for _, label in all_data]

    return run_full_evaluation_from_data(codes, labels, method, progress_callback)

def run_full_evaluation_from_data(codes, labels, method='tfidf', progress_callback=None):
    if method in ('tfidf', 'tf-idf'):
        tokenized_codes = tokenize_codes(codes)
        X, vectorizer = extract_features(tokenized_codes, progress_callback=progress_callback)
    elif method == 'codebert':
        X = extract_codebert_features(codes, progress_callback=progress_callback)
        vectorizer = None
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")

    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  Pass vectorizer into train_models so it gets returned properly
    models, vectorizer, metrics_dict = train_models(X_train, y_train, X_test, y_test, vectorizer=vectorizer)

    for name, model in models.items():
        metrics_dict[name]["model"] = model

    if method in ('tfidf', 'tf-idf') and vectorizer is not None:
        metrics_dict["__vectorizer__"] = {"vectorizer": vectorizer}

    return metrics_dict
