
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

IN_PATH = "../outputs/preprocessed_tasks.csv"
OUT_DIR = "../outputs"
TEXT_COL = "clean_text"
LABEL_COL = "category"  # what we're classifying tasks into


def load_data():
    df = pd.read_csv(IN_PATH)
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")]
    print(f"Loaded {len(df)} rows for classification, target = '{LABEL_COL}'")
    print(df[LABEL_COL].value_counts())
    return df


def extract_features(df):
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df[TEXT_COL])
    y = df[LABEL_COL]
    print(f"TF-IDF matrix shape: {X.shape}")
    return X, y, vectorizer


def evaluate_model(name, model, X_test, y_test, labels):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    fname = f"{OUT_DIR}/confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    X, y, vectorizer = extract_features(df)
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    results = []

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    results.append(evaluate_model("Naive Bayes", nb, X_test, y_test, labels))

    svm = LinearSVC(random_state=42, max_iter=5000)
    svm.fit(X_train, y_train)
    results.append(evaluate_model("SVM", svm, X_test, y_test, labels))

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/classification_results.csv", index=False)
    print(f"\nSaved comparison table -> {OUT_DIR}/classification_results.csv")
    print(results_df)

    # save the best model + vectorizer for reuse
    best = results_df.loc[results_df["f1"].idxmax(), "model"]
    best_model = nb if best == "Naive Bayes" else svm
    joblib.dump(best_model, f"{OUT_DIR}/best_classifier.pkl")
    joblib.dump(vectorizer, f"{OUT_DIR}/tfidf_vectorizer.pkl")
    print(f"\nBest model ({best}) saved -> {OUT_DIR}/best_classifier.pkl")


if __name__ == "__main__":
    main()