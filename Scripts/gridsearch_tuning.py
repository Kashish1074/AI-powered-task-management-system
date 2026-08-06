
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import time

IN_PATH = "../outputs/preprocessed_tasks.csv"
OUT_DIR = "../outputs"
TEXT_COL = "clean_text"
CATEGORY_COL = "category"
LABEL_COL = "priority"
PRIORITY_ORDER = ["Low", "Medium", "High", "Urgent"]

PARAM_GRID = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [8, 12, 16, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
}


def load_data():
    df = pd.read_csv(IN_PATH)
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")]
    return df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [CATEGORY_COL]),
        ]
    )
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    return Pipeline([("preprocess", preprocessor), ("clf", model)])


def evaluate(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    labels_present = [l for l in PRIORITY_ORDER if l in set(y_test) | set(preds)]
    print(classification_report(y_test, preds, labels=labels_present, zero_division=0))
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, preds, labels_present


def main():
    df = load_data()
    X = df[[TEXT_COL, CATEGORY_COL]]
    y = df[LABEL_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # --- Baseline (Day 15/16 settings, for honest comparison) ---
    baseline_pipeline = build_pipeline()
    baseline_pipeline.set_params(clf__n_estimators=200, clf__max_depth=12)
    baseline_pipeline.fit(X_train, y_train)
    baseline_result, _, _ = evaluate("Baseline RF (untuned)", baseline_pipeline, X_test, y_test)

    # --- GridSearchCV tuning ---
    print(f"\nRunning GridSearchCV over {len(PARAM_GRID['clf__n_estimators']) * len(PARAM_GRID['clf__max_depth']) * len(PARAM_GRID['clf__min_samples_split']) * len(PARAM_GRID['clf__min_samples_leaf'])} combinations x 5-fold CV — this may take a few minutes on CPU...")
    start = time.time()
    grid_pipeline = build_pipeline()
    search = GridSearchCV(
        grid_pipeline, PARAM_GRID, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\nGridSearchCV finished in {elapsed:.1f}s")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1: {search.best_score_:.3f}")

    tuned_result, preds, labels_present = evaluate("Tuned RF (GridSearchCV)", search.best_estimator_, X_test, y_test)

    # --- Compare and decide which to keep ---
    results_df = pd.DataFrame([baseline_result, tuned_result])
    results_df.to_csv(f"{OUT_DIR}/priority_tuning_comparison.csv", index=False)
    print(f"\n=== Comparison ===\n{results_df}")

    if tuned_result["f1"] > baseline_result["f1"]:
        print(f"\nTuning IMPROVED F1 ({baseline_result['f1']:.3f} -> {tuned_result['f1']:.3f}). Saving tuned model as best.")
        joblib.dump(search.best_estimator_, f"{OUT_DIR}/best_priority_model.pkl")
        final_preds, final_labels = preds, labels_present
        final_name = "Tuned RF"
    else:
        print(f"\nTuning did NOT improve F1 ({baseline_result['f1']:.3f} vs {tuned_result['f1']:.3f}). "
              f"Keeping baseline model — this is a valid result: the untuned defaults were "
              f"already well-suited to this dataset size, and tuning found nothing better "
              f"within the grid searched.")
        final_preds = baseline_pipeline.predict(X_test)
        final_labels = [l for l in PRIORITY_ORDER if l in set(y_test) | set(final_preds)]
        final_name = "Baseline RF"

    cm = confusion_matrix(y_test, final_preds, labels=final_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=final_labels, yticklabels=final_labels)
    plt.title(f"Final Priority Model ({final_name}) — Confusion Matrix")
    plt.ylabel("True Priority")
    plt.xlabel("Predicted Priority")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_priority_final.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR}/confusion_matrix_priority_final.png")


if __name__ == "__main__":
    main()