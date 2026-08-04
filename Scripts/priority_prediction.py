
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib

IN_PATH = "../outputs/preprocessed_tasks.csv"
OUT_DIR = "../outputs"
TEXT_COL = "clean_text"
CATEGORY_COL = "category"
LABEL_COL = "priority"
PRIORITY_ORDER = ["Low", "Medium", "High", "Urgent"]


def load_data():
    df = pd.read_csv(IN_PATH)
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].str.strip() != "")]
    print(f"Loaded {len(df)} rows for priority prediction")
    print(df[LABEL_COL].value_counts().reindex(PRIORITY_ORDER))
    return df


def build_pipeline(model):
    # TF-IDF on text + one-hot on category, combined via ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [CATEGORY_COL]),
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("clf", model)])
    return pipeline


def evaluate(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    labels_present = [l for l in PRIORITY_ORDER if l in set(y_test) | set(preds)]
    print(classification_report(y_test, preds, labels=labels_present, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=labels_present)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels_present, yticklabels=labels_present)
    plt.title(f"{name} — Priority Confusion Matrix")
    plt.ylabel("True Priority")
    plt.xlabel("Predicted Priority")
    plt.tight_layout()
    fname = f"{OUT_DIR}/confusion_matrix_priority_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved {fname}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    X = df[[TEXT_COL, CATEGORY_COL]]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    results = []
    pipelines = {}

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
    )
    rf_pipeline = build_pipeline(rf)
    rf_pipeline.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf_pipeline, X_test, y_test))
    pipelines["Random Forest"] = rf_pipeline

    # --- XGBoost (needs numeric-encoded labels) ---
    label_encoder = LabelEncoder()
    label_encoder.fit(PRIORITY_ORDER)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric="mlogloss"
    )
    xgb_pipeline = build_pipeline(xgb)
    xgb_pipeline.fit(X_train, y_train_enc)

    # evaluate() expects string labels for the report/plot, so decode predictions back
    preds_enc = xgb_pipeline.predict(X_test)
    preds = label_encoder.inverse_transform(preds_enc)
    xgb_result = {
        "model": "XGBoost",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
    }
    print(f"\n=== XGBoost ===")
    print(f"Accuracy:  {xgb_result['accuracy']:.3f}")
    print(f"Precision: {xgb_result['precision']:.3f}")
    print(f"Recall:    {xgb_result['recall']:.3f}")
    print(f"F1:        {xgb_result['f1']:.3f}")
    labels_present = [l for l in PRIORITY_ORDER if l in set(y_test) | set(preds)]
    print(classification_report(y_test, preds, labels=labels_present, zero_division=0))

    cm = confusion_matrix(y_test, preds, labels=labels_present)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels_present, yticklabels=labels_present)
    plt.title("XGBoost — Priority Confusion Matrix")
    plt.ylabel("True Priority")
    plt.xlabel("Predicted Priority")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_priority_xgboost.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR}/confusion_matrix_priority_xgboost.png")

    results.append(xgb_result)
    pipelines["XGBoost"] = xgb_pipeline

    # --- Compare and save ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/priority_classification_results.csv", index=False)
    print(f"\n=== Comparison ===")
    print(results_df)
    print(f"\nSaved -> {OUT_DIR}/priority_classification_results.csv")

    best_name = results_df.loc[results_df["f1"].idxmax(), "model"]
    best_pipeline = pipelines[best_name]
    joblib.dump(best_pipeline, f"{OUT_DIR}/best_priority_model.pkl")
    if best_name == "XGBoost":
        joblib.dump(label_encoder, f"{OUT_DIR}/priority_label_encoder.pkl")
    print(f"\nBest model ({best_name}) saved -> {OUT_DIR}/best_priority_model.pkl")


if __name__ == "__main__":
    main()