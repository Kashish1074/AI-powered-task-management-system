
import re
import json
import joblib
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUT_DIR = "../outputs"

# --- NLP cleaning (mirrors nlp_preprocessing.py exactly, so predictions
# use the same preprocessing the models were trained on) ---
stemmer = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[a-zA-Z]+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


# --- Workload-aware assignment scoring (mirrors workload_assignment.py) ---
AFFINITY_WEIGHT = 1.0
WORKLOAD_WEIGHT = 1.0
URGENCY_PENALTY = 0.6
OVERLOAD_THRESHOLD = 1.15


def score_agent(agent, task_category, task_priority, affinity, workload, avg_load):
    cat_affinity = affinity[agent].get(task_category, 0)
    normalized_load = workload[agent] / avg_load if avg_load > 0 else 0
    score = AFFINITY_WEIGHT * cat_affinity - WORKLOAD_WEIGHT * normalized_load
    is_overloaded = workload[agent] > avg_load * OVERLOAD_THRESHOLD
    is_urgent = task_priority in ("High", "Urgent")
    if is_overloaded and is_urgent:
        score -= URGENCY_PENALTY
    return score


def recommend_assignee(task_category, task_priority, affinity, workload):
    agents = list(workload.keys())
    avg_load = np.mean(list(workload.values()))
    scores = {
        agent: score_agent(agent, task_category, task_priority, affinity, workload, avg_load)
        for agent in agents
    }
    best_agent = max(scores, key=scores.get)
    cat_pct = affinity[best_agent].get(task_category, 0) * 100
    load = workload[best_agent]
    load_vs_avg = "above" if load > avg_load else "below" if load < avg_load else "at"
    reasoning = (
        f"{best_agent} handles {cat_pct:.0f}% of their tasks in '{task_category}', "
        f"currently has {load} open tasks ({load_vs_avg} team average of {avg_load:.1f})"
    )
    return best_agent, reasoning, scores


class TaskPipeline:
    """Loads all trained models once, then predicts on new task descriptions."""

    def __init__(self, out_dir=OUT_DIR):
        self.category_model = joblib.load(f"{out_dir}/best_classifier.pkl")
        self.category_vectorizer = joblib.load(f"{out_dir}/tfidf_vectorizer.pkl")
        self.priority_model = joblib.load(f"{out_dir}/best_priority_model.pkl")

        # priority model may be XGBoost (needs label decoding) or Random Forest
        # (predicts string labels directly) — check which
        try:
            self.priority_label_encoder = joblib.load(f"{out_dir}/priority_label_encoder.pkl")
        except FileNotFoundError:
            self.priority_label_encoder = None

        with open(f"{out_dir}/agent_profiles.json") as f:
            profiles = json.load(f)
        self.affinity = profiles["affinity"]
        self.workload = profiles["workload"]

    def predict(self, description: str) -> dict:
        clean = clean_text(description)

        # 1. Category
        text_vec = self.category_vectorizer.transform([clean])
        category = self.category_model.predict(text_vec)[0]

        # 2. Priority (model expects a DataFrame with clean_text + category cols)
        priority_input = pd.DataFrame({"clean_text": [clean], "category": [category]})
        priority_pred = self.priority_model.predict(priority_input)[0]
        if self.priority_label_encoder is not None and isinstance(priority_pred, (int, np.integer)):
            priority = self.priority_label_encoder.inverse_transform([priority_pred])[0]
        else:
            priority = priority_pred

        # 3. Recommended assignee
        assignee, reasoning, scores = recommend_assignee(
            category, priority, self.affinity, self.workload
        )

        return {
            "description": description,
            "predicted_category": category,
            "predicted_priority": priority,
            "recommended_assignee": assignee,
            "reasoning": reasoning,
        }


def main():
    pipeline = TaskPipeline()

    test_descriptions = [
        "Application crashes when checkout button is clicked on the payment page",
        "Please add dark mode support to the admin dashboard",
        "URGENT: production deployment is broken on the auth server",
        "Update the documentation for the reporting API endpoint",
        "Users report the search feature stops responding during peak load",
    ]

    print("=== Integrated Pipeline Test ===\n")
    results = []
    for desc in test_descriptions:
        result = pipeline.predict(desc)
        results.append(result)
        print(f"Task: {result['description']}")
        print(f"  -> Category: {result['predicted_category']}")
        print(f"  -> Priority: {result['predicted_priority']}")
        print(f"  -> {result['reasoning']}")
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/integrated_pipeline_demo.csv", index=False)
    print(f"Saved demo results -> {OUT_DIR}/integrated_pipeline_demo.csv")


if __name__ == "__main__":
    main()