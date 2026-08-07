
import sys
import os
import json
import pandas as pd
import streamlit as st

# integrated_pipeline.py lives in scripts/, add it to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from integrated_pipeline import TaskPipeline

st.set_page_config(page_title="AI Task Management System", page_icon="📋", layout="centered")

OUT_DIR = "outputs"


@st.cache_resource
def load_pipeline():
    return TaskPipeline(out_dir=OUT_DIR)


st.title("📋 AI-Powered Task Management System")
st.caption("NLP-based task classification, priority prediction, and workload-aware assignment")

try:
    pipeline = load_pipeline()
except FileNotFoundError as e:
    st.error(
        f"Model files not found: {e}\n\n"
        "Make sure you've run the full pipeline first (generate_synthetic_data.py, "
        "eda.py, nlp_preprocessing.py, feature_extraction_classification.py, "
        "priority_prediction.py, gridsearch_tuning.py, workload_assignment.py) "
        "so the 'outputs/' folder has the trained models."
    )
    st.stop()

tab_predict, tab_metrics, tab_workload = st.tabs(
    ["🔍 Analyze Task", "📊 Model Performance", "👥 Team Workload"]
)

# ============================================================
# TAB 1: Predict
# ============================================================
with tab_predict:
    st.subheader("Submit a new task")
    description = st.text_area(
        "Task description",
        placeholder="e.g. Application crashes when checkout button is clicked on the payment page",
        height=100,
    )

    if st.button("Analyze Task", type="primary"):
        if not description.strip():
            st.warning("Please enter a task description.")
        else:
            with st.spinner("Analyzing..."):
                result = pipeline.predict(description)

            st.success("Task analyzed")

            col1, col2, col3 = st.columns(3)
            col1.metric("Category", result["predicted_category"])
            col2.metric("Priority", result["predicted_priority"])
            col3.metric("Assigned To", result["recommended_assignee"])

            st.info(f"**Why this assignment:** {result['reasoning']}")

# ============================================================
# TAB 2: Model Performance
# ============================================================
with tab_metrics:
    st.subheader("Category Classifier")
    try:
        cat_results = pd.read_csv(f"{OUT_DIR}/classification_results.csv")
        st.dataframe(cat_results.style.format({
            "accuracy": "{:.1%}", "precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"
        }), use_container_width=True)
        best_cat = cat_results.loc[cat_results["f1"].idxmax(), "model"]
        st.caption(f"Best model: **{best_cat}**")
    except FileNotFoundError:
        st.warning("Run feature_extraction_classification.py to generate these results.")

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists(f"{OUT_DIR}/confusion_matrix_svm.png"):
            st.image(f"{OUT_DIR}/confusion_matrix_svm.png", caption="SVM Confusion Matrix")
    with col_b:
        if os.path.exists(f"{OUT_DIR}/confusion_matrix_naive_bayes.png"):
            st.image(f"{OUT_DIR}/confusion_matrix_naive_bayes.png", caption="Naive Bayes Confusion Matrix")

    st.divider()
    st.subheader("Priority Classifier")
    try:
        pri_results = pd.read_csv(f"{OUT_DIR}/priority_tuning_comparison.csv")
        st.dataframe(pri_results.style.format({
            "accuracy": "{:.1%}", "precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"
        }), use_container_width=True)
        st.caption("Tuned via GridSearchCV — see report for parameter search details.")
    except FileNotFoundError:
        st.warning("Run gridsearch_tuning.py to generate these results.")

    if os.path.exists(f"{OUT_DIR}/confusion_matrix_priority_final.png"):
        st.image(f"{OUT_DIR}/confusion_matrix_priority_final.png", caption="Priority Confusion Matrix (Final Model)")

# ============================================================
# TAB 3: Team Workload
# ============================================================
with tab_workload:
    st.subheader("Workload-Aware Assignment: Balancing Demonstration")
    st.caption(
        "Comparing random task assignment vs this project's scoring-based "
        "assignment (category affinity − workload penalty − urgency penalty) "
        "on the same simulated batch of 150 tasks."
    )

    if os.path.exists(f"{OUT_DIR}/workload_balance_comparison.png"):
        st.image(f"{OUT_DIR}/workload_balance_comparison.png", use_container_width=True)
    else:
        st.warning("Run workload_assignment.py to generate this comparison.")

    st.divider()
    st.subheader("Current Team Profiles")
    try:
        with open(f"{OUT_DIR}/agent_profiles.json") as f:
            profiles = json.load(f)
        workload_df = pd.DataFrame(
            list(profiles["workload"].items()), columns=["Agent", "Current Open Tasks"]
        ).sort_values("Current Open Tasks", ascending=False)
        st.dataframe(workload_df, use_container_width=True, hide_index=True)

        st.caption("Category affinity per agent (% of their tasks historically in each category):")
        affinity_df = pd.DataFrame(profiles["affinity"]).T
        st.dataframe(
            affinity_df.style.format("{:.0%}").background_gradient(cmap="Greens", axis=1),
            use_container_width=True,
        )
    except FileNotFoundError:
        st.warning("Run workload_assignment.py to generate agent profiles.")

st.divider()
st.caption(
    "This is a portfolio/academic project using a synthetic dataset. "
    "See the project README for methodology and known limitations."
)