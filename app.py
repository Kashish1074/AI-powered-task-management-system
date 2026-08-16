

import sys
import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TaskFlow AI", page_icon="◆", layout="centered")

# integrated_pipeline.py lives in scripts/, add it to the import path.
# Try both casings — Windows treats "scripts" and "Scripts" as the same
# folder, but Streamlit Cloud runs on case-sensitive Linux, so this avoids
# breaking depending on how the folder happened to get committed to GitHub.
_base_dir = os.path.dirname(__file__)
_scripts_dir = None
for _folder_name in ("scripts", "Scripts"):
    _candidate = os.path.join(_base_dir, _folder_name)
    if os.path.isdir(_candidate):
        _scripts_dir = _candidate
        break

if _scripts_dir is None:
    st.error(
        "Could not find the 'scripts' folder next to app.py. "
        f"Looked in: {_base_dir}. Check that the scripts folder is "
        "actually committed to your GitHub repo."
    )
    st.stop()

sys.path.append(_scripts_dir)
from integrated_pipeline import TaskPipeline

OUT_DIR = "outputs"

# Priority -> color mapping, used consistently across the whole dashboard
# (this is the signature visual element — every priority mention anywhere
# uses this exact color, so it becomes a learned visual language)
PRIORITY_COLORS = {
    "Low": "#6B8CAE",
    "Medium": "#3E6990",
    "High": "#C97B3B",
    "Urgent": "#B84C3C",
}

# ============================================================
# GLOBAL STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Top header band */
.tf-header {
    background: linear-gradient(135deg, #1B2A41 0%, #29394F 100%);
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 28px;
}
.tf-header h1 {
    color: #FFFFFF;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 28px;
    margin: 0;
    letter-spacing: -0.3px;
}
.tf-header p {
    color: #A9BBCC;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    margin: 6px 0 0 0;
    letter-spacing: 0.2px;
}

/* Section labels — eyebrow style */
.tf-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: #5C6B7A;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}
.tf-section-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 19px;
    color: #1B2A41;
    margin-bottom: 16px;
}

/* Result card with priority-coded left stripe (signature element) */
.tf-result-card {
    background: #FFFFFF;
    border: 1px solid #E1E4E8;
    border-left: 5px solid var(--stripe-color, #3E6990);
    border-radius: 6px;
    padding: 20px 24px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(27, 42, 65, 0.06);
}
.tf-result-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 14px;
}
.tf-result-field {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #5C6B7A;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}
.tf-result-value {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 20px;
    color: #1B2A41;
}
.tf-result-reasoning {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    color: #3E4C5E;
    border-top: 1px solid #EEF0F2;
    padding-top: 12px;
    margin-top: 4px;
}

/* Priority pill */
.tf-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    color: #FFFFFF;
}

/* Tabs — more office-toolbar feel */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid #E1E4E8;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    color: #5C6B7A;
    padding: 10px 4px;
}
.stTabs [aria-selected="true"] {
    color: #1B2A41 !important;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    border-radius: 5px;
}

/* Footer */
.tf-footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8895A3;
    text-align: center;
    padding: 20px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="tf-header">
    <h1>◆ TaskFlow AI</h1>
    <p>NLP CLASSIFICATION · PRIORITY PREDICTION · WORKLOAD-AWARE ASSIGNMENT</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    return TaskPipeline(out_dir=OUT_DIR)


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
    ["Analyze Task", "Model Performance", "Team Workload"]
)

# ============================================================
# TAB 1: Predict
# ============================================================
with tab_predict:
    st.markdown('<div class="tf-eyebrow">New Task Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-title">Submit a task for triage</div>', unsafe_allow_html=True)

    description = st.text_area(
        "Task description",
        placeholder="e.g. Application crashes when checkout button is clicked on the payment page",
        height=100,
        label_visibility="collapsed",
    )

    if st.button("Analyze Task", type="primary"):
        if not description.strip():
            st.warning("Please enter a task description.")
        else:
            with st.spinner("Analyzing..."):
                result = pipeline.predict(description)

            stripe = PRIORITY_COLORS.get(result["predicted_priority"], "#3E6990")
            st.markdown(f"""
            <div class="tf-result-card" style="--stripe-color: {stripe};">
                <div class="tf-result-grid">
                    <div>
                        <div class="tf-result-field">Category</div>
                        <div class="tf-result-value">{result['predicted_category']}</div>
                    </div>
                    <div>
                        <div class="tf-result-field">Priority</div>
                        <div class="tf-result-value">
                            <span class="tf-pill" style="background:{stripe};">{result['predicted_priority']}</span>
                        </div>
                    </div>
                    <div>
                        <div class="tf-result-field">Assigned To</div>
                        <div class="tf-result-value">{result['recommended_assignee'].title()}</div>
                    </div>
                </div>
                <div class="tf-result-reasoning">{result['reasoning']}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 2: Model Performance
# ============================================================
with tab_metrics:
    st.markdown('<div class="tf-eyebrow">Category Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-title">Task Category Classifier</div>', unsafe_allow_html=True)
    try:
        cat_results = pd.read_csv(f"{OUT_DIR}/classification_results.csv")
        st.dataframe(cat_results.style.format({
            "accuracy": "{:.1%}", "precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"
        }), width='stretch', hide_index=True)
        best_cat = cat_results.loc[cat_results["f1"].idxmax(), "model"]
        st.caption(f"Best model: {best_cat}")
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
    st.markdown('<div class="tf-eyebrow">Priority Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-title">Priority Prediction (Tuned)</div>', unsafe_allow_html=True)
    try:
        pri_results = pd.read_csv(f"{OUT_DIR}/priority_tuning_comparison.csv")
        st.dataframe(pri_results.style.format({
            "accuracy": "{:.1%}", "precision": "{:.1%}", "recall": "{:.1%}", "f1": "{:.1%}"
        }), width='stretch', hide_index=True)
        st.caption("Tuned via GridSearchCV — see report for parameter search details.")
    except FileNotFoundError:
        st.warning("Run gridsearch_tuning.py to generate these results.")

    if os.path.exists(f"{OUT_DIR}/confusion_matrix_priority_final.png"):
        st.image(f"{OUT_DIR}/confusion_matrix_priority_final.png", caption="Priority Confusion Matrix (Final Model)")

# ============================================================
# TAB 3: Team Workload
# ============================================================
with tab_workload:
    st.markdown('<div class="tf-eyebrow">Load Balancing</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-title">Workload-Aware Assignment</div>', unsafe_allow_html=True)
    st.caption(
        "Comparing random task assignment vs this project's scoring-based "
        "assignment (category affinity − workload penalty − urgency penalty) "
        "on the same simulated batch of 150 tasks."
    )

    if os.path.exists(f"{OUT_DIR}/workload_balance_comparison.png"):
        st.image(f"{OUT_DIR}/workload_balance_comparison.png", width='stretch')
    else:
        st.warning("Run workload_assignment.py to generate this comparison.")

    st.divider()
    st.markdown('<div class="tf-eyebrow">Roster</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-section-title">Current Team Profiles</div>', unsafe_allow_html=True)
    try:
        with open(f"{OUT_DIR}/agent_profiles.json") as f:
            profiles = json.load(f)
        workload_df = pd.DataFrame(
            list(profiles["workload"].items()), columns=["Agent", "Current Open Tasks"]
        ).sort_values("Current Open Tasks", ascending=False)
        workload_df["Agent"] = workload_df["Agent"].str.title()
        st.dataframe(workload_df, width='stretch', hide_index=True)

        st.caption("Category affinity per agent (% of their tasks historically in each category):")
        affinity_df = pd.DataFrame(profiles["affinity"]).T
        affinity_df.index = affinity_df.index.str.title()
        st.dataframe(
            affinity_df.style.format("{:.0%}").background_gradient(cmap="Blues", axis=1),
            width='stretch',
        )
    except FileNotFoundError:
        st.warning("Run workload_assignment.py to generate agent profiles.")

st.markdown(
    '<div class="tf-footer">TASKFLOW AI · ACADEMIC PORTFOLIO PROJECT · SYNTHETIC DATASET · '
    'SEE README FOR METHODOLOGY</div>',
    unsafe_allow_html=True,
)