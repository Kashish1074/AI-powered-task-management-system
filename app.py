

import sys
import os
import streamlit as st

# integrated_pipeline.py lives in scripts/, add it to the import path
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from integrated_pipeline import TaskPipeline

st.set_page_config(page_title="AI Task Management System", page_icon="📋", layout="centered")


@st.cache_resource
def load_pipeline():
    return TaskPipeline(out_dir="outputs")


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

st.divider()
st.caption(
    "This is a portfolio/academic project using a synthetic dataset. "
    "See the project README for methodology and known limitations."
)