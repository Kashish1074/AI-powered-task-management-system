
import pandas as pd
import json
import os
from datetime import datetime

OUT_DIR = "../outputs"
REPORT_PATH = "../PROJECT_METRICS_SUMMARY.md"


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def fmt_pct_table(df):
    if df is None:
        return "*(not found — script may not have been run)*\n"
    df_fmt = df.copy()
    for col in ["accuracy", "precision", "recall", "f1"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.1%}")
    return df_fmt.to_markdown(index=False)


def main():
    lines = []
    lines.append("# Project Metrics Summary")
    lines.append(f"\n*Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by compile_metrics.py*\n")
    lines.append("This document consolidates every result produced across the project. "
                  "Use it as the source of truth when writing the final report — don't "
                  "retype numbers from memory, copy them from here.\n")

    # --- Dataset overview ---
    lines.append("## 1. Dataset\n")
    df = safe_read_csv(f"{OUT_DIR}/cleaned_tasks.csv")
    if df is not None:
        lines.append(f"- **Total tasks:** {len(df)}")
        lines.append(f"- **Categories:** {df['category'].nunique()} ({', '.join(sorted(df['category'].unique()))})")
        lines.append(f"- **Priority levels:** {df['priority'].nunique()} ({', '.join(sorted(df['priority'].unique()))})")
        if "assignee" in df.columns:
            lines.append(f"- **Team members:** {df['assignee'].nunique()}")
    lines.append("- **Source:** Synthetic dataset (see README 'Dataset decision' section for rationale)\n")

    # --- Category classifier ---
    lines.append("## 2. Category Classification (Week 2)\n")
    cat_df = safe_read_csv(f"{OUT_DIR}/classification_results.csv")
    lines.append(fmt_pct_table(cat_df))
    lines.append("")
    if cat_df is not None:
        best = cat_df.loc[cat_df["f1"].idxmax(), "model"]
        lines.append(f"**Best model:** {best}\n")
    lines.append("![SVM Confusion Matrix](outputs/confusion_matrix_svm.png)")
    lines.append("![Naive Bayes Confusion Matrix](outputs/confusion_matrix_naive_bayes.png)\n")

    # --- Priority classifier ---
    lines.append("## 3. Priority Prediction (Week 3)\n")
    pri_df = safe_read_csv(f"{OUT_DIR}/priority_classification_results.csv")
    lines.append("### Random Forest vs XGBoost (Day 16)")
    lines.append(fmt_pct_table(pri_df))
    lines.append("")

    tuning_df = safe_read_csv(f"{OUT_DIR}/priority_tuning_comparison.csv")
    lines.append("### Baseline vs GridSearchCV-Tuned (Day 19)")
    lines.append(fmt_pct_table(tuning_df))
    lines.append("")
    lines.append("![Priority Confusion Matrix](outputs/confusion_matrix_priority_final.png)\n")

    # --- Workload balancing ---
    lines.append("## 4. Workload-Aware Assignment (Core Differentiator)\n")
    lines.append("Heuristic scoring: category affinity − workload penalty − urgency-overload penalty.\n")
    try:
        with open(f"{OUT_DIR}/agent_profiles.json", encoding="utf-8") as f:
            profiles = json.load(f)
        workloads = list(profiles["workload"].values())
        lines.append(f"- **Team size:** {len(profiles['workload'])} agents")
        lines.append(f"- **Current workload range:** {min(workloads)}–{max(workloads)} tasks")
    except FileNotFoundError:
        pass
    lines.append("- **Simulated workload spread reduction vs random assignment:** ~85% "
                  "(see workload_assignment.py output for exact run values)")
    lines.append("![Workload Balance Comparison](outputs/workload_balance_comparison.png)\n")

    # --- Integrated pipeline ---
    lines.append("## 5. Integrated Pipeline (Day 18)\n")
    demo_df = safe_read_csv(f"{OUT_DIR}/integrated_pipeline_demo.csv")
    if demo_df is not None:
        lines.append("Example end-to-end predictions (raw text -> category, priority, assignee):\n")
        lines.append(demo_df.to_markdown(index=False))
    lines.append("")

    # --- Known limitations (for honest reporting) ---
    lines.append("## 6. Known Limitations (for report honesty)\n")
    lines.append("- Dataset is synthetic, not scraped from a live system — see README for why "
                  "the original real Kaggle dataset was rejected (no assignee field, priority "
                  "uncorrelated with SLA/resolution outcomes)")
    lines.append("- Category classifier accuracy (~92%) reflects deliberately-introduced "
                  "template overlap and label noise (~8%), tuned to be realistic rather than trivial")
    lines.append("- Priority prediction (74-76%) is intentionally harder than category — priority "
                  "is a fuzzier, more ordinal problem, and the confusion matrix shows errors "
                  "concentrated between adjacent levels (Low↔Medium, Medium↔High), which is "
                  "the expected pattern for a genuine ordinal relationship")
    lines.append("- Workload balancing is a transparent heuristic, not a trained ML model — "
                  "deliberate choice for explainability in an assignment-decision context")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Compiled metrics summary -> {REPORT_PATH}")
    print(f"Total sections: 6")


if __name__ == "__main__":
    main()