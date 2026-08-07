# AI-Powered Task Management System

Intelligent task management system using NLP/ML to classify, prioritize, and
assign tasks based on user behavior, deadlines, and workload.

## Project structure
```
task_mgmt_project/
├── data/            # raw dataset(s) go here
├── outputs/         # EDA plots, cleaned/preprocessed CSVs, model results
├── scripts/         # pipeline scripts
└── requirements.txt
```

## Status: Day 21 — Week 3 complete
Category classification, priority prediction, and workload-aware assignment
are all built, tested, and integrated into a single end-to-end pipeline.
Week 4 (dashboard + final report) is next.

## Week 3 summary (final results)
- **Category classifier**: SVM/Naive Bayes, ~92% accuracy (7 categories, with
  realistic confusion between related categories like Feature Request/UI-UX)
- **Priority classifier**: Random Forest, tuned via GridSearchCV,
  74% → 76% accuracy (F1 0.741 → 0.760). Confusion concentrated between
  *adjacent* priority levels (Low↔Medium, Medium↔High) — evidence the model
  learned a genuine ordinal relationship, not noise.
- **Workload-aware assignment (core differentiator)**: heuristic scoring
  (category affinity − workload penalty − urgency-overload penalty).
  Reduced workload spread by ~85% compared to random assignment in
  simulation, with fully explainable per-assignment reasoning.
- **Integrated pipeline**: single function (`TaskPipeline.predict()`) takes
  raw task text and returns category, priority, and recommended assignee
  with reasoning — this is what the Week 4 dashboard will call directly.

## Dataset decision (Day 4)
We initially planned to use a real Kaggle helpdesk-ticket dataset. During Day 3-4
EDA, it became clear that the dataset's `priority` labels showed no real
relationship to SLA breach rate or resolution time (both were nearly flat across
Low/Medium/High/Urgent) — a strong sign the priority field was randomly assigned
rather than reflecting real support outcomes. The dataset also had no
assignee/agent column, which is required for this project's core differentiator
(workload-aware task assignment).

**Decision: switched to the synthetic task dataset** (`scripts/generate_synthetic_data.py`),
which has controlled, meaningful correlations (priority tied to category and
deadline pressure) and includes both `assignee` and `deadline` fields. This is
documented here for transparency — the final report should note that results are
based on a synthetic dataset designed to reflect realistic task-management patterns,
not scraped from a live system.

## Setup
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Plan
**Week 1** — dataset sourcing, EDA, NLP preprocessing ✅
**Week 2** — TF-IDF feature extraction, Naive Bayes/SVM classification, evaluation ✅
**Week 3** — priority prediction (Random Forest/XGBoost), workload balancing, hyperparameter tuning ✅
**Week 4** — dashboard, final report, performance metrics compilation

## Mid-Project Review checklist (End of Week 2) — complete
- [x] Cleaned and preprocessed dataset
- [x] Task classifier (Naive Bayes/SVM) trained and evaluated
- [x] EDA visualizations completed
- [x] GitHub repo pushed