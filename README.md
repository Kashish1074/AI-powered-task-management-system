# AI-Powered Task Management System

An intelligent task management system that uses NLP and machine learning to automatically classify, prioritize, and assign incoming tasks — going beyond simple classification to recommend *who* should handle each task based on category expertise and current workload.

## Live Demo
🔗 ****

## What it does

Given a raw task description, the system:
1. **Classifies** the task into one of 7 categories (Bug, Feature Request, Documentation, UI/UX, Backend, DevOps, Testing)
2. **Predicts priority** (Low, Medium, High, Urgent)
3. **Recommends an assignee** using a transparent, explainable scoring formula that balances category fit against current team workload — not just "who's best," but "who's best *and* has capacity"

Example:
> Input: *"URGENT: production deployment is broken on the auth server"*
> Output: Category: `DevOps` · Priority: `Urgent` · Assigned to: `Rohan` — *"handles 12% of their tasks in 'DevOps', currently has 121 open tasks (below team average)"*

## Key results

| Component | Metric | Result |
|---|---|---|
| Category classifier (SVM) | Accuracy | 92.5% |
| Priority classifier (tuned Random Forest) | Accuracy | 75.6% |
| Workload-aware assignment | Workload spread reduction vs. random | ~85% |

Full metrics, confusion matrices, and discussion: see `PROJECT_METRICS_SUMMARY.md` and `Final_Project_Report.docx`.

## Why workload-aware assignment (the core differentiator)

Most task classification projects stop at predicting a label. This system goes further: it decides *who* handles the task, using a heuristic — not a black-box model — so every recommendation is explainable:

```
score(agent, task) = affinity_weight × category_affinity(agent, category)
                      − workload_weight × normalized_current_load(agent)
                      − urgency_penalty (if agent is overloaded AND task is High/Urgent)
```

This is a deliberate design choice: for a decision that affects real people's workload, explainability matters more than squeezing out a marginal accuracy gain from a more complex model.

## Project structure

```
task_mgmt_project/
├── data/                        # synthetic dataset
├── outputs/                     # EDA plots, cleaned/preprocessed data, model results
├── scripts/
│   ├── generate_synthetic_data.py       # dataset generation
│   ├── eda.py                           # exploratory analysis + cleaning
│   ├── nlp_preprocessing.py             # text cleaning pipeline
│   ├── feature_extraction_classification.py  # TF-IDF + category classifier
│   ├── priority_prediction.py           # Random Forest / XGBoost priority model
│   ├── gridsearch_tuning.py             # hyperparameter tuning
│   ├── workload_assignment.py           # workload-aware scoring logic
│   ├── integrated_pipeline.py           # end-to-end prediction pipeline
│   └── compile_metrics.py               # results compilation
├── app.py                       # Streamlit dashboard ("TaskFlow AI")
├── .streamlit/config.toml       # dashboard theme
├── PROJECT_METRICS_SUMMARY.md   # auto-compiled results reference
├── Final_Project_Report.docx    # full project report
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Running the full pipeline

```bash
cd scripts
python generate_synthetic_data.py
python eda.py
python nlp_preprocessing.py
python feature_extraction_classification.py
python priority_prediction.py
python gridsearch_tuning.py
python workload_assignment.py
python integrated_pipeline.py
python compile_metrics.py
cd ..
streamlit run app.py
```

## Tech stack

- **NLP**: TF-IDF, Porter stemming, scikit-learn stopwords
- **ML**: Naive Bayes, SVM (category), Random Forest, XGBoost (priority), GridSearchCV
- **Dashboard**: Streamlit
- **Data**: Synthetic dataset with deliberately realistic difficulty (overlapping category vocabulary, ~8% label noise) rather than trivially clean separability

## Dataset note

This project uses a synthetic dataset rather than a scraped real-world one. A real Kaggle helpdesk-ticket dataset was evaluated first and rejected for two documented reasons: its priority labels showed no measurable relationship to actual outcomes (SLA breach rate, resolution time), and it lacked an assignee field needed for the workload-balancing feature. The synthetic dataset was built to have realistic, learnable-but-imperfect structure instead. 

## Author

Kashish 