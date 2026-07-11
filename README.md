# AI-Powered Task Management System — Week 1

## What's in here
```
task_mgmt_project/
├── data/                       # raw data (synthetic + your Kaggle download)
├── outputs/                    # EDA plots, cleaned/preprocessed CSVs
├── scripts/
│   ├── generate_synthetic_data.py   # fallback/test dataset (800 tasks)
│   ├── eda.py                       # EDA + cleaning + plots
│   └── nlp_preprocessing.py         # tokenize/stopwords/stem
└── requirements.txt
```

## Status: pipeline tested and working
I ran all three scripts end-to-end on the synthetic dataset — 800 tasks → cleaned →
6 EDA plots → preprocessed text. No errors, no internet-download dependencies
(the stemmer and stopword list are both built-in, so this will run on your
Windows machine with zero nltk_data download hassle).

## Finding a real dataset on Kaggle
Since I don't have live internet access to Kaggle from this sandbox, search these
terms yourself on kaggle.com (you already have the Kaggle CLI set up from your
HAM10000 project, so `kaggle datasets download -d <slug>` will work the same way):

- `"jira issue" priority dataset`
- `"github issues" classification`
- `bug report priority prediction`
- `software defect dataset priority`
- `helpdesk ticket classification` (Zendesk/support-ticket datasets also work well —
  they have the same shape: text description → category → priority)

**What to look for in a good candidate:** a text field (title/description/summary),
a category or issue-type label, and ideally a priority/severity label. Assignee and
deadline fields are a bonus (nice-to-have for the workload-balancing part in Week 3,
not essential — you can synthesize deadlines from creation dates if missing).

## Swapping in your real dataset
Once you've downloaded a CSV:
1. Drop it into `data/`.
2. Open `scripts/eda.py`, change `DATA_PATH` to your file, and update the
   `COL_*` variables at the top to match your actual column names.
3. Re-run `eda.py` then `nlp_preprocessing.py` — same commands, no other
   code changes needed. The scripts are written to be column-name-agnostic
   as long as you set those constants correctly.

## Running locally (VS Code)
```bash
cd scripts
pip install -r ../requirements.txt
python generate_synthetic_data.py   # skip once you have real data
python eda.py
python nlp_preprocessing.py
```

## Week 1 deliverables checklist
- [x] Dataset (synthetic, ready to swap for Kaggle data)
- [x] EDA — category distribution, priority distribution, priority-by-category,
      assignee workload, description length, deadline urgency (6 plots in `outputs/`)
- [x] Cleaning — duplicates and empty descriptions dropped
- [x] NLP preprocessing — lowercase, tokenize, stopword removal, stemming
      (no downloads required)

## Week 2: Feature extraction + classification
`scripts/feature_extraction_classification.py` does:
- TF-IDF vectorization (unigrams + bigrams, max 3000 features)
- Naive Bayes and SVM (LinearSVC) trained on an 80/20 split
- Accuracy, precision, recall, F1, per-class report, confusion matrix per model
- Best model + vectorizer saved as `.pkl` for reuse in Week 3/4

**Note on current results:** both models hit 100% accuracy on the synthetic
dataset — expected, since the synthetic templates make categories trivially
separable by keyword overlap. This is a pipeline validation result, not a
real-world result. Once the real Kaggle dataset is swapped in (see above),
re-run this script and expect materially lower, more realistic numbers —
that's what should go in the actual mid-project review report.

## Mid-Project Review checklist (End of Week 2)
- [x] Cleaned and preprocessed dataset
- [x] Task classifier (Naive Bayes/SVM) trained and evaluated — pipeline verified,
      pending re-run on real data
- [x] EDA visualizations completed
- [ ] GitHub repo pushed (local repo initialized, see below)

## Next (Week 3 preview)
Priority prediction with Random Forest/XGBoost + GridSearchCV, and workload
balancing logic.
