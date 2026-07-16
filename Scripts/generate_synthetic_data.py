
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Bug", "Feature Request", "Documentation", "UI/UX", "Backend", "DevOps", "Testing"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
ASSIGNEES = ["arjun", "priya", "rohan", "sneha", "vikram", "neha"]

# Categories that realistically get confused with each other (used for label noise)
CONFUSABLE_PAIRS = {
    "Bug": ["Testing"],
    "Testing": ["Bug", "DevOps"],
    "Backend": ["DevOps", "Bug"],
    "DevOps": ["Backend", "Testing"],
    "UI/UX": ["Feature Request"],
    "Feature Request": ["UI/UX"],
    "Documentation": ["Feature Request"],
}

# Shared vocabulary pool — appears across multiple categories' templates so
# TF-IDF can't rely on one exclusive keyword per category
SHARED_VERBS = ["fix", "update", "check", "review", "resolve", "handle", "investigate", "address"]
SHARED_NOUNS = ["issue", "task", "item", "ticket", "request", "problem", "case"]

TEMPLATES = {
    "Bug": [
        "Application crashes when {x} is clicked on the {y} page",
        "Login fails with error 500 after {x} attempts",
        "Memory leak detected in {y} module during {x}",
        "Data not saving correctly when {x} is submitted",
        "Need to {verb} the {noun} where {y} breaks under {x}",
        "Unexpected behavior in {y} when using {x}",
        "Users report {y} stops responding during {x}",
    ],
    "Feature Request": [
        "Add dark mode support to the {y} page",
        "Implement export to CSV for {y} reports",
        "Allow users to filter {y} by {x}",
        "Add notification system for {x} events",
        "Please {verb} the {noun} to support {y} customization",
        "Requesting new option for {y} during {x}",
    ],
    "Documentation": [
        "Update API docs for the {y} endpoint",
        "Write onboarding guide for {y} module",
        "Document deployment steps for {x} environment",
        "Need to {verb} the {noun} explaining {y} setup",
        "Docs missing for {y} configuration under {x}",
    ],
    "UI/UX": [
        "Redesign the {y} dashboard for better readability",
        "Fix alignment issue on {y} page for {x} screen size",
        "Improve accessibility of {y} form fields",
        "Please {verb} the {noun} layout for {y}",
        "Visual inconsistency on {y} during {x}",
    ],
    "Backend": [
        "Optimize database query for {y} table",
        "Refactor {y} service to reduce latency during {x}",
        "Add caching layer to {y} API",
        "Need to {verb} the {noun} affecting {y} performance",
        "Server-side {y} logic fails under {x}",
    ],
    "DevOps": [
        "Set up CI/CD pipeline for {y} service",
        "Fix broken deployment on {x} server",
        "Configure auto-scaling for {y} during peak {x}",
        "Please {verb} the {noun} related to {y} infrastructure",
        "Deployment {noun} affecting {y} during {x}",
    ],
    "Testing": [
        "Write unit tests for {y} module",
        "Fix flaky integration test in {y} during {x}",
        "Increase test coverage for {y} edge cases",
        "Need to {verb} the {noun} around {y} test failures",
        "QA found {noun} in {y} during {x}",
    ],
}

FILLERS_X = ["submit button", "checkout", "multiple", "peak load", "mobile", "signup", "monthly"]
FILLERS_Y = ["payment", "user profile", "search", "admin", "reporting", "auth", "notifications"]


def build_description(category):
    template = random.choice(TEMPLATES[category])
    desc = template.format(
        x=random.choice(FILLERS_X),
        y=random.choice(FILLERS_Y),
        verb=random.choice(SHARED_VERBS),
        noun=random.choice(SHARED_NOUNS),
    )
    return desc


def generate_row(task_id, noise_rate=0.08):
    true_category = random.choice(CATEGORIES)
    desc = build_description(true_category)

    # controlled label noise: ~8% of rows get labeled as a confusable
    # neighboring category instead of the one the text was generated for —
    # mimics real-world ambiguous/mislabeled tickets
    if random.random() < noise_rate and true_category in CONFUSABLE_PAIRS:
        label_category = random.choice(CONFUSABLE_PAIRS[true_category])
    else:
        label_category = true_category

    # priority correlates loosely with the TRUE category (not the noisy label,
    # since priority would realistically follow the actual nature of the issue)
    if true_category == "Bug":
        priority = random.choices(PRIORITIES, weights=[10, 25, 40, 25])[0]
    elif true_category == "Documentation":
        priority = random.choices(PRIORITIES, weights=[50, 35, 10, 5])[0]
    else:
        priority = random.choices(PRIORITIES, weights=[20, 40, 30, 10])[0]

    created = pd.Timestamp("2025-01-01") + pd.Timedelta(days=random.randint(0, 300))
    deadline_offset = {"Urgent": 2, "High": 5, "Medium": 10, "Low": 20}[priority]
    deadline = created + pd.Timedelta(days=deadline_offset + random.randint(-1, 3))

    return {
        "task_id": task_id,
        "title": desc[:40],
        "description": desc,
        "category": label_category,
        "priority": priority,
        "assignee": random.choice(ASSIGNEES),
        "created_date": created.date(),
        "deadline": deadline.date(),
        "estimated_hours": round(np.random.gamma(2, 3), 1),
    }


def main(n_rows=800, out_path="../data/synthetic_tasks.csv"):
    rows = [generate_row(i) for i in range(1, n_rows + 1)]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic tasks -> {out_path}")
    print(df.head())
    print(f"\nCategory distribution:\n{df['category'].value_counts()}")
    return df


if __name__ == "__main__":
    main()