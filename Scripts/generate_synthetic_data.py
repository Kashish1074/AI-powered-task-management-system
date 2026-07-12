"""
generate_synthetic_data.py
---------------------------
Creates a small synthetic task dataset that mimics a Jira/Trello export.
Use this ONLY to test the pipeline end-to-end before you swap in the real
Kaggle dataset. Once you have real data, point eda.py / preprocessing.py
at that CSV instead (see README.md for the required column names).
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = ["Bug", "Feature Request", "Documentation", "UI/UX", "Backend", "DevOps", "Testing"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
ASSIGNEES = ["arjun", "priya", "rohan", "sneha", "vikram", "neha"]

TEMPLATES = {
    "Bug": [
        "Application crashes when {x} is clicked on the {y} page",
        "Login fails with error 500 after {x} attempts",
        "Memory leak detected in {y} module during {x}",
        "Data not saving correctly when {x} is submitted",
    ],
    "Feature Request": [
        "Add dark mode support to the {y} page",
        "Implement export to CSV for {y} reports",
        "Allow users to filter {y} by {x}",
        "Add notification system for {x} events",
    ],
    "Documentation": [
        "Update API docs for the {y} endpoint",
        "Write onboarding guide for {y} module",
        "Document deployment steps for {x} environment",
    ],
    "UI/UX": [
        "Redesign the {y} dashboard for better readability",
        "Fix alignment issue on {y} page for {x} screen size",
        "Improve accessibility of {y} form fields",
    ],
    "Backend": [
        "Optimize database query for {y} table",
        "Refactor {y} service to reduce latency during {x}",
        "Add caching layer to {y} API",
    ],
    "DevOps": [
        "Set up CI/CD pipeline for {y} service",
        "Fix broken deployment on {x} server",
        "Configure auto-scaling for {y} during peak {x}",
    ],
    "Testing": [
        "Write unit tests for {y} module",
        "Fix flaky integration test in {y} during {x}",
        "Increase test coverage for {y} edge cases",
    ],
}

FILLERS_X = ["submit button", "checkout", "multiple", "peak load", "mobile", "signup", "monthly"]
FILLERS_Y = ["payment", "user profile", "search", "admin", "reporting", "auth", "notifications"]


def generate_row(task_id):
    category = random.choice(CATEGORIES)
    template = random.choice(TEMPLATES[category])
    desc = template.format(x=random.choice(FILLERS_X), y=random.choice(FILLERS_Y))

    # priority correlates loosely with category (bugs skew urgent, docs skew low)
    if category == "Bug":
        priority = random.choices(PRIORITIES, weights=[10, 25, 40, 25])[0]
    elif category == "Documentation":
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
        "category": category,
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
    return df


if __name__ == "__main__":
    main()
