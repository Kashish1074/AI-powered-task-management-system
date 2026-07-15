

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- CONFIG: adjust these to match your actual Kaggle CSV's column names ----
DATA_PATH = "../data/synthetic_tasks.csv"
OUT_DIR = "../outputs"
COL_DESCRIPTION = "description"
COL_CATEGORY = "category"
COL_PRIORITY = "priority"
COL_ASSIGNEE = "assignee"
COL_CREATED = "created_date"
COL_DEADLINE = "deadline"
COL_SLA_BREACHED = None       
COL_RESOLUTION_HOURS = None   
# -------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")


def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print("\nColumns:", list(df.columns))
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDtypes:\n", df.dtypes)
    return df


def basic_cleaning(df):
    before = len(df)
    df = df.drop_duplicates()
    # drop rows with no description text — useless for NLP
    if COL_DESCRIPTION in df.columns:
        df = df[df[COL_DESCRIPTION].notna() & (df[COL_DESCRIPTION].str.strip() != "")]
    print(f"\nCleaning: {before} -> {len(df)} rows after dropping duplicates/empty descriptions")
    return df.reset_index(drop=True)


def plot_category_distribution(df):
    if COL_CATEGORY not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    order = df[COL_CATEGORY].value_counts().index
    sns.countplot(data=df, y=COL_CATEGORY, order=order, palette="viridis")
    plt.title("Task Category Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/category_distribution.png", dpi=120)
    plt.close()
    print("Saved category_distribution.png")


def plot_priority_distribution(df):
    if COL_PRIORITY not in df.columns:
        return
    plt.figure(figsize=(6, 5))
    order = ["Low", "Medium", "High", "Urgent"]
    order = [o for o in order if o in df[COL_PRIORITY].unique()]
    sns.countplot(data=df, x=COL_PRIORITY, order=order, palette="rocket")
    plt.title("Priority Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/priority_distribution.png", dpi=120)
    plt.close()
    print("Saved priority_distribution.png")


def plot_priority_by_category(df):
    if COL_CATEGORY not in df.columns or COL_PRIORITY not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    ct = pd.crosstab(df[COL_CATEGORY], df[COL_PRIORITY])
    ct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))
    plt.title("Priority Breakdown by Category")
    plt.ylabel("Task Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/priority_by_category.png", dpi=120)
    plt.close()
    print("Saved priority_by_category.png")


def plot_assignee_workload(df):
    # No real assignee column in this dataset — synthetic agents get added in Week 3
    # for the workload-balancing feature. Skipping this plot for now.
    if not COL_ASSIGNEE or COL_ASSIGNEE not in df.columns:
        print("Skipped assignee_workload.png (no assignee column in this dataset)")
        return
    plt.figure(figsize=(8, 5))
    df[COL_ASSIGNEE].value_counts().plot(kind="bar", color="teal")
    plt.title("Task Count per Assignee (Current Workload)")
    plt.ylabel("Task Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/assignee_workload.png", dpi=120)
    plt.close()
    print("Saved assignee_workload.png")


def plot_sla_breach_by_priority(df):
    if not COL_SLA_BREACHED or COL_SLA_BREACHED not in df.columns or COL_PRIORITY not in df.columns:
        print("Skipped sla_breach_by_priority.png (no SLA breach column in this dataset)")
        return
    # sla_breached may be stored as real bool, or as strings like "Yes"/"No",
    # "True"/"False", "1"/"0" — normalize all of these to 0/1 before averaging
    raw = df[COL_SLA_BREACHED]
    if raw.dtype == bool:
        breached = raw.astype(int)
    else:
        breached = (
            raw.astype(str).str.strip().str.lower()
            .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        )
        unmapped = breached.isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} rows in {COL_SLA_BREACHED} had unrecognized values, dropped from this plot")
        breached = breached.dropna()

    plt.figure(figsize=(7, 5))
    order = ["Low", "Medium", "High", "Urgent"]
    order = [o for o in order if o in df[COL_PRIORITY].unique()]
    breach_rate = breached.groupby(df.loc[breached.index, COL_PRIORITY]).mean().reindex(order)
    breach_rate.plot(kind="bar", color="firebrick")
    plt.title("SLA Breach Rate by Priority")
    plt.ylabel("Breach Rate")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sla_breach_by_priority.png", dpi=120)
    plt.close()
    print("Saved sla_breach_by_priority.png")


def plot_resolution_time_by_priority(df):
    if not COL_RESOLUTION_HOURS or COL_RESOLUTION_HOURS not in df.columns or COL_PRIORITY not in df.columns:
        print("Skipped resolution_time_by_priority.png (no resolution time column in this dataset)")
        return
    plt.figure(figsize=(8, 5))
    order = ["Low", "Medium", "High", "Urgent"]
    order = [o for o in order if o in df[COL_PRIORITY].unique()]
    sns.boxplot(data=df, x=COL_PRIORITY, y=COL_RESOLUTION_HOURS, order=order, palette="mako")
    plt.title("Resolution Time (hours) by Priority")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/resolution_time_by_priority.png", dpi=120)
    plt.close()
    print("Saved resolution_time_by_priority.png")


def plot_description_length(df):
    if COL_DESCRIPTION not in df.columns:
        return
    lengths = df[COL_DESCRIPTION].astype(str).apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=30, kde=True, color="slateblue")
    plt.title("Task Description Length (words)")
    plt.xlabel("Word Count")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/description_length.png", dpi=120)
    plt.close()
    print("Saved description_length.png")
    print(f"\nDescription length stats:\n{lengths.describe()}")


def plot_deadline_urgency(df):
    # No deadline column in this dataset — using SLA breach + resolution time instead
    # (see plot_sla_breach_by_priority and plot_resolution_time_by_priority below)
    if not COL_DEADLINE or COL_CREATED not in df.columns or COL_DEADLINE not in df.columns:
        print("Skipped deadline_urgency.png (no deadline column in this dataset)")
        return
    created = pd.to_datetime(df[COL_CREATED], errors="coerce")
    deadline = pd.to_datetime(df[COL_DEADLINE], errors="coerce")
    days_to_deadline = (deadline - created).dt.days
    plt.figure(figsize=(8, 5))
    sns.histplot(days_to_deadline.dropna(), bins=30, color="crimson")
    plt.title("Days Given to Complete Task (created -> deadline)")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/deadline_urgency.png", dpi=120)
    plt.close()
    print("Saved deadline_urgency.png")


def main():
    df = load_data(DATA_PATH)
    df = basic_cleaning(df)

    plot_category_distribution(df)
    plot_priority_distribution(df)
    plot_priority_by_category(df)
    plot_assignee_workload(df)
    plot_description_length(df)
    plot_deadline_urgency(df)
    plot_sla_breach_by_priority(df)
    plot_resolution_time_by_priority(df)

    cleaned_path = f"{OUT_DIR}/cleaned_tasks.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"\nSaved cleaned dataset -> {cleaned_path}")
    print(f"\nAll EDA plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()