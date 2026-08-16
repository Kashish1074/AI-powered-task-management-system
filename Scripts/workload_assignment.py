

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json

IN_PATH = "../outputs/cleaned_tasks.csv"
OUT_DIR = "../outputs"
CATEGORY_COL = "category"
PRIORITY_COL = "priority"
ASSIGNEE_COL = "assignee"

AFFINITY_WEIGHT = 1.0
WORKLOAD_WEIGHT = 1.0
URGENCY_PENALTY = 0.6
OVERLOAD_THRESHOLD = 1.15  # agent counts as "overloaded" if load > 1.15x team average

random.seed(42)
np.random.seed(42)


def load_data():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} tasks with existing assignments")
    return df


def build_agent_profiles(df):
    """Compute each agent's category affinity (historical proportion of
    tasks handled per category) and current workload (open task count)."""
    agents = sorted(df[ASSIGNEE_COL].unique())
    categories = sorted(df[CATEGORY_COL].unique())

    # affinity: for each agent, what fraction of THEIR tasks are in each category
    affinity = {}
    for agent in agents:
        agent_tasks = df[df[ASSIGNEE_COL] == agent]
        total = len(agent_tasks)
        affinity[agent] = {
            cat: (agent_tasks[CATEGORY_COL] == cat).sum() / total if total > 0 else 0
            for cat in categories
        }

    # current workload: how many tasks each agent currently has (using full
    # dataset as a stand-in for "current open tasks" for this simulation)
    workload = df[ASSIGNEE_COL].value_counts().to_dict()
    for agent in agents:
        workload.setdefault(agent, 0)

    return agents, categories, affinity, workload


def score_agent(agent, task_category, task_priority, affinity, workload, avg_load):
    cat_affinity = affinity[agent].get(task_category, 0)
    normalized_load = workload[agent] / avg_load if avg_load > 0 else 0

    score = AFFINITY_WEIGHT * cat_affinity - WORKLOAD_WEIGHT * normalized_load

    is_overloaded = workload[agent] > avg_load * OVERLOAD_THRESHOLD
    is_urgent = task_priority in ("High", "Urgent")
    if is_overloaded and is_urgent:
        score -= URGENCY_PENALTY

    return score


def recommend_assignee(task_category, task_priority, agents, affinity, workload):
    avg_load = np.mean(list(workload.values()))
    scores = {
        agent: score_agent(agent, task_category, task_priority, affinity, workload, avg_load)
        for agent in agents
    }
    best_agent = max(scores, key=scores.get)
    return best_agent, scores


def explain_recommendation(agent, task_category, task_priority, affinity, workload):
    avg_load = np.mean(list(workload.values()))
    cat_pct = affinity[agent].get(task_category, 0) * 100
    load = workload[agent]
    load_vs_avg = "above" if load > avg_load else "below" if load < avg_load else "at"
    return (
        f"Assigned to {agent} — handles {cat_pct:.0f}% of their tasks in "
        f"'{task_category}', currently has {load} open tasks ({load_vs_avg} "
        f"team average of {avg_load:.1f})"
    )


def simulate_and_compare(df, agents, categories, affinity, workload_start):
    """Simulate assigning a held-out batch of tasks two ways: random vs
    scored. Track resulting workload spread for each approach."""
    sample = df.sample(n=min(150, len(df)), random_state=42).reset_index(drop=True)

    # --- Random assignment simulation ---
    random_workload = {a: 0 for a in agents}
    for _, row in sample.iterrows():
        agent = random.choice(agents)
        random_workload[agent] += 1

    # --- Scored assignment simulation (workload updates as we go, so the
    # system actually balances load in real time rather than scoring once) ---
    scored_workload = dict(workload_start)  # start from realistic existing load
    scored_workload = {a: 0 for a in agents}  # reset to isolate this batch's balancing
    assignments_log = []
    for _, row in sample.iterrows():
        agent, scores = recommend_assignee(
            row[CATEGORY_COL], row[PRIORITY_COL], agents, affinity, scored_workload
        )
        scored_workload[agent] += 1
        assignments_log.append({
            "category": row[CATEGORY_COL],
            "priority": row[PRIORITY_COL],
            "assigned_to": agent,
        })

    random_std = np.std(list(random_workload.values()))
    scored_std = np.std(list(scored_workload.values()))

    print("\n=== Workload Balance Comparison (150 simulated new tasks) ===")
    print(f"Random assignment  — tasks per agent: {random_workload}")
    print(f"  Std deviation: {random_std:.2f}")
    print(f"Scored assignment  — tasks per agent: {scored_workload}")
    print(f"  Std deviation: {scored_std:.2f}")
    improvement = (1 - scored_std / random_std) * 100 if random_std > 0 else 0
    print(f"\nWorkload spread reduced by {improvement:.1f}% vs random assignment")

    # plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    pd.Series(random_workload).sort_index().plot(kind="bar", ax=axes[0], color="salmon")
    axes[0].set_title(f"Random Assignment\n(std dev = {random_std:.2f})")
    axes[0].set_ylabel("Tasks Assigned")
    pd.Series(scored_workload).sort_index().plot(kind="bar", ax=axes[1], color="seagreen")
    axes[1].set_title(f"Workload-Aware Assignment\n(std dev = {scored_std:.2f})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/workload_balance_comparison.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR}/workload_balance_comparison.png")

    return assignments_log, random_std, scored_std


def main():
    df = load_data()
    agents, categories, affinity, workload = build_agent_profiles(df)

    print(f"\nAgents: {agents}")
    print(f"Categories: {categories}")
    print(f"\nCurrent workload (task count): {workload}")

    # save agent affinity profiles for reuse in the Week 4 dashboard
    with open(f"{OUT_DIR}/agent_profiles.json", "w", encoding="utf-8") as f:
        json.dump({"affinity": affinity, "workload": workload}, f, indent=2)
    print(f"Saved agent profiles -> {OUT_DIR}/agent_profiles.json")

    # demo: recommend an assignee for a few example new tasks
    print("\n=== Example Recommendations ===")
    examples = [
        ("Bug", "Urgent"),
        ("Documentation", "Low"),
        ("DevOps", "High"),
    ]
    for cat, pri in examples:
        best_agent, scores = recommend_assignee(cat, pri, agents, affinity, workload)
        print(f"\nNew task — category: {cat}, priority: {pri}")
        print(f"  {explain_recommendation(best_agent, cat, pri, affinity, workload)}")

    # demonstrate workload balancing vs random assignment
    simulate_and_compare(df, agents, categories, affinity, workload)


if __name__ == "__main__":
    main()