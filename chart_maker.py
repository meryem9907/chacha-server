import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data from your table (percent values)
# -----------------------------
scores = ["ROUGE-1", "ROUGE-L", "BERTScore"]
conditions = ["High-2B", "High-8B", "Low-2B", "Low-8B"]
res_2b = ["High-2B", "Low-2B"]
res_8b = [ "High-8B", "Low-8B"]



data = {
    ("High-2B", "F1"):        [33.36, 33.16, 88.70],
    ("High-2B", "Precision"): [32.19, 32.05, 87.62],
    ("High-2B", "Recall"):    [49.88, 49.52, 89.86],

    ("High-8B", "F1"):        [59.21, 59.16, 95.34],
    ("High-8B", "Precision"): [60.33, 60.29, 95.48],
    ("High-8B", "Recall"):    [60.36, 60.29, 95.25],

    ("Low-2B", "F1"):         [28.58, 28.47, 88.06],
    ("Low-2B", "Precision"):  [26.48, 26.40, 86.86],
    ("Low-2B", "Recall"):     [47.21, 47.00, 89.34],

    ("Low-8B", "F1"):         [55.71, 55.60, 95.34],
    ("Low-8B", "Precision"):  [56.93, 56.86, 95.49],
    ("Low-8B", "Recall"):     [58.93, 58.72, 95.26],
}

data_2b = {
    ("High-2B", "F1"):        [33.36, 33.16, 88.70],
    ("High-2B", "Precision"): [32.19, 32.05, 87.62],
    ("High-2B", "Recall"):    [49.88, 49.52, 89.86],

    ("Low-2B", "F1"):         [28.58, 28.47, 88.06],
    ("Low-2B", "Precision"):  [26.48, 26.40, 86.86],
    ("Low-2B", "Recall"):     [47.21, 47.00, 89.34],
}

data_8b = {

    ("High-8B", "F1"):        [59.21, 59.16, 95.34],
    ("High-8B", "Precision"): [60.33, 60.29, 95.48],
    ("High-8B", "Recall"):    [60.36, 60.29, 95.25],


    ("Low-8B", "F1"):         [55.71, 55.60, 95.34],
    ("Low-8B", "Precision"):  [56.93, 56.86, 95.49],
    ("Low-8B", "Recall"):     [58.93, 58.72, 95.26],
}

# -----------------------------
# Build a tidy dataframe
# -----------------------------
rows = []
for (cond, metric), vals in data.items():
    for score, val in zip(scores, vals):
        rows.append({"Score": score, "Condition": cond, "Metric": metric, "Value": float(val)})

df = pd.DataFrame(rows)

# Optional: wide view (like Excel)
wide = (
    df.pivot_table(index="Score", columns=["Condition", "Metric"], values="Value")
      .reindex(scores)
)
print("\nWide table:\n")
print(wide.round(2))

# -----------------------------
# Plot helpers
# -----------------------------
def grouped_bar_chart(metric_name: str, title: str):
    """Grouped bars: Score on X, 4 conditions as series."""
    sub = df[df["Metric"] == metric_name].copy()
    pivot = sub.pivot(index="Score", columns="Condition", values="Value").reindex(scores)

    x = np.arange(len(scores))
    n = len(conditions)
    #n = len(res_2b)
    width = 0.10

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, cond in enumerate(conditions):
        ax.bar(x + (i - (n - 1) / 2) * width, pivot[cond].values, width, label=cond)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.set_ylabel("Value (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()


def delta_chart(metric_name: str, title: str):
    """
    Δ chart: (Low - High) per model size.
    Negative = drop on low resolution.
    """
    sub = df[df["Metric"] == metric_name].copy()
    pivot = sub.pivot(index="Score", columns="Condition", values="Value").reindex(scores)
    print(pivot)

    delta_2b = pivot["Low-2B"] - pivot["High-2B"]
    delta_8b = pivot["Low-8B"] - pivot["High-8B"]

    x = np.arange(len(scores))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, delta_2b.values, width, label="Δ 2B (Low-High)")
    ax.bar(x + width / 2, delta_8b.values, width, label="Δ 8B (Low-High)")
    ax.axhline(0, linewidth=1)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.set_ylabel("Δ (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 1) Clear comparison charts
# -----------------------------
#grouped_bar_chart("F1", "F1 (%) by Score and Resolution")
#grouped_bar_chart("Precision", "Precision (%) by Score and Resolution")
#grouped_bar_chart("Recall", "Recall (%) by Score and Resolution")

# -----------------------------
# 2) Resolution impact charts (Low - High)
# -----------------------------
delta_chart("F1", "Resolution impact on F1: Δ = Low − High (negative = drop)")
delta_chart("Precision", "Resolution impact on Precision: Δ = Low − High (negative = drop)")
delta_chart("Recall", "Resolution impact on Recall: Δ = Low − High (negative = drop)")
