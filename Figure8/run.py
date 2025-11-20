import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data (assumed in seconds)
df = pd.read_csv("benchmark_times.csv")

# Convert seconds → hours
time_cols = ["LLM", "Unit Test", "SMT", "Autotuning", "Evaluation"]
df[time_cols] = np.round(df[time_cols] / 3600, 1)

# Prepare
operators = df["Operator"].tolist()
data = df[time_cols].values
x = np.arange(len(operators))
bar_width = 0.6

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bottoms = np.zeros(len(operators))
for idx, label in enumerate(time_cols):
    ax.bar(x, data[:, idx], bar_width, bottom=bottoms, label=label)
    bottoms += data[:, idx]

# Annotate total times in hours
for xpos, total in zip(x, bottoms):
    ax.text(xpos, total, f"{total:.1f}", ha="center", va="bottom")

# Decorations
ax.set_ylabel("Time (Hours)", fontsize=14)
ax.set_title("Breakdown of QiMeng-Xpiler's Compilation Time", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(operators, rotation=45, ha="right", fontsize=12)

# Plot average line (in hours)
average = bottoms.mean()
ax.axhline(average, color="gray", linestyle="--", linewidth=1)
ax.text(
    1,
    average,
    f"Average: {average:.1f}",
    color="gray",
    va="bottom",
    ha="left",
    fontsize=12,
)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig("benchmark_times.png")
plt.show()
