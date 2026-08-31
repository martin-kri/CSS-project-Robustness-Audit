import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configuration ────────────────────────────────────────────────────────
models = ["Llama_3.1_8B_Instruct", "gemma_3_4b_it"] 
model_titles = ["Llama 3.1 (8B)", "Gemma 3 (4B)"]

prompts = ["change_output_format", "change_order_swap"]
prompt_titles = ["Output Format Change", "Order Swap"]

# Mengel (2018) Payoffs. Mengel's own listing repeats (250, 50, 750, 150);
# dropped here as a duplicate, same 21 unique matrices as prisoners_dilemma.py.
PAYOFFS = [
    (400, 200, 450, 200), (400, 10, 450, 200),
    (400, 200, 800, 200), (400, 10, 800, 200),
    (400, 100, 450, 120), (400, 100, 450, 200),
    (10, 1, 90, 5),       (10, 5, 90, 5),
    (150, 40, 850, 50),   (150, 5, 850, 95),
    (250, 15, 750, 85),   (250, 5, 750, 95),
    (250, 50, 750, 150),  (250, 100, 750, 160),
    (10, 2, 110, 3),      (10, 1, 110, 9),
    (150, 50, 850, 100),
    (400, 100, 600, 120), (400, 100, 600, 200),
    (400, 100, 1200, 120),(400, 100, 1200, 200)
]

# ── Data Loading & Aggregation ───────────────────────────────────────────
all_data = []
for model in models:
    for prompt in prompts:
        for (a, b, c, d) in PAYOFFS:
            file_path = f"prisoners_game-data/Prisoners_Dilemma_{model}_payoffs_{a}_{b}_{c}_{d}_{prompt}_temp1.0.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['model'] = model
                df['prompt_type'] = prompt
                df['a'], df['b'], df['c'], df['d'] = a, b, c, d
                all_data.append(df)

if not all_data:
    raise FileNotFoundError("No CSV files found. Check your directory and filenames.")

full_df = pd.concat(all_data, ignore_index=True)

# Map "a" to Cooperate and "b" to Defect
full_df["Strategy"] = full_df["choice"].map({"a": "Cooperate", "b": "Defect"})
full_df["Cooperate_Indicator"] = np.where(full_df["Strategy"] == "Cooperate", 1, 0)

# RISK, same definition as analysis_prisoner.py: (d - b) / d
full_df["RISK"] = (full_df["d"] - full_df["b"]) / full_df["d"]

# Collapse to one cooperation rate per (model, prompt, matrix), matching
# the aggregation level `cell` uses in analysis_prisoner.py, one point per
# matrix, not one point per raw draw.
cell = (
    full_df.groupby(["model", "prompt_type", "a", "b", "c", "d", "RISK"])
    ["Cooperate_Indicator"].mean()
    .reset_index()
    .rename(columns={"Cooperate_Indicator": "coop"})
)

# ── Plotting: cooperation vs. RISK, one panel per model ─────────────────
sns.set_theme(style="ticks")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5.5), sharex=True, sharey=True)

condition_style = {
    "change_output_format": {"color": "#d62728", "marker": "s", "label": "Output Format Change", "offset": 0.0},
    "change_order_swap": {"color": "#2ca02c", "marker": "^", "label": "Order Swap", "offset": 0.012},
}

rng = np.random.RandomState(42)

# Mengel (2018) human regression line: Cooperate = 0.455 - 0.269 * RISK,
# her reported constant and RISK coefficient from Table 1. Holds temptation
# and efficiency at whatever level is implicit in her own regression, this
# is a reference line for the slope, not a claim that it predicts this data.
risk_range = np.linspace(0, 1, 100)
mengel_line = 0.455 - 0.269 * risk_range

for i, model in enumerate(models):
    ax = axes[i]
    for prompt in prompts:
        style = condition_style[prompt]
        subset = cell[(cell["model"] == model) & (cell["prompt_type"] == prompt)]
        jitter = rng.uniform(-0.006, 0.006, size=len(subset))
        ax.scatter(
            subset["RISK"] + style["offset"] + jitter, subset["coop"],
            color=style["color"], marker=style["marker"], label=style["label"],
            s=60, alpha=0.75, edgecolor="black", linewidth=0.5, zorder=3,
        )

    ax.axhline(1.0, color="#1f77b4", linewidth=2.5, label="Baseline (100% on every matrix)", zorder=2)
    ax.plot(risk_range, mengel_line, color="gray", linestyle="--", linewidth=1.5,
             label="Mengel (2018) human slope", zorder=2)

    ax.set_title(model_titles[i], fontsize=13, fontweight="bold")
    ax.set_xlabel("RISK = (d − b) / d", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.0)

axes[0].set_ylabel("Cooperation Rate (per matrix)", fontsize=12, fontweight="bold")
axes[1].legend(loc="lower left", fontsize=9, framealpha=0.9)

plt.tight_layout()
sns.despine()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/Prisoners_Dilemma_Risk_Scatter.pdf", dpi=300, bbox_inches="tight")
print("Graph saved successfully as 'plots/Prisoners_Dilemma_Risk_Scatter.pdf'\n")