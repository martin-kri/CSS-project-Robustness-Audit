import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Configuration ────────────────────────────────────────────────────────
models = ["Llama_3.1_8B_Instruct", "gemma_3_4b_it"] 
model_titles = ["Llama 3.1 (8B)", "Gemma 3 (4B)"]

prompts = ["baseline", "change_output_format", "change_order_swap"]
prompt_titles = ["Baseline", "Output Format Change", "Order Swap"]

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
                all_data.append(df)

if not all_data:
    raise FileNotFoundError("No CSV files found. Check your directory and filenames.")

full_df = pd.concat(all_data, ignore_index=True)

# Map "a" to Cooperate and "b" to Defect
full_df["Strategy"] = full_df["choice"].map({"a": "Cooperate", "b": "Defect"})

full_df["Cooperate_Indicator"] = np.where(full_df["Strategy"] == "Cooperate", 1, 0)

# ── Plotting: Frequency of Strategies ────────────────────────────────────
sns.set_theme(style="ticks")
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)

for i, model in enumerate(models):
    for j, prompt in enumerate(prompts):
        ax = axes[i, j]
        subset = full_df[(full_df['model'] == model) & (full_df['prompt_type'] == prompt)]
        
        if not subset.empty:
            # Calculate probabilities
            counts = subset['Strategy'].value_counts(normalize=True).reindex(["Cooperate", "Defect"], fill_value=0)
            
            sns.barplot(
                x=counts.index, 
                y=counts.values, 
                hue=counts.index,
                legend=False,
                ax=ax, 
                palette=["#2ca02c", "#d62728"], # Green for Coop, Red for Defect
                edgecolor="black",
                alpha=0.8
            )
            
            # Annotate overall cooperation rate
            coop_rate = counts.get("Cooperate", 0)
            ax.text(0.5, 0.9, f"Cooperation Rate: {coop_rate:.1%}", 
                    transform=ax.transAxes, ha='center', va='top', 
                    fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            
        if i == 0:
            ax.set_title(prompt_titles[j], fontsize=13, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f"{model_titles[i]}\nProbability", fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel("")
            
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0, 1, 0.94])
sns.despine()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/Prisoners_Dilemma_Strategies.pdf", dpi=300, bbox_inches='tight')
print("Graph saved successfully as 'plots/Prisoners_Dilemma_Strategies.pdf'\n")