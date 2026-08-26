import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ── Configuration ────────────────────────────────────────────────────────
models = ["Llama_3.1_8B_Instruct", "gemma_3_4b_it"]
model_titles = ["Llama 3.1 (8B)", "Gemma 3 (4B)"]

prompts = ["baseline", "change_output_format", "change_order_swap"]
prompt_titles = ["Baseline", "Output Format Change", "Order Swap"]

# Baselines
HUMAN_MEAN = 2.83
BROOKINS_MEAN = 4.83

sns.set_theme(style="ticks")

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
# ── Plotting Loop ────────────────────────────────────────────────────────
for i, model in enumerate(models):
    for j, prompt in enumerate(prompts):

        filename = f"dictator_game-data/Dictator_game_{model}_{prompt}_temp1.0.csv"
        ax = axes[i, j]

        if os.path.exists(filename):
            df = pd.read_csv(filename)

            # Plot histogram
            sns.histplot(
                data=df,
                x="allocation",
                bins=range(12),
                discrete=True,
                stat="probability",
                ax=ax,
                color="#1f77b4" if i == 0 else "#d62728",  # Navy for Llama, Crimson for Gemma
                edgecolor="black",
                alpha=0.7
            )

            # Model Mean
            model_mean = df['allocation'].mean()
            mean_color = "#003f5c" if i == 0 else "#8b0000"
            ax.axvline(model_mean, color=mean_color, linestyle='-', linewidth=2.5,
                       label=f"Current LLM Mean: {model_mean:.2f}€")

            # Human Mean (Green Dotted)
            ax.axvline(HUMAN_MEAN, color='#2ca02c', linestyle='dotted', linewidth=2,
                       label=f"Human (Engel): {HUMAN_MEAN}€")

            # Brookins GPT-3.5 Mean (Grey Dashed)
            ax.axvline(BROOKINS_MEAN, color='grey', linestyle='dashed', linewidth=2,
                       label=f"GPT-3.5 (Brookins): {BROOKINS_MEAN}€")

            # Add Legend
            ax.legend(loc='upper right', fontsize=9, framealpha=1.0)

        else:
            ax.text(0.5, 0.5, 'Data File Not Found', ha='center', va='center')

        # Set titles and labels
        if i == 0:
            ax.set_title(prompt_titles[j], fontsize=13, fontweight='bold')
        if j == 0:
            ax.set_ylabel(f"{model_titles[i]}\nProbability", fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Offer Amount (€)", fontsize=11)
        ax.set_xticks(range(11))

        # Strict Probability Y-Axis Formatting
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Clean up spacing
plt.tight_layout(rect=[0, 0, 1, 0.94])
sns.despine()

# Save the plot
os.makedirs("graphs", exist_ok=True)
plt.savefig("graphs/dictator_allocations.pdf", dpi=300, bbox_inches='tight')
print("Graph saved successfully as 'graphs/dictator_allocations.pdf'")