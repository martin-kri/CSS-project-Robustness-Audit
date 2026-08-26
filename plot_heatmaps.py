import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────────
models = ["Llama_3.1_8B_Instruct", "gemma_3_4b_it"]
prompts = ["baseline", "change_output_format", "change_order_swap"]
offers = list(range(11))

row_labels = [
    "Llama 3.1 (8B)\n(Baseline)",
    "Llama 3.1 (8B)\n(Output Format Change)",
    "Llama 3.1 (8B)\n(Order Swap)",
    "Gemma 3 (4B)\n(Baseline)",
    "Gemma 3 (4B)\n(Output Format Change)",
    "Gemma 3 (4B)\n(Order Swap)",
]

# ── Evaluation Setups ─────────────────────────────────────────────────────
evaluations = [
    {
        "type": "LOGPROBS",
        "file_template": "ultimatum_game-data/Ultimatum_game_LOGPROBS_{model}_{prompt}_offer{offer}.csv",
        "target_col": "p(accept)",
        "title_suffix": "First-Token Log-Probabilities",
        "cbar_label": "Probability of Acceptance [ p(accept) ]",
        "output_file": "graphs/Ultimatum_Game_Heatmap_LOGPROBS.pdf",
    },
    {
        "type": "TEXT",
        "file_template": "ultimatum_game-data/Ultimatum_game_{model}_{prompt}_offer{offer}_temp1.0.csv",
        "target_col": "choice",
        "title_suffix": "Text Generation (Temperature 1.0)",
        "cbar_label": "Acceptance Rate",
        "output_file": "graphs/Ultimatum_Game_Heatmap_TEXT.pdf",
    },
]

# ── Processing Loop ───────────────────────────────────────────────────────
for eval_config in evaluations:
    print(f"Generating heatmap for {eval_config['type']}...")
    heatmap_data = np.zeros((len(row_labels), len(offers)))

    row_idx = 0
    for model in models:
        for prompt in prompts:
            for col_idx, offer in enumerate(offers):
                file_path = eval_config["file_template"].format(
                    model=model, prompt=prompt, offer=offer
                )

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    target = eval_config["target_col"]

                    if target in df.columns:
                        if eval_config["type"] == "TEXT":
                            choice = df[target].astype(str).str.strip().str.lower()
                            choice = choice[choice != "invalid"]
                            if len(choice) > 0:
                                acceptance_rate = choice.str.rstrip("s").eq("accept").mean()
                            else:
                                acceptance_rate = np.nan
                            heatmap_data[row_idx, col_idx] = acceptance_rate
                        else:
                            heatmap_data[row_idx, col_idx] = df[target].mean()
                    else:
                        heatmap_data[row_idx, col_idx] = np.nan
                else:
                    heatmap_data[row_idx, col_idx] = np.nan
            row_idx += 1

    # ── Masking for Instrument Failure in Logprobs ──────────────────────────
    # Create mask matrix (True = masked / hidden from heatmap)
    mask = np.zeros_like(heatmap_data, dtype=bool)

    if eval_config["type"] == "LOGPROBS":
        # Row 3: Gemma 3 Baseline | Row 5: Gemma 3 Order Swap
        failed_rows = [3, 5]
        for r in failed_rows:
            mask[r, :] = True

    # ── Plotting ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=1,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={
            "orientation": "horizontal",
            "label": eval_config["cbar_label"],
            "pad": 0.1,
        },
    )

    # ── Draw Hatched Patches for Masked Rows ────────────────────────────────
    if eval_config["type"] == "LOGPROBS":
        for r in failed_rows:
            # Add gray hatched background across the entire row
            rect = patches.Rectangle(
                (0, r),
                len(offers),
                1,
                facecolor="#ececec",
                edgecolor="#7f7f7f",
                hatch="///",
                linewidth=1,
                zorder=2,
            )
            ax.add_patch(rect)

            # Centered label badge explaining the mask
            ax.text(
                len(offers) / 2.0,
                r + 0.5,
                "Instrument Failure (Zero Probability Mass / Unusable Logprobs)",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#333333",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="white",
                    ec="#7f7f7f",
                    lw=1.2,
                ),
                zorder=3,
            )

    # ── Formatting ───────────────────────────────────────────────────────────
    ax.set_xticks([x + 0.5 for x in range(len(offers))])
    ax.set_xticklabels([f"${x}" for x in offers], fontsize=12)
    ax.set_yticks([y + 0.5 for y in range(len(row_labels))])
    ax.set_yticklabels(row_labels, rotation=0, fontsize=12)

    plt.xlabel("Offer Amount ($)", fontsize=14, fontweight="bold", labelpad=10)
    plt.ylabel(
        "Model & Prompt Condition", fontsize=14, fontweight="bold", labelpad=10
    )
    plt.title(
        f"Ultimatum Game Acceptance: {eval_config['title_suffix']}",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(eval_config["output_file"], dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved successfully as '{eval_config['output_file']}'\n")