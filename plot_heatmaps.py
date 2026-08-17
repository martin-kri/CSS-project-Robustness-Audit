import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
    "Gemma 3 (4B)\n(Order Swap)"
]

# ── Evaluation Setups ─────────────────────────────────────────────────────
evaluations = [
    {
        "type": "LOGPROBS",
        "file_template": "Ultimatum_game_LOGPROBS_{model}_{prompt}_offer{offer}.csv",
        "target_col": "p(accept)",
        "title_suffix": "First-Token Log-Probabilities",
        "cbar_label": "Probability of Acceptance [ p(accept) ]",
        "output_file": "Ultimatum_Game_Heatmap_LOGPROBS.pdf"
    },
    {
        "type": "TEXT",
        "file_template": "Ultimatum_game_{model}_{prompt}_offer{offer}_temp1.0.csv", 
        "target_col": "choice", 
        "title_suffix": "Text Generation (Temperature 1.0)",
        "cbar_label": "Acceptance Rate",
        "output_file": "Ultimatum_Game_Heatmap_TEXT.pdf"
    }
]

# ── Processing Loop ───────────────────────────────────────────────────────
for eval_config in evaluations:
    print(f"Generating heatmap for {eval_config['type']}...")
    heatmap_data = np.zeros((len(row_labels), len(offers)))
    
    row_idx = 0
    for model in models:
        for prompt in prompts:
            for col_idx, offer in enumerate(offers):
                file_path = eval_config["file_template"].format(model=model, prompt=prompt, offer=offer)
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    target = eval_config["target_col"]
                    
                    if target in df.columns:
                        if eval_config["type"] == "TEXT":
                            # Convert "accept" strings to 1s and "reject" to 0s to calculate the percentage
                            acceptance_rate = (df[target].astype(str).str.strip().str.lower() == 'accept').mean()
                            heatmap_data[row_idx, col_idx] = acceptance_rate
                        else:
                            # Logprobs are already numeric, just take the mean
                            heatmap_data[row_idx, col_idx] = df[target].mean()
                    else:
                        heatmap_data[row_idx, col_idx] = np.nan
                else:
                    heatmap_data[row_idx, col_idx] = np.nan
            row_idx += 1

    # ── Plotting ─────────────────────────────────────────────────────────────
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(
        heatmap_data, 
        cmap="Blues", 
        annot=True,
        fmt=".2f",
        linewidths=1, 
        cbar_kws={"orientation": "horizontal", "label": eval_config["cbar_label"], "pad": 0.1}
    )

    # ── Formatting ───────────────────────────────────────────────────────────
    ax.set_xticks([x + 0.5 for x in range(len(offers))])
    ax.set_xticklabels([f"${x}" for x in offers], fontsize=12)
    ax.set_yticks([y + 0.5 for y in range(len(row_labels))])
    ax.set_yticklabels(row_labels, rotation=0, fontsize=12)

    plt.xlabel("Offer Amount ($)", fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel("Model & Prompt Condition", fontsize=14, fontweight='bold', labelpad=10)

    plt.tight_layout()
    plt.savefig(eval_config["output_file"], dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"Heatmap saved successfully as '{eval_config['output_file']}'\n")