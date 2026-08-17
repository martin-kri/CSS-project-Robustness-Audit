import pandas as pd
import os

# ── Configuration ────────────────────────────────────────────────────────
models = ["Llama_3.1_8B_Instruct", "gemma_3_4b_it"]
model_titles = ["Llama 3.1 (8B)", "Gemma 3"]

prompts = ["baseline", "change_output_format", "change_order_swap"]
prompt_titles = ["Baseline", "Output Format Change", "Order Swap"]

print("=== Dictator Game: Histogram Probabilities ===\n")

# ── Data Extraction Loop ─────────────────────────────────────────────────
for i, model in enumerate(models):
    for j, prompt in enumerate(prompts):
        filename = f"Dictator_game_{model}_{prompt}_temp1.0.csv"
        
        print(f"--- {model_titles[i]} | {prompt_titles[j]} ---")
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            
            # 1. The Vertical Line (Mean)
            model_mean = df['allocation'].mean()
            print(f"Mean Allocation: {model_mean:.2f}€")
            
            # 2. The Bar Heights (Probabilities)
            probs = df['allocation'].value_counts(normalize=True).sort_index()
            
            print("Bar Heights (Probability):")
            for alloc in range(11):  # Offers from 0€ to 10€
                # .get() returns 0.0 if the model never made that specific offer
                prob = probs.get(alloc, 0.0) 
                print(f"  {alloc}€: {prob:.4f}")
        else:
            print("Data File Not Found.")
            
        print("") # Blank line for readability