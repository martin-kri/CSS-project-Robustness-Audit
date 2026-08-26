import glob, pandas as pd

frames = []
for f in glob.glob("ultimatum_game-data/Ultimatum_game_LOGPROBS_*.csv"):
    df = pd.read_csv(f)
    df["model"] = "Llama 3.1" if "llama" in f.lower() else "Gemma 3"
    frames.append(df)

d = pd.concat(frames, ignore_index=True)

for model, group in d.groupby("model"):
    print(f"=== {model} : Mean Z-Coverage (%) ===")
    piv = group.groupby(["prompt_type", "offer"])["z_coverage"].mean().unstack() * 100
    
    # Displays as percentage with 6 decimal places
    formatted = piv.map(lambda x: f"{x:.6f}%" if pd.notnull(x) else "nan")
    print(formatted)
    print("\n")