import re
import pandas as pd

df = pd.read_csv("dictator_game-data/gpt_dictator_results.csv")

# Extract the number allocated/transferred to the recipient
df["allocation"] = (
    df["model_answer"]
    .str.extract(r"(?:transfer|allocate)\s*(\d+)", flags=re.IGNORECASE)[0]
    .astype(float)
)

mean_allocation = df["allocation"].mean()
valid_responses = df["allocation"].dropna().count()

print(f"Mean allocation: {mean_allocation:.2f}€")
print(f"Valid responses parsed: {valid_responses} / {len(df)}")