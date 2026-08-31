import glob
import pandas as pd

files_with_retries = []

for f in sorted(glob.glob("prisoners_game-data/Prisoners_Dilemma_*.csv")):
    d = pd.read_csv(f)
    if "attempt" not in d.columns or d.empty:
        continue

    # Calculate attempts spent on each iteration from the cumulative counter
    attempts_taken = d["attempt"].diff().fillna(d["attempt"].iloc[0]).astype(int)
    retries = d[attempts_taken > 1]

    if not retries.empty:
        retry_info = [
            f"Iteration {row.get('iteration', idx + 1)}: took {attempts_taken[idx]} attempts"
            for idx, row in retries.iterrows()
        ]
        files_with_retries.append((f, retry_info))

if not files_with_retries:
    print("All files succeeded on the first attempt.")
else:
    for filename, details in files_with_retries:
        print(f"{filename}:")
        for item in details:
            print(f"  - {item}")
        print()