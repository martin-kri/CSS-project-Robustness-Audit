import glob
import pandas as pd

files = glob.glob("ultimatum_game-data/Ultimatum_game_gemma*_temp1.0.csv")
for f in sorted(files):
    df = pd.read_csv(f)
    col = "response_raw" if "response_raw" in df.columns else "response"
    
    # Check for null, empty strings, or suspiciously short outputs (< 3 chars)
    bad_rows = df[df[col].isna() | (df[col].astype(str).str.strip().isin(["", '""', "''"])) | (df[col].astype(str).str.len() < 3)]
    
    if not bad_rows.empty:
        print(f"{f}: Found {len(bad_rows)} degenerate/empty rows")
    else:
        print(f"{f}: Clean (no empty outputs)")