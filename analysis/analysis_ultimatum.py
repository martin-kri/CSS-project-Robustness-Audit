"""
analysis_ultimatum.py
Statistical analysis of the Ultimatum Game results.

Run:  python analysis_ultimatum.py
Needs: stats_helpers.py in the same folder.

Five parts:
  1. Parser audit, including a check that 'accepts' was folded into 'accept'
  2. The two measurement instruments side by side
  3. H2: do text generation and token probability disagree?
  4. Does acceptance rise with the offer, as it does for humans?
  5. How much does swapping a surname change the answer?
"""

import numpy as np
import pandas as pd
from scipy import stats

from stats_helpers import fmt_p, holm, load_all, model_from_filename

TEXT_GLOB = "ultimatum_game-data/Ultimatum_game_*_temp*.csv"
LOGP_GLOB = "ultimatum_game-data/Ultimatum_game_LOGPROBS_*.csv"
THRESHOLD_PP = 10.0


def accept_flag(series):
    """
    Fold 'accepts' into 'accept' and 'rejects' into 'reject'.
    The generation script stored all four strings separately, so testing
    choice == "accept" would silently count every plural as a rejection.
    """
    return series.astype(str).str.lower().str.rstrip("s").eq("accept")


def main():
    txt = load_all(TEXT_GLOB, model_from_filename)
    lgp = load_all(LOGP_GLOB, model_from_filename)

    # Cells where the scored tokens hold almost no probability mass are
    # reported as instrument failure, not as a result.
    Z_THRESHOLD = 0.5
    z_by_cell = lgp.groupby(["model", "prompt_type"]).z_coverage.mean()
    invalid_cells = z_by_cell[z_by_cell < Z_THRESHOLD].index.tolist()

    print("\nLog-probability validity by cell (mean z_coverage):")
    print(z_by_cell.round(4).to_string())
    if invalid_cells:
        print(f"\nBelow {Z_THRESHOLD}, treated as instrument failure, not a result:")
        for cell in invalid_cells:
            print(f"  {cell}")

    # 1. Parser audit.
    print("=" * 70)
    print("1. PARSER AUDIT")
    print("=" * 70)
    print("\n  Raw values stored in the `choice` column of the text files:")
    print(txt.choice.value_counts().to_string())
    plurals = txt.choice.astype(str).str.lower().isin(["accepts", "rejects"]).sum()
    print(f"\n  {plurals} plural forms found and folded in.")

    print("\n  Unreadable responses, by condition:")
    inv = (txt.assign(bad=txt.choice.eq("invalid"))
              .groupby(["model", "prompt_type"]).bad.agg(["mean", "sum", "size"]))
    inv.columns = ["invalid_rate", "invalid_n", "total_n"]
    print(inv.to_string(float_format=lambda x: f"{x:.4f}"))

    txt = txt[txt.choice != "invalid"].copy()
    txt["accept"] = accept_flag(txt.choice).astype(int)

    # 2. The two instruments: mean_p (average p(accept)), hard_rate (share with
    #    p(accept) > 0.5), and text (share of generated answers saying accept).
    #    text and hard_rate are both proportions, so that's the fair comparison.
    lgp["accept_hard"] = accept_flag(lgp.choice).astype(int)
    grid = (lgp.groupby(["model", "prompt_type", "offer"])
               .agg(mean_p=("p(accept)", "mean"), hard_rate=("accept_hard", "mean"))
               .reset_index())
    text_rate = txt.groupby(["model", "prompt_type", "offer"]).accept.mean().rename("text")
    m = grid.set_index(["model", "prompt_type", "offer"]).join(text_rate).reset_index()

    print("\n" + "=" * 70)
    print("2. ACCEPTANCE BY OFFER, UNDER EACH INSTRUMENT")
    print("=" * 70)
    for col, label in [("text", "generated text"),
                       ("hard_rate", "token probability, thresholded at 0.5"),
                       ("mean_p", "token probability, averaged")]:
        print(f"\n  {label}")
        print(m.pivot_table(index=["model", "prompt_type"], columns="offer",
                            values=col).round(2).to_string())

    # 3. H2: MAD is the mean absolute gap between the two instruments across
    #    the eleven offers. With only 11 paired items the Wilcoxon test floors
    #    at p=0.000977, so it can't rank gap sizes; MAD is the number that matters.
    print("\n" + "=" * 70)
    print("3. H2: DO THE TWO INSTRUMENTS DISAGREE?")
    print(f"   MAD = mean absolute gap across the 11 offers. Threshold: {THRESHOLD_PP} points.")
    print("   NOTE: the p-value bottoms out at 0.000977 with 11 offers. Read MAD.")
    print("=" * 70)
    family = []
    for (mm, c), g in m.groupby(["model", "prompt_type"]):
        if (mm, c) in invalid_cells:
            continue
        print(f"\n  {mm} {c}")
        for col, label in [("hard_rate", "vs thresholded probability"),
                           ("mean_p", "vs averaged probability")]:
            mad = (g.text - g[col]).abs().mean() * 100
            p = stats.wilcoxon(g.text, g[col]).pvalue
            verdict = "EXCEEDS threshold" if mad >= THRESHOLD_PP else "below threshold"
            floor = "  (at the test's floor)" if p <= 0.000978 else ""
            print(f"    {label:28s} MAD = {mad:6.2f} points   {fmt_p(p)}{floor}"
                  f"   -> {verdict}")
            family.append((f"{mm} {c} {label}", p))

    print("\n  Rank reversal: does the instrument change which model looks more generous?")
    for c in m.prompt_type.unique():
        sub = m[m.prompt_type == c]
        pt = sub.pivot(index="offer", columns="model", values="text")
        pl = sub.pivot(index="offer", columns="model", values="mean_p")
        if pt.shape[1] < 2:
            continue
        flips = int((np.sign(pt.iloc[:, 1] - pt.iloc[:, 0])
                     != np.sign(pl.iloc[:, 1] - pl.iloc[:, 0])).sum())
        print(f"    {c:22s} the ordering flips at {flips} of {len(pt)} offers")
    holm(family, "Ultimatum Game, H2")

    # 4. Offer sensitivity: Spearman correlation between offer and acceptance.
    print("\n" + "=" * 70)
    print("4. DOES ACCEPTANCE RISE WITH THE OFFER, AS IT DOES FOR HUMANS?")
    print("=" * 70)
    for (mm, c), g in m.groupby(["model", "prompt_type"]):
        row = f"  {mm:6s} {c:22s}"
        for col in ["text", "mean_p"]:
            rho, p = stats.spearmanr(g.offer, g[col])
            row += f"   {col}: rho {rho:+.3f} ({fmt_p(p)})"
        print(row)
    print("\n  Aher et al. found this rising pattern only in their largest model.")

    # 5. Name sensitivity: does acceptance vary by proposer/responder race group?
    print("\n" + "=" * 70)
    print("5. HOW MUCH DOES THE NAME MATTER?")
    print("=" * 70)
    for (mm, c), g in txt.groupby(["model", "prompt_type"]):
        by = g.groupby(["proposer_race", "responder_race"]).accept.mean()
        rng = (by.max() - by.min()) * 100
        flag = "  <-- exceeds threshold" if rng >= THRESHOLD_PP else ""
        print(f"  {mm:6s} {c:22s} acceptance ranges {by.min():.3f} to {by.max():.3f} "
              f"across 25 name-group pairs (spread {rng:.1f} points){flag}")
    print("\n  Sensitivity to the name, not a bias claim: the responder surname is")
    print("  drawn once per group, so the cells are not fully independent.")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()