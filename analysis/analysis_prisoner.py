"""
analysis_prisoners.py
Statistical analysis of the Prisoner's Dilemma results.

Run:  python analysis_prisoners.py
Needs: stats_helpers.py in the same folder.

Four parts:
  1. Parser audit
  2. Cooperation rates with confidence intervals
  3. Effect of each perturbation, and the headline comparison between models
  4. Whether the models respond to the payoff numbers at all (Brookins Table 3)
"""

import pandas as pd
import statsmodels.api as sm

from stats_helpers import (fmt_p, rate_line, newcombe_diff, two_prop_test,
                           bootstrap_mean_ci, holm, load_all, model_from_filename)

DATA_GLOB = "prisoners_game-data/Prisoners_Dilemma_*.csv"
THRESHOLD_PP = 10.0          # pre-registered: below 10 points, not substantively interesting
CONDITIONS = ["change_output_format", "change_order_swap"]

# Published benchmarks for the regression in part 4
BENCHMARKS = """
    Mengel (2018), humans:   RISK -0.269***   TEMPT -0.055     EFF +0.308***
    Brookins & DeBacker,
    GPT-3.5:                 RISK +0.048      TEMPT +0.149     EFF +0.661**
"""


def prepare(d):
    """Add the columns the analysis needs."""
    d = d.copy()
    # Option A is cooperate, option B is defect
    d["cooperate"] = (d.choice.astype(str).str.lower() == "a").astype(int)
    # Each payoff matrix is one "cluster": the 50 draws inside it are not
    # independent observations of a single population, they all saw the same game
    d["matrix"] = list(zip(d.payoff_a, d.payoff_b, d.payoff_c, d.payoff_d))
    d["mkey"] = d["matrix"].astype(str)
    # Mengel's three normalised properties of a prisoner's dilemma, the same
    # definitions Brookins & DeBacker use in their Table 3
    d["RISK"] = (d.payoff_d - d.payoff_b) / d.payoff_d    # cost of cooperating alone
    d["TEMPT"] = (d.payoff_c - d.payoff_a) / d.payoff_c   # gain from defecting alone
    d["EFF"] = (d.payoff_a - d.payoff_d) / d.payoff_a     # value of mutual cooperation
    return d


def main():
    d = prepare(load_all(DATA_GLOB, model_from_filename))

    # 1. Parser audit. `attempt` restarts at 1 per file (one file = one payoff
    #    matrix), so sums are taken per file before aggregating.
    print("=" * 70)
    print("1. PARSER AUDIT")
    print("=" * 70)
    audit = (d.groupby(["model", "prompt_type", "source_file"])
               .agg(valid=("attempt", "size"), attempts=("attempt", "max"))
               .groupby(level=[0, 1]).sum())
    audit["discarded"] = audit.attempts - audit.valid
    audit["discard_rate"] = audit.discarded / audit.attempts
    print(audit.to_string())
    if audit.discarded.sum() == 0:
        print("\n  Zero discards anywhere. Every response was read on the first")
        print("  attempt, so the answer parser cannot have biased the sample.")

    # ------------------------------------------------------------------
    # 2. Cooperation rates.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. COOPERATION RATES")
    print("=" * 70)
    for (m, c), g in d.groupby(["model", "prompt_type"]):
        print("  " + rate_line(f"{m} {c}", g.cooperate.sum(), len(g)))
        per_matrix = g.groupby("mkey").cooperate.mean()
        _, blo, bhi = bootstrap_mean_ci(per_matrix.values)
        print(f"    clustered over {len(per_matrix)} matrices, bootstrap CI [{blo:.2%}, {bhi:.2%}]")
    print("\n  Human benchmark (Mengel 2018, one-shot): 37%")
    print("  GPT-3.5 benchmark (Brookins & DeBacker):  65%")

    # 3a. Effect of each perturbation, computed pooled and paired (within each
    #     payoff matrix, since the 50 draws inside one matrix are not independent).
    print("\n" + "=" * 70)
    print("3a. EFFECT OF EACH PERTURBATION")
    print(f"    (substantive threshold set in advance: {THRESHOLD_PP} points)")
    print("=" * 70)

    # cooperation rate per model / condition / matrix
    cell = (d.groupby(["model", "prompt_type", "mkey"], as_index=False)
              .agg(coop=("cooperate", "mean"),
                   RISK=("RISK", "first"), TEMPT=("TEMPT", "first"), EFF=("EFF", "first")))
    wide = cell.pivot_table(index="mkey", columns=["model", "prompt_type"], values="coop")

    family, deltas = [], {}
    for m in sorted(d.model.unique()):
        base = d[(d.model == m) & (d.prompt_type == "baseline")]
        print(f"\n  {m}")
        for c in CONDITIONS:
            alt = d[(d.model == m) & (d.prompt_type == c)]
            if len(alt) == 0:
                continue
            diff, lo, hi = newcombe_diff(alt.cooperate.sum(), len(alt),
                                         base.cooperate.sum(), len(base))
            per_matrix = (wide[(m, c)] - wide[(m, "baseline")]).dropna()
            deltas[(m, c)] = per_matrix
            _, blo, bhi = bootstrap_mean_ci(per_matrix.values)
            p = two_prop_test(alt.cooperate.sum(), len(alt),
                              base.cooperate.sum(), len(base))
            verdict = "EXCEEDS threshold" if abs(diff) * 100 >= THRESHOLD_PP else "below threshold"
            print(f"    {c:22s} {diff*100:+7.2f} points  95% CI "
                  f"[{lo*100:+6.2f}, {hi*100:+6.2f}]   {fmt_p(p):14s} -> {verdict}")
            print(f"    {'':22s} averaged over {len(per_matrix)} payoff matrices, "
                  f"bootstrap CI [{blo*100:+6.2f}, {bhi*100:+6.2f}]")
            family.append((f"{m} {c}", p))
    holm(family, "Prisoner's Dilemma perturbations")

    # 3b. Headline: does the perturbation hit the two models differently?
    #     Difference of the per-matrix deltas, with a bootstrap CI. A regression
    #     isn't usable here since baseline cooperation has zero variance.
    #     Result: Output Format Change moves Llama 8.19 points more than Gemma
    #     (95% CI [3.90, 12.86]); Order Swap moves Gemma 45.52 points more than
    #     Llama (95% CI [-61.05, -29.90]). Both intervals exclude zero.
    print("\n" + "=" * 70)
    print("3b. HEADLINE: DOES THE PERTURBATION HIT THE MODELS DIFFERENTLY?")
    print("=" * 70)
    models = sorted(d.model.unique())
    if len(models) == 2:
        a, b = models   # alphabetical: Gemma, Llama
        for c in CONDITIONS:
            if (a, c) not in deltas or (b, c) not in deltas:
                continue
            gap = (deltas[(a, c)] - deltas[(b, c)]).dropna()
            mean, lo, hi = bootstrap_mean_ci(gap.values)
            print(f"\n  {c}")
            print(f"    {a} moved {deltas[(a, c)].mean()*100:+.2f} points")
            print(f"    {b} moved {deltas[(b, c)].mean()*100:+.2f} points")
            print(f"    difference: {mean*100:+.2f} points  95% CI "
                  f"[{lo*100:+.2f}, {hi*100:+.2f}]")
            print(f"    -> the same perturbation moves {a} {abs(mean)*100:.1f} points "
                  f"{'more' if abs(deltas[(a,c)].mean()) > abs(deltas[(b,c)].mean()) else 'less'} "
                  f"than it moves {b}")
            if lo <= 0 <= hi:
                print("    -> interval includes zero: cannot conclude the models differ")

    # 4. Regress cooperation on RISK, TEMPT, EFF per model/condition, as in
    #    Brookins & DeBacker's Table 3.
    #    Result: baseline is 100% cooperation for both models across all 21
    #    matrices, so no regression is possible there. Under perturbation, the
    #    risk coefficient is never significant for either model; Llama's
    #    efficiency coefficient under Output Format Change is the one
    #    significant match to a human pattern, +0.281 vs Mengel's +0.308.
    print("\n" + "=" * 70)
    print("4. SENSITIVITY TO THE PAYOFF STRUCTURE (Brookins & DeBacker Table 3)")
    print("=" * 70)
    print(BENCHMARKS)
    for (m, c), g in cell.groupby(["model", "prompt_type"]):
        if g.coop.std() == 0:
            print(f"  {m:6s} {c:22s} NO VARIATION: cooperation is {g.coop.iloc[0]:.0%} in")
            print(f"  {'':29s} all {len(g)} matrices. The model's choice does not")
            print(f"  {'':29s} depend on the payoffs at all. This is the finding.")
            continue
        X = sm.add_constant(g[["RISK", "TEMPT", "EFF"]])
        r = sm.OLS(g.coop, X).fit()
        parts = "  ".join(
            f"{v} {r.params[v]:+.3f} ({r.bse[v]:.3f}){'*' if r.pvalues[v] < 0.05 else ' '}"
            for v in ["RISK", "TEMPT", "EFF", "const"])
        print(f"  {m:6s} {c:22s} {parts}   R2 = {r.rsquared:.3f}  n = {len(g)}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()