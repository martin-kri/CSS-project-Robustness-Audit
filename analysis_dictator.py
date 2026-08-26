"""
analysis_dictator.py
Statistical analysis of the Dictator Game results.

Run:  python analysis_dictator.py
Needs: stats_helpers.py in the same folder.

Produces four things, in the order they should appear in the paper:
  1. How many responses the answer-parser threw away, and whether that matters
  2. Mean allocation with a confidence interval, plus the shape of the
     distribution (which the mean hides)
  3. Whether each prompt perturbation moved the result, judged against the
     1 EUR threshold set in advance
  4. How the models compare to the human benchmark from Engel (2011)
"""

import pandas as pd
from scipy import stats

from stats_helpers import (fmt_p, wilson, bootstrap_mean_ci,
                           perm_test_means, holm, load_all, model_from_filename)

DATA_GLOB = "dictator_game-data/Dictator_*.csv"
THRESHOLD_EUR = 1.0
ENDOWMENT = 10


ENGEL_MEAN = 2.83            # 28.3% of a 10 EUR endowment
ENGEL_GIVES_NOTHING = 0.36   # share of humans who transfer zero
ENGEL_GIVES_HALF = 0.17      # share who transfer exactly half

CONDITIONS = ["change_output_format", "change_order_swap"]


def main():
    d = load_all(DATA_GLOB, model_from_filename)
    d = d.dropna(subset=["allocation"])

    # 1. Parser audit: how many responses were discarded, and whether that
    #    could bias the retained sample.
    print("=" * 70)
    print("1. PARSER AUDIT")
    print("=" * 70)
    audit = (d.groupby(["model", "prompt_type", "source_file"])
               .agg(valid=("attempt", "size"), attempts=("attempt", "max"))
               .groupby(level=[0, 1]).sum())
    audit["discarded"] = audit.attempts - audit.valid
    audit["discard_rate"] = audit.discarded / audit.attempts

    for (m, c), row in audit.iterrows():
        g = d[(d.model == m) & (d.prompt_type == c)]
        n_kept, n_lost = len(g), int(row.discarded)
        obs = g.allocation.mean()
        if n_lost == 0:
            print(f"  {m:6s} {c:22s} discarded 0 of {int(row.attempts)}. "
                  f"No filtering, no bias possible.")
            continue
        lo = (obs * n_kept + 0 * n_lost) / (n_kept + n_lost)
        hi = (obs * n_kept + ENDOWMENT * n_lost) / (n_kept + n_lost)
        print(f"  {m:6s} {c:22s} discarded {n_lost} of {int(row.attempts)} "
              f"({row.discard_rate:.1%}). Observed mean {obs:.2f}; worst case "
              f"if all discards were 0 EUR: {lo:.2f}, if all were 10 EUR: {hi:.2f}")

    # 2. Central result: mean, interval, and distribution shape, plus the full
    #    bar-height breakdown (share of responses at each amount from 0 to 10 EUR).
    print("\n" + "=" * 70)
    print("2. ALLOCATIONS: MEAN, INTERVAL, AND SHAPE")
    print("=" * 70)
    for (m, c), g in d.groupby(["model", "prompt_type"]):
        mean, lo, hi = bootstrap_mean_ci(g.allocation)
        p0 = (g.allocation == 0).mean()
        p5 = (g.allocation == 5).mean()
        print(f"  {m:6s} {c:22s} mean = {mean:.2f} EUR  95% CI [{lo:.2f}, {hi:.2f}]"
              f"   gives 0: {p0:5.1%}   gives 5: {p5:5.1%}   n = {len(g)}")
        probs = g.allocation.value_counts(normalize=True).reindex(range(ENDOWMENT + 1), fill_value=0.0)
        print("    " + "  ".join(f"{k}EUR:{v:.3f}" for k, v in probs.items()))
    print(f"\n  Human benchmark (Engel 2011): mean {ENGEL_MEAN} EUR, "
          f"{ENGEL_GIVES_NOTHING:.0%} give nothing, {ENGEL_GIVES_HALF:.0%} give half.")

    # 3. Effect of each perturbation: permutation test on the mean, KS test on shape.
    #    Result: Gemma's format-constraint shift (-3.40 EUR) clears the 1 EUR
    #    threshold; Llama's (+0.98 EUR) does not by mean alone, despite
    #    collapsing to a single point at 5 EUR (KS D=0.326, p<0.001).
    print("\n" + "=" * 70)
    print("3. EFFECT OF EACH PERTURBATION")
    print(f"   (substantive threshold set in advance: {THRESHOLD_EUR} EUR)")
    print("=" * 70)
    family = []
    for m, g in d.groupby("model"):
        base = g.loc[g.prompt_type == "baseline", "allocation"]
        print(f"\n  {m}  (baseline mean {base.mean():.2f} EUR)")
        for c in CONDITIONS:
            alt = g.loc[g.prompt_type == c, "allocation"]
            if len(alt) == 0:
                continue
            diff, p = perm_test_means(alt, base)
            ks, p_ks = stats.ks_2samp(base, alt)
            verdict = "EXCEEDS threshold" if abs(diff) >= THRESHOLD_EUR else "below threshold"
            print(f"    {c:22s}")
            print(f"      mean shifts {diff:+.2f} EUR   {fmt_p(p):14s} -> {verdict}")
            print(f"      distribution shape: KS D = {ks:.3f}   {fmt_p(p_ks)}")
            if abs(diff) < THRESHOLD_EUR and ks > 0.2:
                print("      NOTE: the mean barely moves but the shape changes a lot.")
            family.append((f"{m} {c}", p))
    holm(family, "Dictator Game perturbations")

    # 4. Comparison to humans: share giving 0 and giving half, against Engel's
    #    benchmark (binomial test), since Engel's distribution is bimodal.
    print("\n" + "=" * 70)
    print("4. AGAINST THE HUMAN BENCHMARK (Engel 2011)")
    print("=" * 70)
    for (m, c), g in d.groupby(["model", "prompt_type"]):
        n = len(g)
        out = [f"  {m:6s} {c:22s}"]
        for label, k, human in [("gives 0", (g.allocation == 0).sum(), ENGEL_GIVES_NOTHING),
                                ("gives 5", (g.allocation == 5).sum(), ENGEL_GIVES_HALF)]:
            p = stats.binomtest(int(k), n, human).pvalue
            lo, hi = wilson(k, n)
            out.append(f"    {label}: {k/n:6.1%} CI [{lo:5.1%}, {hi:5.1%}] "
                       f"vs human {human:.0%}   {fmt_p(p)}")
        print("\n".join(out))

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()