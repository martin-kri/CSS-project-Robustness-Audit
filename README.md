# CSS-project-Robustness-Audit
GitHub repository for the project of Computational Social Science 2025/2026 course at the University of Trento

## Structure

- `games/` — the four generation sweeps plus the Ultimatum name-pair builder
- `analysis/` — the statistics reported in the paper, plus the shared helper module
- `plotting/` — builds the four figures into `plots/`
- `paper-checks/` — not part of the pipeline; each script sources one specific claim in the text
- `dictator_game-data/`, `prisoners_game-data/`, `ultimatum_game-data/`, `plots/` — shared data and output folders, read/written by scripts in every folder above

**Run every script from this top-level folder** (e.g. `python games/dictator_game.py`, not `cd games && python dictator_game.py`). All file paths inside the scripts are relative to this folder, not to the script's own location, since the data and output folders are shared across `games/`, `analysis/`, `plotting/`, and `paper-checks/`.

## Setup

Python 3.10 or later, with the packages listed in `requirements.txt`:
```
pip install -r requirements.txt
```
`torch` is pinned to the CUDA 12.4 build used on the server these sweeps ran on. If your machine has a different CUDA version (or no GPU), install `torch` separately for your setup before installing the rest.

You will also need a Hugging Face access token with permission for the two gated models:
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/google/gemma-3-4b-it

Copy `.env.example` to `.env` and fill in your token:
```
HF_TOKEN=your_token_here
```

All four generation sweeps (Dictator Game, Prisoner's Dilemma, Ultimatum Game text and log-probability) run one model at a time: near the top of each script, comment out the `model_id` line for Llama and uncomment the one for Gemma, then run it again. Every generation script skips a condition if its output CSV already exists, so re-running only fills in what's missing. These sweeps ran on GPU (a single V100 for everything except the Ultimatum log-probability sweep, which used an L40S); expect the full set to take hours, not minutes.

## Pipeline (run these to reproduce the results)

### Dictator Game
1. `games/dictator_game.py` — 3 prompt conditions x 500 iterations, per model, into `dictator_game-data/`
2. `analysis/analysis_dictator.py` — the statistics reported in Results: parser audit, means and distribution shape, perturbation tests, comparison to Engel (2011)
3. `plotting/plot_dictator.py` — builds `plots/dictator_allocations.pdf` (Figure 1)

### Prisoner's Dilemma
1. `games/prisoners_dilemma.py` — 3 prompt conditions x 21 payoff matrices x 50 iterations, per model, into `prisoners_game-data/`
2. `analysis/analysis_prisoner.py` — cooperation rates, perturbation effects, and the payoff-structure regression against Mengel (2018) / Brookins & DeBacker (2024)
3. `plotting/plot_prisoner.py` — builds `plots/Prisoners_Dilemma_Strategies.pdf` (Figure 2)

### Ultimatum Game
1. `games/create_dataset.py` — builds `ultimatum_game-data/experiment_pairs.csv`, 10,000 name pairs from the surname list used in Aher et al. (2023), fixed seed
2. `games/ultimatum_game-text_generation.py` and `games/ultimatum_game-logprobs.py` — both read `ultimatum_game-data/experiment_pairs.csv`; run each once per model into `ultimatum_game-data/`
3. `analysis/analysis_ultimatum.py` — the two-instrument comparison (H2), offer-sensitivity, and name-sensitivity checks
4. `plotting/plot_heatmaps.py` — builds `plots/Ultimatum_Game_Heatmap_LOGPROBS.pdf` and `plots/Ultimatum_Game_Heatmap_TEXT.pdf` (Figures 3-4)

`analysis/stats_helpers.py` holds the statistical functions shared by the three `analysis_*.py` scripts and is not run on its own.

## paper-checks/ — used only to pull specific numbers cited in the text

These aren't part of the pipeline above and don't need to be run to reproduce the headline results. Each one was used to check or source a specific claim made in the paper.

- `Brookins-DeBacker_gpt-mean.py` — computes the 4.83€ GPT-3.5 mean allocation from `gpt_dictator_results.csv`, used as the reference line in `plotting/plot_dictator.py`.
- `discards-prisoners_dillema.py` — confirms that every Prisoner's Dilemma payoff matrix reached its 50-response target on the first pass, with no resampling (Analytical Approach).
- `ulti_gemma-empty.py` — checks Gemma's text-generation responses under float16 for empty or malformed output (Analytical Approach).
- `ulti-logprobs-analysis.py` — prints the mean validity rate (z_coverage) per model and condition; the basis for the log-probability validity numbers reported in Analytical Approach and for excluding Gemma's baseline and order-swap log-probability cells as instrument failure.

## Data & output folders

- `dictator_game-data/`, `prisoners_game-data/`, `ultimatum_game-data/` — raw response CSVs, one file per model/condition (Prisoner's Dilemma: per payoff matrix; Ultimatum: per offer)
- `plots/` — the four PDF figures embedded in the paper; the three `plotting/` scripts save directly into this folder (created automatically if it doesn't exist)
