# CSS-project-Robustness-Audit
GitHub repository for the project of Computational Social Science 2025/2026 course at the University of Trento


For running the python scripts I have used Python 3.10.12 with libraries as in the requirements.txt
You will also need an access token from [huggingface.co](https://huggingface.co) and permission to access the models from: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct & https://huggingface.co/google/gemma-3-4b-it

## For the Dictator Game:

at line 13 in the script `dictator_game.py` You will need to replace `YOUR_TOKEN_FROM_HUGGINGFACE` with Your access token from huggingface. Then, the script can be run with one model at a time. To run the second model: `gemma-3-4b-it`, You need to comment line 16 and uncomment line 17. For generating the visualization run `plot_dictator.py`


## For the Prisoner's Dilemma:

at line 15 in the script `prisoners_dilemma.py` You will need to replace `YOUR_TOKEN_FROM_HUGGINGFACE` with Your access token from huggingface. Then, the script can be run with one model at a time. To run the second model: `gemma-3-4b-it`, You need to comment line 18 and uncomment line 19. For generating the visualization run `plot_prisoner.py`


## For the Ultimatum Game:

first, `create_dataset.py` needs to be run to create `experiment_pairs.csv`, to get the name pairs that Aher et al. used in their 2023 research paper. Then both `ultimatum_game-logprobs.py` and `ultimatum_game-text_generation.py` can be run. 

For `ultimatum_game-logprobs.py`: at line 15 in the script You will need to replace `YOUR_TOKEN_FROM_HUGGINGFACE` with Your access token from huggingface. Then, the script can be run with one model at a time. To run the second model: `gemma-3-4b-it`, You need to comment line 17 and uncomment line 18.

For `ultimatum_game-text_generation.py`: at line 15 in the script You will need to replace `YOUR_TOKEN_FROM_HUGGINGFACE` with Your access token from huggingface. Then, the script can be run with one model at a time. To run the second model: `gemma-3-4b-it`, You need to comment line 18 and uncomment line 19.

Then for generating the heatmaps, `plot_heatmaps.py` needs to be run.
For additional information about probabilities, `probability_extraction_ult-game.py` can be run, this was used for getting more accurate numbers while writing the project's text.
