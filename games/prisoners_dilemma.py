import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import pandas as pd
import torch
import transformers
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

# --- MODEL SELECTION ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "google/gemma-3-4b-it"

num_iter = 50
MAX_ATTEMPTS = num_iter * 5   # every payoff matrix reached 50/50 valid responses on the first pass
TEMPERATURES = [1.0]
CANDIDATES = ["a", "b"]
PROMPT_TYPES = ["baseline", "change_output_format", "change_order_swap"]

# ── System Parameter (Brookins & DeBacker) ────────────────────────────────────
SYSTEM_PROMPT = (
    "You are playing a single shot game as the Player 1 (described below). "
    "Please pretend that you are a human in this game."
)

# ── Mengel (2018) Payoff Matrices (a, b, c, d) ─────────────────────────────────
# Mengel's own listing repeats (250, 50, 750, 150). For this project that's a
# duplicate, not a second condition, so it's dropped: 21 unique matrices below.
PAYOFFS = [
    (400, 200, 450, 200), (400, 10, 450, 200),
    (400, 200, 800, 200), (400, 10, 800, 200),
    (400, 100, 450, 120), (400, 100, 450, 200),
    (10, 1, 90, 5), (10, 5, 90, 5),
    (150, 40, 850, 50), (150, 5, 850, 95),
    (250, 15, 750, 85), (250, 5, 750, 95),
    (250, 50, 750, 150), (250, 100, 750, 160),
    (10, 2, 110, 3), (10, 1, 110, 9),
    (150, 50, 850, 100),
    (400, 100, 600, 120), (400, 100, 600, 200),
    (400, 100, 1200, 120), (400, 100, 1200, 200)
]


# ── Robustness Prompt Builder ─────────────────────────────────────────────────
def build_prompt(a: int, b: int, c: int, d: int, prompt_type: str) -> str:
    if prompt_type == "baseline":
        return (
            "You can select one of the two choices: A or B. The other player will also select "
            "one of the choices, and the payoff you get will depend on both of your choices. "
            "Payoff is determined as follows:\n"
            f"1. If you both choose A: Both get {a} euro.\n"
            f"2. If you both choose B: Both get {d} euro.\n"
            f"3. If you choose A, the other player chooses B: You get {b} euro, the other gets {c} euro.\n"
            f"4. If you choose B, the other player chooses A: You get {c} euro, the other gets {b} euro.\n"
            "Note that you and the other player make choices simultaneously, so you cannot "
            "know her choice before you choose. "
            "Please pretend that you are a human in this single-shot game.\n"
            "Tell me which choice you would make, A or B. Do not explain your reasoning."
        )
    elif prompt_type == "change_output_format":
        return (
            "You can select one of the two choices: A or B. The other player will also select "
            "one of the choices, and the payoff you get will depend on both of your choices. "
            "Payoff is determined as follows:\n"
            f"1. If you both choose A: Both get {a} euro.\n"
            f"2. If you both choose B: Both get {d} euro.\n"
            f"3. If you choose A, the other player chooses B: You get {b} euro, the other gets {c} euro.\n"
            f"4. If you choose B, the other player chooses A: You get {c} euro, the other gets {b} euro.\n"
            "Note that you and the other player make choices simultaneously, so you cannot "
            "know her choice before you choose. "
            "Please pretend that you are a human in this single-shot game.\n"
            "Answer with only the letter 'A' or 'B'."
        )
    elif prompt_type == "change_order_swap":
        return (
            "You can select one of the two choices: A or B. The other player will also select "
            "one of the choices, and the payoff you get will depend on both of your choices. "
            "Payoff is determined as follows:\n"
            f"1. If you choose A, the other player chooses B: You get {b} euro, the other gets {c} euro.\n"
            f"2. If you choose B, the other player chooses A: You get {c} euro, the other gets {b} euro.\n"
            f"3. If you both choose A: Both get {a} euro.\n"
            f"4. If you both choose B: Both get {d} euro.\n"
            "Note that you and the other player make choices simultaneously, so you cannot "
            "know her choice before you choose. "
            "Please pretend that you are a human in this single-shot game.\n"
            "Tell me which choice you would make, A or B. Do not explain your reasoning."
        )


# ── Smart Response matching ───────────────────────────────────────────────────
def get_valid_choice(text: str) -> str | None:
    if not text: return None
    text_snippet = str(text).lower()

    # Check if 'a' or 'b' exists as standalone letters in the output
    found = [c for c in CANDIDATES if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', text_snippet)]
    return found[0] if len(found) == 1 else None


# ── Auth & Model loading ──────────────────────────────────────────────────────
login(token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True
)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


def get_terminators(pipeline):
    raw = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]
    return [t for t in raw if t is not None and t != pipeline.tokenizer.unk_token_id]


# ── Main Logic ──────────────────────────────────────────────────────────
terminators = get_terminators(pipeline)
total_combinations = len(TEMPERATURES) * len(PAYOFFS) * len(PROMPT_TYPES)
current_combo = 0

print(f"\nStarting Prisoner's Dilemma sweep across {len(PAYOFFS)} payoff matrices...")
print(f"Total Combinations to run: {total_combinations}")

for temp in TEMPERATURES:
    for (a, b, c, d) in PAYOFFS:
        for p_type in PROMPT_TYPES:
            current_combo += 1
            output_name = model_id.split("/")[-1].replace("-", "_")
            # Save files dynamically based on the exact payoff matrix
            file_path = f"prisoners_game-data/Prisoners_Dilemma_{output_name}_payoffs_{a}_{b}_{c}_{d}_{p_type}_temp{temp}.csv"

            if os.path.exists(file_path):
                print(f"  [{current_combo}/{total_combinations} SKIP – file exists] {file_path}")
                continue

            print(f"\n{'=' * 70}")
            print(
                f"  [{current_combo}/{total_combinations}] Payoffs: ({a},{b},{c},{d}) | Temp: {temp} | Prompt: {p_type}")
            print(f"{'=' * 70}")

            prompt_text = build_prompt(a, b, c, d, p_type)

            # Explicitly load the system prompt alongside the user prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text}
            ]

            responses = []
            valid_count = 0
            attempts = 0

            with tqdm(total=num_iter, desc="Generating") as pbar:
                while valid_count < num_iter and attempts < MAX_ATTEMPTS:
                    attempts += 1
                    try:
                        outputs = pipeline(
                            messages,
                            max_new_tokens=30,
                            eos_token_id=terminators,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id
                        )

                        raw_output = outputs[0]["generated_text"][-1]["content"].strip()
                        choice = get_valid_choice(raw_output)

                        pbar.write(f"[attempt {attempts}] raw: {raw_output!r} -> choice: {choice}")


                        if choice is not None:
                            valid_count += 1
                            pbar.update(1)
                            responses.append({
                                "iteration": valid_count,
                                "attempt": attempts,
                                "temperature": temp,
                                "prompt_type": p_type,
                                "payoff_a": a, "payoff_b": b, "payoff_c": c, "payoff_d": d,
                                "response_raw": raw_output,
                                "choice": choice,
                            })
                    except Exception as e:
                        print(f"    Error: {e}")
                        time.sleep(0.1)

            if valid_count > 0:
                pd.DataFrame(responses).to_csv(file_path, index=False)
                print(f"  → Saved {valid_count} valid iterations: {file_path}")

print("\nAll combinations completed successfully.")
