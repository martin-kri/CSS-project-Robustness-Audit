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

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

# --- MODEL SELECTION ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "google/gemma-3-4b-it"

num_iter = 500
MAX_ATTEMPTS = num_iter * 5   # highest attempt count reached in practice: 514/2500 (Llama, order swap)
TEMPERATURES = [1.0]

# ── Robustness Prompts ────────────────────────────────────────────────────────
PROMPTS = {
    "baseline": (
        "This task is about dividing money between yourself and another person to whom "
        "you are randomly matched. You do not know this other person, and you will not "
        "knowingly meet him/her.\n\n"
        "You have been randomly assigned the role of the \"allocator.\" The other person is "
        "in the role of the \"recipient.\"\n\n"
        "You are endowed with 10€, the recipient is endowed with 0€.\n\n"
        "You can decide how much of your 10€ endowment to transfer to the recipient. "
        "You can choose any amount between 0€ and 10€. The recipient receives the "
        "amount that you decide to transfer to him/her; you receive the amount that you "
        "decide not to transfer and thus to keep.\n\n"
        "How much of your 10€ endowment do you want to transfer to the recipient?\n\n"
        "Just tell me the allocation, not your reasoning."
    ),
    "change_output_format": (
        "This task is about dividing money between yourself and another person to whom "
        "you are randomly matched. You do not know this other person, and you will not "
        "knowingly meet him/her.\n\n"
        "You have been randomly assigned the role of the \"allocator.\" The other person is "
        "in the role of the \"recipient.\"\n\n"

        "You are endowed with 10€, the recipient is endowed with 0€.\n\n"
        "You can decide how much of your 10€ endowment to transfer to the recipient. "
        "You can choose any amount between 0€ and 10€. The recipient receives the "
        "amount that you decide to transfer to him/her; you receive the amount that you "
        "decide not to transfer and thus to keep.\n\n"
        "How much of your 10€ endowment do you want to transfer to the recipient?\n\n"
        "Answer with the number only."
    ),
    "change_order_swap": (
        "This task is about dividing money between yourself and another person to whom "
        "you are randomly matched. You do not know this other person, and you will not "
        "knowingly meet him/her.\n\n"
        "The other person is assigned the role of the \"recipient\" and you have been "
        "assigned the role of the \"allocator.\"\n\n"
        "You are endowed with 10€, the recipient is endowed with 0€.\n\n"
        "You can decide how much of your 10€ endowment to transfer to the recipient. "
        "You can choose any amount between 0€ and 10€. The recipient receives the "
        "amount that you decide to transfer to him/her; you receive the amount that you "
        "decide not to transfer and thus to keep.\n\n"
        "How much of your 10€ endowment do you want to transfer to the recipient?\n\n"
        "Just tell me the allocation, not your reasoning."
    )
}
# ── Number matching ───────────────────────────────────────────────
NUMBER_MAP = {
    "0": 0, "zero": 0, "1": 1, "one": 1, "2": 2, "two": 2, "3": 3, "three": 3,
    "4": 4, "four": 4, "5": 5, "five": 5, "6": 6, "six": 6, "7": 7, "seven": 7,
    "8": 8, "eight": 8, "9": 9, "nine": 9, "10": 10, "ten": 10,
}
SORTED_KEYS = sorted(NUMBER_MAP.keys(), key=len, reverse=True)

def get_valid_allocation(text):
    if not text: return None
    text_lower = str(text).lower()
    found_values = set()
    for token in SORTED_KEYS:
        pattern = r'(?<![a-z0-9])' + re.escape(token) + r'(?![a-z0-9])'
        if re.search(pattern, text_lower):
            found_values.add(NUMBER_MAP[token])
    if len(found_values) == 1:
        return found_values.pop()
    return None

# ── Auth & Model loading ────────────────────────────────────────────────────
login(token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True
)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_terminators(pipeline):
    raw = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),    # For Llama
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")  # For Gemma
    ]
    return [t for t in raw if t is not None and t != pipeline.tokenizer.unk_token_id]

# ── Main Logic ──────────────────────────────────────────────────────────
def run_temperature(temp, prompt_name, prompt_text, terminators):
    output_name = model_id.split("/")[-1].replace("-", "_")
    file_path = f"dictator_game-data/Dictator_game_{output_name}_{prompt_name}_temp{temp}.csv"

    if os.path.exists(file_path):
        print(f"\n  [SKIP – file exists] {file_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"  Dictator Game  |  Prompt: {prompt_name}  |  temp={temp}")
    print(f"{'=' * 70}")

    messages = [{"role": "user", "content": prompt_text}]
    responses = []
    valid_count = 0
    attempts = 0

    while valid_count < num_iter and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            outputs = pipeline(
                messages,
                max_new_tokens=40,
                eos_token_id=terminators,
                do_sample=True, temperature=temp, top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            raw_output = outputs[0]["generated_text"][-1]["content"].strip()
            allocation = get_valid_allocation(raw_output)

            if allocation is not None:
                valid_count += 1
                responses.append({
                    "iteration": valid_count,
                    "attempt": attempts,
                    "temperature": temp,
                    "prompt_type": prompt_name,
                    "response": raw_output,
                    "allocation": allocation,
                })
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.1)

    if valid_count > 0:
        pd.DataFrame(responses).to_csv(file_path, index=False)
        print(f"  Saved {valid_count} valid iterations: {file_path}")

# Run
terminators = get_terminators(pipeline)
for temp in TEMPERATURES:
    for prompt_name, prompt_text in PROMPTS.items():
        run_temperature(temp, prompt_name, prompt_text, terminators)
print("\nAll done!")
