import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import itertools
import pandas as pd
import torch
import transformers
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_TOKEN_FROM_HUGGINGFACE")

# --- MODEL SELECTION ---
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "google/gemma-3-4b-it" 

MAX_ATTEMPTS_PER_PROMPT = 3
MAX_PAIRS_TO_RUN = None
TEMPERATURES = [1.0]
OFFERS = list(range(0, 11))
CANDIDATES = ["accept", "reject", "accepts", "rejects"]
PROMPT_TYPES = ["baseline", "change_output_format", "change_order_swap"]

# ── Robustness Prompt Builder ─────────────────────────────────────────────────
def build_prompt(offer: int, proposer: str, responder: str, prompt_type: str) -> str:
    proposer_keeps = 10 - offer
    
    if prompt_type == "baseline":
        return (
            f"In the following scenario, {responder} had to decide whether to accept or reject the proposal.\n\n"
            f"Scenario: {proposer} is given $10. {proposer} will propose how to split the money between "
            f"themselves and {responder}. Then {responder} will decide whether to accept or reject "
            f"{proposer}'s proposal. If {responder} accepts, then {proposer} and {responder} get the money "
            f"as they agreed to split. If {responder} rejects, then {proposer} and {responder} both "
            f"receive nothing. {proposer} takes ${proposer_keeps} for themselves and offers {responder} ${offer}.\n\n"
            f"Answer: {responder} decides to"
        )
        
    elif prompt_type == "change_output_format":
        return (
            f"In the following scenario, {responder} had to decide whether to accept or reject the proposal.\n\n"
            f"Scenario: {proposer} is given $10. {proposer} will propose how to split the money between "
            f"themselves and {responder}. Then {responder} will decide whether to accept or reject "
            f"{proposer}'s proposal. If {responder} accepts, then {proposer} and {responder} get the money "
            f"as they agreed to split. If {responder} rejects, then {proposer} and {responder} both "
            f"receive nothing. {proposer} takes ${proposer_keeps} for themselves and offers {responder} ${offer}.\n\n"
            f"Answer with only the word 'accept' or 'reject'."
        )
        
    elif prompt_type == "change_order_swap":
        return (
            f"In the following scenario, {responder} had to decide whether to accept or reject the proposal.\n\n"
            f"Scenario: {proposer} is given $10. {proposer} will propose how to split the money between "
            f"themselves and {responder}. Then {responder} will decide whether to accept or reject "
            f"{proposer}'s proposal. If {responder} rejects, then {proposer} and {responder} both "
            f"receive nothing. If {responder} accepts, then {proposer} and {responder} get the money "
            f"as they agreed to split. {proposer} takes ${proposer_keeps} for themselves and offers {responder} ${offer}.\n\n"
            f"Answer: {responder} decides to"
        )


# ── Smart Response matching ───────────────────────────────────────
def get_valid_choice(text: str) -> str | None:
    if not text: return None
    text_snippet = str(text).lower()
    found = [c for c in CANDIDATES if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', text_snippet)]
    return found[0] if len(found) == 1 else None

# ── Auth & Model loading ──────────────────────────────────────────────────────
login(token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, use_safetensors=True
)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
pipeline.call_count = 0 

def get_terminators(pipeline):
    raw = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return [t for t in raw if t is not None]

# ── Main dataset load & Sweep ─────────────────────────────────────────────────
df_pairs = pd.read_csv("experiment_pairs.csv")
name_pairs = df_pairs.to_dict('records')[:MAX_PAIRS_TO_RUN] if MAX_PAIRS_TO_RUN else df_pairs.to_dict('records')

terminators = get_terminators(pipeline)
combinations = list(itertools.product(TEMPERATURES, OFFERS, PROMPT_TYPES))
total = len(combinations)

print(f"\nStarting text-generation sweep across {len(name_pairs)} name pairs per combination...")
print(f"Total Combinations to run: {total}")

for idx, (temp, offer, p_type) in enumerate(combinations, 1):
    output_name = model_id.split("/")[-1].replace("-", "_")
    file_path = f"Ultimatum_game_{output_name}_{p_type}_offer{offer}_temp{temp}.csv"

    if os.path.exists(file_path):
        print(f"  [{idx}/{total} SKIP – file exists] {file_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"  [{idx}/{total}] Offer = ${offer} | Temp = {temp} | Prompt = {p_type}")
    print(f"{'=' * 70}")

    responses = []

    for pair_idx, pair in tqdm(enumerate(name_pairs, 1), total=len(name_pairs), desc="Generating"):
        proposer, responder = pair["proposer_name"], pair["responder_name"]
        prompt = build_prompt(offer, proposer, responder, p_type)
        messages = [{"role": "user", "content": prompt}]

        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        valid_response_found = False
        attempts = 0
        raw_output, choice = "", None

        while not valid_response_found and attempts < MAX_ATTEMPTS_PER_PROMPT:
            attempts += 1
            try:
                outputs = pipeline(
                    formatted_prompt,
                    max_new_tokens=18,
                    eos_token_id=terminators,
                    do_sample=True, 
                    temperature=temp, 
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                raw_output = outputs[0]["generated_text"].strip()
                choice = get_valid_choice(raw_output)
                
                if choice is not None:
                    valid_response_found = True
                    
            except Exception as e:
                time.sleep(0.5)

        choice = choice if valid_response_found else "invalid"
        
        # VISUAL INSPECTION
        if pair_idx <= 2 or pair_idx % 2500 == 0:
            tqdm.write(f"\n  [INSPECTING PAIR {pair_idx}] {proposer} -> {responder} (Matched: {str(choice).upper()})")
            tqdm.write(f"  Raw Output: {raw_output!r}")
            tqdm.write(f"  {'-' * 40}")

        record = pair.copy()
        record.update({
            "temperature": temp, "offer": offer, "prompt_type": p_type,
            "proposer_keeps": 10 - offer, "response_raw": raw_output,
            "choice": choice, "attempts_needed": attempts
        })
        responses.append(record)

    pd.DataFrame(responses).to_csv(file_path, index=False)
    print(f"  → Saved dataset: {file_path}")

print("\nAll combinations completed successfully.")
