import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import itertools
import pandas as pd
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_TOKEN_FROM_HUGGINGFACE")

# Model Selection
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "google/gemma-3-4b-it"

OFFERS = list(range(0, 11))
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
# ── Model & Tokenizer Setup ──────────────────────────────────────────────────
login(token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True
)
model.eval()

# Helper to locate the exact target token IDs for 'accept' and 'reject'
def get_target_token_id(word: str):
    # Try tokenizing with a leading space (standard after a prompt) and without
    tok_space = tokenizer.encode(" " + word, add_special_tokens=False)
    tok_no_space = tokenizer.encode(word, add_special_tokens=False)
    return tok_space[0] if len(tok_space) > 0 else tok_no_space[0]

accept_id = get_target_token_id("accept")
reject_id = get_target_token_id("reject")

print(f"Target Token IDs -> 'accept': {accept_id}, 'reject': {reject_id}")

# ── Dataset Loading ───────────────────────────────────────────────────────────
df_pairs = pd.read_csv("experiment_pairs.csv")
name_pairs = df_pairs.to_dict('records')

combinations = list(itertools.product(OFFERS, PROMPT_TYPES))
total = len(combinations)

print(f"\nRunning high-speed LOGPROBS sweep over {len(name_pairs)} name pairs...")

for idx, (offer, p_type) in enumerate(combinations, 1):
    output_name = model_id.split("/")[-1].replace("-", "_")
    file_path = f"Ultimatum_game_LOGPROBS_{output_name}_{p_type}_offer{offer}.csv"

    if os.path.exists(file_path):
        print(f"  [{idx}/{total} SKIP – file exists] {file_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"  [{idx}/{total}] Offer = ${offer} | Prompt = {p_type}")
    print(f"{'=' * 70}")

    responses = []

    for pair in tqdm(name_pairs, desc="Evaluating Logits"):
        proposer, responder = pair["proposer_name"], pair["responder_name"]
        prompt_text = build_prompt(offer, proposer, responder, p_type)
        
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Logits for the next predicted token position
            next_token_logits = outputs.logits[0, -1, :]

        logit_accept = next_token_logits[accept_id].item()
        logit_reject = next_token_logits[reject_id].item()

        # Binary Softmax over accept vs reject
        probs = torch.softmax(torch.tensor([logit_accept, logit_reject]), dim=0)
        p_accept = probs[0].item()
        p_reject = probs[1].item()
        
        # Categorical decision based on highest probability
        choice = "accept" if p_accept >= 0.5 else "reject"

        record = pair.copy()
        record.update({
            "offer": offer, 
            "prompt_type": p_type,
            "proposer_keeps": 10 - offer, 
            "logit_accept": logit_accept,
            "logit_reject": logit_reject,
            "p(accept)": p_accept,
            "p(reject)": p_reject,
            "choice": choice
        })
        responses.append(record)

    pd.DataFrame(responses).to_csv(file_path, index=False)
    print(f"  → Saved dataset: {file_path}")

print("\nLogprobs sweep completed successfully.")