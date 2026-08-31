import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import itertools
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

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

# ── Tokenizer Diagnostic Check ────────────────────────────────────────────────
# Scoring only " accept"/" reject" (leading space, mid-sentence spelling) undercounts:
# after a chat template the model starts a new word with no leading space. Including
# unspaced and capitalised variants recovers a validity rate of 97.8-99.9% for Llama
# across all three prompts, and 100% for Gemma under output-format change. Gemma's
# validity under baseline and order-swap stays under 0.01% even with the correction
# (see ulti-logprobs-analysis.py) and those two cells are excluded from analysis.
def get_target_token_ids(word):
    ids = set()
    for variant in (word, word.capitalize(), " " + word, " " + word.capitalize()):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if len(enc) == 1:                      # only single-token spellings
            ids.add(enc[0])
    if not ids:
        raise ValueError(f"'{word}' is not a single token for this tokenizer")
    return sorted(ids)

accept_ids = get_target_token_ids("accept")
reject_ids = get_target_token_ids("reject")
print("accept:", [(i, repr(tokenizer.decode([i]))) for i in accept_ids])
print("reject:", [(i, repr(tokenizer.decode([i]))) for i in reject_ids])

# ── Dataset Loading ───────────────────────────────────────────────────────────
df_pairs = pd.read_csv("ultimatum_game-data/experiment_pairs.csv")
name_pairs = df_pairs.to_dict('records')

combinations = list(itertools.product(OFFERS, PROMPT_TYPES))
total = len(combinations)

print(f"Running LOGPROBS sweep over {len(name_pairs)} name pairs...")

for idx, (offer, p_type) in enumerate(combinations, 1):
    output_name = model_id.split("/")[-1].replace("-", "_")
    file_path = f"ultimatum_game-data/Ultimatum_game_LOGPROBS_{output_name}_{p_type}_offer{offer}.csv"

    if os.path.exists(file_path):
        print(f"  [{idx}/{total} SKIP – file exists] {file_path}")
        continue

    print(f"\n{'=' * 70}")
    print(f"  [{idx}/{total}] Offer = ${offer} | Prompt = {p_type}")
    print(f"{'=' * 70}")

    responses = []

    for pair_idx, pair in enumerate(tqdm(name_pairs, desc="Evaluating Logits")):
        proposer, responder = pair["proposer_name"], pair["responder_name"]
        prompt_text = build_prompt(offer, proposer, responder, p_type)
        
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        # Full vocabulary probabilities
        vocab_probs = torch.softmax(next_token_logits, dim=-1)
        z_mass = (vocab_probs[accept_ids].sum() + vocab_probs[reject_ids].sum()).item()

        # Diagnostic print for the first pair of each offer/prompt combination
        if pair_idx == 0:
            top5_indices = torch.topk(vocab_probs, 5).indices.tolist()
            top5_tokens = [tokenizer.decode([i]) for i in top5_indices]
            top5_probs = [round(vocab_probs[i].item(), 4) for i in top5_indices]
            print(f"\n[Diagnostic Pair 1] Z coverage (accept+reject): {z_mass:.4f}")
            print(f"[Diagnostic Pair 1] Top 5 next tokens: {list(zip(top5_tokens, top5_probs))}\n")

        logit_accept = next_token_logits[accept_ids].max().item()
        logit_reject = next_token_logits[reject_ids].max().item()

        # Binary Softmax normalized between accept vs reject
        p_a = vocab_probs[accept_ids].sum().item()
        p_r = vocab_probs[reject_ids].sum().item()
        Z = p_a + p_r                      # validity rate: how much mass is on a real decision
        p_accept = p_a / Z
        p_reject = p_r / Z
        
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
            "z_coverage": z_mass,
            "choice": choice
        })
        responses.append(record)

    pd.DataFrame(responses).to_csv(file_path, index=False)
    print(f"  → Saved dataset: {file_path}")

print("\nLogprobs sweep completed successfully.")