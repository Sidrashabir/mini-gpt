"""
Evaluate a trained model on the held-out TEST split.

Reports loss, perplexity and bits-per-character (BPC). BPC is tokenizer-independent,
so it lets you compare a char model, a BPE model and a pretrained Hugging Face model
fairly. Perplexities from different tokenizers are NOT comparable.

    python evaluate.py --run runs/shakespeare_char_small
    python evaluate.py --run runs/shakespeare_char_small --compare_hf gpt2
"""
import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from utils import load_run


@torch.no_grad()
def eval_mini_gpt(model, tok, test_text, device, stride=None):
    """Sliding-window evaluation over the whole test set."""
    ids = torch.tensor(tok.encode(test_text), dtype=torch.long)
    block = model.cfg.block_size
    stride = stride or block // 2
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(ids) - 1, stride):
        end = min(start + block, len(ids) - 1)
        x = ids[start:end].unsqueeze(0).to(device)
        y = ids[start + 1:end + 1].unsqueeze(0).to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits[0], y[0], reduction="none")
        # only score tokens not already scored by the previous window
        scored_from = 0 if start == 0 else block - stride
        total_nll += nll[scored_from:].sum().item()
        total_tokens += nll[scored_from:].numel()
        if end == len(ids) - 1:
            break
    return total_nll, total_tokens


@torch.no_grad()
def eval_hf(model_name, test_text, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    ids = torch.tensor(hf_tok.encode(test_text), dtype=torch.long)
    block = min(512, hf_model.config.max_position_embeddings)
    stride = block // 2
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(ids) - 1, stride):
        end = min(start + block, len(ids) - 1)
        x = ids[start:end].unsqueeze(0).to(device)
        y = ids[start + 1:end + 1].to(device)
        logits = hf_model(x).logits[0]
        nll = F.cross_entropy(logits, y, reduction="none")
        scored_from = 0 if start == 0 else block - stride
        total_nll += nll[scored_from:].sum().item()
        total_tokens += nll[scored_from:].numel()
        if end == len(ids) - 1:
            break
    params = sum(p.numel() for p in hf_model.parameters())
    return total_nll, total_tokens, params


def summarize(name, total_nll, total_tokens, n_chars, params):
    loss = total_nll / total_tokens
    return {"model": name, "params_M": round(params / 1e6, 2), "test_loss": round(loss, 4),
            "perplexity": round(math.exp(loss), 2),
            "bits_per_char": round(total_nll / n_chars / math.log(2), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=None, help="defaults to the data folder used in training")
    p.add_argument("--compare_hf", default=None, help="e.g. gpt2 or distilgpt2 (needs transformers)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model, tok, ckpt, device = load_run(args.run, args.device)
    data_dir = args.data or ckpt["train_cfg"]["data"]
    with open(os.path.join(data_dir, "test.txt"), encoding="utf-8") as f:
        test_text = f.read()

    results = []
    nll, n = eval_mini_gpt(model, tok, test_text, device)
    results.append(summarize(f"Mini-GPT ({tok.name})", nll, n, len(test_text), model.num_params()))

    if args.compare_hf:
        nll, n, params = eval_hf(args.compare_hf, test_text, device)
        results.append(summarize(f"{args.compare_hf} (pretrained, zero-shot)", nll, n, len(test_text), params))

    print(f"\nTest set: {len(test_text):,} characters\n")
    print(f"{'model':38s} {'params':>8s} {'loss':>8s} {'ppl':>9s} {'BPC':>7s}")
    for r in results:
        print(f"{r['model']:38s} {r['params_M']:>7.2f}M {r['test_loss']:>8.4f} "
              f"{r['perplexity']:>9.2f} {r['bits_per_char']:>7.4f}")
    print("\nLower BPC is better. Compare models using BPC, not perplexity.")

    with open(os.path.join(args.run, "eval.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
