"""
Inference engine used by both the web app and the REST API.

Loads a model folder (created by export.py) or a training run folder, and offers:
  - stream():       token-by-token generation with a KV-cache
  - generate():     full text in one call
  - next_token():   the model's top-k candidates for the next token, with probabilities
  - attention():    attention weights of one layer for a prompt

Model location is resolved in this order:
  1. the path you pass in
  2. env var MODEL_DIR (default "model")
  3. env var MODEL_REPO: a Hugging Face Hub repo id to download the model from
"""
import json
import os
import threading
import time

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import load_tokenizer


def resolve_model_dir(path=None):
    path = path or os.environ.get("MODEL_DIR", "model")
    if os.path.exists(os.path.join(path, "model.pt")) or os.path.exists(os.path.join(path, "ckpt.pt")):
        return path
    repo = os.environ.get("MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN"))
    raise FileNotFoundError(
        f"No model found in '{path}'. Run `python export.py --run runs/<your_run>` first, "
        "or set MODEL_REPO to a Hugging Face model repo."
    )


class Engine:
    def __init__(self, model_dir=None, device="cpu", threads=None):
        model_dir = resolve_model_dir(model_dir)
        if threads:
            torch.set_num_threads(threads)
        self.device = device

        if os.path.exists(os.path.join(model_dir, "model.pt")):
            ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location=device, weights_only=False)
            card_path = os.path.join(model_dir, "model_card.json")
            self.card = json.load(open(card_path)) if os.path.exists(card_path) else {}
        else:  # a raw training run folder
            ckpt = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device, weights_only=False)
            self.card = {"val_loss": ckpt.get("val_loss")}

        self.model = GPT(GPTConfig(**ckpt["model_cfg"]))
        state = {k: v.float() for k, v in ckpt["model"].items()}  # handles fp16 exports
        self.model.load_state_dict(state)
        self.model.to(device).eval()
        self.tok = load_tokenizer(os.path.join(model_dir, "tokenizer.json"))
        self.cfg = self.model.cfg
        self._attn_lock = threading.Lock()  # attention capture mutates model state

    # ---------- info ----------
    def info(self):
        return {
            "name": self.card.get("name", "Mini-GPT"),
            "parameters": self.model.num_params(),
            "n_layer": self.cfg.n_layer,
            "n_head": self.cfg.n_head,
            "n_embd": self.cfg.n_embd,
            "context_length": self.cfg.block_size,
            "vocab_size": self.cfg.vocab_size,
            "tokenizer": self.tok.name,
            "dataset": self.card.get("dataset"),
            "val_loss": self.card.get("val_loss"),
            "eval": self.card.get("eval"),
            "trained_iters": self.card.get("iters"),
        }

    # ---------- helpers ----------
    def _encode_prompt(self, prompt):
        ids = self.tok.encode(prompt or "\n") or self.tok.encode("\n")
        return ids[-self.cfg.block_size:]

    @staticmethod
    def _filter(logits, temperature, top_k, top_p):
        logits = logits / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
        return logits

    # ---------- generation ----------
    @torch.no_grad()
    def stream(self, prompt, max_new_tokens=300, temperature=0.8, top_k=50, top_p=None, seed=None,
               stop=None):
        """Yields decoded text pieces as they are generated."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed if seed is not None else int(time.time() * 1000) % (2**31))

        ids = self._encode_prompt(prompt)
        context = list(ids)
        logits, kvs = self.model.forward_cached(torch.tensor([ids], device=self.device))
        produced = ""
        for _ in range(max_new_tokens):
            probs = F.softmax(self._filter(logits, temperature, top_k, top_p), dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=gen).item())
            context.append(next_id)
            piece = self.tok.decode([next_id])
            produced += piece
            yield piece
            if stop and stop in produced:
                return

            if kvs[0][0].size(2) >= self.cfg.block_size:
                # Cache full: slide the window, keeping the most recent half as context.
                keep = context[-(self.cfg.block_size // 2):]
                logits, kvs = self.model.forward_cached(torch.tensor([keep], device=self.device))
            else:
                logits, kvs = self.model.forward_cached(
                    torch.tensor([[next_id]], device=self.device), past_kvs=kvs)

    def generate(self, prompt, **kwargs):
        start = time.perf_counter()
        pieces = list(self.stream(prompt, **kwargs))
        elapsed = time.perf_counter() - start
        return {
            "prompt": prompt,
            "completion": "".join(pieces),
            "tokens_generated": len(pieces),
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(len(pieces) / elapsed, 1) if elapsed > 0 else None,
        }

    # ---------- explainability ----------
    @torch.no_grad()
    def next_token(self, prompt, k=10, temperature=1.0):
        ids = self._encode_prompt(prompt)
        logits, _ = self.model.forward_cached(torch.tensor([ids], device=self.device))
        probs = F.softmax(logits[0] / max(temperature, 1e-5), dim=-1)
        top = torch.topk(probs, min(k, probs.numel()))
        entropy = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
        return {
            "candidates": [{"token": self.tok.decode([int(i)]), "probability": round(float(p), 5)}
                           for p, i in zip(top.values, top.indices)],
            "entropy_bits": round(entropy, 3),
        }

    @torch.no_grad()
    def attention(self, prompt, layer=0, max_tokens=48):
        ids = self._encode_prompt(prompt)[:max_tokens]
        layer = max(0, min(layer, self.cfg.n_layer - 1))
        with self._attn_lock:
            self.model.set_store_attention(True)
            try:
                self.model(torch.tensor([ids], device=self.device))
                att = self.model.blocks[layer].attn.last_attn[0]  # (heads, T, T)
            finally:
                self.model.set_store_attention(False)
        tokens = [self.tok.decode([i]) for i in ids]
        return {"layer": layer, "tokens": tokens, "weights": att.tolist()}
