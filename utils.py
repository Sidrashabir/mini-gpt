"""Shared helpers for loading a trained run."""
import os

import torch

from model import GPT, GPTConfig
from tokenizer import load_tokenizer


def pick_device(choice="auto"):
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_run(run_dir, device="auto"):
    """Load model + tokenizer from a run folder created by train.py."""
    device = pick_device(device)
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tok = load_tokenizer(os.path.join(run_dir, "tokenizer.json"))
    return model, tok, ckpt, device
