"""Creates a tiny untrained model in ./model so CI can build and smoke-test the Docker image."""
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer, save_tokenizer

if not os.path.exists("model/model.pt"):
    os.makedirs("model", exist_ok=True)
    tok = CharTokenizer("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;!?'\n")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32)
    torch.save({"model": GPT(cfg).state_dict(), "model_cfg": cfg.__dict__}, "model/model.pt")
    save_tokenizer(tok, "model/tokenizer.json")
    print("dummy model written to model/")
