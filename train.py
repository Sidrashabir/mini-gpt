"""
Train Mini-GPT.

Examples:
    python train.py --preset tiny                     # CPU-friendly, minutes
    python train.py --preset small                    # free Colab/Kaggle GPU, ~10M params
    python train.py --preset small --n_layer 4        # override any setting (ablations!)
    python train.py --data data/shakespeare_bpe --preset small --out_dir runs/bpe
"""
import argparse
import csv
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig
from tokenizer import load_tokenizer

PRESETS = {
    # ~0.8M params: runs on a laptop CPU in a few minutes, good for checking everything works
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32,
                 max_iters=2000, eval_interval=250, lr=1e-3, dropout=0.1, warmup_iters=100),
    # ~10.7M params: the main model, ~15-25 min on a T4 GPU
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64,
                  max_iters=5000, eval_interval=500, lr=1e-3, dropout=0.2, warmup_iters=200),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/shakespeare_char")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--preset", choices=PRESETS.keys(), default="tiny")
    # Anything below overrides the preset when given
    for name, typ in [("n_layer", int), ("n_head", int), ("n_embd", int), ("block_size", int),
                      ("batch_size", int), ("max_iters", int), ("eval_interval", int),
                      ("lr", float), ("dropout", float), ("warmup_iters", int)]:
        p.add_argument(f"--{name}", type=typ, default=None)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile for extra GPU speed")
    args = p.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in cfg:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    for k, v in vars(args).items():
        if k not in cfg:
            cfg[k] = v
    if cfg["out_dir"] is None:
        cfg["out_dir"] = os.path.join("runs", f"{os.path.basename(cfg['data'])}_{cfg['preset']}")
    return cfg


def pick_device(choice):
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    cfg = get_args()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device(cfg["device"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    with open(os.path.join(cfg["data"], "meta.json")) as f:
        meta = json.load(f)
    tok = load_tokenizer(os.path.join(cfg["data"], "tokenizer.json"))
    # keep the tokenizer next to the checkpoint so a run folder is self-contained
    shutil.copy(os.path.join(cfg["data"], "tokenizer.json"), os.path.join(cfg["out_dir"], "tokenizer.json"))
    data = {s: np.memmap(os.path.join(cfg["data"], f"{s}.bin"), dtype=meta["dtype"], mode="r")
            for s in ("train", "val")}

    def get_batch(split):
        d = data[split]
        ix = np.random.randint(0, len(d) - cfg["block_size"] - 1, cfg["batch_size"])
        x = torch.from_numpy(np.stack([d[i: i + cfg["block_size"]].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([d[i + 1: i + 1 + cfg["block_size"]].astype(np.int64) for i in ix]))
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

    model_cfg = GPTConfig(vocab_size=meta["vocab_size"], block_size=cfg["block_size"],
                          n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                          dropout=cfg["dropout"])
    model = GPT(model_cfg).to(device)
    print(f"Device: {device} | parameters: {model.num_params() / 1e6:.2f}M")

    optimizer = model.configure_optimizer(cfg["lr"], cfg["weight_decay"])

    # Mixed precision on GPU: bfloat16 if supported, else float16 with a grad scaler
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    train_model = torch.compile(model) if cfg["compile"] else model

    def lr_at(it):
        """Linear warmup, then cosine decay down to min_lr."""
        max_lr, min_lr = cfg["lr"], cfg["lr"] * cfg["min_lr_ratio"]
        if it < cfg["warmup_iters"]:
            return max_lr * (it + 1) / cfg["warmup_iters"]
        progress = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
        return min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (max_lr - min_lr)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(cfg["eval_iters"])
            for k in range(cfg["eval_iters"]):
                x, y = get_batch(split)
                with ctx:
                    _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def sample(n_tokens=200):
        model.eval()
        start = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
        text = tok.decode(model.generate(start, n_tokens, temperature=0.8, top_k=50)[0].tolist())
        model.train()
        return text

    log_path = os.path.join(cfg["out_dir"], "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    best_val = float("inf")
    t0 = time.time()
    for it in range(cfg["max_iters"] + 1):
        lr = lr_at(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"]:
            losses = estimate_loss()
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {elapsed:.0f}s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                        f"{lr:.6f}", f"{elapsed:.1f}"])
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "model_cfg": asdict(model_cfg),
                            "train_cfg": cfg, "iter": it, "val_loss": best_val},
                           os.path.join(cfg["out_dir"], "ckpt.pt"))
            if it > 0:
                print("--- sample ---\n" + sample() + "\n--------------")

        if it == cfg["max_iters"]:
            break

        x, y = get_batch("train")
        with ctx:
            _, loss = train_model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

    with open(os.path.join(cfg["out_dir"], "config.json"), "w") as f:
        json.dump({"train_cfg": cfg, "model_cfg": asdict(model_cfg),
                   "best_val_loss": best_val, "params": model.num_params()}, f, indent=2)
    print(f"Done. Best val loss {best_val:.4f}. Checkpoint: {cfg['out_dir']}/ckpt.pt")


if __name__ == "__main__":
    main()
