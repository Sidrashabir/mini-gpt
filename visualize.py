"""
Plots for your README:
  1. Training curves (train vs val loss, learning rate)
  2. Attention heatmaps for every head in a chosen layer

    python visualize.py --run runs/shakespeare_char_small --prompt "To be, or not to be" --layer 0
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import load_run


def plot_training(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "log.csv"))))
    it = [int(r["iter"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(it, [float(r["train_loss"]) for r in rows], label="train")
    ax1.plot(it, [float(r["val_loss"]) for r in rows], label="val")
    ax1.set_xlabel("iteration"); ax1.set_ylabel("cross-entropy loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(it, [float(r["lr"]) for r in rows], color="tab:green")
    ax2.set_xlabel("iteration"); ax2.set_title("Learning rate (warmup + cosine)"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(run_dir, "training_curves.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")


@torch.no_grad()
def plot_attention(model, tok, device, prompt, layer, run_dir):
    ids = tok.encode(prompt)[: model.cfg.block_size]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.set_store_attention(True)
    model(x)
    model.set_store_attention(False)

    att = model.blocks[layer].attn.last_attn[0]  # (n_head, T, T)
    labels = [repr(tok.decode([i]))[1:-1].replace(" ", "␣") for i in ids]  # show spaces
    n_head = att.size(0)
    cols = min(3, n_head)
    rows = (n_head + cols - 1) // cols
    size = max(3.5, len(ids) * 0.28)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows), squeeze=False)
    for h in range(rows * cols):
        ax = axes[h // cols][h % cols]
        if h >= n_head:
            ax.axis("off"); continue
        ax.imshow(att[h].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"layer {layer} · head {h}")
        ax.set_xticks(range(len(ids))); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(ids))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("attends to (key)"); ax.set_ylabel("query token")
    fig.tight_layout()
    path = os.path.join(run_dir, f"attention_layer{layer}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    plot_training(args.run)
    model, tok, _, device = load_run(args.run, args.device)
    for layer in sorted({args.layer, model.cfg.n_layer - 1}):
        plot_attention(model, tok, device, args.prompt, layer, args.run)


if __name__ == "__main__":
    main()
