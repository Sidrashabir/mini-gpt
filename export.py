"""
Turn a training run into a small, deployable model folder.

    python export.py --run runs/shakespeare_char_small
    python export.py --run runs/shakespeare_char_small --half            # half the file size
    python export.py --run runs/shakespeare_char_small --push_to_hub yourname/mini-gpt-shakespeare

Output (default ./model):
    model.pt          weights + architecture config (no training state)
    tokenizer.json
    model_card.json   metrics and training details shown in the app's "About" tab
    README.md         Hugging Face model card (used if you push to the Hub)
"""
import argparse
import json
import os
import shutil

import torch


def pretty_dataset(path):
    name = os.path.basename(path.rstrip("/"))
    for suffix in ("_char", "_bpe"):
        name = name.removesuffix(suffix)
    return "Tiny Shakespeare" if name == "shakespeare" else name.replace("_", " ").title()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default="model")
    p.add_argument("--name", default="Mini-GPT")
    p.add_argument("--examples", nargs="*", default=None,
                   help='example prompts shown in the app, e.g. --examples "ROMEO:" "Once upon a time"')
    p.add_argument("--dataset_name", default=None, help='display name, e.g. "Tiny Shakespeare"')
    p.add_argument("--half", action="store_true", help="store weights in float16")
    p.add_argument("--push_to_hub", default=None, help="Hugging Face model repo id, e.g. user/mini-gpt")
    args = p.parse_args()

    ckpt = torch.load(os.path.join(args.run, "ckpt.pt"), map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if args.half:
        state = {k: v.half() for k, v in state.items()}
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": state, "model_cfg": ckpt["model_cfg"]}, os.path.join(args.out, "model.pt"))
    shutil.copy(os.path.join(args.run, "tokenizer.json"), os.path.join(args.out, "tokenizer.json"))

    eval_path = os.path.join(args.run, "eval.json")
    tcfg = ckpt["train_cfg"]
    n_params = sum(v.numel() for k, v in state.items() if k != "pos_emb.weight" and k != "lm_head.weight")
    card = {
        "name": args.name,
        "dataset": args.dataset_name or pretty_dataset(tcfg["data"]),
        "parameters": n_params,
        "iters": ckpt["iter"],
        "val_loss": round(ckpt["val_loss"], 4),
        "example_prompts": args.examples,
        "eval": json.load(open(eval_path)) if os.path.exists(eval_path) else None,
        "model_cfg": ckpt["model_cfg"],
        "training": {k: tcfg[k] for k in ("batch_size", "lr", "dropout", "max_iters",
                                          "weight_decay", "warmup_iters") if k in tcfg},
    }
    with open(os.path.join(args.out, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)

    m = ckpt["model_cfg"]
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(f"""---
license: mit
tags: [pytorch, gpt, transformer, from-scratch, text-generation]
---
# {args.name}

A {n_params / 1e6:.1f}M-parameter decoder-only transformer written from scratch in PyTorch
and trained on `{card['dataset']}`.

| Setting | Value |
|---|---|
| Layers / heads / embedding | {m['n_layer']} / {m['n_head']} / {m['n_embd']} |
| Context length | {m['block_size']} tokens |
| Vocabulary | {m['vocab_size']} |
| Best validation loss | {card['val_loss']} |

Load it with the `inference.Engine` class from the project repository.
This is an educational model: it imitates the style of its training text and does not
produce factual or reliable content.
""")

    size_mb = os.path.getsize(os.path.join(args.out, "model.pt")) / 1e6
    print(f"Exported to {args.out}/ ({size_mb:.1f} MB)")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push_to_hub, repo_type="model")
        print(f"Pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
