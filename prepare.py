"""
Download (or load) a text corpus, tokenize it, and split it into train/val/test.

Examples:
    python prepare.py                                   # Tiny Shakespeare, char-level
    python prepare.py --tokenizer bpe                   # Tiny Shakespeare, GPT-2 BPE
    python prepare.py --input my_corpus.txt --name poems  # your own text file
"""
import argparse
import json
import os
import urllib.request

import numpy as np

from tokenizer import build_tokenizer, save_tokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="path to a .txt file (default: download Tiny Shakespeare)")
    p.add_argument("--name", default="shakespeare", help="dataset folder name under data/")
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    args = p.parse_args()

    out_dir = os.path.join("data", f"{args.name}_{args.tokenizer}")
    os.makedirs(out_dir, exist_ok=True)

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()
    else:
        raw_path = os.path.join("data", "tinyshakespeare.txt")
        if not os.path.exists(raw_path):
            print(f"Downloading Tiny Shakespeare -> {raw_path}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, raw_path)
        with open(raw_path, encoding="utf-8") as f:
            text = f.read()

    # Split the raw text first (contiguous chunks), so val/test are truly unseen passages.
    n = len(text)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    train_text = text[: n - n_val - n_test]
    val_text = text[n - n_val - n_test: n - n_test]
    test_text = text[n - n_test:]

    # Char vocabulary is built from ALL text so val/test have no unknown characters.
    tok = build_tokenizer(args.tokenizer, text)
    dtype = np.uint16 if tok.vocab_size < 65535 else np.uint32

    stats = {"tokenizer": args.tokenizer, "vocab_size": tok.vocab_size}
    for split, split_text in [("train", train_text), ("val", val_text), ("test", test_text)]:
        ids = np.array(tok.encode(split_text), dtype=dtype)
        ids.tofile(os.path.join(out_dir, f"{split}.bin"))
        # raw test text is kept for bits-per-character evaluation and the HF comparison
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write(split_text)
        stats[f"{split}_tokens"] = int(len(ids))
        stats[f"{split}_chars"] = len(split_text)

    stats["dtype"] = np.dtype(dtype).name
    save_tokenizer(tok, os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
