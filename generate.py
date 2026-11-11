"""
Generate text from a trained model.

    python generate.py --run runs/shakespeare_char_small --prompt "ROMEO:" --tokens 500
"""
import argparse

import torch

from utils import load_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder created by train.py")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok, _, device = load_run(args.run, args.device)
    ids = tok.encode(args.prompt) or tok.encode("\n")
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(x, args.tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p)
        print(f"===== sample {i + 1} =====")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
