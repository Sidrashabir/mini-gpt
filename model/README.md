---
license: mit
tags: [pytorch, gpt, transformer, from-scratch, text-generation]
---
# Mini-GPT

A 0.8M-parameter decoder-only transformer written from scratch in PyTorch
and trained on `Tiny Shakespeare`.

| Setting | Value |
|---|---|
| Layers / heads / embedding | 4 / 4 / 128 |
| Context length | 128 tokens |
| Vocabulary | 65 |
| Best validation loss | 2.0429 |

Load it with the `inference.Engine` class from the project repository.
This is an educational model: it imitates the style of its training text and does not
produce factual or reliable content.
