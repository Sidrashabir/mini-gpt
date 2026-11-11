"""
Two tokenizers so you can compare them as an experiment:

- CharTokenizer: every unique character is a token. Tiny vocabulary, no dependencies,
  easiest to understand. Built from the training text itself.
- BPETokenizer: GPT-2's byte-pair encoding via `tiktoken` (vocab 50,257). Fewer tokens
  per sentence, but a much bigger embedding table to learn.
"""
import json


class CharTokenizer:
    name = "char"

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            # Drop characters never seen in training instead of crashing.
            text = "".join(ch for ch in text if ch in self.stoi)
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": "".join(self.chars)}


class BPETokenizer:
    name = "bpe"

    def __init__(self):
        import tiktoken  # imported lazily so char mode works without it
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids])

    def to_dict(self):
        return {"type": "bpe"}


def build_tokenizer(kind, text=None):
    if kind == "char":
        if text is None:
            raise ValueError("CharTokenizer needs the training text to build its vocabulary")
        return CharTokenizer(text)
    if kind == "bpe":
        return BPETokenizer()
    raise ValueError(f"unknown tokenizer: {kind}")


def tokenizer_from_dict(d):
    if d["type"] == "char":
        return CharTokenizer(d["chars"])
    return BPETokenizer()


def save_tokenizer(tok, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, ensure_ascii=False)


def load_tokenizer(path):
    with open(path, encoding="utf-8") as f:
        return tokenizer_from_dict(json.load(f))
