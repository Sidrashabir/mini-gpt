"""
Mini-GPT: a decoder-only transformer written from scratch in PyTorch.

Every piece of the architecture is implemented by hand (no nn.Transformer):
token + positional embeddings, multi-head causal self-attention, feed-forward
MLP, pre-LayerNorm residual blocks, weight tying and autoregressive sampling.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256   # maximum context length (tokens)
    n_layer: int = 6        # number of transformer blocks
    n_head: int = 6         # attention heads per block
    n_embd: int = 384       # embedding / hidden size
    dropout: float = 0.2
    bias: bool = False      # bias in Linear/LayerNorm (False is slightly better and faster)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token may only attend to earlier tokens."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # One linear layer produces queries, keys and values for all heads at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular mask: position i can see positions 0..i only.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

        # When True, the manual path is used and attention weights are kept for visualization.
        self.store_attn = False
        self.last_attn = None

    def forward(self, x: torch.Tensor, past_kv=None, return_kv: bool = False):
        """
        past_kv: optional (k, v) cached from earlier steps, used during generation.
        With a cache, x holds only the NEW token(s); keys/values of old tokens are reused
        instead of recomputed (the "KV-cache" used by every production LLM).
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            assert T == 1, "with a KV-cache, feed one new token at a time"
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        # A single new token may attend to every cached token, so no mask is needed then.
        causal = past_kv is None

        if self.store_attn:
            # Explicit scaled dot-product attention: softmax(QK^T / sqrt(d)) V
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if causal:
                att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            self.last_attn = att.detach().cpu()
            att = self.attn_dropout(att)
            y = att @ v
        else:
            # Same maths, fused kernel (much faster on GPU). Tested equal in tests/.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=self.attn_dropout.p if self.training else 0.0
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        y = self.resid_dropout(self.proj(y))
        return (y, (k, v)) if return_kv else y


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer block with pre-LayerNorm and residual connections."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, return_kv: bool = False):
        if return_kv:
            a, kv = self.attn(self.ln1(x), past_kv=past_kv, return_kv=True)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, kv
        x = x + self.attn(self.ln1(x))   # tokens communicate
        x = x + self.mlp(self.ln2(x))    # each token "thinks" on its own
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: input embedding and output projection share weights.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale residual projections so activations don't grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def forward_cached(self, idx: torch.Tensor, past_kvs=None):
        """
        Inference-only forward pass that returns a KV-cache.
        First call: idx is the whole prompt (prefill). Later calls: idx is one new token.
        Returns logits for the last position and the updated cache.
        """
        B, T = idx.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        assert past_len + T <= self.cfg.block_size, "cache is full: re-prefill with a shorter context"
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_kvs = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, past_kv=None if past_kvs is None else past_kvs[i], return_kv=True)
            new_kvs.append(kv)
        logits = self.lm_head(self.ln_f(x[:, -1, :]))
        return logits, new_kvs

    def set_store_attention(self, flag: bool):
        for block in self.blocks:
            block.attn.store_attn = flag

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95)):
        """AdamW with weight decay on matrices only (not on biases or LayerNorm)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """Autoregressive sampling with temperature, top-k and nucleus (top-p) filtering."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]          # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # last position only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()  # always keep the top token
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
