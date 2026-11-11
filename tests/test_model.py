"""
Unit tests that prove the model is implemented correctly.
Run with:  python -m pytest tests -v
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

CFG = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def make_model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_output_shapes():
    model = make_model()
    x = torch.randint(0, CFG.vocab_size, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, CFG.vocab_size)
    assert loss.dim() == 0


def test_initial_loss_is_near_uniform():
    # A fresh model should be "maximally unsure": loss ~ ln(vocab_size)
    model = make_model()
    x = torch.randint(0, CFG.vocab_size, (8, 16))
    y = torch.randint(0, CFG.vocab_size, (8, 16))  # random targets (not x: tied weights favour copying)
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3


def test_causality_future_tokens_do_not_leak():
    # Changing the LAST token must not change predictions at earlier positions.
    model = make_model()
    x = torch.randint(0, CFG.vocab_size, (1, 12))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
    l1, _ = model(x)
    l2, _ = model(x2)
    assert torch.allclose(l1[0, :-1], l2[0, :-1], atol=1e-5)
    assert not torch.allclose(l1[0, -1], l2[0, -1])


def test_manual_attention_matches_fused_kernel():
    model = make_model()
    x = torch.randint(0, CFG.vocab_size, (2, 16))
    fused, _ = model(x)
    model.set_store_attention(True)
    manual, _ = model(x)
    assert torch.allclose(fused, manual, atol=1e-5)
    att = model.blocks[0].attn.last_attn
    assert torch.allclose(att.sum(-1), torch.ones_like(att.sum(-1)), atol=1e-5)  # rows sum to 1
    assert torch.all(att.triu(diagonal=1) == 0)  # no attention to the future


def test_weight_tying():
    model = make_model()
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_can_overfit_a_single_batch():
    # A correct model + optimizer should memorize one small batch.
    torch.manual_seed(0)
    model = GPT(CFG).train()
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, CFG.vocab_size, (4, 16))
    y = torch.roll(x, -1, dims=1)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1


def test_generate_length_and_context_cropping():
    model = make_model()
    x = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(x, max_new_tokens=40, top_k=10, top_p=0.9)  # longer than block_size
    assert out.shape == (1, 41)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"
