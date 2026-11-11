"""
Tests for the inference engine and the REST API, using a tiny randomly-initialised model
so they run in seconds without any training.
"""
import os
import sys

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import build_router  # noqa: E402
from inference import Engine  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from tokenizer import CharTokenizer, save_tokenizer  # noqa: E402

TEXT = "abcdefghijklmnopqrstuvwxyz ,.:\nABCDEFGHIJKLMNOPQRSTUVWXYZ"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    tok = CharTokenizer(TEXT)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(cfg)
    torch.save({"model": model.state_dict(), "model_cfg": cfg.__dict__}, d / "model.pt")
    save_tokenizer(tok, d / "tokenizer.json")
    return str(d)


@pytest.fixture(scope="module")
def engine(model_dir):
    return Engine(model_dir)


@pytest.fixture(scope="module")
def client(engine):
    app = FastAPI()
    app.include_router(build_router(engine))
    return TestClient(app)


def test_kv_cache_matches_full_forward(engine):
    """Logits with the KV-cache must equal logits from recomputing everything."""
    ids = torch.randint(0, engine.cfg.vocab_size, (1, 20))
    logits, kvs = engine.model.forward_cached(ids[:, :10])
    for t in range(10, 20):
        full, _ = engine.model(ids[:, : t + 1])
        logits, kvs = engine.model.forward_cached(ids[:, t: t + 1], past_kvs=kvs)
        assert torch.allclose(logits, full[:, -1], atol=1e-5)


def test_stream_is_reproducible_with_seed(engine):
    a = "".join(engine.stream("ab", max_new_tokens=50, seed=7))
    b = "".join(engine.stream("ab", max_new_tokens=50, seed=7))
    assert a == b and len(a) == 50


def test_stream_handles_text_longer_than_context(engine):
    out = list(engine.stream("hello " * 20, max_new_tokens=100, seed=1))  # prompt + output > block_size
    assert len(out) == 100


def test_stop_sequence(engine):
    out = "".join(engine.stream("a", max_new_tokens=500, seed=3, stop="e"))
    assert out.endswith("e") or len(out) == 500


def test_next_token_probabilities(engine):
    r = engine.next_token("abc", k=5)
    probs = [c["probability"] for c in r["candidates"]]
    assert len(probs) == 5 and probs == sorted(probs, reverse=True) and sum(probs) <= 1.0001


def test_api_health_and_model(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/model").json()["n_layer"] == 2


def test_api_generate(client):
    r = client.post("/api/generate", json={"prompt": "abc", "max_new_tokens": 20, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["tokens_generated"] == 20 and len(body["completion"]) == 20


def test_api_stream(client):
    with client.stream("POST", "/api/generate/stream", json={"prompt": "a", "max_new_tokens": 10}) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]" and len(lines) == 11


def test_api_validation(client):
    assert client.post("/api/generate", json={"prompt": "a", "max_new_tokens": 0}).status_code == 422
    assert client.post("/api/generate", json={"prompt": "a", "temperature": 5}).status_code == 422
    assert client.post("/api/attention", json={"prompt": "a", "layer": 99}).status_code == 422


def test_api_attention(client):
    r = client.post("/api/attention", json={"prompt": "hello", "layer": 1}).json()
    assert len(r["weights"]) == 2 and len(r["tokens"]) == 5
