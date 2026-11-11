"""
Web interface for Mini-GPT. Built by build_ui() and served by server.py.

Tabs:
  Write              streaming text generation with style presets
  Next-token lab     see the probability of every candidate next token, step through generation
  Attention explorer heatmaps of what each attention head looks at
  About & API        model card, metrics, how it works, how to call the API
"""
import json
import time

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inference import Engine  # noqa: E402

DEFAULT_PROMPTS = ["ROMEO:", "JULIET:\nO", "KING RICHARD III:\nNow is the", "First Citizen:\nWe are"]

STYLES = {
    "Focused":  dict(temperature=0.5, top_k=20, top_p=1.0),
    "Balanced": dict(temperature=0.8, top_k=50, top_p=1.0),
    "Creative": dict(temperature=1.1, top_k=0, top_p=0.95),
}

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#hero h1 { font-size: 2rem; margin-bottom: 0.2rem; }
#output textarea { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.95rem; line-height: 1.5; }
.stat { font-size: 0.85rem; opacity: 0.75; }
"""


def show_token(t: str) -> str:
    """Make whitespace visible in labels."""
    return repr(t)[1:-1].replace(" ", "␣") or "∅"


def build_ui(engine: Engine) -> gr.Blocks:
    info = engine.info()
    prompts = engine.card.get("example_prompts") or DEFAULT_PROMPTS

    # ---------------- callbacks ----------------
    def apply_style(style):
        s = STYLES[style]
        return s["temperature"], s["top_k"], s["top_p"]

    def write(prompt, max_tokens, temperature, top_k, top_p, seed):
        seed = int(seed) if seed is not None and int(seed) >= 0 else None
        text, n, start = prompt, 0, time.perf_counter()
        for piece in engine.stream(prompt, max_new_tokens=int(max_tokens), temperature=temperature,
                                   top_k=int(top_k) or None, top_p=top_p if top_p < 1 else None,
                                   seed=seed):
            text += piece
            n += 1
            if n % 4 == 0:  # refresh every few tokens for smooth streaming
                yield text, f"{n} tokens · {n / (time.perf_counter() - start):.0f} tokens/sec"
        elapsed = time.perf_counter() - start
        yield text, f"{n} tokens in {elapsed:.1f}s · {n / max(elapsed, 1e-6):.0f} tokens/sec"

    def lab_predict(prompt, temperature):
        r = engine.next_token(prompt, k=10, temperature=temperature)
        label = {show_token(c["token"]): c["probability"] for c in r["candidates"]}
        h = r["entropy_bits"]
        mood = "very confident" if h < 1 else "fairly sure" if h < 2.5 else "uncertain"
        return label, f"Uncertainty (entropy): **{h:.2f} bits**, so the model is {mood} about what comes next."

    def lab_step(prompt, temperature, greedy):
        if greedy:
            prompt += engine.next_token(prompt, k=1)["candidates"][0]["token"]
        else:
            prompt += "".join(engine.stream(prompt, max_new_tokens=1, temperature=temperature, top_k=None))
        label, msg = lab_predict(prompt, temperature)
        return prompt, label, msg

    def attention_plot(prompt, layer, head):
        r = engine.attention(prompt, layer=int(layer), max_tokens=40)
        w = np.array(r["weights"])
        mat = w.mean(axis=0) if head == "All heads (average)" else w[int(head.split()[-1])]
        tokens = [show_token(t) for t in r["tokens"]]
        n = len(tokens)
        size = min(10, max(4.5, n * 0.28))
        fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(mat, cmap="magma", vmin=0, vmax=max(mat.max(), 1e-6))
        ax.set_xticks(range(n), tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(n), tokens, fontsize=8)
        ax.set_xlabel("…looks at this earlier token")
        ax.set_ylabel("This token…")
        ax.set_title(f"Layer {int(layer)} · {head}")
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ---------------- layout ----------------
    with gr.Blocks(title="Mini-GPT · a transformer built from scratch") as demo:
        gr.Markdown(
            f"# Mini-GPT\n"
            f"A {info['parameters'] / 1e6:.1f}M-parameter GPT-style language model written from scratch "
            f"in PyTorch and trained on **{info['dataset'] or 'Shakespeare'}**. "
            f"Watch it write, then look inside to see how it decides.",
            elem_id="hero",
        )

        with gr.Tabs():
            # ---- Write ----
            with gr.Tab("✍️ Write"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt = gr.Textbox(label="Start the text", value=prompts[0], lines=3)
                        gr.Examples(prompts, inputs=prompt, label="Try a prompt")
                        style = gr.Radio(list(STYLES), value="Balanced", label="Writing style")
                        with gr.Accordion("Advanced settings", open=False):
                            max_tokens = gr.Slider(50, 1000, value=400, step=50, label="Length (tokens)")
                            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature",
                                                    info="Higher = more surprising")
                            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k",
                                              info="Sample only from the k most likely tokens (0 = off)")
                            top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Top-p (nucleus)",
                                              info="Sample from the smallest set covering p of the probability")
                            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)",
                                             info="Use a fixed number to get the same text every time")
                        with gr.Row():
                            go = gr.Button("Generate", variant="primary")
                            stop = gr.Button("Stop")
                    with gr.Column(scale=3):
                        output = gr.Textbox(label="Mini-GPT writes…", lines=20, elem_id="output",
                                            buttons=["copy"])
                        stats = gr.Markdown(elem_classes="stat")

                style.change(apply_style, style, [temperature, top_k, top_p])
                run = go.click(write, [prompt, max_tokens, temperature, top_k, top_p, seed], [output, stats])
                stop.click(None, cancels=[run])

            # ---- Next-token lab ----
            with gr.Tab("🔬 Next-token lab"):
                gr.Markdown("A language model never writes a whole sentence at once. It predicts a probability "
                            "for **every possible next token**, picks one, and repeats. Type something to see "
                            "its top guesses, then step forward one token at a time.")
                with gr.Row():
                    with gr.Column():
                        lab_prompt = gr.Textbox(label="Text so far", value="To be, or not to b", lines=4)
                        lab_temp = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature",
                                             info="Watch it flatten or sharpen the probabilities")
                        with gr.Row():
                            lab_go = gr.Button("Predict", variant="primary")
                            lab_greedy = gr.Button("Add most likely")
                            lab_sample = gr.Button("Add sampled")
                    with gr.Column():
                        lab_label = gr.Label(label="Top 10 next-token candidates", num_top_classes=10)
                        lab_msg = gr.Markdown()
                lab_go.click(lab_predict, [lab_prompt, lab_temp], [lab_label, lab_msg])
                lab_temp.release(lab_predict, [lab_prompt, lab_temp], [lab_label, lab_msg])
                lab_greedy.click(lambda p, t: lab_step(p, t, True), [lab_prompt, lab_temp],
                                 [lab_prompt, lab_label, lab_msg])
                lab_sample.click(lambda p, t: lab_step(p, t, False), [lab_prompt, lab_temp],
                                 [lab_prompt, lab_label, lab_msg])

            # ---- Attention explorer ----
            with gr.Tab("🧠 Attention explorer"):
                gr.Markdown("Self-attention lets each token look back at earlier tokens. Each row is a token, "
                            "and bright cells show which earlier tokens it paid attention to. Early layers "
                            "often focus on the last few characters; later layers reach further back.")
                with gr.Row():
                    att_prompt = gr.Textbox(label="Text (up to 40 tokens)", value="To be, or not to be", scale=3)
                    att_layer = gr.Slider(0, info["n_layer"] - 1, value=info["n_layer"] - 1, step=1,
                                          label="Layer", scale=1)
                    att_head = gr.Dropdown(["All heads (average)"] + [f"Head {h}" for h in range(info["n_head"])],
                                           value="All heads (average)", label="Head", scale=1)
                att_go = gr.Button("Show attention", variant="primary")
                att_plot = gr.Plot(label="Attention weights")
                att_go.click(attention_plot, [att_prompt, att_layer, att_head], att_plot)
                att_layer.release(attention_plot, [att_prompt, att_layer, att_head], att_plot)
                att_head.change(attention_plot, [att_prompt, att_layer, att_head], att_plot)

            # ---- About & API ----
            with gr.Tab("📄 About & API"):
                ev = info.get("eval") or []
                eval_md = ""
                if ev:
                    rows = "\n".join(f"| {r['model']} | {r['params_M']}M | {r['perplexity']} | "
                                     f"{r['bits_per_char']} |" for r in ev)
                    eval_md = ("### Test-set results\n| Model | Params | Perplexity | Bits per char ↓ |\n"
                               "|---|---|---|---|\n" + rows)
                curl_body = json.dumps({"prompt": "ROMEO:", "max_new_tokens": 200, "temperature": 0.8})
                gr.Markdown(f"""
### Model
| | |
|---|---|
| Architecture | Decoder-only transformer (GPT), implemented from scratch |
| Parameters | {info['parameters'] / 1e6:.2f}M |
| Layers · heads · embedding size | {info['n_layer']} · {info['n_head']} · {info['n_embd']} |
| Context window | {info['context_length']} tokens |
| Tokenizer | {info['tokenizer']} (vocabulary {info['vocab_size']}) |
| Validation loss | {info['val_loss']} |

{eval_md}

### How it works
Text is split into tokens, each token becomes a vector, and the vectors pass through
{info['n_layer']} transformer blocks. Each block uses **masked multi-head self-attention** (tokens look
at earlier tokens only) followed by a small feed-forward network. The final layer outputs a probability
for every token in the vocabulary. Generation uses a **KV-cache**, so earlier tokens are never recomputed.

### Limitations
This is a small educational model. It learns the *style* of its training text, not facts, so its
output is often odd and never reliable information.

### Use the API
Interactive documentation is at [`/docs`](/docs). Replace `BASE_URL` with this app's address.

```bash
curl -X POST BASE_URL/api/generate -H "Content-Type: application/json" -d '{curl_body}'
```

```python
import requests
r = requests.post("BASE_URL/api/next-token", json={{"prompt": "To be, or not to b", "k": 5}})
print(r.json())
```
""")

        demo.load(lab_predict, [lab_prompt, lab_temp], [lab_label, lab_msg])
    return demo
