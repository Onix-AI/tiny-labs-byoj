# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx",
#     "marimo",
#     "openai",
#     "python-dotenv",
#     "torch",
#     "transformers",
# ]
#
# [tool.marimo.runtime]
# watcher_on_save = "autorun"
#
# [tool.marimo.save]
# autosave = "off"
# ///
"""BYOJ talk: live demo. Turn any LLM into a Jev, then do it offline on a tiny model."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import os
    import time

    import httpx
    import marimo as mo
    import torch
    from dotenv import load_dotenv
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv(mo.notebook_dir() / ".env")  # OPENROUTER_API_KEY from a .env next to this notebook (gitignored)
    return AutoModelForCausalLM, AutoTokenizer, OpenAI, httpx, math, mo, os, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Bring Your Own Jev: demo

    A decision is a **question** about a **state**, with a fixed set of **options**.
    We never let the model write an answer: we read **one token's** probabilities.
    """)
    return


@app.cell
def _():
    DIGITS = "0123456789"


    def build_messages(state, question, options):
        """The same prompt for every model: the options become single digits."""
        listing = "\n".join(f"{d}) {o}" for d, o in zip(DIGITS, options))
        prompt = f'Message:\n"""\n{state}\n"""\n\nQuestion about the message: {question}\n{listing}\n\nAnswer with a single digit.'
        return [{"role": "user", "content": prompt}]

    return DIGITS, build_messages


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Any LLM → Jev

    One token out (`max_tokens=1`), and ask for its **logprobs**.
    """)
    return


@app.cell
def _(DIGITS, OpenAI, build_messages, math, os, time):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY", "missing"))


    def jev_api(state, question, options, model):
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(state, question, options),
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={"provider": {"require_parameters": True}},  # only route to providers that return logprobs
        )
        ms = (time.perf_counter() - start) * 1000

        logprobs = response.choices[0].logprobs
        if logprobs is None or not logprobs.content:
            raise ValueError(f"{model} returned no logprobs")

        top = {t.token: t.logprob for t in logprobs.content[0].top_logprobs}
        probs = [math.exp(top.get(d, -math.inf)) for d in DIGITS[: len(options)]]
        total = sum(probs) or 1.0
        return {o: p / total for o, p in zip(options, probs)}, ms

    return (jev_api,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The real thing: Jev

    TypeSafe's Jev, through OpenRouter's Decisions API. Same key, no prompt: just the state and a typed question.
    """)
    return


@app.cell
def _(httpx, os, time):
    def jev_typesafe(state, question, options, model="~typesafe/jev-latest"):
        start = time.perf_counter()
        response = httpx.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', 'missing')}"},
            json={
                "model": model,
                "state": state,
                "questions": {
                    "answer": {"type": "choice", "instructions": question, "criteria": {o: o for o in options}},
                },
            },
            timeout=30,
        )
        ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        probs = response.json()["answers"]["answer"]["probabilities"]
        return {o: probs[o] for o in options}, ms

    return (jev_typesafe,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Fully offline: a tiny local model

    Same prompt, same idea: one forward pass, then read the logits of the option digits.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, torch):
    MODEL = "Qwen/Qwen3-1.7B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16 if device == "mps" else torch.float32)
    model = model.to(device).eval()
    return device, model, tokenizer


@app.cell
def _(DIGITS, build_messages, device, model, time, tokenizer, torch):
    @torch.no_grad()
    def jev_local(state, question, options):
        start = time.perf_counter()
        text = tokenizer.apply_chat_template(
            build_messages(state, question, options),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0, -1]  # next-token logits, no generation

        digit_ids = [tokenizer.encode(d, add_special_tokens=False)[0] for d in DIGITS[: len(options)]]
        probs = logits[digit_ids].float().softmax(-1).tolist()
        ms = (time.perf_counter() - start) * 1000
        return dict(zip(options, probs)), ms


    jev_local("warm up", "Is this a warm up?", ["no", "yes"])  # first call on the GPU is slow
    return (jev_local,)


@app.cell
def _(mo):
    def bars(probs, title=""):
        """Probabilities as a small horizontal bar chart."""
        top = max(probs, key=probs.get)
        rows = "".join(
            f'''<div style="display:flex;align-items:center;gap:0.6em;margin:0.2em 0">
              <div style="width:9em;text-align:right">{o}</div>
              <div style="flex:1;background:#f3f4f6;border-radius:4px">
                <div style="width:{p * 100:.1f}%;height:1.2em;border-radius:4px;background:{"#a626a4" if o == top else "#d8b4d8"}"></div>
              </div>
              <div style="width:3.5em;font-weight:700">{p:.0%}</div>
            </div>'''
            for o, p in probs.items()
        )
        return mo.Html(f'<div><div style="font-weight:600;margin-bottom:0.3em">{title}</div>{rows}</div>')

    return (bars,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Try it
    """)
    return


@app.cell
def _(mo):
    state = mo.ui.text_area(
        value="I keep waking up at 3am and can't fall back asleep. Any tips?", label="State", full_width=True
    )
    question = mo.ui.text(value="Is this about health and wellness?", label="Question", full_width=True)
    options = mo.ui.text(value="no, yes", label="Options (comma separated)", full_width=True)
    api_model = mo.ui.dropdown(
        options=[
            "openai/gpt-4o-mini",
            "google/gemma-4-31b-it",
            "anthropic/claude-haiku-5.5",  # closed: no logprobs
        ],
        value="openai/gpt-4o-mini",
        label="LLM via OpenRouter",
    )
    run = mo.ui.run_button(label="Run")
    mo.vstack([state, question, options, mo.hstack([api_model, run], justify="start")])
    return api_model, options, question, run, state


@app.cell
def _(api_model, bars, jev_api, jev_local, jev_typesafe, mo, options, question, run, state):
    mo.stop(not run.value, mo.md("*Press Run.*"))

    _options = [o.strip() for o in options.value.split(",") if o.strip()]
    _local, _local_ms = jev_local(state.value, question.value, _options)
    _panels = [bars(_local, f"Qwen3-1.7B, offline · {_local_ms:.0f} ms")]
    try:
        _api, _api_ms = jev_api(state.value, question.value, _options, api_model.value)
        _panels.append(bars(_api, f"{api_model.value} · {_api_ms:.0f} ms"))
    except Exception as e:
        _panels.append(mo.md(f"**{api_model.value}**: {e}"))
    try:
        _jev, _jev_ms = jev_typesafe(state.value, question.value, _options)
        _panels.append(bars(_jev, f"Jev · {_jev_ms:.0f} ms"))
    except Exception as e:
        _panels.append(mo.md(f"**Jev**: {e}"))
    mo.hstack(_panels, widths="equal", gap=3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Head to head

    Some questions have a right answer. Some **can't** have one, and we still get a distribution.
    """)
    return


@app.cell
def _():
    TASKS = [
        ("Can you help me reset my password?", "Is this about health and wellness?", ["no", "yes"]),
        ("What's the best way to fix a flat bike tire?", "Is this about health and wellness?", ["no", "yes"]),
        ("I keep waking up at 3am and can't fall back asleep. Any tips?", "Is this about health and wellness?", ["no", "yes"]),
        ("How much protein should I eat after a workout?", "Is this about health and wellness?", ["no", "yes"]),
        ("I keep waking up at 3am and can't fall back asleep. Any tips?", "Which topic is it about?", ["sleep", "nutrition", "fitness"]),
        ("I flip a fair coin.", "How does it land?", ["heads", "tails"]),
        ("I flip a fair coin.", "How does it land?", ["tails", "heads"]),  # same question, options swapped
        ("It's October 2026.", "Will it rain in Montreal exactly 3 weeks from today?", ["no", "yes"]),
        ("The US midterm elections are in November 2026.", "Which party wins the House?", ["Democrats", "Republicans"]),
    ]
    return (TASKS,)


@app.cell
def _(api_model, mo):
    run_tasks = mo.ui.run_button(label=f"Run all tasks (Qwen3-1.7B vs {api_model.value} vs Jev)")
    run_tasks
    return (run_tasks,)


@app.cell
def _(TASKS, api_model, jev_api, jev_local, jev_typesafe, mo, run_tasks):
    mo.stop(not run_tasks.value)

    def _fmt(probs):
        return " · ".join(f"{o} {p:.0%}" for o, p in probs.items())

    _rows = []
    for _state, _question, _options in TASKS:
        _local, _local_ms = jev_local(_state, _question, _options)
        try:
            _api, _api_ms = jev_api(_state, _question, _options, api_model.value)
            _api_cell, _api_ms_cell = _fmt(_api), f"{_api_ms:.0f}"
        except Exception as e:
            _api_cell, _api_ms_cell = f"error: {e}", "-"
        try:
            _jev, _jev_ms = jev_typesafe(_state, _question, _options)
            _jev_cell, _jev_ms_cell = _fmt(_jev), f"{_jev_ms:.0f}"
        except Exception as e:
            _jev_cell, _jev_ms_cell = f"error: {e}", "-"
        _rows.append(
            {
                "question": f"{_question} ({_state})",
                "Qwen3-1.7B": _fmt(_local),
                "ms (local)": f"{_local_ms:.0f}",
                api_model.value: _api_cell,
                "ms (api)": _api_ms_cell,
                "Jev": _jev_cell,
                "ms (Jev)": _jev_ms_cell,
            }
        )
    mo.ui.table(_rows, selection=None)
    return


if __name__ == "__main__":
    app.run()
