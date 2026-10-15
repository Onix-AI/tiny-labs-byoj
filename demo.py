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
    import random
    import time
    from dataclasses import dataclass

    import httpx
    import marimo as mo
    import torch
    from dotenv import load_dotenv
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv(mo.notebook_dir() / ".env")  # OPENROUTER_API_KEY, gitignored
    autorun = os.environ.get("DEMO_AUTORUN") == "1"  # run everything without pressing buttons (for exports)
    return AutoModelForCausalLM, AutoTokenizer, OpenAI, autorun, dataclass, httpx, math, mo, os, random, time, torch


@app.cell
def _(mo):
    mo.md(r"""
    # Bring Your Own Jev

    A decision: a **state**, a **question**, a fixed set of **options**.
    We never let the model write an answer. We read the probability of **one token**.
    """)
    return


@app.cell
def _(dataclass, math):
    @dataclass
    class Choice:
        instructions: str
        options: list[str]


    def prompt(state: str, choice: Choice) -> str:
        options = "\n".join(f"{i}) {option}" for i, option in enumerate(choice.options))
        return f'Message:\n"""\n{state}\n"""\n\n{choice.instructions}\n{options}\n\nAnswer with a single digit.'


    def softmax(options: list[str], logits: list[float]) -> dict[str, float]:
        exps = [math.exp(x) for x in logits]
        return {option: e / sum(exps) for option, e in zip(options, exps)}

    return Choice, prompt, softmax


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Any LLM → Jev

    Ask for **one token**, with its **logprobs**.
    """)
    return


@app.cell
def _(OpenAI, math, os, prompt, softmax):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))


    def jev_api(state, choice, model="openai/gpt-4o-mini"):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt(state, choice)}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={"provider": {"require_parameters": True}},  # only providers that return logprobs
        )
        top = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
        logits = [top.get(str(i), -math.inf) for i in range(len(choice.options))]
        return softmax(choice.options, logits)

    return (jev_api,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The real thing: Jev

    TypeSafe's Jev, through OpenRouter's Decisions API. No prompt, just a typed question.
    """)
    return


@app.cell
def _(httpx, os):
    def jev_typesafe(state, choice):
        response = httpx.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"},
            json={
                "model": "~typesafe/jev-latest",
                "state": state,
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": choice.instructions,
                        "criteria": {option: option for option in choice.options},
                    }
                },
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["answers"]["answer"]["probabilities"]

    return (jev_typesafe,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Fully offline: a tiny local model

    One forward pass, then read the logits of the option digits. No generation, no network.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, torch):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16).to(device).eval()
    return device, model, tokenizer


@app.cell
def _(device, model, prompt, softmax, tokenizer, torch):
    @torch.no_grad()
    def jev_local(state, choice):
        messages = [{"role": "user", "content": prompt(state, choice)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        next_token_logits = model(**inputs).logits[0, -1]
        digits = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(len(choice.options))]
        return softmax(choice.options, next_token_logits[digits].tolist())

    return (jev_local,)


@app.cell
def _(jev_api, jev_local, jev_typesafe, mo, time):
    def run_all(state, choice, api_model):
        """Run every backend on one decision: {name: (probabilities or error, ms)}."""
        backends = {
            "Qwen3-1.7B (offline)": lambda: jev_local(state, choice),
            api_model: lambda: jev_api(state, choice, api_model),
            "Jev": lambda: jev_typesafe(state, choice),
        }
        results = {}
        for name, run in backends.items():
            start = time.perf_counter()
            try:
                results[name] = (run(), (time.perf_counter() - start) * 1000)
            except Exception as e:
                results[name] = (f"error: {e}", None)
        return results


    def bars(probs):
        """Probabilities as a small horizontal bar chart."""
        top = max(probs, key=probs.get)
        rows = "".join(
            f'''<div style="display:flex;align-items:center;gap:0.6em;margin:0.2em 0">
              <div style="width:8em;text-align:right">{option}</div>
              <div style="flex:1;background:#f3f4f6;border-radius:4px">
                <div style="width:{p * 100:.1f}%;height:1.2em;border-radius:4px;background:{"#a626a4" if option == top else "#d8b4d8"}"></div>
              </div>
              <div style="width:3em;font-weight:700">{p:.0%}</div>
            </div>'''
            for option, p in probs.items()
        )
        return mo.Html(rows)

    return bars, run_all


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Try it
    """)
    return


@app.cell
def _(mo):
    state = mo.ui.text_area(value="I keep waking up at 3am and can't fall back asleep. Any tips?", label="State", full_width=True)
    question = mo.ui.text(value="Is this about health and wellness?", label="Question", full_width=True)
    options = mo.ui.text(value="no, yes", label="Options", full_width=True)
    api_model = mo.ui.dropdown(
        ["openai/gpt-4o-mini", "google/gemma-4-31b-it"],
        value="openai/gpt-4o-mini",
        label="LLM",
    )
    run = mo.ui.run_button(label="Run")
    mo.vstack([state, question, options, mo.hstack([api_model, run], justify="start")])
    return api_model, options, question, run, state


@app.cell
def _(Choice, api_model, autorun, bars, mo, options, question, run, run_all, state):
    mo.stop(not (run.value or autorun), mo.md("*Press Run.*"))

    _choice = Choice(question.value, [o.strip() for o in options.value.split(",")])
    _panels = []
    for _name, (_result, _ms) in run_all(state.value, _choice, api_model.value).items():
        _title = mo.md(f"**{_name}**" + (f" · {_ms:.0f} ms" if _ms else ""))
        _panels.append(mo.vstack([_title, bars(_result) if isinstance(_result, dict) else mo.md(_result)]))
    mo.hstack(_panels, widths="equal", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Head to head

    Some questions have a right answer. Some **can't** have one, and we still get a distribution.
    """)
    return


@app.cell
def _(Choice):
    health = Choice("Is this about health and wellness?", ["no", "yes"])

    TASKS = [
        ("Can you help me reset my password?", health),
        ("What's the best way to fix a flat bike tire?", health),
        ("I keep waking up at 3am and can't fall back asleep. Any tips?", health),
        ("How much protein should I eat after a workout?", health),
        ("I keep waking up at 3am and can't fall back asleep. Any tips?", Choice("Which topic?", ["sleep", "nutrition", "fitness"])),
        ("I flip a fair coin.", Choice("How does it land?", ["heads", "tails"])),
        ("I flip a fair coin.", Choice("How does it land?", ["tails", "heads"])),
        ("It's October 2026.", Choice("Will it rain in Montreal 3 weeks from today?", ["no", "yes"])),
        ("The US midterms are in November 2026.", Choice("Who wins the House?", ["Democrats", "Republicans"])),
    ]
    return (TASKS,)


@app.cell
def _(mo):
    run_tasks = mo.ui.run_button(label="Run all (live)")
    run_tasks
    return (run_tasks,)


@app.cell
def _(TASKS, api_model, autorun, mo, run_all, run_tasks):
    mo.stop(not (run_tasks.value or autorun))

    def _fmt(result, ms):
        if not isinstance(result, dict):
            return result
        return " · ".join(f"{option} {p:.0%}" for option, p in result.items()) + f"  ({ms:.0f} ms)"

    _rows = []
    for _state, _choice in TASKS:
        _row = {"state": _state, "question": _choice.instructions}
        for _name, (_result, _ms) in run_all(_state, _choice, api_model.value).items():
            _row[_name] = _fmt(_result, _ms)
        _rows.append(_row)
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Can a decision model be random? Approximating π

    Did you know you could approximate pi by just drawing points at random on a grid? This is how it works:

    Draw an (x,y) point at random inside a square, and check if it falls within a circle inscribed in that square. Count the number of circle hits and divide by the number of total hits. This should approximate the area of the circle to the square. From that, we can approximate pi. Check out the figure below for more intuition.

    ![approximate_pi](https://i.stack.imgur.com/g0Ctz.png)

    Since

    ![approx_pi1](https://user-images.githubusercontent.com/18450628/111053532-8d5f5b00-8432-11eb-9be4-2de7ec8cc3e2.jpg)

    we can approximate pi:

    ![approx_pi2](https://user-images.githubusercontent.com/18450628/111053571-d1526000-8432-11eb-9e23-27867c99a54b.png)

    **The twist:** instead of `random.random()`, we ask each model for a random digit **once**, then draw points from its distribution (`x = 0.d₁d₂`, `y = 0.d₃d₄`). A fair model gives 10% per digit, and π ≈ 3.14. Play with N, the number of points to draw, to see how it affects the approximation.
    """)
    return


@app.cell
def _(mo):
    pi_state = mo.ui.text(value="Pick a random digit between 0 and 9.", label="State", full_width=True)
    run_pi = mo.ui.run_button(label="Ask the models")
    n_points = mo.ui.slider(100, 10000, step=100, value=2000, label="N points", show_value=True)
    mo.vstack([pi_state, mo.hstack([run_pi, n_points], justify="start", gap=2)])
    return n_points, pi_state, run_pi


@app.cell
def _(Choice, api_model, autorun, mo, pi_state, run_all, run_pi):
    mo.stop(not (run_pi.value or autorun), mo.md("*Press Ask the models.*"))

    digit = Choice("Pick a random digit.", [str(i) for i in range(10)])
    digit_probs = {}
    for _name, (_result, _ms) in run_all(pi_state.value, digit, api_model.value).items():
        if isinstance(_result, dict):
            digit_probs[_name] = _result
    return (digit_probs,)


@app.cell
def _(mo, random):
    def estimate_pi(draw_digit, n, seed=0):
        """Draw n points from a digit generator; return (pi estimate, points, digits drawn)."""
        rng = random.Random(seed)
        digits = []

        def coordinate():
            d1, d2 = draw_digit(rng), draw_digit(rng)
            digits.extend([d1, d2])
            return (d1 * 10 + d2 + 0.5) / 100

        points = [(coordinate(), coordinate()) for _ in range(n)]
        inside = sum((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.25 for x, y in points)
        return 4 * inside / n, points, digits


    def from_model(probs):
        """A digit generator that samples from a model's distribution."""
        return lambda rng: int(rng.choices(list(probs), weights=list(probs.values()))[0])


    def scatter(points, size=180):
        """Points in the unit square: inside the circle in purple, outside in grey."""
        dots = "".join(
            f'<circle cx="{x * size:.1f}" cy="{(1 - y) * size:.1f}" r="1.3" '
            f'fill="{"#a626a4" if (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.25 else "#9ca3af"}"/>'
            for x, y in points[:3000]
        )
        return mo.Html(
            f'<svg width="{size}" height="{size}" style="border:1px solid #e5e7eb">'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{size / 2}" fill="none" stroke="#383a42"/>{dots}</svg>'
        )

    return estimate_pi, from_model, scatter


@app.cell
def _(bars, digit_probs, estimate_pi, from_model, mo, n_points, scatter):
    _generators = {"random.randint(0, 9)": lambda rng: rng.randint(0, 9)}
    _generators.update({name: from_model(probs) for name, probs in digit_probs.items()})

    _panels = []
    for _name, _draw in _generators.items():
        _pi, _points, _digits = estimate_pi(_draw, n_points.value)
        _counts = {str(d): _digits.count(d) / len(_digits) for d in range(10)}  # what was actually drawn
        _panels.append(mo.vstack([mo.md(f"**{_name}**  \nπ ≈ **{_pi:.3f}**"), scatter(_points), bars(_counts)]))
    mo.hstack(_panels, widths="equal", gap=1.5)
    return


if __name__ == "__main__":
    app.run()
