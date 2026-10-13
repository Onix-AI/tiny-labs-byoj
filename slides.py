# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo"]
# ///
"""BYOJ talk: WIP slides draft. Slides layout lives in layouts/slides.slides.json."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", layout_file="layouts/slides.slides.json")


@app.cell
def _():
    import math

    import marimo as mo

    return math, mo


@app.cell
def _(mo):
    mo.md(r"""
    # Bring Your Own Jev

    *Most of your LLM calls are just if-statements.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What is a decision?

    ```python

    def decide(observation) -> bool:
        ...

    decision = decide(observation)

    if decision:
        print("Do thing")
    else:
        print("Do other thing")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How decision models changed the equation

    The question moved **from the output head into the input**.

    `(state, question, options) → probability per option`, in one pass.

    | type | asks | returns |
    |---|---|---|
    | **choice** | pick one of N | P over options |
    | **noul** | is this true? | P(yes) |
    | **score** | how much? | P over ordered levels |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Structured output + logprobs is basically already Jev

    1. generate text → parse
    2. **structured output** → the format
    3. **+ logprobs** → the distribution
    4. **calibration** → a Jev
    """)
    return


@app.cell
def _(mo):
    temperature = mo.ui.slider(0.5, 5.0, step=0.1, value=1.0, label="temperature")
    return (temperature,)


@app.cell
def _(math, mo, temperature):
    # Toy logits standing in for an LLM's answer-token logits (real model comes later).
    logits = {"angry": 4.0, "neutral": 1.5, "happy": 0.5}
    z = {k: v / temperature.value for k, v in logits.items()}
    norm = sum(math.exp(v) for v in z.values())
    probs = {k: math.exp(v) / norm for k, v in z.items()}
    bars = "\n".join(
        f"| {k} | {'█' * round(p * 40)} {p:.0%} |" for k, p in probs.items()
    )
    mo.vstack([
        mo.md("## One temperature fixes the overconfidence"),
        temperature,
        mo.md(f"| option | P(option) |\n|---|---|\n{bars}"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where Jev still might fail

    - **OOD.** In-distribution 75% → **56%** when the dataset is held out.
    - Multi-step reasoning
    - Closed weights: no audit

    Small, open, cascaded.
    """)
    return


if __name__ == "__main__":
    app.run()
