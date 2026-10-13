# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo"]
#
# [tool.marimo.runtime]
# watcher_on_save = "autorun"
#
# [tool.marimo.save]
# autosave = "off"
# ///
"""BYOJ talk: WIP slides draft. Slides layout lives in layouts/slides.slides.json."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    layout_file="layouts/slides.slides.json",
    css_file="theme.css",
)


@app.cell
def _():
    import math

    import marimo as mo

    return math, mo


@app.cell
def title(mo):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("# Bring Your Own Jev"),
                    mo.md("@jerpint, Jeremy Pinto"),
                ],
                gap=0.5,
            ),
            mo.image(src="public/onix-tiny-labs.png", width=400, rounded=True),
        ],
        justify="space-between",
        align="center",
    )
    return


@app.cell
def _(mo):
    # Full-slide image, synced from Google Slides. Re-renders when the file is updated.
    _slide = mo.watch.file("public/memes/decision-models.png")
    mo.image(_slide.read_bytes(), width="100%")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What is a decision?

    A decision can be summarized as:

    > given an observation, which option should I pick?

    ```python

    decision = decide(observations)

    if decision:
        print("Do expensive thing")
    else:
        print("Do other thing")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What is a decision?

    Deterministic decisions are easy, eg:

    ```python

    def decide(observations: dict) -> bool:
        '''Make a decision given the observations.'''

        return observations["value"] > MAXIMUM_ALLOWED_VALUE


    decision = decide(observations)

    if decision:
        print("Do expensive thing")
    else:
        print("Do other thing")
    ```
    """)
    return



@app.cell
def _(mo):
    mo.md(r"""
    ## What is a decision?

    Semantic decisions are harder to codify, eg:

    ```python

    def decide(observations: dict) -> bool:
        '''Is the user's question relevant?'''

        # no threshold to compare against so you guess some heuristics
        keywords = ["health", "wellness"]
        if observations["user_input"] in keywords:
            return True
        return False


    decision = decide({"question": "Can you help me reset my password?"})

    if decision:
        print(ask_expensive_bot(question))
    else:
        print("Sorry, I can only help with health and wellness questions.")
    ```
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md(r"""
    ## "Decision model" in 2020 (it was bad)

    COVID-19 Q&A: map a user's question to a pre-canned FAQ answer... **if it's relevant**.
    """),
        mo.hstack(
            [
                mo.md(r"""
    - Fine-tuned **BERT**, specific to this one use case
    - Out-of-distribution training data was **really hard** to get
    """),
                mo.image(src="public/figures/covid-qa-schematic.png", height=380),
            ],
            widths=[1, 1.4],
            align="center",
        ),
        mo.md(r"""
    <small>Bronzi, Pinto, Ghosn, Subakan, Sharma, Lu, Reddy. *A Question Answering System in Response to the COVID-19 Crisis.* [Montreal AI Symposium 2020](https://montrealaisymposium.wordpress.com/accepted-submissions-2020/)</small>
    """),
    ])
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
    ## Everyone can build one: the Jev Decision Index

    [Community leaderboard](https://huggingface.co/spaces/multimodalart/jev-decision-index) of open Jev reproductions (snapshot 2026-10-12):

    - **148** models scored on 37 public benchmarks + private tests
    - **7** open models already beat Jev
    - **AnyJev L0 · Gemma-4-31B** (logit readout, *no training*) ties Jev: 60.2 vs 60.1, but calibration error **0.24** vs **0.07**

    *Structured output + logprobs gets you the accuracy. Calibration makes it a Jev.*

    📚 Every reproduction, trick and explainer in one place: [the Jev reproductions tracker](https://multimodalart-jev-decision-index.static.hf.space/news.html) (~65 entries)
    """)
    return


@app.cell
def _(mo):
    mo.Html(
        '<iframe src="https://multimodalart-jev-decision-index.static.hf.space/index.html" '
        'style="width:100%;height:75vh;border:0;border-radius:8px"></iframe>'
    )
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
