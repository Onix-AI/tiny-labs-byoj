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
    html_head_file="head.html",
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
        print("Do the thing")
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
        return any(k in observations["question"].lower() for k in keywords)


    question = "Can you help me reset my password?"
    decision = decide({"question": question})

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
    ## "Decision model" in 2020 (it was hard)

    COVID-19 Q&A: map a user's question to a pre-canned FAQ answer... **if it's relevant**.
    """),
        mo.hstack(
            [
                mo.md(r"""
    - Had to fine-tune for every use-case
    - Training data was **really hard** to get
    """),
                mo.image(
                    src="public/figures/covid-qa-schematic.png",
                    # cap the height so the slide fits (a 16:9 slide is 56.25cqw tall)
                    style={"max-height": "28cqw", "width": "auto", "max-width": "100%"},
                ),
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
    ## LLMs changed the game

    You could now just *ask* a model, with the right context:

    ```python

    prompt = f'''Is this question about health and wellness?
    Question: {question}

    Answer in JSON: {{"relevant": true | false}}
    PLEASE OUTPUT VALID JSON OR I LOSE MY JOB'''

    response = client.chat.completions.create(
        model="any-llm",
        messages=[{"role": "user", "content": prompt}],
    )
    decision = json.loads(response.choices[0].message.content)["relevant"]
    ```

    Works fine... until it replies

    > `Sure! Here's your JSON:`

    💀
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Structured outputs: the format is guaranteed

    ```python

    class Relevance(BaseModel):
        relevant: bool


    response = client.chat.completions.parse(
        model="any-llm",
        messages=[{"role": "user", "content": f"Is this about health and wellness? {question}"}],
        response_format=Relevance,
    )
    decision = response.choices[0].message.parsed.relevant
    ```

    Valid JSON every time, no begging required.

    But **how sure** is it?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Structured output + logprobs is basically already Jev

    ```python

    response = client.chat.completions.parse(
        model="any-llm",
        messages=messages,
        response_format=Relevance,
        logprobs=True,
        top_logprobs=2,  # also return the 2 most likely tokens + their logprobs
    )

    response.choices[0].message.parsed
    # Relevance(relevant=True)   ← same answer whether it was 99% or 51% sure

    # logprobs at the true / false token:
    # [TopLogprob(token='true',  logprob=-0.03),   # ≈ 97%
    #  TopLogprob(token='false', logprob=-3.5)]    # ≈ 3%
    ```

    Structured output picks the answer. **Logprobs tell you how sure it was.**
    """)
    return


@app.cell
def _(mo):
    # Accuracy vs cost chart; re-renders when the file is updated.
    _fig = mo.watch.file("public/figures/llm-latency-cost.jpg")
    mo.vstack(
        [
            mo.md(r"""
    ## The problem: LLMs are slow and expensive deciders
    """),
            mo.hstack(
                [
                    mo.md(r"""
    - **Latency**: LLMs **generate** every token in the response, a decision only generates **one token**

    - **Model**: LLMs have to learn to **generate** text, a decision model only needs to **understand** it
    """),
                    mo.vstack(
                        [
                            mo.image(_fig.read_bytes(), style={"max-height": "36cqw", "width": "auto", "max-width": "100%"}),
                            mo.md(r"""
    <small>Source: [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe, Oct 12, 2026</small>
    """),
                        ],
                        align="center",
                        gap=0.25,
                    ),
                ],
                widths=[1, 1.5],
                align="center",
                gap=3,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    # Full-slide image, synced from Google Slides (page 3). Re-renders when the file is updated.
    _slide = mo.watch.file("public/figures/encoder-vs-decoder.png")
    mo.vstack(
        [
            mo.md(r"""
    ## Model architecture

    A decision model can be "just" a **BERT-style encoder**: smaller and easier to train. LLMs need a **decoder** to generate. Both are transformers.
    """),
            mo.vstack(
                [
                    mo.image(
                        _slide.read_bytes(),
                        style={"max-height": "36cqw", "width": "auto", "max-width": "100%"},
                    ),
                    mo.md(r"""
    <small>Figures adapted from Sebastian Raschka, [Language Models for Text Classification: From Bag-of-Words to Jev](https://magazine.sebastianraschka.com/p/classifier-history-and-jev) (originals: BERT, Devlin et al. 2019; Transformer, Vaswani et al. 2017)</small>
    """),
                ],
                align="center",
                gap=0.25,
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    ## How decision models changed the equation

    The question moved **from the output head into the input**: one state, many questions, one call.
    """),
            mo.hstack(
                [
                    mo.md(r"""
    | type | asks | returns |
    |---|---|---|
    | **noul** | is this true? | P(yes) |
    | **choice** | pick one of N | P over options |
    | **score** | how much? | P over ordered levels |
    """),
                    mo.md(r"""
    ```python

    response = client.system_one(
        state=user_message,
        questions={
            "is_relevant": Noul(instructions="Is this about health and wellness?"),
            "topic": Choice(
                instructions="Which topic is it about?",
                criteria={"sleep": "...", "nutrition": "...", "fitness": "..."},
            ),
            "urgency": Score(
                instructions="How urgent is it?",
                criteria=["not urgent", "somewhat urgent", "very urgent"],
            ),
        },
    )

    response.nouls["is_relevant"].noul    # 0.97
    response.choices["topic"].choice      # "sleep"
    response.scores["urgency"].score      # 0.33
    ```
    """),
                ],
                widths=[1, 1.6],
                align="center",
                gap=2,
            ),
            mo.md(r"""
    <small>API shape: [TypeSafe docs, quickstart](https://docs.typesafe.ai/introduction/quickstart) (`POST /v1/systemone`)</small>
    """),
        ],
        gap=0.5,
    )
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
