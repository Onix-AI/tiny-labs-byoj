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
    import base64
    import math

    import marimo as mo

    return base64, math, mo


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
        '''Is the value above the threshold?'''

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
    # Announcement banner, synced from Google Slides (page 5). Re-renders when the file is updated.
    _banner = mo.watch.file("public/figures/jev-announcement.png")
    mo.vstack(
        [
            mo.md(r"""
    ## What is Jev?
    """),
            mo.hstack(
                [
                    mo.md(r"""
    - A **decision model** from **TypeSafe**: *"unstructured state in, typed probabilistic decisions out"*
    - They call it a **System One model**: fast, structured decisions that software can use directly
    - **Closed source**: we get an API, not the weights or the architecture
    - It **started the hype**: open replications, guides and leaderboards within days
    """),
                    mo.image(
                        _banner.read_bytes(),
                        style={"max-height": "30cqw", "width": "auto", "max-width": "100%", "border-radius": "8px"},
                    ),
                ],
                widths=[1.2, 1],
                align="center",
                gap=2,
            ),
            mo.md(r"""
    <small>Source: [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe</small>
    """),
        ],
        gap=0.75,
    )
    return


@app.cell
def _(mo):
    def decision_flow(stage):
        """The "when to use a decision model" flowchart, revealed in stages.

        1: decision model vs LLM. 2: + "what kind of answer?" (LLM greyed out).
        3: + Noul / Choice / Score.
        """
        lines = [
            "flowchart LR",
            'Q{"Fixed set of options,<br/>known in advance?"}',
            'Q -- "<b>yes</b>" --> D["<b>Decision model</b><br/>classification · true or false · choice"]',
            'Q -- "<b>no</b>" --> L["<b>LLM</b><br/>text · code · open-ended"]',
        ]
        if stage >= 2:
            lines.append('D --> T{"What kind<br/>of answer?"}')
        if stage >= 3:
            lines += [
                'T -- "yes / no" --> N["<b>Noul</b><br/>P(yes)"]',
                'T -- "one of N" --> C["<b>Choice</b><br/>P per option"]',
                'T -- "a level on a scale" --> S["<b>Score</b><br/>P per level"]',
            ]
        lines += [
            "classDef question fill:#ffffff,stroke:#383a42,stroke-width:2px,color:#383a42",
            "classDef decision fill:#f6e9f6,stroke:#a626a4,stroke-width:2px,color:#383a42",
            "classDef llm fill:#fdf3e1,stroke:#c18401,stroke-width:2px,color:#383a42",
            "classDef muted fill:#f3f4f6,stroke:#d1d5db,stroke-width:2px,color:#9ca3af",
            "class Q question",
            "class D decision",
        ]
        if stage >= 2:
            lines += [
                "class T question",
                "class L muted",
                "linkStyle 1 stroke:#d1d5db,color:#9ca3af",  # the "no" arrow to the LLM
            ]
        else:
            lines.append("class L llm")
        if stage >= 3:
            lines.append("class N,C,S decision")
        return mo.mermaid(
            "\n    ".join(lines),
            theme="base",
            theme_variables={
                "fontSize": "18px",
                "fontFamily": "-apple-system, BlinkMacSystemFont, Inter, Helvetica Neue, Arial, sans-serif",
                "lineColor": "#383a42",
                "edgeLabelBackground": "#ffffff",
            },
        )

    return (decision_flow,)

@app.cell
def _(decision_flow, mo):
    mo.vstack([mo.md(r"""
    ## When to use a decision model
    """), decision_flow(1)], gap=1)
    return

@app.cell
def _(decision_flow, mo):
    mo.vstack([mo.md(r"""
    ## What kind of answer?
    """), decision_flow(2)], gap=1)
    return

@app.cell
def _(mo):
    # Full-slide image, synced from Google Slides (page 4). Re-renders when the file is updated.
    _slide = mo.watch.file("public/figures/typesafe-primitives.png")
    mo.vstack(
        [
            mo.md(r"""
    ## What kind of answer?
    """),
            mo.vstack(
                [
                    mo.image(
                        _slide.read_bytes(),
                        style={"max-height": "32cqw", "width": "auto", "max-width": "100%"},
                    ),
                    mo.md(r"""
    <small>Source: [docs.typesafe.ai](https://docs.typesafe.ai/introduction)</small>
    """),
                ],
                align="center",
                gap=0.25,
            ),
            mo.md(r"""
    Jev set the standard for what the API surface could look like for decision models.
    """),
        ],
        gap=0.75,
    )
    return


@app.cell
def _(decision_flow, mo):
    mo.vstack([mo.md(r"""
    ## What kind of answer?
    """), decision_flow(3)], gap=1)
    return

@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Jev data structures

    The question moved **from the output head into the input**: one state, many questions, one call.
    """),
            mo.hstack(
                [
                    mo.md(r"""
    ```python

    @dataclass
    class Noul:      # is this true?
        instructions: str
        # -> noul: P(yes)


    @dataclass
    class Choice:    # pick one of N
        instructions: str
        criteria: dict[str, str]
        # -> choice + probabilities


    @dataclass
    class Score:     # how much? (ordered classes)
        instructions: str
        criteria: list[str]
        # -> score + probabilities
    ```
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
                widths=[1, 1.5],
                align="center",
                gap=1.5,
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
    mo.vstack(
        [
            mo.md(r"""
    ## A real Jev response

    *"Our API integration started returning 500 errors on every request about 20 minutes ago, and we can't process any customer orders until this is fixed."*
    """),
            mo.hstack(
                [
                    mo.md(r"""
    ```python

    "department": {      # Choice
        "choice": "technical",
        "confidence": 1.0,
        "probabilities": {
            "technical": 1.0,
            "billing": 0.0,
            "sales": 0.0,
        },
    },
    "is_urgent": {       # Noul
        "noul": 0.99,
    },
    ```
    """),
                    mo.md(r"""
    ```python

    "frustration": {     # Score
        "score": 0.67,
        "confidence": 0.5,
        "legend": {
            "0": "Calm, just stating facts",
            "1": "Frustrated but civil",
            "2": "Very angry, strong language",
        },
        "probabilities": {"0": 0.33, "1": 0.67, "2": 0.0},
    },
    ```
    """),
                ],
                widths=[1, 1.4],
                gap=1.5,
            ),
            mo.md(r"""
    <small>Example response from the [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart) (`jev-1.13.0`, 428 input / 73 output tokens)</small>
    """),
        ],
        gap=0.5,
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
    """),
            mo.hstack(
                [
                    mo.md(r"""
    A decision model can be "just" a **BERT-style encoder**: smaller and easier to train.

    LLMs need a **decoder** to generate.

    Both are transformers.
    """),
                    mo.image(
                        _slide.read_bytes(),
                        style={"max-height": "40cqw", "width": "auto", "max-width": "100%"},
                    ),
                ],
                widths=[1, 2.6],
                align="center",
                gap=1.5,
            ),
            mo.md(r"""
    <small>Figures adapted from Sebastian Raschka, [Language Models for Text Classification: From Bag-of-Words to Jev](https://magazine.sebastianraschka.com/p/classifier-history-and-jev) (originals: BERT, Devlin et al. 2019; Transformer, Vaswani et al. 2017)</small>
    """),
        ],
        gap=0.25,
    )
    return


@app.cell
def _(math, mo, temperature):
    # Toy answer logits for "Which topic is this question about?" (the real model comes later).
    _logits = {"sleep": 4.0, "nutrition": 1.5, "fitness": 1.0, "off-topic": 0.2}
    _z = {k: v / temperature.value for k, v in _logits.items()}
    _norm = sum(math.exp(v) for v in _z.values())
    _probs = {k: math.exp(v) / _norm for k, v in _z.items()}
    _top = max(_probs, key=_probs.get)

    _cols = "".join(
        f'''<div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;height:100%">
          <div style="font-weight:700;margin-bottom:0.3em">{p:.0%}</div>
          <div style="width:100%;height:{p * 85:.1f}%;border-radius:6px 6px 0 0;background:{"#a626a4" if k == _top else "#d8b4d8"}"></div>
        </div>'''
        for k, p in _probs.items()
    )
    _labels = "".join(f'<div style="flex:1;text-align:center">{k}</div>' for k in _probs)
    _chart = mo.Html(
        f'''<div style="font-size:1.6cqw;width:100%">
          <div style="display:flex;align-items:flex-end;gap:1.5cqw;height:20cqw;border-bottom:2px solid #383a42;padding:0 0.5cqw">{_cols}</div>
          <div style="display:flex;gap:1.5cqw;padding:0.4em 0.5cqw 0;color:#383a42">{_labels}</div>
        </div>'''
    )

    mo.vstack(
        [
            mo.md(r"""
    ## Calibration: can you trust the 90%?
    """),
            mo.hstack(
                [
                    mo.md(r"""
    A model is **calibrated** if, when it says **90%**, it's right **90%** of the time.

    - It says **90%** on 10 reviews, but only **8** are right: **overconfident**
    - Calibrated, it says **80%** on those same reviews: same accuracy, **honest probability**

    Why it matters: decisions run on thresholds. *"Only call the expensive bot if P(relevant) > 0.7"* only works if 0.7 means 0.7.

    Raw LLM logprobs are usually **overconfident**.
    """),
                    mo.vstack(
                        [
                            mo.md("*Which topic is this question about?*"),
                            _chart,
                            temperature,
                        ],
                        gap=1,
                    ),
                ],
                widths=[1.1, 1],
                align="center",
                gap=2.5,
            ),
            mo.md(r"""
    <small>Example adapted from Sebastian Raschka, [From Bag-of-Words to Jev](https://magazine.sebastianraschka.com/p/classifier-history-and-jev)</small>
    """),
        ],
        gap=0.75,
    )
    return

@app.cell
def _(mo):
    temperature = mo.ui.slider(0.5, 5.0, step=0.1, value=1.0, label="temperature")
    return (temperature,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Where decision models might fail

    They're still **smaller models**, with a weaker **world model**:

    - **Ambiguous, long-context reasoning**: read a whole murder mystery novel, *who is the murderer?*
    - **Stochastic events**: *"Will it rain 3 weeks from now?"*
    - **Calibration**: *"Simulate a coin flip / a skewed die"*, ask 1,000 times: do the probabilities match?

    So how good are the open ones?
    """)
    return


@app.cell
def _(base64, mo):
    # Leaderboard screenshot (cropped to the chart); clicking it opens the live page.
    _img = base64.b64encode(mo.watch.file("public/figures/jev-decision-index.png").read_bytes()).decode()
    mo.vstack(
        [
            mo.md(r"""
    ## Everyone can build one: the Jev Decision Index

    **10+** open models already score above Jev
    """),
            mo.Html(
                '<a href="https://huggingface.co/spaces/multimodalart/jev-decision-index" target="_blank">'
                f'<img src="data:image/png;base64,{_img}" '
                'style="display:block;margin:0 auto;max-height:36cqw;max-width:100%;border-radius:8px"></a>'
            ),
            mo.md(r"""
    [huggingface.co/spaces/multimodalart/jev-decision-index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
    """).center(),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bring your own Jev

    Two ways to build one, live:

    1. **Any LLM → Jev**: structured output + logprobs, through OpenRouter
        - Caveat: not every model exposes logprobs today, **especially closed models**
    2. **A tiny local model, fully offline**: read the answer straight off the logits of a small Qwen

    Then we throw some questions at them and see how they do.

    **Demo time.**
    """)
    return


if __name__ == "__main__":
    app.run()
