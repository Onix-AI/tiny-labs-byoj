# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "accelerate",
#     "httpx",
#     "huggingface_hub",
#     "openai",
#     "pillow",
#     "python-dotenv",
#     "qev[torch] @ git+https://github.com/OmniJev/OneJev.git@6f1b57c4105312cfa71aa2ac0415deab918f4284",
#     "safetensors",
#     "torch",
#     "torchvision",
#     "transformers>=5.10",
# ]
# ///
"""Reproduce the results of the blog post "Is your decision model calibrated?".

Asks five decision models two questions with known answers:

  coin   "I flip a fair coin. How does it land?", once with heads listed first, once with tails first
  digit  "Pick a random digit between 0 and 9."

and writes every model's probabilities to a JSON file, the one the post's interactive figures read.

    uv run blog/calibration.py                        # all models -> blog/results.json
    uv run blog/calibration.py --only qwen onejev     # a subset (merged into an existing file)

Backends:
  qwen      Qwen3-1.7B, local, probability of each option's digit token (same as demo.py)
  gpt       openai/gpt-4o-mini via OpenRouter, logprobs of one token (same as demo.py)
  jev       Jev via the OpenRouter Decisions API
  onejev    OneJev1.5-9B, local, with the model's own qev engine
  clef      Cloudflare clef-flash (9B), local, with the model's own joint_schema_model.py

gpt and jev need OPENROUTER_API_KEY (in .env or the environment). The local models download
~3.4 GB (Qwen) and ~19 GB each (OneJev, clef-flash) and run one at a time, on CUDA, Apple Silicon
(mps) or CPU. The hosted models can change behind the API, so their numbers may drift a little.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Pinned revisions: the exact weights used for the post.
QWEN = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
ONEJEV = ("OmniJev/OneJev1.5-9B", "ec2f276eab207810f576fb1f2d68c2acabd3eb71")
CLEF = ("Cloudflare/clef-flash", "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3")

# (key, state, instructions, options)
QUESTIONS = [
    ("heads_first", "I flip a fair coin.", "How does it land?", ["heads", "tails"]),
    ("tails_first", "I flip a fair coin.", "How does it land?", ["tails", "heads"]),
    ("digit", "Pick a random digit between 0 and 9.", "Pick a random digit.", [str(i) for i in range(10)]),
]


# The LLM backends read a decision from a single token, exactly like demo.py.
def prompt(state: str, instructions: str, options: list[str]) -> str:
    listed = "\n".join(f"{i}) {option}" for i, option in enumerate(options))
    return f'Message:\n"""\n{state}\n"""\n\n{instructions}\n{listed}\n\nAnswer with a single digit.'


def softmax(options: list[str], logits: list[float]) -> dict[str, float]:
    exps = [math.exp(x) for x in logits]
    return {option: e / sum(exps) for option, e in zip(options, exps)}


def request(state: str, instructions: str, options: list[str], model: str) -> dict:
    """A Jev / System One request body with one choice question."""
    return {
        "model": model,
        "state": state,
        "questions": {
            "answer": {"type": "choice", "instructions": instructions, "criteria": {o: o for o in options}}
        },
    }


def qwen():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN[0], revision=QWEN[1])
    model = AutoModelForCausalLM.from_pretrained(QWEN[0], revision=QWEN[1], dtype=torch.float16).to(DEVICE).eval()

    @torch.no_grad()
    def ask(state, instructions, options):
        messages = [{"role": "user", "content": prompt(state, instructions, options)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        logits = model(**tokenizer(text, return_tensors="pt").to(DEVICE)).logits[0, -1]
        digits = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(len(options))]
        return softmax(options, logits[digits].tolist())

    return ask


def gpt():
    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

    def ask(state, instructions, options):
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt(state, instructions, options)}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
            extra_body={"provider": {"require_parameters": True}},  # only providers that return logprobs
        )
        top = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
        return softmax(options, [top.get(str(i), -math.inf) for i in range(len(options))])

    return ask


def jev():
    def ask(state, instructions, options):
        response = httpx.post(
            "https://openrouter.ai/api/alpha/decisions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=request(state, instructions, options, "~typesafe/jev-latest"),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["answers"]["answer"]["probabilities"]

    return ask


def onejev():
    from qev.cli import _find_calibration
    from qev.mm_engine import MMDecisionEngine, local_model_dir
    from qev.schema import SystemOneRequest
    from qev.calibrate import Calibration
    from huggingface_hub import snapshot_download

    path = snapshot_download(ONEJEV[0], revision=ONEJEV[1])
    calibration_path = _find_calibration(local_model_dir(path))
    engine = MMDecisionEngine(
        path,
        device=DEVICE,
        dtype="bfloat16",
        calibration=Calibration.load(calibration_path) if calibration_path else None,
        cuda_graphs=DEVICE == "cuda",
    )

    def ask(state, instructions, options):
        body = request(state, instructions, options, "OneJev1.5-9B")
        response, _ = engine.decide(SystemOneRequest.model_validate(body), debias=1)  # qev serve's default
        return response.model_dump()["answers"]["answer"]["probabilities"]

    return ask


def clef():
    from huggingface_hub import snapshot_download

    path = snapshot_download(CLEF[0], revision=CLEF[1])
    sys.path.insert(0, path)
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(path, device=DEVICE)

    def ask(state, instructions, options):
        return systemone(model, processor, request(state, instructions, options, "clef-flash"))["answers"]["answer"][
            "probabilities"
        ]

    return ask


BACKENDS = {"qwen": ("Qwen3-1.7B", qwen), "gpt": ("GPT-4o mini", gpt), "jev": ("Jev", jev),
            "onejev": ("OneJev1.5-9B", onejev), "clef": ("Clef-Flash", clef)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(BACKENDS), default=list(BACKENDS))
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    results.setdefault("coin", {"prompt": "I flip a fair coin. How does it land?", "models": []})
    results.setdefault("digit", {"prompt": "Pick a random digit between 0 and 9.", "models": []})
    results["date"] = time.strftime("%Y-%m-%d")

    for key in args.only:
        name, load = BACKENDS[key]
        print(f"{name}: loading", flush=True)
        ask = load()
        answers = {q: ask(state, instructions, options) for q, state, instructions, options in QUESTIONS}
        coin = {"name": name, "headsFirst": round(answers["heads_first"]["heads"], 4),
                "tailsFirst": round(answers["tails_first"]["heads"], 4)}
        digit = {"name": name, "probabilities": [round(answers["digit"][str(i)], 4) for i in range(10)]}
        for section, row in (("coin", coin), ("digit", digit)):
            rows = [r for r in results[section]["models"] if r["name"] != name]
            results[section]["models"] = rows + [row]
        print(f"{name}: P(heads) {coin['headsFirst']:.2f} heads first, {coin['tailsFirst']:.2f} tails first;"
              f" digits {digit['probabilities']}", flush=True)
        del ask
        gc.collect()  # free one local model before loading the next
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()

    order = [name for name, _ in BACKENDS.values()]
    for section in ("coin", "digit"):
        results[section]["models"].sort(key=lambda r: order.index(r["name"]) if r["name"] in order else len(order))
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
