# Workflow

Slides and demo are one marimo notebook: `slides.py` (cells) + `layouts/slides.slides.json` (slide types, speaker notes; matched to cells by order).

## Setup (once)

Only [uv](https://docs.astral.sh/uv/) is needed.

```bash
git clone git@github.com:Onix-AI/tiny-labs-byoj.git
cd tiny-labs-byoj
```

## Edit

```bash
uvx marimo edit --sandbox slides.py
```

- Cmd-K → "Slides" for the deck view. Arrows: left/right = main path, up/down = sub-slides.
- Saving in the browser writes `slides.py` and `layouts/`. If the files change on disk, marimo prompts to reload.
- `S` opens speaker view.

## Present / export

```bash
uvx marimo run --sandbox slides.py                # present
uvx marimo export pdf slides.py --as=slides   # PDF
```

## Version

Commit `slides.py` and `layouts/` together, at small checkpoints:

```bash
git add slides.py layouts/ && git commit -m "slides 1-3: draft" && git push
```

New dependencies go in the `# /// script` header at the top of `slides.py`; `--sandbox` installs them in an isolated env.
