#!/usr/bin/env bash
# Rebuild the static HTML copies of the slides and the demo in static/.
# Run it after any change to slides.py or demo.py, and commit the result.
# The demo export runs every cell for real: it needs OPENROUTER_API_KEY in .env
# and downloads Qwen3-1.7B on the first run.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p static
uvx marimo export html --sandbox slides.py -o static/slides.html
DEMO_AUTORUN=1 uvx marimo export html --sandbox demo.py -o static/demo.html
ls -lh static/*.html
