#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run optimum-cli export onnx --task text2text-generation-with-past --model out/konjunktiv-t5 out/onnx-fp32
uv run python scripts/quantize.py
