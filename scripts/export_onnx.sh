#!/usr/bin/env bash
# Export a fine-tuned checkpoint to the web layout.
#   scripts/export_onnx.sh [checkpoint] [web_model_dir]
# Defaults: out/konjunktiv-t5 → web/model; the large model is out/konjunktiv-t5-large → web/model-large.
set -euo pipefail

cd "$(dirname "$0")/.."

checkpoint="${1:-out/konjunktiv-t5}"
web_dir="${2:-web/model}"
export_dir="out/onnx-fp32${web_dir#web/model}"

rm -rf "$export_dir"
uv run optimum-cli export onnx --task text2text-generation-with-past --model "$checkpoint" "$export_dir"
uv run python scripts/quantize.py "$export_dir" "$web_dir"
