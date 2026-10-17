#!/usr/bin/env bash
# Publish one training round: ONNX export + quantize, parity smoke, site deploy, live byte check,
# commit + push, GitHub release with the web model archive.
#   scripts/release_round.sh <round-number> "<release notes>"
# Expects the trained checkpoint and its entry in out/rounds.json (the page's training statistics are built
# from that file). A round with "variant": "large" in rounds.json is flan-t5-large: its checkpoint is
# out/konjunktiv-t5-large.round<N> and it ships as web/model-large next to the default web/model.
# Every network/long step has a hard timeout and no TTY input: an interactive pager once blocked a
# release for 40 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

round="$1"
notes="$2"
tag="model-round${round}"
site="https://konjunktiv.jonas-strassel.de"
archive="/tmp/konjunktiv-model-round${round}.tar.gz"
export GIT_PAGER=cat PAGER=cat GH_PAGER=cat GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0

node -e "const r=require('./out/rounds.json'); if(!r.some(x=>x.round===${round})) process.exit(1)" \
  || { echo "add round ${round} to out/rounds.json first" >&2; exit 1; }
variant=$(node -p "require('./out/rounds.json').find(x=>x.round===${round}).variant ?? 'base'")
if [ "$variant" = large ]; then
  model="out/konjunktiv-t5-large.round${round}" web_dir="web/model-large" base_name="flan-t5-large"
else
  model="out/konjunktiv-t5.round${round}" web_dir="web/model" base_name="flan-t5-base"
fi
[ -d "$model" ] || { echo "missing $model" >&2; exit 1; }
# Put an already released round back live (after a regressing round): deploy and commit, keep its release.
redeploy=false
timeout 120 gh release view "$tag" < /dev/null > /dev/null 2>&1 && redeploy=true
timeout 3000 bash scripts/export_onnx.sh "$model" "$web_dir" < /dev/null
# The parity reference is the PyTorch checkpoint's own challenge predictions. The heredoc is python's stdin,
# so this step needs no `< /dev/null` (a second redirect would replace the script with an empty one).
reference="/tmp/konjunktiv-pred-challenge-round${round}.jsonl"
timeout 1200 uv run python - "$model" "$reference" <<'EOF'
import json, sys
sys.path.insert(0, "scripts")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from eval_model import generate
from train import load_rows
ckpt, out = sys.argv[1:3]
rows = load_rows("data/challenge_20.jsonl")
preds = generate(AutoModelForSeq2SeqLM.from_pretrained(ckpt), AutoTokenizer.from_pretrained(ckpt), [r["source"] for r in rows])
with open(out, "w", encoding="utf-8") as fh:
    for r, p in zip(rows, preds):
        fh.write(json.dumps({"source": r["source"], "pred": p}, ensure_ascii=False) + "\n")
EOF
[ -s "$reference" ] || { echo "no PyTorch reference at $reference" >&2; exit 1; }
timeout 1200 node scripts/smoke.mjs "${web_dir#web/}" "$reference" < /dev/null
rm -f "$reference"

timeout 600 npm run build < /dev/null
timeout 1800 wrangler deploy < /dev/null
# Right after a deploy some edge locations still serve the previous chunks (round 13 saw the old encoder once,
# identical a minute later), so retry for up to 5 minutes before calling the deploy broken.
live_ok() {
  for f in onnx/encoder_model.onnx onnx/decoder_model_merged_quantized.onnx tokenizer.json; do
    timeout 900 curl -sSf -o /tmp/konj-live.bin "$site/${web_dir#web/}/$f" < /dev/null || return 1
    cmp -s /tmp/konj-live.bin "$web_dir/$f" || { echo "live $f differs from $web_dir/$f" >&2; return 1; }
  done
}
for attempt in 1 2 3 4 5 6; do
  live_ok && break
  [ "$attempt" = 6 ] && { rm -f /tmp/konj-live.bin; exit 1; }
  sleep 60
done
rm -f /tmp/konj-live.bin
echo "live model files identical"

git add -A README.md .gitignore data out/README.md out/metrics*.json out/rounds.json scripts web
git diff --cached --quiet || git commit -q -m "Round ${round}: ${notes%%.*}"
timeout 300 git push -q < /dev/null

if $redeploy; then
  echo "redeployed existing release $tag"
else
  timeout 900 tar -C "$(dirname "$web_dir")" -czf "$archive" "$(basename "$web_dir")"
  timeout 3000 gh release create "$tag" "$archive" --title "Model round ${round} (${base_name})" --notes "$notes" < /dev/null
  rm -f "$archive"
fi
timeout 120 gh release view "$tag" --json assets -q '.assets[] | "\(.name) \(.size) \(.state)"' < /dev/null
