#!/usr/bin/env bash
# Publish one training round: ONNX export + quantize, parity smoke, site deploy, live byte check,
# commit + push, GitHub release with the web model archive.
#   scripts/release_round.sh <round-number> "<release notes>"
# Expects the trained checkpoint at out/konjunktiv-t5.round<N> and its entry in out/rounds.json (the page's
# training statistics are built from that file).
# Every network/long step has a hard timeout and no TTY input: an interactive pager once blocked a
# release for 40 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

round="$1"
notes="$2"
model="out/konjunktiv-t5.round${round}"
tag="model-round${round}"
site="https://konjunktiv.jonas-strassel.de"
archive="/tmp/konjunktiv-model-round${round}.tar.gz"
export GIT_PAGER=cat PAGER=cat GH_PAGER=cat GH_PROMPT_DISABLED=1 GIT_TERMINAL_PROMPT=0

[ -d "$model" ] || { echo "missing $model" >&2; exit 1; }
node -e "const r=require('./out/rounds.json'); if(!r.some(x=>x.round===${round})) process.exit(1)" \
  || { echo "add round ${round} to out/rounds.json first" >&2; exit 1; }
# Put an already released round back live (after a regressing round): deploy and commit, keep its release.
redeploy=false
timeout 120 gh release view "$tag" < /dev/null > /dev/null 2>&1 && redeploy=true
rm -rf out/onnx-fp32
timeout 1800 uv run optimum-cli export onnx --task text2text-generation-with-past --model "$model" out/onnx-fp32 < /dev/null
timeout 1200 uv run python scripts/quantize.py < /dev/null
timeout 600 node scripts/smoke.mjs < /dev/null

timeout 600 npm run build < /dev/null
timeout 1800 wrangler deploy < /dev/null
# Right after a deploy some edge locations still serve the previous chunks (round 13 saw the old encoder once,
# identical a minute later), so retry for up to 5 minutes before calling the deploy broken.
live_ok() {
  for f in onnx/encoder_model.onnx onnx/decoder_model_merged_quantized.onnx tokenizer.json; do
    timeout 600 curl -sSf -o /tmp/konj-live.bin "$site/model/$f" < /dev/null || return 1
    cmp -s /tmp/konj-live.bin "web/model/$f" || { echo "live $f differs from web/model/$f" >&2; return 1; }
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
  timeout 600 tar -C web -czf "$archive" model
  timeout 3000 gh release create "$tag" "$archive" --title "Model round ${round} (flan-t5-base)" --notes "$notes" < /dev/null
  rm -f "$archive"
fi
timeout 120 gh release view "$tag" --json assets -q '.assets[] | "\(.name) \(.size) \(.state)"' < /dev/null
