# Konjunktiv – direkte Rede → indirekte Rede, offline im Browser

**Demo: https://konjunktiv.jonas-strassel.de**

A small fine-tuned language model that rewrites German first-person statements into reported speech
(indirekte Rede, Konjunktiv I with Konjunktiv II fallback):

```
sprecher: Er | Ich bin 18 Jahre alt und in Nürnberg geboren
→ Er sei 18 Jahre alt und sei in Nürnberg geboren worden.
```

The model runs entirely client-side with [Transformers.js](https://github.com/huggingface/transformers.js) and
ONNX Runtime Web (WASM). The site is static files only; no text is sent anywhere. After the first load the
browser keeps the model in Cache Storage and the page works offline.

## How it works

| Step | What | Script |
|---|---|---|
| Sources | ~49k German `ich`/`wir` sentences from [Tatoeba](https://tatoeba.org) plus templated sentences for every tense and LLM-written sentences for hard cases (separable verbs, rare verbs, subordinate clauses, dative pronouns, capital umlauts) | `collect_sentences.py`, `templates.py`, `add_targeted_sources.py` |
| Labels | An LLM rewrites every sentence into indirect speech under fixed rules (`RULES` in `synthesize_pairs.py`) | `synthesize_pairs.py` (run in an agent kernel with `completion()`) |
| Filter | A second LLM pass judges every pair against the same rules; rejects are re-checked twice | `synthesize_pairs.py` |
| Tokenizer | Flan-T5's SentencePiece vocab lacks capital Ä/Ö/ẞ; they are added as real pieces | `build_tokenizer.py` |
| Train | Fine-tune [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (248M params), 3 epochs; the optional larger model is [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) (783M) on the same data. `KONJUNKTIV_BASE_MODEL`, `KONJUNKTIV_TOKENIZER`, `KONJUNKTIV_OUTPUT_DIR` and `KONJUNKTIV_SEED` select base, tokenizer, output and seed | `train.py` |
| Average | The published model is the uniform weight average of the last four rounds; single rounds swing (each fixes its targets and breaks other cases), the average keeps the gains | `average_models.py` |
| Evaluate | Exact match on 1,000 held-out pairs, hand-written challenge/probe sets, LLM-judged audit on unseen Tatoeba sentences and on 22 held-out fictional asylum-hearing and police-interview accounts; the shipped ONNX models are scored on the same test set | `eval_model.py`, `probe.py`, `audit_predict.py`, `score_accounts.py`, `audit_judge.py`, `eval_web.mjs` |
| Export | ONNX export; decoder int8 per-channel with the output projection kept fp32, encoder fp32 | `export_onnx.sh`, `quantize.py` |
| Web | Static page with two models to choose from: `web/model/` (flan-t5-base, default, ~680 MB) and `web/model-large/` (flan-t5-large, opt-in, ~1.9 GB). Runtime vendored, files > 20 MiB split into `/chunks/` for Cloudflare Workers assets and reassembled by a tiny Worker; `_headers` makes the page cross-origin isolated so ONNX Runtime can use WASM threads; training statistics and model sizes on the page come from `out/rounds.json` and the model folders | `vendor_runtime.mjs`, `build_site.mjs`, `worker/index.js` |
| Publish | Export, deploy, live byte check, commit and GitHub release per round | `release_round.sh` |

Run log, measured numbers and every deviation from the original plan: [`out/README.md`](out/README.md).

## Reproduce

Tools come from [mise](https://mise.jdx.dev) (`mise.toml`), Python dependencies from [uv](https://docs.astral.sh/uv/)
(`pyproject.toml`; torch is pinned to the ROCm 7.2 wheels for the AMD Radeon 890M iGPU the base rounds trained on —
switch the `pytorch-rocm` index to `https://download.pytorch.org/whl/cpu` or a CUDA index for other machines).
Round 19 (flan-t5-large) trained on an AMD Instinct MI50 (gfx906), which needs the ROCm 6.3 wheels
(`torch==2.9.1+rocm6.3`); ROCm 7.x builds ship no kernels for that card.

```bash
mise install && uv sync && npm install
uv run python scripts/collect_sentences.py
# label + filter: see the docstring of scripts/synthesize_pairs.py (needs an LLM completion() helper)
uv run python scripts/build_tokenizer.py
uv run python scripts/check_data.py
uv run python scripts/train.py
uv run python scripts/eval_model.py
bash scripts/export_onnx.sh          # → web/model/
npm run smoke                         # Node parity check against the PyTorch predictions
npm run build                         # → dist/
npm run deploy                        # Cloudflare Workers (wrangler.jsonc)
```

The large model: train with `KONJUNKTIV_BASE_MODEL=google/flan-t5-large KONJUNKTIV_OUTPUT_DIR=out/konjunktiv-t5-large`,
export with `bash scripts/export_onnx.sh out/konjunktiv-t5-large web/model-large`. `release_round.sh` handles a round
whose `out/rounds.json` entry has `"variant": "large"` the same way.

The trained ONNX models are attached to the GitHub releases, so the site can be rebuilt without training:
unpack the base archive into `web/model/` and the large one into `web/model-large/`, then run `npm run build`.

## Scope

- Speakers `Er` and `Sie` only; `wir` becomes plural `sie`.
- Tenses: Präsens, Perfekt, Präteritum, Futur I, `geboren`.
- Not supported: questions, imperatives, direct address (`du`/`ihr`/`Sie`), shifting deictic words (`gestern` stays `gestern`).
- Like any model it makes mistakes on rare words and long sentences; proofread the output.

## License

Code: [MIT](LICENSE) © Jonas Strassel.

Third-party material has its own terms and is **not** covered by the MIT license:

- Base model `google/flan-t5-base`: Apache-2.0. The fine-tuned weights are a derivative and are distributed under Apache-2.0.
- Tatoeba sentences in `data/`: CC BY 2.0 FR, © Tatoeba contributors (https://tatoeba.org).
- Transformers.js and ONNX Runtime Web (vendored into the built site): Apache-2.0 and MIT respectively.
