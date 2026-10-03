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
| Train | Fine-tune [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) (248M params), 3 epochs; `KONJUNKTIV_BASE_MODEL=google/flan-t5-small` for a faster, weaker run | `train.py` |
| Average | The published model is the uniform weight average of the last four rounds; single rounds swing (each fixes its targets and breaks other cases), the average keeps the gains | `average_models.py` |
| Evaluate | Exact match on 1,000 held-out pairs, hand-written challenge/probe sets, LLM-judged audit on unseen Tatoeba sentences and on 22 held-out fictional asylum-hearing and police-interview accounts | `eval_model.py`, `probe.py`, `audit_predict.py`, `audit_judge.py` |
| Export | ONNX export; decoder int8 per-channel with the output projection kept fp32, encoder fp32 | `export_onnx.sh`, `quantize.py` |
| Web | Static page; runtime vendored, files > 20 MiB split for Cloudflare Workers assets and reassembled by a tiny Worker; training statistics on the page come from `out/rounds.json` | `vendor_runtime.mjs`, `build_site.mjs`, `worker/index.js` |
| Publish | Export, deploy, live byte check, commit and GitHub release per round | `release_round.sh` |

Run log, measured numbers and every deviation from the original plan: [`out/README.md`](out/README.md).

## Reproduce

Tools come from [mise](https://mise.jdx.dev) (`mise.toml`), Python dependencies from [uv](https://docs.astral.sh/uv/)
(`pyproject.toml`; torch is pinned to the ROCm 7.2 wheels because training ran on an AMD Radeon 890M iGPU —
switch the `pytorch-rocm` index to `https://download.pytorch.org/whl/cpu` or a CUDA index for other machines).

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

The trained ONNX model is attached to the GitHub releases, so the site can be rebuilt without training:
unpack the release archive into `web/model/` and run `npm run build`.

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
