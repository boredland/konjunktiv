# Konjunktiv-T5 run log

synthesis pass rate: 0.950 (37,378 of 39,340 synthesized pairs kept; gate 0.6, no few-shot revision needed)
kept pairs: tatoeba 34,077 / template 3,301 (template share 0.088); splits train 34,406 / val 1,000 / test 1,000

data pipeline deviations from plan:
- Tatoeba candidates after filters: 130,527; capped to a deterministic sample of 36,000 (random.Random(42)) to bound LLM synthesis cost. Filters unchanged.
- Quality filter: judge_batch() stalled harness-side (batches stayed running with done=0, cost=0; even a 2-state probe never settled), so judging ran through completion() with the same rules 1–8 as a strict boolean verdict (p_ok 1.0/0.0, rows tagged judge="completion"). The KEEP_THRESHOLD of 0.8 applies unchanged. The first 64 pairs were judged by judge_batch before the stall (pass rate 0.875 on that sample).

environment: tools via mise (mise.toml: python 3.12, node 24, uv); Python deps via uv (pyproject.toml, torch from the PyTorch ROCm 7.2 index). Training runs on the Radeon 890M iGPU (gfx1150): 0.47 s per optimizer step vs ~1.1 s on 12 CPU threads. Run scripts with `uv run python scripts/<name>.py`.

training config deviation: torch.set_num_threads(12) instead of 24. The CPU has 12 physical cores with SMT; 24 threads measured 26.5 s per optimizer step vs 0.66 s with 12 (flan-t5-small, batch 16, grad-accum 2). At 24 threads the first run projected ~25 h.

web model deviation: fp32 encoder + q8 merged decoder (195 MB) instead of q8/q8 (94 MB). Measured in Transformers.js on test[:300]: q8/q8 0.797 exact, fp32/q8 0.863 (= fp32/fp32), q8/fp32 0.807. The quantized encoder swapped rare words ("Einblick" → "Überblick"). Deployed at https://konjunktiv.jonas-strassel.de; files >20 MiB are split by scripts/build_site.mjs (Workers asset limit 25 MiB) and reassembled by worker/index.js.
web/model size: 195M

## Round 2 (retrain for rare verbs, separable Präteritum, capital umlauts)

tokenizer: scripts/build_tokenizer.py adds ▁Ä ▁Ö Ä Ö ẞ as real SentencePiece pieces (out/tokenizer, 32105 tokens). Added-token variants round-trip in Python but Transformers.js inserts a word break after them ("Ö sterreich"). New embedding rows start from similar existing pieces (▁Ö ← mean(▁O, ö)); a shared mean init left ▁Ä/▁Ö at cosine 0.99997 and the model wrote "Äkologie". Sentinels are moved by name (T5 numbers them in reverse).
targeted sources (scripts/add_targeted_sources.py, appended, round-1 ids/speakers/splits unchanged): 1,916 unused Tatoeba sentences matching separable-Präteritum / -ierte / capital Ä-Ö patterns, plus 1,350 sentences written by the agy bridge (Gemini) in data/raw/agy_sentences*.json.
judge: first-pass rejects of the new pairs were re-judged twice; kept on 2/2 ok (78 of 135 and 96 of 167 flipped). Round-1 pairs were not re-judged.
kept pairs: 40,516 of 42,606 (tatoeba 35,937 / template 3,301 / agy 1,278); train 37,295, val and test unchanged (1,000 each, all round-1 rows).
decoder quantization: per-channel q8 with lm_head kept fp32 (109 MB decoder, web/model 242 MB). Per-tensor q8 turned "ausgereicht" into "ausgebrochen"; per-channel q8 including lm_head turned "beschlagnahmt" into "beschlaggenommen". Browser WASM on test[:300]: per-channel 0.870 / regression 10/12, per-channel + fp32 lm_head 0.873 / 11/12 (same as the full fp32 decoder at 233 MB).

round-2 metrics (PyTorch, out/metrics.json): test_exact_match 0.897 (round 1: 0.899), indicative leak 0.0, challenge 20/20, regression set (data/regression_round2.jsonl) 11/12 (round 1: 4/12). Remaining miss: "Wir kauften im Supermarkt ein." → "ein gekauft" (eingekauft appears in 5 training targets).
web/model size: 242M
