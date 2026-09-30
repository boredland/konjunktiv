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

## Rounds 3–6 (audit-driven fix loop)

Each round: hand-written probe sets with gold targets (data/probe_separable.jsonl, data/probe_subclause.jsonl), an LLM-judged audit of 600 unseen Tatoeba sentences (ids in data/audit_ids.json, never used as sources), new targeted sources for the error classes found, relabelling of inconsistent existing pairs, retrain.
round 3 (flan-t5-small): rule 9 (every finite verb in subordinate clauses converted). 1,877 of 2,058 changed subordinate-clause labels passed a 2/2 re-judge. test 0.905, probes: separable 37/40, subclause 14/24.
round 4 (flan-t5-small): rules 10–11 (dative uns/mir → ihnen/ihm/ihr, never sich; modal verbs in subordinate clauses; sein-Perfekt for verbs of motion), 900 agy sentences. test 0.913, separable 35/40, subclause 18/24. Remaining errors were invented participles ("eingelud", "zusammengewerkt"), so the base model was switched.
round 5 (flan-t5-base, 248M, 1.9 s/step on the iGPU, 2 h 6 min): val loss 0.0309 (small: 0.0508). test 0.946, separable 39/40, subclause 23/24, regression 10/12: both misses moved a fronted adverbial behind the pronoun ("Er habe vom 1. bis …"). 131 training pairs did the same, contradicting rule 8 and 1,007 consistent pairs; 123 were reordered deterministically (the label with only the fronted phrase restored), 7 dropped.
round 6 (flan-t5-base): 800 agy sentences (weak -r/-l/-hl stems, wir + weak Präsens, mögen/sollen in dependent clauses, dative mir/uns with gehören/gehen/rühren, colloquial "ich hol/hab", Präteritum relative clauses); 784 kept. kept pairs 46,912 of 49,159, train 43,205. The first run died at epoch 2 (harness job lost); train.py now resumes from the last epoch checkpoint. val loss 0.0297.
judge: since round 4 the judge fails schema validation on 20-row audit batches; scripts/audit_judge.py now uses 4-row batches and a 3-vote majority (single verdicts: 46 of 76 round-4 flags had an empty or identical fix).

round-6 metrics (PyTorch): test_exact_match 0.941 (round 5: 0.946), indicative leak 0.0, challenge 20/20, regression 12/12, separable 38/40, subclause 24/24. Audit majority flags 12/546 in-scope (0.022; round 5: 16). Remaining: "Wir kaufen jeden Freitag ein." → "würden … einkaufen" (würde-form instead of the -te form), a dropped repeated auxiliary in coordination, rare invented participles ("aufgehinget").
web model (Transformers.js, same dtypes as the site): identical probe results to PyTorch (94/96), ~200 ms/sentence in Node.
web/model size: 650M (encoder fp32 439 MB, decoder q8 per-channel with fp32 lm_head 239 MB)

## Round 7 (user-reported interview sentences)

regression set: the three user sentences (gegenübergesessen, schwergefallen, der Atem gestockt) added to data/regression_round2.jsonl; two generated sources overlapping them were held out of training.
targeted sources: 700 agy sentences (data/raw/agy_sentences7.json): Präteritum with dative experiencer mir/uns and rare verbs (stocken, versagen, schwinden …), "es fiel mir schwer / tat mir leid", dative position verbs (gegenübersitzen, zuhören), wir + separable weak Präsens, coordinated verbs needing their own auxiliary. 94 more pronoun-first labels after a fronted phrase were reordered, 3 dropped. train 43,826.
round-7 metrics (PyTorch): val loss 0.0289, test_exact_match 0.944, challenge 20/20, regression 15/15, separable 40/40, subclause 23/24 ("klinge" for "klingle"). Audit majority flags 7/548 (0.013). Web model identical to PyTorch on all 99 probe rows.
remaining audit flags: indicative left in a few subordinate/infinitive constructions ("neigen dazu", "der sich gegen sie richtet"), rare invented forms ("ausblierten", "erschlage" for "ergreife", "aufgewachsen sei" for "aufgehängt habe").
