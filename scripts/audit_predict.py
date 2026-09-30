"""Predict on held-out Tatoeba sentences never used as sources, for an LLM-judged audit.

The sampled ids are written to data/audit_ids.json and add_targeted_sources.py skips them, so the
audit stays held out across retraining rounds. Output: out/audit_preds.jsonl.
Usage: uv run python scripts/audit_predict.py [N]
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from collect_sentences import OUT as SOURCES, filter_sentences, normalize
from eval_model import generate
from train import OUTPUT_DIR

AUDIT_IDS = Path("data/audit_ids.json")
PREDS = Path("out/audit_preds.jsonl")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    if AUDIT_IDS.exists():
        audit = json.loads(AUDIT_IDS.read_text())
    else:
        used = {json.loads(l)["id"] for l in SOURCES.open(encoding="utf-8")}
        pool = [(sid, normalize(t)) for sid, _, t in filter_sentences(5, 25) if sid not in used]
        rng = random.Random(7)
        audit = [{"id": sid, "direct": t, "speaker": rng.choice(["Er", "Sie"])} for sid, t in rng.sample(pool, n)]
        AUDIT_IDS.write_text(json.dumps(audit, ensure_ascii=False, indent=0))
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    preds = generate(model, tokenizer, [f"sprecher: {a['speaker']} | {a['direct']}" for a in audit])
    with PREDS.open("w", encoding="utf-8") as out:
        for a, p in zip(audit, preds):
            out.write(json.dumps({**a, "indirect": p}, ensure_ascii=False) + "\n")
    print(f"wrote {PREDS} ({len(audit)} rows)")


if __name__ == "__main__":
    main()
