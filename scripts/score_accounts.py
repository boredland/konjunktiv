"""Predict the 393 held-out account sentences (data/probe_hearings.json, data/probe_police.json) for a checkpoint.

Writes one JSONL row per sentence with a stable id, so audit_judge.audit() can grade it and two checkpoints can be
diffed sentence by sentence. Sentences are split and filtered exactly like the training narratives
(add_targeted_sources.py), which is how the 393 in out/rounds.json were counted.
    uv run python scripts/score_accounts.py out/konjunktiv-t5.round17 out/accounts_round17.jsonl
"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from collect_sentences import FORBIDDEN, RE_SECOND
from eval_model import generate

ACCOUNT_FILES = (Path("data/probe_hearings.json"), Path("data/probe_police.json"))
SENTENCE_END = re.compile(r"(?<=[.!])\s+(?=[A-ZÄÖÜ])")


def account_rows():
    rows = []
    for path in ACCOUNT_FILES:
        for narrative in json.loads(path.read_text(encoding="utf-8")):
            sentences = [s.strip() for p in narrative["text"].split("\n\n") for s in SENTENCE_END.split(p) if s.strip()]
            for i, text in enumerate(sentences):
                if 4 <= len(text.split()) <= 40 and text.endswith(".") and not (set(text) & FORBIDDEN) and not RE_SECOND.search(text):
                    rows.append({"id": f"{narrative['id']}-{i}", "direct": text, "speaker": narrative["speaker"]})
    return rows


def main():
    checkpoint, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    rows = account_rows()
    preds = generate(model, tokenizer, [f"sprecher: {r['speaker']} | {r['direct']}" for r in rows])
    with out_path.open("w", encoding="utf-8") as out:
        for r, p in zip(rows, preds):
            out.write(json.dumps({**r, "indirect": p}, ensure_ascii=False) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
