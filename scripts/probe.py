"""Run the PyTorch checkpoint on hand-written probe files and list every miss.

Usage: uv run python scripts/probe.py [data/probe_*.jsonl ...]  (default: all probe + regression files)
"""
import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from eval_model import generate, normalize
from train import OUTPUT_DIR, load_rows

DEFAULT = sorted(Path("data").glob("probe_*.jsonl")) + [Path("data/regression_round2.jsonl")]


def main():
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULT
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    total = hits = 0
    for path in paths:
        rows = load_rows(path)
        preds = generate(model, tokenizer, [r["source"] for r in rows])
        misses = [(r, p) for r, p in zip(rows, preds) if normalize(p) != normalize(r["target"])]
        total += len(rows)
        hits += len(rows) - len(misses)
        print(f"{path}: {len(rows) - len(misses)}/{len(rows)}")
        for r, p in misses:
            print(f"  {r['source']}\n    got:  {p}\n    want: {r['target']}")
    print(f"TOTAL {hits}/{total}")


if __name__ == "__main__":
    main()
