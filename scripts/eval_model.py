"""Evaluate the fine-tuned model on test + challenge; gate with one continuation run."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train import (
    MAX_LENGTH,
    OUTPUT_DIR,
    TRAIN_PATH,
    VAL_PATH,
    append_readme,
    drop_over_length,
    load_rows,
    train_round,
)

TEST_PATH = Path("data/test.jsonl")
CHALLENGE_PATH = Path("data/challenge_20.jsonl")
# user-reported failures (rare verbs, separable Präteritum, capital umlauts); reported, not gated
REGRESSION_PATH = Path("data/regression_round2.jsonl")
METRICS_PATH = Path("out/metrics.json")
PRED_CHALLENGE_PATH = Path("out/pred_challenge.jsonl")

INDICATIVE_FORMS = {
    "ist",
    "hat",
    "wird",
    "war",
    "hatte",
    "kann",
    "muss",
    "will",
    "geht",
    "kommt",
    "sind",
    "haben",
    "werden",
}
SUBJECT_PRONOUNS = {"er", "sie", "wir"}
GEN_BATCH = 16


def normalize(text):
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    return " ".join(text.split())


def first_verb_after_subject(output):
    # The reported clause is verb-second, so the finite verb sits right after the subject.
    words = re.findall(r"\w+", output.lower())
    if len(words) > 1 and words[0] in SUBJECT_PRONOUNS:
        return words[1]
    return words[0] if words else ""


def generate(model, tokenizer, sources):
    predictions = []
    for start in range(0, len(sources), GEN_BATCH):
        batch = tokenizer(
            sources[start : start + GEN_BATCH], max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            output_ids = model.generate(**batch, num_beams=1, max_new_tokens=64)
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return predictions


def evaluate(model, tokenizer, test_rows, challenge_rows):
    test_preds = generate(model, tokenizer, [row["source"] for row in test_rows])
    challenge_preds = generate(model, tokenizer, [row["source"] for row in challenge_rows])
    regression_rows = load_rows(REGRESSION_PATH)
    regression_preds = generate(model, tokenizer, [row["source"] for row in regression_rows])
    regression_misses = [
        {"source": row["source"], "target": row["target"], "pred": pred}
        for row, pred in zip(regression_rows, regression_preds)
        if normalize(pred) != normalize(row["target"])
    ]

    metrics = {
        "test_exact_match": sum(
            normalize(pred) == normalize(row["target"]) for pred, row in zip(test_preds, test_rows)
        )
        / len(test_rows),
        "test_indicative_leak_rate": sum(
            first_verb_after_subject(pred) in INDICATIVE_FORMS for pred in test_preds
        )
        / len(test_preds),
        "challenge_exact": sum(
            normalize(pred) == normalize(row["target"]) for pred, row in zip(challenge_preds, challenge_rows)
        ),
        "regression_exact": f"{len(regression_rows) - len(regression_misses)}/{len(regression_rows)}",
        "regression_misses": regression_misses,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with PRED_CHALLENGE_PATH.open("w", encoding="utf-8") as fh:
        for row, pred in zip(challenge_rows, challenge_preds):
            fh.write(json.dumps({"source": row["source"], "pred": pred}, ensure_ascii=False) + "\n")
    return metrics


def gate_ok(metrics):
    return metrics["test_exact_match"] >= 0.85 and metrics["challenge_exact"] >= 15


def main():
    parser = argparse.ArgumentParser(description="Evaluate Konjunktiv-T5.")
    parser.add_argument("--case", help="convert a single source string and print the output")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)

    if args.case is not None:
        print(generate(model, tokenizer, [args.case])[0])
        return

    test_rows = load_rows(TEST_PATH)
    challenge_rows = load_rows(CHALLENGE_PATH)
    metrics = evaluate(model, tokenizer, test_rows, challenge_rows)
    print(json.dumps(metrics, indent=2))
    if gate_ok(metrics):
        return

    train_rows = drop_over_length(load_rows(TRAIN_PATH), tokenizer, "train")
    val_rows = drop_over_length(load_rows(VAL_PATH), tokenizer, "val")
    train_round(
        model, tokenizer, train_rows, val_rows, num_train_epochs=2, learning_rate=1e-4, output_dir=OUTPUT_DIR
    )

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
    metrics = evaluate(model, tokenizer, test_rows, challenge_rows)
    print(json.dumps(metrics, indent=2))
    if not gate_ok(metrics):
        append_readme(
            "eval gate failed after continuation: "
            f"test_exact_match={metrics['test_exact_match']:.3f}, "
            f"test_indicative_leak_rate={metrics['test_indicative_leak_rate']:.3f}, "
            f"challenge_exact={metrics['challenge_exact']}"
        )


if __name__ == "__main__":
    main()
