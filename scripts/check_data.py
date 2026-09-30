"""Verification 1: integrity checks over the synthesized split files."""

import json
import random
from pathlib import Path

CANONICAL_SOURCE = "sprecher: Er | Ich bin 18 Jahre alt und in Nürnberg geboren"
CANONICAL_TARGET = "Er sei 18 Jahre alt und sei in Nürnberg geboren worden."
SOURCE_PREFIXES = ("sprecher: Er | ", "sprecher: Sie | ")


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    train = load("data/train.jsonl")
    val = load("data/val.jsonl")
    test = load("data/test.jsonl")
    challenge = load("data/challenge_20.jsonl")

    train_ids = {row["id"] for row in train}
    val_ids = {row["id"] for row in val}
    test_ids = {row["id"] for row in test}
    assert not (train_ids & test_ids), "train and test share ids"
    assert not (train_ids & val_ids), "train and val share ids"
    assert not (val_ids & test_ids), "val and test share ids"

    assert len(train) >= 25000, f"len(train) = {len(train)} < 25000"

    kept = train + val + test
    template_share = sum(row.get("origin") == "template" for row in kept) / len(kept)
    assert template_share <= 0.15, f"template share {template_share:.3f} > 0.15"

    for name, rows in (("train", train), ("val", val), ("test", test), ("challenge", challenge)):
        for row in rows:
            assert row["source"].startswith(SOURCE_PREFIXES), f"{name}: bad source format {row['source']!r}"

    assert len(challenge) == 20, f"challenge has {len(challenge)} lines, expected 20"
    assert challenge[0]["source"] == CANONICAL_SOURCE, "challenge line 1 source is not the canonical pair"
    assert challenge[0]["target"] == CANONICAL_TARGET, "challenge line 1 target is not the canonical pair"

    for row in random.Random(42).sample(train, 5):
        print(f"{row['source']}  ->  {row['target']}")

    print(f"check_data: OK ({len(kept)} kept pairs, template share {template_share:.3f})")


if __name__ == "__main__":
    main()
