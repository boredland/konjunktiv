"""Fine-tune Flan-T5 on direct -> indirect speech pairs (Konjunktiv-T5)."""

import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# flan-t5-base (248M) instead of -small (77M): the download budget allows ~1 GB, and the small model's
# remaining errors (invented participles, rare verbs) are capacity errors that more data did not remove.
# Override with KONJUNKTIV_BASE_MODEL=google/flan-t5-small for a faster run.
BASE_MODEL = os.environ.get("KONJUNKTIV_BASE_MODEL", "google/flan-t5-base")
OUTPUT_DIR = Path("out/konjunktiv-t5")
README_PATH = Path("out/README.md")
TRAIN_PATH = Path("data/train.jsonl")
VAL_PATH = Path("data/val.jsonl")

# Model input contract: these bytes must stay identical to what web/main.js feeds the model.
SOURCE_PREFIX = "sprecher: "
SOURCE_SEPARATOR = " | "
SPEAKERS = ("Er", "Sie")

MAX_LENGTH = 64

GUARD_AFTER_STEPS = 50
GUARD_BUDGET_S = 4 * 60 * 60


class ThroughputGuard(TrainerCallback):
    """Stops the run when the projected wall time exceeds the CPU time budget."""

    def __init__(self):
        self.tripped = False
        self.projected_s = None
        self._checked = False
        self._started_at = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self._started_at = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self._checked or state.global_step < GUARD_AFTER_STEPS:
            return
        self._checked = True
        steps_per_second = state.global_step / (time.monotonic() - self._started_at)
        self.projected_s = state.max_steps / steps_per_second
        if self.projected_s > GUARD_BUDGET_S:
            self.tripped = True
            control.should_training_stop = True


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer):
        sources = tokenizer([row["source"] for row in rows], max_length=MAX_LENGTH, truncation=True)
        targets = tokenizer(text_target=[row["target"] for row in rows], max_length=MAX_LENGTH, truncation=True)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in sources["input_ids"]]
        self.attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in sources["attention_mask"]]
        self.labels = [torch.tensor(ids, dtype=torch.long) for ids in targets["input_ids"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        source = row["source"]
        if not source.startswith(SOURCE_PREFIX):
            raise ValueError(f"{path}: source must start with {SOURCE_PREFIX!r}: {source!r}")
        speaker = source[len(SOURCE_PREFIX):].split(SOURCE_SEPARATOR, 1)[0]
        if speaker not in SPEAKERS:
            raise ValueError(f"{path}: speaker must be one of {SPEAKERS}, got {speaker!r}")
    return rows


def drop_over_length(rows, tokenizer, label):
    source_lengths = [len(ids) for ids in tokenizer([row["source"] for row in rows], truncation=False)["input_ids"]]
    target_lengths = [
        len(ids) for ids in tokenizer(text_target=[row["target"] for row in rows], truncation=False)["input_ids"]
    ]
    kept = [
        row
        for row, source_len, target_len in zip(rows, source_lengths, target_lengths)
        if source_len <= MAX_LENGTH and target_len <= MAX_LENGTH
    ]
    print(f"{label}: dropped {len(rows) - len(kept)} of {len(rows)} rows longer than {MAX_LENGTH} tokens")
    return kept


def append_readme(line):
    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    with README_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def train_round(model, tokenizer, train_rows, val_rows, *, num_train_epochs, learning_rate, output_dir, guard=None):
    """Run one fine-tuning round; returns False when the throughput guard stopped it."""
    # 12 physical cores with SMT: 24 threads oversubscribe and measured ~40x slower per step (26.5s vs 0.66s)
    torch.set_num_threads(12)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        bf16=False,
        fp16=False,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=PairDataset(train_rows, tokenizer),
        eval_dataset=PairDataset(val_rows, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
        callbacks=[guard] if guard is not None else None,
    )
    # A base-size run takes ~2 h; if the process dies, continue from the last epoch checkpoint instead of from scratch
    last = get_last_checkpoint(str(output_dir)) if Path(output_dir).is_dir() else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)
    if guard is not None and guard.tripped:
        return False
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return True


# Built by scripts/build_tokenizer.py: base vocab + SentencePiece pieces for Ä/Ö/ẞ.
TOKENIZER_DIR = Path("out/tokenizer")
BASE_SP_PIECES = 32000
# Each new piece starts from the mean of existing pieces it resembles. A shared init (e.g. the vocab
# mean) left ▁Ä and ▁Ö at cosine 0.99997 after training, and the model wrote "Äkologie" for "Ökologie".
NEW_PIECE_SEEDS = {"▁Ä": ["▁A", "ä"], "▁Ö": ["▁O", "ö"], "Ä": ["A", "ä"], "Ö": ["O", "ö"], "ẞ": ["ß"]}


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    added = len(tokenizer) - len(AutoTokenizer.from_pretrained(BASE_MODEL))
    if added:
        # New pieces sit right after the 32000 base pieces, which shifts the 100 sentinels up by `added`.
        # T5 numbers sentinels in reverse (<extra_id_0> is the last id), so move their rows by name.
        # The matrix has 128 rows, so everything fits without resizing.
        base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        sentinels = [f"<extra_id_{i}>" for i in range(100)]
        old_ids = base_tok.convert_tokens_to_ids(sentinels)
        new_ids = tokenizer.convert_tokens_to_ids(sentinels)
        for emb in (model.get_input_embeddings().weight, model.get_output_embeddings().weight):
            with torch.no_grad():
                emb[new_ids] = emb[old_ids].clone()
                for piece, seeds in NEW_PIECE_SEEDS.items():
                    seed_ids = base_tok.convert_tokens_to_ids(seeds)
                    assert base_tok.unk_token_id not in seed_ids, f"seed for {piece} is not in the base vocab"
                    emb[tokenizer.convert_tokens_to_ids(piece)] = emb[seed_ids].mean(dim=0)
    return tokenizer, model


def main():
    tokenizer, model = load_base()
    train_rows = drop_over_length(load_rows(TRAIN_PATH), tokenizer, "train")
    val_rows = drop_over_length(load_rows(VAL_PATH), tokenizer, "val")

    guard = ThroughputGuard()
    completed = train_round(
        model, tokenizer, train_rows, val_rows, num_train_epochs=3, learning_rate=3e-4, output_dir=OUTPUT_DIR, guard=guard
    )
    if completed:
        return

    line = (
        f"training config deviation: throughput guard projected {guard.projected_s / 3600:.1f} h at "
        f"{GUARD_AFTER_STEPS} optimizer steps (> 4 h); restarting with num_train_epochs=2 "
        "and the first 15,000 train rows"
    )
    print(line)
    append_readme(line)

    _, model = load_base()
    train_round(
        model,
        tokenizer,
        train_rows[:15_000],
        val_rows,
        num_train_epochs=2,
        learning_rate=3e-4,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
