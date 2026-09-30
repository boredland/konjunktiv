"""Collect German direct-speech source sentences from Tatoeba plus templates → data/sources.jsonl."""
import bz2
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from templates import generate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "deu_sentences.tsv.bz2"
OUT = ROOT / "data" / "sources.jsonl"
URL = "https://downloads.tatoeba.org/exports/per_language/deu/deu_sentences.tsv.bz2"

RE_SUBJ = re.compile(r"\b(ich|wir)\b", re.I)
TATOEBA_CAP = 36000
RE_SECOND = re.compile(r"\b(du|dich|dir|dein\w*|euch|euer\w*)\b", re.I)
FORBIDDEN = set('?"„“»«:')
PRONOUNS = re.compile(r"^(ich|wir|mein\w*|unser\w*)$", re.I)


def download():
    if RAW.exists() and RAW.stat().st_size > 1_000_000:
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(URL, RAW)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"download failed: {URL}: {e}") from e


def normalize(text):
    text = " ".join(text.split())
    if text.endswith("!"):
        text = text[:-1] + "."
    return text


def filter_sentences(lo, hi):
    counts = {}

    def step(name, it):
        lst = list(it)
        counts[name] = len(lst)
        print(f"  {name}: {len(lst)}", flush=True)
        return lst

    with bz2.open(RAW, "rt", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f]
    rows = step("total", (r for r in rows if len(r) == 3))
    rows = step(f"len {lo}-{hi}", (r for r in rows if lo <= len(r[2].split()) <= hi))
    rows = step("ich/wir", (r for r in rows if RE_SUBJ.search(r[2])))
    rows = step("ends ./!", (r for r in rows if r[2].endswith((".", "!"))))
    rows = step("no forbidden chars", (r for r in rows if not (set(r[2]) & FORBIDDEN)))
    rows = step("no 2nd person", (r for r in rows if not RE_SECOND.search(r[2])))

    def not_imperative(text):
        first = text.split()[0].strip(",")
        if PRONOUNS.match(first):
            return True
        return not first.lower().endswith(("t", "e"))

    rows = step("no imperative", (r for r in rows if not_imperative(r[2])))
    return rows


def main():
    download()
    print("tatoeba filters:")
    rows = filter_sentences(5, 25)
    if len(rows) < 20000:
        print("fewer than 20000; widening to 4-30")
        rows = filter_sentences(4, 30)
    # Tatoeba yields ~130k candidates; LLM synthesis budget only needs ~36k for the ≥27k kept-pair target
    if len(rows) > TATOEBA_CAP:
        rows = random.Random(42).sample(rows, TATOEBA_CAP)
        print(f"  capped to deterministic sample of {TATOEBA_CAP}")
    seen = set()
    n_t = n_tpl = 0
    with OUT.open("w", encoding="utf-8") as out:
        for sid, _, text in rows:
            text = normalize(text)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps({"id": sid, "direct": text, "origin": "tatoeba"}, ensure_ascii=False) + "\n")
            n_t += 1
        for i, text in enumerate(generate()):
            text = normalize(text)
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps({"id": f"tpl-{i}", "direct": text, "origin": "template"}, ensure_ascii=False) + "\n")
            n_tpl += 1
    print(f"wrote {OUT}: tatoeba={n_t} template={n_tpl}")


if __name__ == "__main__":
    main()
