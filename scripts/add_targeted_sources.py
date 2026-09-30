"""Append targeted sources to data/sources.jsonl for later training rounds.

Round 2 targets: separable verbs in Präteritum ("reichte … aus" → "ausgebrochen"), rare verbs
("absolvierte" → "gearbeitet"), words starting with Ä/Ö. Round 3 targets (held-out audit, see
scripts/audit_predict.py): malformed participles ("ein gekauft", "beobacht", "gelagt"), subordinate
clauses left in the indicative, and dative uns/mir turned into reflexive sich.
Sources: unused Tatoeba sentences matching those patterns, plus LLM-written sentences in
data/raw/agy_sentences*.json. Appending keeps every existing id, speaker assignment and split unchanged.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from collect_sentences import FORBIDDEN, OUT, RE_SECOND, filter_sentences, normalize  # noqa: E402

# one file per request to the agy bridge; the file stem prefixes the ids so batches never collide
AGY_FILES = sorted(Path("data/raw").glob("agy_sentences*.json"))
AUDIT_IDS = Path("data/audit_ids.json")
PARTICLES = ("ab|an|auf|aus|bei|ein|fest|fort|her|hin|los|mit|nach|vor|weg|zu|zurück|zusammen|statt|teil|um|dar"
             "|herum|heraus|vorbei|weiter")
STRONG_PAST = ("fand|fanden|fuhr|fuhren|ging|gingen|kam|kamen|rief|riefen|sah|sahen|stand|standen|nahm|nahmen|gab"
               "|gaben|hielt|hielten|zog|zogen|brach|brachen|fing|fingen|trat|traten|lud|luden|schlug|schlugen"
               "|sprach|sprachen|wies|wiesen|schloss|schlossen|stieg|stiegen|trug|trugen|warf|warfen|lief|liefen"
               "|fiel|fielen|ließ|ließen|bot|boten|sprang|sprangen|griff|griffen|brachte|brachten|dachte|dachten")
TARGETED = [
    re.compile(rf"\b(?:\w+te|\w+ten|{STRONG_PAST})\b[^,.;]*\s(?:{PARTICLES})[.,;]"),
    re.compile(r"\b(?:ich|wir)\s+\w{3,}ierten?\b|\b\w{3,}ierten?\s+(?:ich|wir)\b", re.I),
    re.compile(r"[ÄÖ]"),
    # Präteritum with -et- stem or irregular participle, where the model produced "beobacht", "gelagt"
    re.compile(r"\b(?:ich|wir)\s+(?:\w+teten?|lag|lagen|saß|saßen|band|banden|hing|hingen|traf|trafen|bat|baten|befand|befanden)\b", re.I),
    # subordinate clause in Präsens with a third-person subject that is not the speaker
    re.compile(r",\s*(?:dass|daß|weil|wenn|sobald|was|wie|ob)\s+(?:er|sie|es|man|jemand|Tom|Maria|der \w+|die \w+|das \w+)\s[^,.]*\w+t[.,]"),
    # non-reflexive dative uns/mir
    re.compile(r"\b(?:uns|mir)\s+(?:fehlt|fehlte|gefällt|gefiel|gehört|gehörte|schmeckt|schmeckte|hilft|half|geht|ging)\b|\b(?:fehlt|fehlte|gefällt|gefiel|gehört|gehörte|schmeckt|schmeckte|geht|ging)\s+(?:es\s+)?(?:uns|mir)\b", re.I),
]
FIRST_PERSON = re.compile(r"\b(ich|mich|mir|mein\w*|wir|uns|unser\w*)\b", re.I)


def main():
    existing = [json.loads(line) for line in OUT.open(encoding="utf-8")]
    seen_ids = {r["id"] for r in existing}
    # audit sentences must never become training data
    if AUDIT_IDS.exists():
        seen_ids |= {a["id"] for a in json.loads(AUDIT_IDS.read_text(encoding="utf-8"))}
    seen_text = {r["direct"].lower() for r in existing}
    added = []

    def add(sid, text, origin):
        text = normalize(text)
        if sid in seen_ids or text.lower() in seen_text:
            return
        seen_ids.add(sid)
        seen_text.add(text.lower())
        added.append({"id": sid, "direct": text, "origin": origin})

    for sid, _, text in filter_sentences(5, 25):
        if any(p.search(normalize(text)) for p in TARGETED):
            add(sid, text, "tatoeba")
    n_tatoeba = len(added)

    rejected = 0
    for agy_file in AGY_FILES:
        prefix = agy_file.stem.replace("agy_sentences", "agy") or "agy"
        for i, text in enumerate(json.loads(agy_file.read_text(encoding="utf-8"))):
            # the agy brief asked for these constraints; enforce them instead of trusting the output
            ok = (5 <= len(text.split()) <= 25 and text.endswith(".") and not (set(text) & FORBIDDEN)
                  and not RE_SECOND.search(text) and FIRST_PERSON.search(text))
            if ok:
                add(f"{prefix}-{i}", text, "agy")
            else:
                rejected += 1

    with OUT.open("a", encoding="utf-8") as out:
        for r in added:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(added)} sources (tatoeba {n_tatoeba}, agy {len(added) - n_tatoeba}; agy rejected {rejected})")


if __name__ == "__main__":
    main()
