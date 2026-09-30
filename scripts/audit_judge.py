"""Categorize the model's held-out predictions (out/audit_preds.jsonl) with the LLM judge.

Run in the eval kernel after `%load scripts/synthesize_pairs.py` (needs completion(), RULES, _run_pool):
    %load scripts/audit_judge.py
    audit()          # writes out/audit_report.json, prints counts and every majority flag
Categories follow the round-2 audit; project conventions the judge must accept are stated explicitly,
because an unconstrained judge flagged allowed forms (-te Konjunktiv II, original word order).
A single verdict is noisy (round 4: 46 of 76 flags had an empty or identical "fix"), so each row is judged in
`votes` independent rounds and counts as an error only when a majority names the same error category.
Batches stay at 4 rows: since round 4 the judge fails schema validation on 20-row batches.
"""
import json
from collections import Counter, defaultdict

AUDIT_PREDS = ROOT / "out" / "audit_preds.jsonl"
AUDIT_REPORT = ROOT / "out" / "audit_report.json"
CATS = ["ok", "participle_or_verb_form", "wrong_or_invented_word", "auxiliary_haben_sein", "tense_not_converted",
        "indicative_left", "pronoun_or_case", "missing_or_extra_word", "out_of_scope_input"]
NOT_ERRORS = ("ok", "out_of_scope_input")
AUDIT_SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object",
    "properties": {"id": {"type": "string"}, "category": {"type": "string", "enum": CATS},
                   "problem": {"type": "string"}, "corrected": {"type": "string"}},
    "required": ["id", "category", "problem", "corrected"]}}}, "required": ["verdicts"]}


def audit_prompt(batch):
    lines = "\n".join(json.dumps({"id": r["id"], "direct": r["direct"], "speaker": r["speaker"], "indirect": r["indirect"]},
                                 ensure_ascii=False) for r in batch)
    return f"""{RULES}

Verbindliche Konventionen dieses Projekts (gelten als KORREKT, nicht beanstanden):
- Schwache Verben im Konjunktiv II mit -te/-ten-Form sind erlaubt ("Wir wohnen" → "Sie wohnten"); würde-Umschreibung ebenfalls.
- Regel 8 geht vor Regel 7: die Wortstellung des Originals bleibt, das Pronomen muss NICHT an den Satzanfang.
- Satzzeichen im Satzinneren (!, …) dürfen bleiben. "möchte" darf bleiben.
- Anredeformen mit du/ihr/Sie und Imperative gehören zu out_of_scope_input.

Prüfe jede Modellausgabe `indirect`. Wähle genau eine Kategorie:
ok | participle_or_verb_form (falsch gebildetes Partizip/Verbform, z. B. "ein gekauft", "beobacht", "gelagt", falscher Numerus) |
wrong_or_invented_word (Wort ersetzt oder erfunden) | auxiliary_haben_sein | tense_not_converted (Präteritum/Präsens nicht umgesetzt
oder falsche Zeitstufe) | indicative_left (Indikativ im Haupt- oder Nebensatz übrig) | pronoun_or_case (falsches Pronomen/Kasus,
z. B. uns→sich statt ihnen) | missing_or_extra_word | out_of_scope_input (Eingabe ist Imperativ, Frage oder spricht eine zweite
Person an – dann egal was ausgegeben wurde).
Nur echte Fehler beanstanden. Antworte nur als JSON:
{{"verdicts": [{{"id": "<id>", "category": "<kategorie>", "problem": "<kurz, leer wenn ok>", "corrected": "<korrekte Fassung, leer wenn ok>"}}]}}

{lines}"""


def audit(path=AUDIT_PREDS, label="audit", votes=3, batch_size=4):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ballots = defaultdict(list)

    for round_no in range(votes):
        seen = set()

        def on_result(batch, res):
            want = {r["id"] for r in batch}
            try:
                if isinstance(res, str):
                    res = json.loads(res)
                for v in (res or {}).get("verdicts", []):
                    if v.get("id") in want and v.get("category") in CATS and v["id"] not in seen:
                        seen.add(v["id"])
                        ballots[v["id"]].append(v)
            except Exception as e:  # noqa: BLE001
                print("parse", e)
            return [r for r in batch if r["id"] not in seen]

        _run_pool([rows[i:i + batch_size] for i in range(0, len(rows), batch_size)],
                  lambda b: completion(audit_prompt(b), model="default", schema=AUDIT_SCHEMA), on_result,
                  f"{label}-v{round_no + 1}")

    need = votes // 2 + 1
    report = []
    for r in rows:
        cats = Counter(v["category"] for v in ballots.get(r["id"], []))
        if not cats:
            continue
        cat, n = cats.most_common(1)[0]
        if n < need:
            cat = "ok"
        fix = next((v["corrected"] for v in ballots[r["id"]] if v["category"] == cat and v["corrected"].strip()), "")
        report.append({**r, "category": cat, "corrected": fix, "votes": dict(cats)})
    AUDIT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(r["category"] for r in report)
    errors = [r for r in report if r["category"] not in NOT_ERRORS]
    in_scope = len(report) - counts["out_of_scope_input"]
    print(f"judged {len(report)}/{len(rows)} rows, {votes} votes each")
    print(dict(counts))
    print(f"majority flags: {len(errors)}/{in_scope} in-scope ({len(errors) / max(in_scope, 1):.3f})")
    for r in sorted(errors, key=lambda r: r["category"]):
        print(f"[{r['category']}] {r['direct']}\n    got: {r['indirect']}\n    fix: {r['corrected']}")
    return report


print("loaded: audit(path=out/audit_preds.jsonl)")
