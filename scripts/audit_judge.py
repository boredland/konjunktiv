"""Categorize the model's held-out predictions (out/audit_preds.jsonl) with the LLM judge.

Run in the eval kernel after `%load scripts/synthesize_pairs.py` (needs completion(), RULES, _run_pool):
    %load scripts/audit_judge.py
    audit()          # writes out/audit_report.json, prints counts and every genuine flag
Categories follow the round-2 audit; project conventions the judge must accept are stated explicitly,
because an unconstrained judge flagged allowed forms (-te Konjunktiv II, original word order).
"""
import json
from collections import Counter

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


def audit(path=AUDIT_PREDS, label="audit"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    verdicts = {}

    def on_result(batch, res):
        want = {r["id"] for r in batch}
        try:
            if isinstance(res, str):
                res = json.loads(res)
            for v in (res or {}).get("verdicts", []):
                if v.get("id") in want and v.get("category") in CATS:
                    verdicts[v["id"]] = v
        except Exception as e:  # noqa: BLE001
            print("parse", e)
        return [r for r in batch if r["id"] not in verdicts]

    _run_pool([rows[i:i + 20] for i in range(0, len(rows), 20)],
              lambda b: completion(audit_prompt(b), model="default", schema=AUDIT_SCHEMA), on_result, label)
    report = [{**r, **verdicts[r["id"]]} for r in rows if r["id"] in verdicts]
    AUDIT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = Counter(r["category"] for r in report)
    errors = [r for r in report if r["category"] not in NOT_ERRORS]
    in_scope = len(report) - counts["out_of_scope_input"]
    print(dict(counts))
    print(f"genuine flags: {len(errors)}/{in_scope} in-scope ({len(errors) / max(in_scope, 1):.3f})")
    for r in sorted(errors, key=lambda r: r["category"]):
        print(f"[{r['category']}] {r['direct']}\n    got: {r['indirect']}\n    fix: {r['corrected']}")
    return report


print("loaded: audit(path=out/audit_preds.jsonl)")
