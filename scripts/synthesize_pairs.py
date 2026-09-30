"""Synthesize direct → indirect speech pairs. Run with `%load scripts/synthesize_pairs.py` in the eval kernel:
uses the kernel globals `completion` and `judge_batch`. Do not run with `python`.

Resilience contract (the cell may die at any time; every stage must be safe to re-call):
- data/pairs_raw.jsonl / pairs_judged.jsonl are append-only with per-item flush; ids already scored are skipped.
- judge() persists its judge_batch id in data/judge_state.json and reattaches after a crash instead of
  spawning a duplicate batch (duplicates clog the judge queue and stall everything).
- judge() and synth() work in bounded chunks and return instead of running for hours in one cell.
- judge() cancels its batch on stall or exit (finally), so no orphaned jobs are left behind.

Stages: synth(limit) → data/pairs_raw.jsonl, await judge(limit) → data/pairs_judged.jsonl,
pass_rate(), check_pass_rate(), write_pairs() → data/pairs.jsonl, split() → train/val/test.
"""
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/home/jonass/Documents/passiv/konjunktiv")
DATA = ROOT / "data"
SOURCES = DATA / "sources.jsonl"
RAW = DATA / "pairs_raw.jsonl"
PAIRS = DATA / "pairs.jsonl"
JUDGED = DATA / "pairs_judged.jsonl"
JSTATE = DATA / "judge_state.json"

BATCH = 25
IN_FLIGHT = 32
KEEP_THRESHOLD = 0.8
JUDGE_CHUNK = 250        # states per judge_batch; small enough that a lost cell costs little
DRAIN_TIMEOUT = 30       # seconds per drain() wait
STALL_CYCLES = 10        # no progress for this many drains (≈5 min) → cancel the batch
CALL_DEADLINE = 180      # a completion call older than this is cancelled and its batch requeued
MAX_ATTEMPTS = 3         # per batch; after that its ids are dropped (logged) and retried on the next synth()
SYNTH_STALL = 600        # no batch finished for this long → synth() returns so the cell never hangs

CANONICAL_SRC = "sprecher: Er | Ich bin 18 Jahre alt und in Nürnberg geboren"
CANONICAL_TGT = "Er sei 18 Jahre alt und sei in Nürnberg geboren worden."

RULES = """Regeln (Verhaltensspezifikation):
1. Personenverschiebung je nach Sprecher: `ich`→`er|sie`, `mich`→`ihn|sie`, `mir`→`ihm|ihr`, `mein-`→`sein-|ihr-` (flektiert), `wir`→`sie`, `uns`→`sich` (reflexiv) oder `ihnen` (Dativ), `unser-`→`ihr-`. Sprecher `Er` = maskulin 3. Sg. (er/sein/ihm/ihn), Sprecher `Sie` = feminin 3. Sg. (sie/ihr/ihr/sie). `wir` wird immer zu Plural `sie` (Possessiv `ihr-`, Pluralverb: `seien`, `hätten`).
2. Präsens / Perfekt / Futur I → Konjunktiv I (`sei, habe, werde, gehe, könne, müsse`). Ist die Konjunktiv-I-Form gleich dem Indikativ (vor allem Plural und von der 1. Person abgeleitete Formen wie `sie haben`), dann Konjunktiv II (`hätten`, `würden` + Infinitiv, starke Verben `gingen`).
3. Präteritum → Konjunktiv I Perfekt (`ich ging` → `er sei gegangen`, `ich hatte` → `er habe gehabt`, `ich war` → `er sei gewesen`). Standard-Vergangenheit der indirekten Rede.
4. Zeitstufe beibehalten: Perfekt bleibt Perfekt (`ich habe gearbeitet` → `er habe gearbeitet`).
5. `ich bin … geboren` → `er sei … geboren worden` (Vorgangspassiv Perfekt, Konjunktiv I). Andere `sein`-Perfekt-Verben (`gestorben, gegangen, aufgewachsen`) bekommen nur `sei` und kein `worden`.
6. Teilen sich koordinierte Teilsätze (`… und …`) ein finites Verb, wird das Konjunktiv-Hilfsverb in jedem Teilsatz wiederholt (`Er sei 18 Jahre alt und sei in Nürnberg geboren worden.`).
7. Nur den wiedergegebenen Satz ausgeben: kein `dass`, kein Redeverb (kein „Er sagte, …“), keine Anführungszeichen. Das Subjektpronomen steht am Anfang mit Großbuchstaben, der Satz endet mit `.`.
8. Sonst nichts ändern: gleiche Inhaltswörter, gleiche Wortstellung, abgesehen von dem, was 1–7 und 9–10 verlangen.
9. Auch in Nebensätzen (dass-, weil-, wenn-, als-, sobald-Sätze, Relativsätze, indirekte Fragen) werden ALLE finiten Verben der wiedergegebenen Rede nach Regel 2–4 umgesetzt, egal welches Subjekt sie haben (`dass er kommt` → `dass er komme`, `was er denkt` → `was er denke`, `die sich dort befinden` → `die sich dort befänden`, `als ich ankam` → `als er angekommen sei`). Kein Indikativ bleibt stehen.
10. `uns`/`mir` als Dativobjekt, das sich NICHT auf das Subjekt bezieht, wird `ihnen`/`ihm|ihr`, niemals `sich` (`Uns fehlte die Zeit` → `Ihnen habe die Zeit gefehlt`, `Ich hole uns etwas` → `Er hole ihnen etwas`, `Mir geht es gut` → `Ihm gehe es gut`). Partizipien trennbarer Verben immer zusammen und mit -ge- in der Mitte (`kaufte ein` → `eingekauft`, `schaute an` → `angeschaut`)."""

FEWSHOT = [
    # canonical (rules 5, 6)
    ("Er", "Ich bin 18 Jahre alt und in Nürnberg geboren.", CANONICAL_TGT),
    ("Sie", "Ich bin in Hamburg geboren.", "Sie sei in Hamburg geboren worden."),
    # rule 2: Präsens/Perfekt/Futur → K I; wir → K II fallback
    ("Er", "Ich wohne seit zwei Jahren in Berlin.", "Er wohne seit zwei Jahren in Berlin."),
    ("Sie", "Ich werde morgen nach Wien fahren.", "Sie werde morgen nach Wien fahren."),
    ("Er", "Wir haben keine Zeit und müssen gehen.", "Sie hätten keine Zeit und müssten gehen."),
    ("Sie", "Wir gehen jeden Tag ins Kino.", "Sie gingen jeden Tag ins Kino."),
    # rule 3: Präteritum → K I Perfekt
    ("Er", "Ich ging gestern nach Hause.", "Er sei gestern nach Hause gegangen."),
    ("Sie", "Ich hatte keine Zeit und war müde.", "Sie habe keine Zeit gehabt und sei müde gewesen."),
    # rule 4: Perfekt bleibt Perfekt
    ("Sie", "Ich habe gestern lange gearbeitet.", "Sie habe gestern lange gearbeitet."),
    ("Er", "Wir sind mit dem Zug gekommen.", "Sie seien mit dem Zug gekommen."),
    # rule 5: sein-Perfekt ohne worden
    ("Er", "Mein Vater ist letztes Jahr gestorben.", "Sein Vater sei letztes Jahr gestorben."),
    ("Sie", "Ich bin in Köln aufgewachsen.", "Sie sei in Köln aufgewachsen."),
    # rule 6: coordination, aux repeated
    ("Sie", "Ich bin müde und habe Hunger.", "Sie sei müde und habe Hunger."),
    ("Er", "Ich kann gut schwimmen und will Lehrer werden.", "Er könne gut schwimmen und wolle Lehrer werden."),
    # rule 1: pronoun cases
    ("Er", "Meine Mutter hilft mir immer.", "Seine Mutter helfe ihm immer."),
    ("Sie", "Niemand versteht mich.", "Niemand verstehe sie."),
    ("Er", "Wir freuen uns auf unseren Urlaub.", "Sie freuten sich auf ihren Urlaub."),
    # rule 9: subordinate clauses shift too
    ("Sie", "Ich weiß, dass er morgen kommt.", "Sie wisse, dass er morgen komme."),
    ("Er", "Wir kennen niemanden, der dort wohnt.", "Sie kennten niemanden, der dort wohne."),
    ("Er", "Ich war froh, als ich endlich ankam.", "Er sei froh gewesen, als er endlich angekommen sei."),
    # rule 10: non-reflexive dative, separable participles
    ("Sie", "Uns fehlte das Geld.", "Ihnen habe das Geld gefehlt."),
    ("Er", "Ich hole uns etwas zu trinken.", "Er hole ihnen etwas zu trinken."),
    ("Sie", "Wir kauften im Supermarkt ein.", "Sie hätten im Supermarkt eingekauft."),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "indirect": {"type": "string"}},
                "required": ["id", "indirect"],
            },
        }
    },
    "required": ["pairs"],
}

EXTRA_FEWSHOT = []  # filled by check_pass_rate() when the first 1000 pairs fall below the 0.6 gate


def build_prompt(batch):
    shots = "\n".join(f"- Sprecher {s} | direkt: {d} → indirekt: {i}" for s, d, i in FEWSHOT + EXTRA_FEWSHOT)
    lines = "\n".join(f'{{"id": "{r["id"]}", "sprecher": "{r["speaker"]}", "direkt": "{r["direct"]}"}}' for r in batch)
    return f"""Du wandelst deutsche direkte Rede in indirekte Rede um (Konjunktiv I, Konjunktiv II als Ersatzform).

{RULES}

Beispiele:
{shots}

Wandle jeden der folgenden Sätze für den angegebenen Sprecher um. Antworte ausschließlich als JSON:
{{"pairs": [{{"id": "<id>", "indirect": "<indirekte Rede>"}}, …]}}
Gib genau eine Ausgabe pro id, in derselben Reihenfolge.

{lines}"""


def load_sources():
    rows = [json.loads(l) for l in SOURCES.open(encoding="utf-8")]
    rng = random.Random(42)
    for r in rows:
        r["speaker"] = "Er" if rng.random() < 0.5 else "Sie"
    return rows


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def latest_by_id(rows):
    """Later rows win: crashed judge runs may leave a duplicate id, the freshest score counts."""
    return {r["id"]: r for r in rows}


def _append(fh, obj):
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def _submit(batch):
    return completion(build_prompt(batch), model="default", schema=SCHEMA)


def _parse(batch, res):
    want = {r["id"] for r in batch}
    if isinstance(res, str):
        res = json.loads(res)
    got = {}
    for p in res.get("pairs", []):
        if p.get("id") in want and isinstance(p.get("indirect"), str) and p["indirect"].strip():
            got[p["id"]] = p["indirect"].strip()
    return got


def _run_pool(items, submit, on_result, label):
    """Run completion calls with bounded concurrency; harvest in completion order, cancel+retry past deadline.

    Completion order matters: waiting on the oldest call let one hung completion block every slot for ~20 min.
    Returns (finished, dropped_items); never blocks longer than SYNTH_STALL without progress.
    """
    queue = [(it, 1) for it in items]
    n = len(queue)
    finished = dropped = timeouts = 0
    t0 = last_progress = last_log = time.time()
    inflight = []  # (item, attempt, handle, submitted_at)

    def retry(item, attempt, why):
        nonlocal dropped
        if attempt < MAX_ATTEMPTS:
            queue.append((item, attempt + 1))
        else:
            dropped += len(item)
            print(f"  dropping {len(item)} ids after {attempt} attempts ({why})")

    while queue or inflight:
        while queue and len(inflight) < IN_FLIGHT:
            item, attempt = queue.pop(0)
            try:
                inflight.append((item, attempt, submit(item), time.time()))
            except Exception as e:  # noqa: BLE001
                retry(item, attempt, f"submit: {e}")
        now = time.time()
        still = []
        for item, attempt, h, t_sub in inflight:
            if h.done():
                try:
                    res = h.wait(timeout=1)
                except Exception as e:  # noqa: BLE001
                    res = None
                    print(f"  {label} error (attempt {attempt}): {e}")
                missing = on_result(item, res)
                if missing:
                    retry(missing, attempt, "missing ids")
                else:
                    finished += 1
                last_progress = now
            elif now - t_sub > CALL_DEADLINE:
                h.cancel()
                timeouts += 1
                retry(item, attempt, "deadline")
            else:
                still.append((item, attempt, h, t_sub))
        inflight = still
        if now - last_log > 30:
            el = now - t0
            eta = el / max(finished, 1) * (n - finished)
            print(f"  {label} {finished}/{n} inflight={len(inflight)} queued={len(queue)} "
                  f"timeouts={timeouts} dropped={dropped} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
            last_log = now
        if now - last_progress > SYNTH_STALL:
            for _, _, h, _ in inflight:
                h.cancel()
            print(f"STALLED: no {label} finished for {SYNTH_STALL}s — returning; re-call to resume")
            return finished, dropped
        time.sleep(0.5)
    print(f"{label} done: {finished}/{n} timeouts={timeouts} dropped={dropped}")
    return finished, dropped


def synth(limit=None):
    """Append synthesized pairs to data/pairs_raw.jsonl. Safe to re-call; processes `limit` sources per call."""
    rows = load_sources()
    done = {r["id"] for r in load_jsonl(RAW)}
    todo = [r for r in rows if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"sources={len(rows)} done={len(done)} todo={len(todo)}")
    with RAW.open("a", encoding="utf-8") as out:
        def on_result(batch, res):
            try:
                got = _parse(batch, res) if res is not None else {}
            except Exception as e:  # noqa: BLE001
                print(f"  parse error: {e}")
                got = {}
            for r in batch:
                if r["id"] in got:
                    _append(out, {"id": r["id"], "direct": r["direct"], "speaker": r["speaker"],
                                  "indirect": got[r["id"]], "origin": r["origin"]})
            return [r for r in batch if r["id"] not in got]

        _run_pool([todo[i : i + BATCH] for i in range(0, len(todo), BATCH)], _submit, on_result, "batches")


JUDGE_Q = {
    "ok": {
        "type": "bool",
        "instructions": RULES + "\n\nIs `indirect` a correct rendering of `direct` for this speaker under these rules?",
    }
}


def _attach_or_create(states):
    """Reattach to a persisted judge_batch if possible; else create one and persist its id."""
    if JSTATE.exists():
        try:
            old = json.loads(JSTATE.read_text())
            b = judge_batch.attach(old["id"])
            print(f"reattached judge batch {old['id']}")
            return b
        except Exception as e:  # noqa: BLE001
            print(f"reattach failed ({e}); starting a new batch")
    b = judge_batch(states, JUDGE_Q, intent="Judging indirect speech pairs")
    JSTATE.write_text(json.dumps({"id": b.id}))
    print(f"started judge batch {b.id} over {len(states)} states")
    return b


async def judge(limit=JUDGE_CHUNK):
    """Score unscored (and previously errored) rows in data/pairs_judged.jsonl. Bounded per call, restart-safe."""
    raw = latest_by_id(load_jsonl(RAW))
    judged = latest_by_id(load_jsonl(JUDGED))
    scored_ids = {i for i, r in judged.items() if r.get("p_ok") is not None}
    todo = [r for i, r in raw.items() if i not in scored_ids]
    todo.sort(key=lambda r: r["id"])
    todo = todo[:limit]
    print(f"raw={len(raw)} scored={len(scored_ids)} todo={len(todo)}")
    if not todo:
        JSTATE.unlink(missing_ok=True)
        return
    states = {r["id"]: {"direct": r["direct"], "speaker": r["speaker"], "indirect": r["indirect"]} for r in todo}
    by_id = raw
    b = _attach_or_create(states)
    stall = 0
    last = None
    t0 = time.time()
    try:
        with JUDGED.open("a", encoding="utf-8") as out:
            while True:
                try:
                    items = await b.drain(DRAIN_TIMEOUT)
                except Exception as e:  # noqa: BLE001
                    print(f"drain died: {e} — closing batch, state kept for retry")
                    return
                for k, item in items:
                    r = by_id.get(k)
                    if r is None:
                        continue
                    answers = getattr(item, "answers", None)
                    p = answers["ok"]["bool"] if answers else None
                    _append(out, {**r, "p_ok": p})
                st = b.status()
                progress = st["done"] + st["failed"]
                if items or last is None or progress != last:
                    stall = 0
                    last = progress
                else:
                    stall += 1
                if items or stall % 2 == 0:
                    print(f"  {progress}/{st['total']} failed={st['failed']} stall={stall} "
                          f"elapsed={time.time() - t0:.0f}s", flush=True)
                if progress >= st["total"]:
                    print("judge chunk complete")
                    break
                if stall >= STALL_CYCLES:
                    print(f"STALLED for {stall * DRAIN_TIMEOUT}s — cancelling batch {b.id}; "
                          "unscored ids will be retried on the next call")
                    b.cancel()
                    return
    finally:
        try:
            b.close()
        except Exception:  # noqa: BLE001
            pass
        JSTATE.unlink(missing_ok=True)


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "ok": {"type": "boolean"}},
                "required": ["id", "ok"],
            },
        }
    },
    "required": ["verdicts"],
}


def _judge_prompt(batch):
    lines = "\n".join(json.dumps({"id": r["id"], "direct": r["direct"], "speaker": r["speaker"],
                                  "indirect": r["indirect"]}, ensure_ascii=False) for r in batch)
    return f"""{RULES}

For each item below: is `indirect` a correct rendering of `direct` for this speaker under these rules?
Be strict about rules 1–7; accept any wording that satisfies them. Answer only as JSON:
{{"verdicts": [{{"id": "<id>", "ok": true|false}}, …]}} — exactly one verdict per id.

{lines}"""


def judge_llm(limit=None):
    """Quality filter via completion(), used while judge_batch is unavailable (it stalled with done=0, cost=0).

    Writes p_ok as 1.0/0.0 so KEEP_THRESHOLD applies unchanged; rows carry judge="completion" for provenance.
    """
    raw = latest_by_id(load_jsonl(RAW))
    judged = latest_by_id(load_jsonl(JUDGED))
    scored = {i for i, r in judged.items() if r.get("p_ok") is not None}
    todo = sorted((r for i, r in raw.items() if i not in scored), key=lambda r: r["id"])
    if limit:
        todo = todo[:limit]
    print(f"raw={len(raw)} scored={len(scored)} todo={len(todo)}")
    with JUDGED.open("a", encoding="utf-8") as out:
        def on_result(batch, res):
            want = {r["id"] for r in batch}
            got = {}
            try:
                if isinstance(res, str):
                    res = json.loads(res)
                for v in (res or {}).get("verdicts", []):
                    if v.get("id") in want and isinstance(v.get("ok"), bool):
                        got[v["id"]] = v["ok"]
            except Exception as e:  # noqa: BLE001
                print(f"  parse error: {e}")
            for r in batch:
                if r["id"] in got:
                    _append(out, {**r, "p_ok": 1.0 if got[r["id"]] else 0.0, "judge": "completion"})
            return [r for r in batch if r["id"] not in got]

        submit = lambda b: completion(_judge_prompt(b), model="default", schema=JUDGE_SCHEMA)  # noqa: E731
        _run_pool([todo[i : i + BATCH] for i in range(0, len(todo), BATCH)], submit, on_result, "judged")


def pass_rate(rows=None):
    rows = rows if rows is not None else list(latest_by_id(load_jsonl(JUDGED)).values())
    rows = [r for r in rows if r.get("p_ok") is not None]
    ok = sum(1 for r in rows if r["p_ok"] >= KEEP_THRESHOLD)
    return ok / max(len(rows), 1), len(rows)


def check_pass_rate():
    """Plan gate: first 1000 pairs below 0.6 → add 5 rejected examples to the few-shot and resynthesize them."""
    rows = [r for r in latest_by_id(load_jsonl(JUDGED)).values() if r.get("p_ok") is not None][:1000]
    pr = sum(1 for r in rows if r["p_ok"] >= KEEP_THRESHOLD) / max(len(rows), 1)
    print(f"pass rate on first {len(rows)}: {pr:.3f}")
    if pr >= 0.6:
        return
    rejects = [r for r in rows if r["p_ok"] < KEEP_THRESHOLD][:5]
    for r in rejects:
        EXTRA_FEWSHOT.append((r["speaker"], r["direct"], r["indirect"]))  # corrected outputs added below
    print(f"pass rate below 0.6; {len(rejects)} rejected examples staged for the few-shot block — "
          "fill in corrected outputs, then call resynthesize_ids([...]) for those 1000 ids")


def write_pairs():
    rows = [r for r in latest_by_id(load_jsonl(JUDGED)).values() if r.get("p_ok") is not None]
    kept = [r for r in rows if r["p_ok"] >= KEEP_THRESHOLD]
    with PAIRS.open("w", encoding="utf-8") as out:
        for r in kept:
            _append(out, {"id": r["id"], "source": f"sprecher: {r['speaker']} | {r['direct']}",
                          "target": r["indirect"], "origin": r["origin"]})
    c = Counter(r["origin"] for r in kept)
    print(f"kept {len(kept)}/{len(rows)} ({len(kept) / max(len(rows), 1):.3f}); origins={dict(c)}")
    return kept


def split():
    rows = load_jsonl(PAIRS)
    buckets = {"test": [], "val": [], "train": []}
    for r in rows:
        h = int(hashlib.sha1(r["id"].encode()).hexdigest(), 16) % 100
        if h <= 3:
            buckets["test"].append(r)
        elif h <= 7:
            buckets["val"].append(r)
        else:
            buckets["train"].append(r)
    buckets["test"] = buckets["test"][:1000]
    buckets["val"] = buckets["val"][:1000]
    for name, rs in buckets.items():
        with (DATA / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for r in rs:
                _append(out, r)
        print(f"{name}: {len(rs)}")


print("loaded: synth(limit), judge_llm(limit), await judge(limit), pass_rate(), check_pass_rate(), write_pairs(), split()")
