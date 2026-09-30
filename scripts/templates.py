"""Template sentences covering all five tenses (Präsens, Perfekt, Präteritum, Futur I, geboren/Zustandspassiv)."""
import random

# (infinitiv, ich-präsens, wir-präsens, ich-prät, wir-prät, partizip, perfekt-aux, [objekte])
VERBS = [
    ("sein", "bin", "sind", "war", "waren", "gewesen", "sein", ["müde", "zu Hause", "sehr glücklich", "krank", "in der Stadt"]),
    ("haben", "habe", "haben", "hatte", "hatten", "gehabt", "haben", ["keine Zeit", "viel Arbeit", "einen Hund", "Hunger", "eine Idee"]),
    ("werden", "werde", "werden", "wurde", "wurden", "geworden", "sein", ["Lehrer", "müde", "krank", "langsam alt"]),
    ("gehen", "gehe", "gehen", "ging", "gingen", "gegangen", "sein", ["nach Hause", "in die Schule", "ins Kino", "zur Arbeit", "spazieren"]),
    ("kommen", "komme", "kommen", "kam", "kamen", "gekommen", "sein", ["aus Berlin", "zu spät", "nach Hause", "mit dem Zug"]),
    ("fahren", "fahre", "fahren", "fuhr", "fuhren", "gefahren", "sein", ["nach Hamburg", "mit dem Auto", "in den Urlaub", "zur Arbeit"]),
    ("bleiben", "bleibe", "bleiben", "blieb", "blieben", "geblieben", "sein", ["zu Hause", "in Berlin", "noch eine Woche", "lange wach"]),
    ("arbeiten", "arbeite", "arbeiten", "arbeitete", "arbeiteten", "gearbeitet", "haben", ["in einer Bank", "sehr viel", "als Lehrer", "im Garten"]),
    ("wohnen", "wohne", "wohnen", "wohnte", "wohnten", "gewohnt", "haben", ["in München", "in einer kleinen Wohnung", "bei meinen Eltern", "am Stadtrand"]),
    ("lernen", "lerne", "lernen", "lernte", "lernten", "gelernt", "haben", ["Deutsch", "jeden Tag", "für die Prüfung", "Klavier spielen"]),
    ("können", "kann", "können", "konnte", "konnten", "gekonnt", "haben", ["gut schwimmen", "nicht kommen", "das nicht verstehen", "sehr gut kochen"]),
    ("müssen", "muss", "müssen", "musste", "mussten", "gemusst", "haben", ["viel arbeiten", "früh aufstehen", "nach Hause gehen", "noch lernen"]),
    ("wollen", "will", "wollen", "wollte", "wollten", "gewollt", "haben", ["nach Wien fahren", "ein Buch lesen", "nicht bleiben", "Arzt werden"]),
    ("dürfen", "darf", "dürfen", "durfte", "durften", "gedurft", "haben", ["nicht rauchen", "lange schlafen", "das Auto nehmen", "mitkommen"]),
    ("machen", "mache", "machen", "machte", "machten", "gemacht", "haben", ["die Hausaufgaben", "eine Pause", "einen Fehler", "das Abendessen"]),
    ("sagen", "sage", "sagen", "sagte", "sagten", "gesagt", "haben", ["die Wahrheit", "nichts", "das immer", "es niemandem"]),
    ("sehen", "sehe", "sehen", "sah", "sahen", "gesehen", "haben", ["einen Film", "das Meer", "nichts", "meine Freunde"]),
    ("geben", "gebe", "geben", "gab", "gaben", "gegeben", "haben", ["mein Bestes", "keine Antwort", "dem Kind ein Geschenk", "gern Geld aus"]),
    ("nehmen", "nehme", "nehmen", "nahm", "nahmen", "genommen", "haben", ["den Bus", "ein Taxi", "das Angebot an", "ein Stück Kuchen"]),
    ("finden", "finde", "finden", "fand", "fanden", "gefunden", "haben", ["den Schlüssel", "das gut", "keine Wohnung", "einen neuen Job"]),
    ("denken", "denke", "denken", "dachte", "dachten", "gedacht", "haben", ["oft an dich nicht", "an die Zukunft", "an meine Familie", "viel nach"]),
    ("glauben", "glaube", "glauben", "glaubte", "glaubten", "geglaubt", "haben", ["das nicht", "an das Gute", "ihm", "es kaum"]),
    ("wissen", "weiß", "wissen", "wusste", "wussten", "gewusst", "haben", ["es nicht", "die Antwort", "alles", "nichts davon"]),
    ("kaufen", "kaufe", "kaufen", "kaufte", "kauften", "gekauft", "haben", ["ein neues Auto", "Brot", "ein Haus in Bremen", "die Fahrkarten"]),
    ("essen", "esse", "essen", "aß", "aßen", "gegessen", "haben", ["zu Mittag", "kein Fleisch", "gern Pizza", "einen Apfel"]),
    ("trinken", "trinke", "trinken", "trank", "tranken", "getrunken", "haben", ["Kaffee", "keinen Alkohol", "ein Glas Wasser", "Tee"]),
    ("schlafen", "schlafe", "schlafen", "schlief", "schliefen", "geschlafen", "haben", ["sehr lange", "schlecht", "acht Stunden", "im Hotel"]),
    ("lesen", "lese", "lesen", "las", "lasen", "gelesen", "haben", ["die Zeitung", "ein Buch", "viel", "den Brief"]),
    ("schreiben", "schreibe", "schreiben", "schrieb", "schrieben", "geschrieben", "haben", ["einen Brief", "ein Buch", "eine E-Mail", "den Bericht"]),
    ("spielen", "spiele", "spielen", "spielte", "spielten", "gespielt", "haben", ["Fußball", "Klavier", "mit den Kindern", "Schach"]),
    ("laufen", "laufe", "laufen", "lief", "liefen", "gelaufen", "sein", ["jeden Morgen", "zum Bahnhof", "durch den Park", "zehn Kilometer"]),
    ("fliegen", "fliege", "fliegen", "flog", "flogen", "geflogen", "sein", ["nach Zürich", "morgen", "in den Süden", "mit der Lufthansa"]),
    ("sprechen", "spreche", "sprechen", "sprach", "sprachen", "gesprochen", "haben", ["Deutsch", "mit dem Chef", "sehr leise", "drei Sprachen"]),
    ("helfen", "helfe", "helfen", "half", "halfen", "geholfen", "haben", ["meiner Mutter", "gern", "den Nachbarn", "im Haushalt"]),
    ("verstehen", "verstehe", "verstehen", "verstand", "verstanden", "verstanden", "haben", ["das nicht", "die Frage", "alles", "kein Wort"]),
    ("brauchen", "brauche", "brauchen", "brauchte", "brauchten", "gebraucht", "haben", ["Hilfe", "mehr Zeit", "ein neues Handy", "Ruhe"]),
    ("suchen", "suche", "suchen", "suchte", "suchten", "gesucht", "haben", ["eine Wohnung", "meine Brille", "Arbeit", "den Ausgang"]),
    ("warten", "warte", "warten", "wartete", "warteten", "gewartet", "haben", ["auf den Bus", "schon lange", "auf eine Antwort", "vor der Tür"]),
    ("studieren", "studiere", "studieren", "studierte", "studierten", "studiert", "haben", ["Medizin", "in Leipzig", "Informatik", "im dritten Semester"]),
    ("aufstehen", "stehe", "stehen", "stand", "standen", "aufgestanden", "sein", ["früh auf", "um sechs Uhr auf", "spät auf", "jeden Tag um sieben auf"]),
]

PLACES = ["Nürnberg", "Berlin", "Hamburg", "München", "Köln", "Wien", "Zürich", "Leipzig", "Bremen", "Dresden"]
TIMES = ["heute", "gestern", "morgen", "jetzt", "am Montag", "im Sommer", "letztes Jahr", "nächste Woche", "am Wochenende", "seit zwei Jahren"]
AGES = [7, 12, 16, 18, 21, 25, 30, 34, 42, 50, 63, 71]

PAST_TIMES = {"gestern", "letztes Jahr", "am Montag", "im Sommer", "am Wochenende"}
FUTURE_TIMES = {"morgen", "nächste Woche", "am Montag", "im Sommer", "am Wochenende"}
PRESENT_TIMES = {"heute", "jetzt", "morgen", "seit zwei Jahren", "am Wochenende", "im Sommer"}


def _obj_with_time(rng, obj, time):
    # German default order: time adverbial before the object/complement
    return f"{time} {obj}"


def generate(seed=42):
    rng = random.Random(seed)
    out = []
    for inf, ich_p, wir_p, ich_prat, wir_prat, pp, aux, objs in VERBS:
        for obj in objs:
            # aufstehen: the particle is inside obj; keep the finite-verb/particle pattern intact
            for subj in ("Ich", "Wir"):
                pres = ich_p if subj == "Ich" else wir_p
                prat = ich_prat if subj == "Ich" else wir_prat
                aux_pres = {"haben": ("habe", "haben"), "sein": ("bin", "sind")}[aux][0 if subj == "Ich" else 1]
                werde = "werde" if subj == "Ich" else "werden"
                t_any = rng.choice(sorted(PRESENT_TIMES))
                t_past = rng.choice(sorted(PAST_TIMES))
                t_fut = rng.choice(sorted(FUTURE_TIMES))
                # Präsens
                out.append(f"{subj} {pres} {obj}.")
                out.append(f"{subj} {pres} {_obj_with_time(rng, obj, t_any)}.")
                # Perfekt
                if inf == "aufstehen":
                    out.append(f"{subj} {aux_pres} {obj.replace(' auf', '')} aufgestanden.")
                    out.append(f"{subj} {aux_pres} {t_past} {obj.replace(' auf', '')} aufgestanden.")
                    out.append(f"{subj} {prat} {obj}.")
                    out.append(f"{subj} {prat} {t_past} {obj}.")
                    out.append(f"{subj} {werde} {obj.replace(' auf', '')} aufstehen.")
                    out.append(f"{subj} {werde} {t_fut} {obj.replace(' auf', '')} aufstehen.")
                    continue
                out.append(f"{subj} {aux_pres} {obj} {pp}.")
                out.append(f"{subj} {aux_pres} {_obj_with_time(rng, obj, t_past)} {pp}.")
                # Präteritum
                out.append(f"{subj} {prat} {obj}.")
                out.append(f"{subj} {prat} {_obj_with_time(rng, obj, t_past)}.")
                # Futur I
                out.append(f"{subj} {werde} {obj} {inf}.")
                out.append(f"{subj} {werde} {_obj_with_time(rng, obj, t_fut)} {inf}.")
    # geboren / Zustandspassiv, with and without coordination
    for place in PLACES:
        out.append(f"Ich bin in {place} geboren.")
        out.append(f"Wir sind in {place} geboren.")
        out.append(f"Ich bin in {place} geboren und wohne jetzt in {rng.choice([p for p in PLACES if p != place])}.")
        for age in AGES:
            out.append(f"Ich bin {age} Jahre alt und in {place} geboren.")
            out.append(f"Ich bin {age} Jahre alt.")
            if rng.random() < 0.3:
                out.append(f"Wir sind {age} Jahre alt und in {place} geboren.")
                out.append(f"Ich bin in {place} geboren und {age} Jahre alt.")
    # coordinated clauses with shared subject across tenses
    for _ in range(600):
        v1, v2 = rng.sample(VERBS, 2)
        if "aufstehen" in (v1[0], v2[0]):
            continue
        subj = rng.choice(["Ich", "Wir"])
        i = 0 if subj == "Ich" else 1
        o1, o2 = rng.choice(v1[7]), rng.choice(v2[7])
        kind = rng.choice(["pres", "perf", "prat", "fut"])
        if kind == "pres":
            out.append(f"{subj} {v1[1 + i]} {o1} und {v2[1 + i]} {o2}.")
        elif kind == "perf":
            a1 = {"haben": ("habe", "haben"), "sein": ("bin", "sind")}[v1[6]][i]
            a2 = {"haben": ("habe", "haben"), "sein": ("bin", "sind")}[v2[6]][i]
            out.append(f"{subj} {a1} {o1} {v1[5]} und {a2} {o2} {v2[5]}.")
        elif kind == "prat":
            out.append(f"{subj} {v1[3 + i]} {o1} und {v2[3 + i]} {o2}.")
        else:
            w = "werde" if subj == "Ich" else "werden"
            out.append(f"{subj} {w} {o1} {v1[0]} und {o2} {v2[0]}.")
    # dedupe preserving order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


if __name__ == "__main__":
    sents = generate()
    print(len(sents))
    for s in sents[:10]:
        print(s)
