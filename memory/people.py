"""Person resolution for shared first names ("Sarah" = Sarah Kim or Sarah Patel).

WHY: a question that says only "Sarah" matches both people's records; the right one is usually clear from
the REST of the question ("pricing proposal" -> the Acme contact, "geocoder fix" -> the engineer). Instead
of hand-writing a rule for one name, we derive the people directory from the data, find first names that
map to several people, and score each candidate by how well the question's other words fit the records
that person wrote or is involved in (plus organisation cues: an email-domain stem such as "acme" in
acmefreight.example.com). If one candidate clearly wins we pick it; if it is close we keep both and flag
the ambiguity so the writer can say so instead of guessing or abstaining.

Everything is computed from visible records only, so nothing from after `as_of` can influence it.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from memory.index import tokenize
from memory.queries import QUESTION_WORDS
from memory.store import MemoryStore

_EMAIL_RE = re.compile(r"[\w.+-]+@([\w-]+)\.[\w.-]+")
MIN_EVIDENCE = 2.5   # total idf weight of the question's non-name words that appear in any candidate's records
WIN_RATIO = 2.0      # winner must beat the runner-up by this factor to be picked at all
STRONG_SHARE = 0.85  # above this share the pick is trusted; below it we still search both people


@dataclass
class Person:
    name: str
    first: str
    emails: set[str] = field(default_factory=set)
    org_stems: set[str] = field(default_factory=set)   # e.g. {"acmefreight"} for acmefreight.example.com


@lru_cache(maxsize=4)
def _directory(store_id: int, n_units: int, store: MemoryStore) -> tuple[dict[str, Person], dict[str, set[str]]]:
    """(name -> Person, name -> ids of units the person wrote or is involved in)."""
    people: dict[str, Person] = {}
    for u in store.units:
        names = u.meta.get("slack_user_names") or {}
        emails = u.meta.get("email_names") or {}
        if names or emails:
            for email, name in emails.items():
                if not name or " " not in name.strip():
                    continue  # skip bots / groups ("Linear", "All Brightline")
                first = name.split()[0].lower()
                if first not in email.split("@")[0].lower() and name not in names.values():
                    continue  # organisations ("FreightTech SF Meetup") do not have their name in the mailbox
                p = people.setdefault(name, Person(name=name, first=first))
                p.emails.add(email.lower())
                m = _EMAIL_RE.match(email.lower())
                if m and not m.group(1).startswith("brightline"):
                    p.org_stems.add(m.group(1))
            for name in names.values():
                if name and " " in name.strip():
                    people.setdefault(name, Person(name=name, first=name.split()[0].lower()))
            break
    involved: dict[str, set[str]] = defaultdict(set)
    for u in store.units:
        blob_emails = " ".join(str(v) for k, v in u.meta.items() if k in ("from", "to", "cc", "attendees", "organizer", "target_context")).lower()
        lowered = u.text.lower()
        for name, p in people.items():
            if (u.speaker == name or name.lower() in lowered or any(e in blob_emails for e in p.emails)):
                involved[name].add(u.id)
    return people, involved


def directory(store: MemoryStore):
    return _directory(id(store), len(store.units), store)


def ambiguous_first_names(store: MemoryStore) -> dict[str, list[str]]:
    people, _ = directory(store)
    by_first: dict[str, list[str]] = defaultdict(list)
    for p in people.values():
        by_first[p.first].append(p.name)
    return {f: sorted(ns) for f, ns in by_first.items() if len(ns) > 1}


def resolve(question: str, visible_ids: set[str], store: MemoryStore, idf: dict[str, float]) -> dict:
    """{} when the question has no ambiguous bare first name; otherwise
    {"first", "candidates": [(name, score)], "resolved": name|None, "ambiguous": bool}."""
    people, involved = directory(store)
    amb = ambiguous_first_names(store)
    q_low = question.lower()
    q_tokens = re.findall(r"[a-z0-9]+", q_low)
    for first, names in amb.items():
        if first not in q_tokens:
            continue
        # a surname already in the question disambiguates (Sarah Kim, "Kim's")
        explicit = [n for n in names if all(part.lower() in q_tokens for part in n.split())]
        if explicit:
            return {"first": first, "candidates": [(explicit[0], 1.0)], "resolved": explicit[0], "ambiguous": False, "by": "name"}
        stop = QUESTION_WORDS | {first} | {part.lower() for n in names for part in n.split()}
        words = [w for w in dict.fromkeys(tokenize(question)) if w not in stop]
        # Discriminative fit: for each remaining question word, how much more often does candidate X use it than the
        # other candidates do (rate per token, so a talkative person is not favoured)? Weighted by rarity (idf).
        profiles: dict[str, tuple[dict[str, int], int]] = {}
        for name in names:
            counts: dict[str, int] = defaultdict(int)
            n_tok = 0
            for uid in involved[name] & visible_ids:
                for t in tokenize(store.get(uid).text):
                    counts[t] += 1
                    n_tok += 1
            profiles[name] = (counts, max(n_tok, 1))
        total = 0.0
        fit = {name: 0.0 for name in names}
        for w in words:
            rates = {n: profiles[n][0].get(w, 0) / profiles[n][1] for n in names}
            denom = sum(rates.values())
            if denom <= 0:
                continue
            weight = idf.get(w, 1.0)
            total += weight
            for n in names:
                fit[n] += weight * rates[n] / denom
        scored = []
        for name in names:
            org = sum(1 for stem in people[name].org_stems for w in q_tokens if len(w) >= 4 and stem.startswith(w))
            scored.append((name, (fit[name] / total if total else 0.0) + (0.5 if org else 0.0)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        best, second = scored[0], scored[1]
        enough = total >= MIN_EVIDENCE
        picked = enough and best[1] > 0 and best[1] >= WIN_RATIO * max(second[1], 1e-9)
        strong = picked and best[1] >= STRONG_SHARE
        return {"first": first, "candidates": scored, "resolved": best[0] if picked else None,
                "ambiguous": not strong, "by": "context" if picked else "tie"}
    return {}


def person_ids(name: str, visible_ids: set[str], store: MemoryStore) -> list[str]:
    """Visible unit ids written by / addressed to / mentioning `name`, time-ordered."""
    _, involved = directory(store)
    return sorted(involved.get(name, set()) & visible_ids, key=lambda i: (store.get(i).time, i))
