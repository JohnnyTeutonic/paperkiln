#!/usr/bin/env python3
"""Carry-12: a 12-14 turn chat carry-over benchmark for tiny language models (DRAFT v0).

    python tools/carry12_build.py build [--out corpus/eval/carry12_v0_draft.jsonl]
    python tools/carry12_build.py summary [--set FILE] [--out FILE.json]
    python tools/carry12_build.py budget --vocab V.gguf [--set FILE] [--reply-tokens 24,48] --out FILE.json
    python tools/carry12_build.py decontam --out REPORT.json [--state STATE.json] CORPUS [CORPUS ...]

The long-carry companion of Carry-5 (tools/carry5_build.py; experiments/tiny_chat_long_2026-10/
PREREGISTRATION_DRAFT.md). Jonathan, 24 Oct 2026: tiny chat models should carry a conversation for at
least a dozen turns. Every item is a scripted conversation of 12-14 user turns with a canned short
assistant reply per turn (the GOLD history), one TARGET turn (10-14) whose reply is scored, and four
forced-choice candidates in a shared answer frame, exactly as Carry-5: the slot, topic, instruction and
knowledge templates and every value list are imported from carry5_build, so the two benchmarks differ
only in length, distance and the small talk between the facts.

Families:
  recall       a fact about the user planted at turn 1-4, asked at turn 10-14; 0-3 same-type
               distractor facts about other people, in the early window (turns 1-4) or the late window
               (turn 5 to the turn before the target), alternating by cell.
  correction   the fact planted, then corrected ("..., not X"); early corrections (plant 1-2, fix 3-4:
               the corrected value then waits 6-11 turns) and late corrections (plant 1-3, fix 7-9: the
               wrong value was held for six turns); 0-2 further distractors.
  topic        a topic with a detail at turn 1-4, a digression of small talk ("By the way, ..."), then
               "Anyway, back to ..." at turn 10-14; 0-2 same-type distractors inside the digression.
  load         a fact planted at turn 1-4 beside two facts of OTHER slots about the user (no same-type
               distractor): several facts carried at once.
  instruction  a standing instruction (carry5's four kinds) at turn 1-3 with compliant question-answer
               turns up to the target at 10, 12 or 14. Descriptive: its forced choice is length-confounded
               (Carry-5 pre-registration A2.5); read it from generation.
  knowledge    a common-knowledge question at turn 10-14 after small talk (carry5's 60 questions): does
               a long context alone hurt plain question answering?

Turns that hold no event are natural small talk from hand-written templates (SMALL_TALK, ARCS: two-turn
exchanges on one theme), never model text; turn 1 opens, the last turn after the target closes. No small
talk names a value of any list or touches a slot's subject (names, pets, homes, work, ages, favourites,
trips, cakes, books, gardens, parties, music), so it never acts as an unlabelled distractor.

Controls on every item, as Carry-5: `twin` (the statements and the question in one user turn), `noctx`
(the question alone), `interference`, `distance` (target minus the turn holding the needed fact), and
new: `near`, the same event turns and question with the small talk removed (a short multi-turn
rendering, so twin -> near is the cost of splitting into turns and near -> full is the cost of the
distance), with `near_target` its target turn.

The file is DRAFT v0, never a registered benchmark: corpus/eval/carry12_v0_draft.jsonl carries
"draft": true in every item and is not pinned. The frozen v1 is built by this tool (VERSION and the
counts fixed then) in the commit that freezes the pre-registration, and its sha256 pinned in
corpus/hashes/carry12_v1.sha256. It is an evaluation set: nothing here may enter training data.

`budget` tokenises every item in the served chat template (<|user|>U<|assistant|>A<|endoftext|>) with
the 16k Llama 3-scheme vocabulary, for the gold history and for self history with every earlier
assistant reply assumed to be N tokens; `decontam` runs carry5_build's contamination scan on this set.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import carry5_build as C5  # noqa: E402
from carry5_build import (ACK_FACT, ACK_FIX, ACK_NAME, ACK_OTHER, ACK_TOPIC, ANIMALS, CLOSINGS,  # noqa: E402
                          FAVOURITES, INSTRUCTIONS, KNOWLEDGE, KNOWLEDGE_ASK, LONG, NEUTRAL, ONE_WORD_QA,
                          OPENERS, RELATIVES, SLOTS, TOPICS, VERBOSE, Builder, article, number_words,
                          plant_templates, shuffle_candidates, words)

OUT = os.path.join(ROOT, "corpus", "eval", "carry12_v0_draft.jsonl")
VERSION = "carry12_v0_draft"
SEED = 20261024
TARGETS = (10, 11, 12, 13, 14)
PLANTS = (1, 2, 3, 4)

# ---------------------------------------------------------------- small talk (hand-written)
SMALL_TALK = [
    ("What is a good way to relax in the evening?", "A warm drink and some quiet time can help you unwind."),
    ("Can you suggest a simple dinner idea?", "Pasta with some cheese and herbs is quick and simple."),
    ("How can I sleep better?", "Keep a regular bedtime and put your phone away before bed."),
    ("Any tips for a rainy afternoon?", "You could do a puzzle, watch a film or bake some bread."),
    ("How do I stop feeling so tired after lunch?", "A short walk and a glass of water can help."),
    ("Do you have a tip for staying focused?", "Try short blocks of effort with small breaks in between."),
    ("How can I keep my kitchen tidy?", "Wash up straight after cooking and wipe the counters each night."),
    ("What should I pack for a day at the beach?", "Sun cream, a hat, water and a towel."),
    ("Can you tell me something interesting?", "Honey never really goes off if it is kept sealed."),
    ("Is it good to stretch in the morning?", "Yes, a gentle stretch can wake your body up."),
    ("How do I make a room feel cosier?", "Soft lights, a blanket and a few cushions make a big difference."),
    ("What can I do if I feel a bit low?", "Talking to someone you trust or getting some fresh air can help."),
    ("How often should I wash my towels?", "About once a week is a good rule."),
    ("What is a good snack for the afternoon?", "A handful of nuts or some yoghurt is a nice choice."),
    ("How do I remember to take my keys?", "Keep a small bowl by the door and always put them there."),
    ("Any ideas for a quiet evening in?", "A film, a warm bath or an early night all sound good."),
    ("How do I keep my shoes from smelling?", "Let them dry out fully and sprinkle in a little bicarbonate of soda."),
    ("What is a polite way to end a phone call?", "Thank them for calling and say you need to go."),
    ("How can I wake up earlier?", "Go to bed a little earlier and put your alarm across the room."),
    ("What should I do on a hot day?", "Drink plenty of water and stay in the shade."),
    ("How do I get rid of hiccups?", "Sipping cold water slowly often helps."),
    ("Is it fine to have a lazy day sometimes?", "Yes, rest is important too."),
    ("How can I make my mornings less rushed?", "Get your things ready the night before."),
    ("What is a good way to meet new people?", "Joining a local club or class is a friendly way to start."),
    ("How do I fold a fitted sheet?", "Tuck the corners into each other, then fold it into a square."),
    ("Can you suggest a board game for two people?", "Chess or draughts are great for two."),
    ("How do I keep bread fresh for longer?", "Keep it in a bread bin or freeze a few slices."),
    ("What is a good habit to start?", "Writing down three good things each evening is a nice one."),
    ("How can I stop losing my phone charger?", "Give it one fixed spot and keep it there."),
    ("Why do I feel sleepy on dull days?", "Darker skies can make your body feel ready to rest."),
    ("How long should I boil an egg?", "About seven minutes gives a firm middle."),
    ("How do I clean a window without streaks?", "Use warm water with a little vinegar and dry it with paper."),
    ("Is it better to shower in the morning or at night?", "Either is fine, so pick what helps you feel fresh."),
    ("How can I stay calm before a big day?", "Take slow breaths and get everything ready early."),
    ("What should I do with a pile of newspapers?", "You can recycle them or use them to wrap things."),
    ("How do I stop a door from squeaking?", "A drop of oil on the hinges usually fixes it."),
    ("What is a good way to spend a sunny afternoon?", "Sitting outside with a cold drink sounds lovely."),
    ("How can I drink less fizzy drink?", "Try sparkling water with a slice of cucumber."),
    ("What can I cook with leftover rice?", "Fried rice with an egg and some peas is quick."),
    ("How do I make my bed look neat?", "Smooth the sheet, plump the pillows and fold the blanket back."),
    ("Do you know a good way to cool down a room?", "Close the curtains in the day and open the windows at night."),
    ("What is a nice way to say thank you?", "A short handwritten note means a lot."),
    ("How can I make a long day go faster?", "Break it into small parts and plan a treat for the end."),
    ("What should I keep in a first aid box?", "Plasters, bandages, a pair of scissors and some cream."),
    ("How do I stop my glasses from fogging up?", "Wash them with a drop of soap and let them dry."),
    ("Can you suggest a game to play with friends?", "Charades is fun and needs no equipment."),
]
ARCS = [
    [("It has been raining all morning here.", "Rainy mornings can feel slow. Are you staying inside?"),
     ("Yes, I think I will stay in and tidy up.", "That sounds like a good use of a wet day.")],
    [("I watched a funny film last night.", "Oh, nice. Was it a good one?"),
     ("It was silly, but it made me laugh a lot.", "A good laugh is always worth it.")],
    [("I am trying to drink more water during the day.", "That is a healthy habit. How is it going?"),
     ("I keep forgetting, so I set a reminder.", "A reminder is a clever way to keep it up.")],
    [("I made soup for lunch today.", "Lovely. Was it a hearty one?"),
     ("Yes, thick and warm, just right for today.", "That sounds perfect for a cool day.")],
    [("I could not sleep well last night.", "I am sorry to hear that. Was something keeping you awake?"),
     ("There was a lot of noise outside until late.", "That is annoying. Earplugs might help next time.")],
    [("I started a jigsaw puzzle yesterday.", "That sounds relaxing. Is it a big one?"),
     ("It has a thousand pieces, so it will take a while.", "Take your time and enjoy it.")],
    [("My kettle broke this morning.", "Oh no. Did you manage to make a hot drink?"),
     ("I boiled some water in a pan instead.", "Good thinking, that is a handy trick.")],
    [("I went for a long walk by the sea.", "That sounds refreshing. Was it windy?"),
     ("Very windy, but the fresh air was nice.", "Sea air can really clear your head.")],
    [("I am learning to knit a scarf.", "What a nice skill to learn. Is it going well?"),
     ("It is slow, but I am getting better.", "Practice makes it easier every time.")],
    [("I tried a new recipe for dinner.", "How exciting. Did it turn out well?"),
     ("It was nice, though a bit too salty.", "Next time you can add a little less salt.")],
    [("The bus was late again today.", "That is frustrating. Did it make you late?"),
     ("Only by a few minutes.", "At least it was not too bad.")],
    [("I finally cleaned out my wardrobe.", "Well done! Did you find anything you had forgotten?"),
     ("A warm coat I really like.", "What a nice surprise.")],
    [("I have a bit of a cold today.", "Sorry to hear that. Are you resting?"),
     ("Yes, with lots of hot drinks and tissues.", "Good. Rest is the best thing for a cold.")],
    [("The sky was so clear last night.", "Lovely. Could you see many stars?"),
     ("Lots of them, and the moon was very bright.", "That must have been a beautiful sight.")],
]
EXTRA_OPENERS = [("Hi! Do you have time for a chat?", "Yes, I would love to chat. What is new with you?"),
                 ("Hello again! I hope you are well.", "I am well, thank you. How are you?")]
CLOSING_EXTRA = [("Right, I should get going now.", "Okay! It was lovely talking with you.")]

# A filler must not raise any slot's subject: these words would turn small talk into an unlabelled cue.
SUBJECT_WORDS = {"name", "called", "live", "lives", "home", "work", "works", "job", "old", "age", "favourite",
                 "trip", "travel", "travelling", "holiday", "cake", "baking", "book", "reading", "garden",
                 "planting", "party", "birthday", "music", "lessons", "instrument", "pet", "dog", "cat",
                 "brother", "sister", "cousin", "friend", "neighbour", "aunt", "uncle", "colleague", "mum",
                 "dad"} | set(ANIMALS)


def lower_first(text):
    """'By the way, ' + text: lower-case the first letter unless the first word is I or a proper noun."""
    first = text.split()[0]
    if first in ("I", "I'm") or first[:2] == "I'":
        return text
    return text[0].lower() + text[1:]


class Item:
    """Turns 1..n with their events; compose() fills the rest with small talk."""

    def __init__(self, n_turns, target):
        self.n, self.target = n_turns, target
        self.ev = {t: [] for t in range(1, n_turns + 1)}

    def put(self, turn, ev):
        self.ev[turn].append(ev)


def n_turns_for(target):
    """12 turns at the least, at least one closing turn after the target, at most 14."""
    return min(14, max(12, target + 1))


def compose(item, rng, digression_at=None):
    """[{user, gold}] for turns 1..n. Event turns join their events' text; empty turns before the target
    take small talk (two-turn arcs where two empty turns follow each other, single exchanges otherwise),
    turn 1 an opener; turns after the target small talk then a closing. Nothing repeats in an item."""
    singles = list(SMALL_TALK)
    rng.shuffle(singles)
    arcs = [list(a) for a in ARCS]
    rng.shuffle(arcs)
    closing = rng.choice(CLOSINGS + CLOSING_EXTRA)
    out = [None] * item.n
    t = 1
    while t <= item.n:
        evs = item.ev[t]
        if evs:
            text = " ".join(e["text"] for e in evs)
            gold = next((e["gold"] for e in evs if e.get("main")), evs[-1]["gold"])
            out[t - 1] = {"user": text, "gold": gold, "kind": "+".join(e["kind"] for e in evs)}
            t += 1
            continue
        if t == item.n and t > item.target:
            out[t - 1] = {"user": closing[0], "gold": closing[1], "kind": "closing"}
            t += 1
            continue
        if t == 1:
            q, a = rng.choice(OPENERS + EXTRA_OPENERS)
            out[0] = {"user": q, "gold": a, "kind": "opener"}
            t += 1
            continue
        nxt_empty = t + 1 <= item.n and not item.ev[t + 1] and not (t + 1 == item.n and t + 1 > item.target)
        if nxt_empty and t != item.target - 1 and arcs and rng.random() < 0.5:
            arc = arcs.pop()
            for k, (q, a) in enumerate(arc):
                out[t - 1 + k] = {"user": q, "gold": a, "kind": "small_talk"}
            t += len(arc)
            continue
        q, a = singles.pop()
        out[t - 1] = {"user": q, "gold": a, "kind": "small_talk"}
        t += 1
    if digression_at:
        turn = out[digression_at - 1]
        assert turn["kind"] == "small_talk", turn
        turn["user"] = "By the way, " + lower_first(turn["user"])
        turn["kind"] = "digression"
    return out


def near_of(conv, target):
    """The event turns before the target, in order, then the target turn: the conversation without its
    small talk (openers, digressions and closings removed too)."""
    keep = [{"user": t["user"], "gold": t["gold"]} for t in conv[:target - 1]
            if t["kind"] not in ("opener", "small_talk", "digression", "closing")]
    keep.append({"user": conv[target - 1]["user"], "gold": conv[target - 1]["gold"]})
    return keep


def build_items(seed=SEED):
    b = Builder(seed)
    rng = b.rng
    items = []
    seen = set()

    def slot_values(slot, ctx):
        if slot == "name":
            return C5.NAMES, lambda v: [v.lower()], lambda v: v
        if slot == "pet":
            return C5.PET_NAMES, lambda v: [v.lower()], lambda v: v
        if slot == "city":
            return C5.CITIES, lambda v: [v.lower()], lambda v: v
        if slot == "job":
            return C5.JOBS, lambda v: [v], article
        if slot == "age":
            return list(range(18, 80)), lambda v: number_words(v), str
        return FAVOURITES[ctx["c"]], lambda v: [v], lambda v: v

    def fill(tpl, ctx, **kw):
        return tpl.format(**{**ctx, **kw})

    def unique(make):
        for _ in range(200):
            it = make()
            keys = (("twin", it["twin"]), ("turns", tuple(t["user"] for t in it["turns"])))
            if not any(k in seen for k in keys):
                seen.update(keys)
                items.append(it)
                return
        raise ValueError(f"no unique {it['family']} item after 200 draws")

    def windows(target, taken, where, k):
        """k distinct free turns for distractors: early = 1-4, late = 5 .. target-1."""
        lo, hi = (1, 4) if where == "early" else (5, target - 1)
        free = [t for t in range(lo, hi + 1) if t not in taken]
        if len(free) < k:
            free = [t for t in range(1, target) if t not in taken]
        return sorted(rng.sample(free, k))

    def finish(fam, slot, item, conv, q, ctx_fields, statements):
        target = item.target
        out = {"family": fam, "slot": slot, "n_turns": item.n, "turns": [{"user": t["user"], "gold": t["gold"]}
                                                                           for t in conv],
               "turn_kinds": [t["kind"] for t in conv], "target_turn": target}
        out.update(ctx_fields)
        near = near_of(conv, target)
        out.update({"twin": " ".join(statements + [q]), "near": near, "near_target": len(near)})
        return out

    def slot_draw(slot, n, key_prefix):
        spec = SLOTS[slot]
        ctx = {"a": rng.choice(ANIMALS), "c": list(FAVOURITES)[n % 3]}
        values, acc, show = slot_values(slot, ctx)
        gold = b.cycle((key_prefix, slot, ctx.get("c") if slot == "favourite" else ""), values)
        frame = spec["frame"].replace("{a}", ctx["a"]).replace("{c}", ctx["c"])
        return spec, ctx, values, acc, show, gold, frame

    # ---- recall: slot x plant 1-4 x target 10-14 x interference 0-3, distractor window alternating
    def recall(n, slot, plant, target, inter):
        spec, ctx, values, acc, show, gold, frame = slot_draw(slot, n, "recall")
        dis = b.others(values, {gold}, inter)
        fill_k = b.others(values, {gold, *dis}, 3 - inter)
        rels = rng.sample(NEUTRAL if slot == "name" else RELATIVES, inter)
        where = "early" if (n // 4) % 2 == 0 else "late"  # n // 4 indexes (slot, plant, target): balanced per level
        item = Item(n_turns_for(target), target)
        plant_txt = fill(rng.choice(plant_templates(spec, plant)), ctx, v=show(gold))
        item.put(plant, {"text": plant_txt, "gold": rng.choice(ACK_NAME if slot == "name" else ACK_FACT),
                         "main": True, "kind": "plant"})
        dturns = windows(target, {plant}, where, inter) if inter else []
        dis_txt = []
        for t, d, r in zip(dturns, dis, rels):
            s = fill(rng.choice(spec["distractor"]), ctx, rel=r, v=show(d))
            dis_txt.append((t, s))
            item.put(t, {"text": s, "gold": rng.choice(ACK_OTHER), "main": True, "kind": "distractor"})
        q = fill(rng.choice(spec["question"]), ctx)
        item.put(target, {"text": q, "gold": frame.format(show(gold)), "main": True, "kind": "question"})
        conv = compose(item, rng)
        statements = [s for _, s in sorted([(plant, plant_txt)] + dis_txt)]
        cands, gi, cacc = shuffle_candidates(rng, show(gold), [show(v) for v in dis + fill_k],
                                             [acc(v) for v in [gold] + dis + fill_k])
        return finish("recall", slot, item, conv, q, {
            "fact_turn": plant, "distance": target - plant, "interference": inter, "distractor_turns": dturns,
            "distractor_window": where if inter else None, "distractor_values": [show(v) for v in dis],
            "answers": acc(gold), "frame": frame, "candidates": cands, "gold_index": gi,
            "candidate_answers": cacc, "check": None, "noctx": q}, statements)

    cells = list(itertools.product(SLOTS, PLANTS, TARGETS, (0, 1, 2, 3)))
    for n, (slot, plant, target, inter) in enumerate(cells):
        unique(lambda: recall(n, slot, plant, target, inter))

    # ---- correction: slot x (plant, fix) x target x interference 0-2
    def correction(n, slot, plant, corr, target, inter):
        spec, ctx, values, acc, show, new, frame = slot_draw(slot, n, "fix")
        old = b.others(values, {new}, 1)[0]
        dis = b.others(values, {new, old}, inter)
        fill_k = b.others(values, {new, old, *dis}, 2 - inter)
        rels = rng.sample(NEUTRAL if slot == "name" else RELATIVES, inter)
        item = Item(n_turns_for(target), target)
        plant_txt = fill(rng.choice(plant_templates(spec, plant)), ctx, v=show(old))
        fix_txt = fill(rng.choice(spec["correction"]), ctx, new=show(new), old=show(old))
        item.put(plant, {"text": plant_txt, "gold": rng.choice(ACK_NAME if slot == "name" else ACK_FACT),
                         "main": True, "kind": "plant"})
        item.put(corr, {"text": fix_txt, "gold": rng.choice(ACK_FIX), "main": True, "kind": "correction"})
        dturns = sorted(rng.sample([t for t in range(1, target) if t not in (plant, corr)], inter))
        dis_txt = []
        for t, d, r in zip(dturns, dis, rels):
            s = fill(rng.choice(spec["distractor"]), ctx, rel=r, v=show(d))
            dis_txt.append((t, s))
            item.put(t, {"text": s, "gold": rng.choice(ACK_OTHER), "main": True, "kind": "distractor"})
        q = fill(rng.choice(spec["question"]), ctx)
        item.put(target, {"text": q, "gold": frame.format(show(new)), "main": True, "kind": "question"})
        conv = compose(item, rng)
        statements = [s for _, s in sorted([(plant, plant_txt), (corr, fix_txt)] + dis_txt)]
        cands, gi, cacc = shuffle_candidates(rng, show(new), [show(v) for v in [old] + dis + fill_k],
                                             [acc(v) for v in [new, old] + dis + fill_k])
        return finish("correction", slot, item, conv, q, {
            "fact_turn": corr, "plant_turn": plant, "correction_kind": "early" if corr <= 4 else "late",
            "held_turns": corr - plant, "distance": target - corr, "interference": inter,
            "distractor_turns": dturns, "distractor_values": [show(v) for v in dis], "old_value": show(old),
            "answers": acc(new), "frame": frame, "candidates": cands, "gold_index": gi,
            "candidate_answers": cacc, "check": None, "noctx": q}, statements)

    pairs = ((1, 3), (2, 4), (1, 7), (3, 9))
    cells = list(itertools.product(SLOTS, pairs, TARGETS, (0, 1, 2)))
    for n, (slot, (plant, corr), target, inter) in enumerate(cells):
        unique(lambda: correction(n, slot, plant, corr, target, inter))

    # ---- topic return: kind x plant 1-4 x target 10-14 x interference 0-2
    def topic(kind, plant, target, inter):
        spec = TOPICS[kind]
        values = spec["values"]
        gold = b.cycle(("topic", kind), values)
        dis = b.others(values, {gold}, inter)
        fill_k = b.others(values, {gold, *dis}, 3 - inter)
        rels = rng.sample(RELATIVES, inter)
        item = Item(n_turns_for(target), target)
        plant_txt = rng.choice(spec["plant"]).format(v=gold)
        item.put(plant, {"text": plant_txt, "gold": rng.choice(ACK_TOPIC), "main": True, "kind": "plant"})
        dturns = sorted(rng.sample(range(plant + 2, target), inter))
        dist_sent = []
        for t, d, r in zip(dturns, dis, rels):
            s = rng.choice(spec["distractor"]).format(rel=r, v=d)
            dist_sent.append(s)
            item.put(t, {"text": s, "gold": rng.choice(ACK_OTHER), "main": True, "kind": "distractor"})
        q = spec["question"]
        item.put(target, {"text": q, "gold": spec["frame"].format(gold), "main": True, "kind": "question"})
        conv = compose(item, rng, digression_at=plant + 1)
        twin_q = rng.choice(spec["twin_q"])
        cands, gi, cacc = shuffle_candidates(rng, gold, dis + fill_k, [[v.lower()] for v in [gold] + dis + fill_k])
        out = finish("topic", kind, item, conv, twin_q, {
            "fact_turn": plant, "distance": target - plant, "interference": inter, "distractor_turns": dturns,
            "distractor_values": list(dis), "answers": [gold.lower()], "frame": spec["frame"],
            "candidates": cands, "gold_index": gi, "candidate_answers": cacc, "check": None, "noctx": twin_q},
            [plant_txt] + dist_sent)
        return out

    cells = list(itertools.product(TOPICS, PLANTS, TARGETS, (0, 1, 2)))
    for kind, plant, target, inter in cells:
        unique(lambda: topic(kind, plant, target, inter))

    # ---- load: the queried fact beside two facts of other slots, all in turns 1-4
    def load(n, slot, plant, target):
        spec, ctx, values, acc, show, gold, frame = slot_draw(slot, n, "load")
        fill_k = b.others(values, {gold}, 3)
        item = Item(n_turns_for(target), target)
        plant_txt = fill(rng.choice(plant_templates(spec, plant)), ctx, v=show(gold))
        item.put(plant, {"text": plant_txt, "gold": rng.choice(ACK_NAME if slot == "name" else ACK_FACT),
                         "main": True, "kind": "plant"})
        other_slots = rng.sample([s for s in SLOTS if s != slot], 2)
        oturns = sorted(rng.sample([t for t in PLANTS if t != plant], 2))
        texts = [(plant, plant_txt)]
        for t, s2 in zip(oturns, other_slots):
            sp2 = SLOTS[s2]
            c2 = {"a": rng.choice([a for a in ANIMALS if a != ctx["a"]]),
                  "c": rng.choice([c for c in FAVOURITES if c != ctx["c"]])}
            v2s, _, sh2 = slot_values(s2, c2)
            v2 = rng.choice(v2s)
            txt = fill(rng.choice(plant_templates(sp2, t)), c2, v=sh2(v2))
            texts.append((t, txt))
            item.put(t, {"text": txt, "gold": rng.choice(ACK_NAME if s2 == "name" else ACK_FACT), "main": True,
                         "kind": "load"})
        q = fill(rng.choice(spec["question"]), ctx)
        item.put(target, {"text": q, "gold": frame.format(show(gold)), "main": True, "kind": "question"})
        conv = compose(item, rng)
        cands, gi, cacc = shuffle_candidates(rng, show(gold), [show(v) for v in fill_k],
                                             [acc(v) for v in [gold] + fill_k])
        return finish("load", slot, item, conv, q, {
            "fact_turn": plant, "distance": target - plant, "interference": 0, "load_slots": other_slots,
            "load_turns": oturns, "distractor_turns": [], "distractor_values": [], "answers": acc(gold),
            "frame": frame, "candidates": cands, "gold_index": gi, "candidate_answers": cacc, "check": None,
            "noctx": q}, [s for _, s in sorted(texts)])

    cells = list(itertools.product(SLOTS, PLANTS, TARGETS))
    for n, (slot, plant, target) in enumerate(cells):
        unique(lambda: load(n, slot, plant, target))

    # ---- instruction (descriptive): kind x instruction turn 1-3 x target 10/12/14, twice
    def instruction(kind, at, target):
        spec = INSTRUCTIONS[kind]
        tq = b.cycle(("inst", kind), spec["qa"])
        pool = list(spec["qa"])
        if kind == "short":  # one-word answers comply with "five words or fewer" too
            have = {x[0] for x in pool}
            pool += [x for x in ONE_WORD_QA if x[0] not in have]
        rest = [x for x in pool if x[0] != tq[0]]
        rng.shuffle(rest)
        say = rng.choice(spec["say"])
        item = Item(n_turns_for(target), target)
        item.put(at, {"text": f"{say} {rest[0][0]}", "gold": rest[0][1], "main": True, "kind": "instruction"})
        for k, t in enumerate(range(at + 1, target), start=1):
            item.put(t, {"text": rest[k][0], "gold": rest[k][1], "main": True, "kind": "compliant"})
        q, short, s, ans = tq
        item.put(target, {"text": q, "gold": short, "main": True, "kind": "question"})
        conv = compose(item, rng)
        bad = [t.format(S=s[0].upper() + s[1:], s=s) for t in (LONG if kind == "short" else VERBOSE)]
        cands, gi, _ = shuffle_candidates(rng, short, bad, [ans] * 4)
        out = finish("instruction", kind, item, conv, q, {
            "fact_turn": at, "distance": target - at, "interference": 0, "distractor_turns": [],
            "distractor_values": [], "answers": ans, "frame": "{}", "candidates": cands, "gold_index": gi,
            "candidate_answers": None, "check": {"kind": kind}, "noctx": q}, [say])
        return out

    cells = list(itertools.product(INSTRUCTIONS, (1, 2, 3), (10, 12, 14)))
    for _rep in range(2):
        for kind, at, target in cells:
            unique(lambda: instruction(kind, at, target))

    # ---- knowledge: carry5's 60 questions at turn 10-14 after small talk
    def knowledge(n, q, frame, gold, dis, acc):
        target = TARGETS[n % len(TARGETS)]
        ask = KNOWLEDGE_ASK[n % len(KNOWLEDGE_ASK)].format(q=q)
        item = Item(n_turns_for(target), target)
        item.put(target, {"text": ask, "gold": frame.format(gold), "main": True, "kind": "question"})
        conv = compose(item, rng)
        cands, gi, cacc = shuffle_candidates(rng, gold, dis, [acc] + [[d.lower()] for d in dis])
        out = finish("knowledge", "knowledge", item, conv, ask, {
            "fact_turn": None, "distance": None, "interference": 0, "distractor_turns": [],
            "distractor_values": [], "answers": acc, "frame": frame, "candidates": cands, "gold_index": gi,
            "candidate_answers": cacc, "check": None, "noctx": q}, [])
        return out

    for n, k in enumerate(KNOWLEDGE):
        unique(lambda: knowledge(n, *k))

    counts = {}
    out = []
    for it in items:
        fam = it["family"]
        counts[fam] = counts.get(fam, 0) + 1
        out.append({"id": f"{VERSION}:{fam}:{counts[fam]:04d}", "version": VERSION, "draft": True, **it})
    validate(out)
    return out


def all_values():
    values = set()
    for vs in [C5.NAMES, C5.PET_NAMES, C5.CITIES, C5.JOBS, *FAVOURITES.values(),
               *(t["values"] for t in TOPICS.values())]:
        values |= {w for v in vs for w in words(v)}
    return values


def validate(items):
    """Build-time checks: Carry-5's (four distinct candidates, the gold reply is the frame filled with the
    gold candidate, the needed fact before the target, distractors visible before the target and in the
    twin) plus: 12-14 turns, a target of 10-14, facts at 1-4 for recall/topic/load, small talk naming no
    list value and no slot subject, every item's near rendering ending in its target question."""
    values = all_values()
    canned = [t for pair in OPENERS + EXTRA_OPENERS + SMALL_TALK + CLOSINGS + CLOSING_EXTRA for t in pair]
    canned += [t for arc in ARCS for pair in arc for t in pair]
    canned += ACK_FACT + ACK_NAME + ACK_OTHER + ACK_FIX + ACK_TOPIC
    for text in canned:
        clash = values & set(words(text))
        if clash:
            raise ValueError(f"canned text {text!r} names list values {sorted(clash)}")
    for text in [t for pair in SMALL_TALK + EXTRA_OPENERS + CLOSING_EXTRA for t in pair] + \
            [t for arc in ARCS for pair in arc for t in pair]:
        clash = SUBJECT_WORDS & set(words(text))
        if clash:
            raise ValueError(f"small talk {text!r} raises a slot subject {sorted(clash)}")
    if len({q for q, _ in SMALL_TALK} | {q for arc in ARCS for q, _ in arc}) != len(SMALL_TALK) + 2 * len(ARCS):
        raise ValueError("small talk questions repeat")
    for it in items:
        n, tt = it["n_turns"], it["target_turn"]
        assert len(it["turns"]) == n and 12 <= n <= 14 and 10 <= tt <= 14 and tt <= n, it["id"]
        assert len(set(it["candidates"])) == 4, it["id"]
        gold = it["turns"][tt - 1]["gold"]
        assert gold == it["frame"].format(it["candidates"][it["gold_index"]]), it["id"]
        if it["family"] in ("recall", "topic", "load"):
            assert 1 <= it["fact_turn"] <= 4, it["id"]
        if it["fact_turn"]:
            assert it["fact_turn"] < tt, it["id"]
        if it["family"] != "instruction":
            assert any(a in gold.lower() for a in it["answers"]), it["id"]
        assert len(it["distractor_values"]) == it["interference"], it["id"]
        assert all(t < tt for t in it["distractor_turns"]), it["id"]
        before = " ".join(t["user"] for t in it["turns"][:tt - 1])
        for v in it["distractor_values"] + ([it["old_value"]] if "old_value" in it else []):
            assert re.search(r"\b" + re.escape(v) + r"\b", before), (it["id"], v)
            assert v in it["candidates"], (it["id"], v)
            assert re.search(r"\b" + re.escape(v) + r"\b", it["twin"]), (it["id"], v)
        assert it["candidates"][it["gold_index"]] not in it["distractor_values"], it["id"]
        assert it["near"][-1]["user"] == it["turns"][tt - 1]["user"], it["id"]
        assert it["near_target"] == len(it["near"]), it["id"]
        kinds = it["turn_kinds"]
        assert kinds[tt - 1].endswith("question"), it["id"]
        assert all(k != "closing" for k in kinds[:tt]), it["id"]


def dump(items):
    return "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")


def summarise(items):
    by = {}
    for it in items:
        f = by.setdefault(it["family"], {"items": 0})
        f["items"] += 1
        keys = [("slot", it["slot"]), ("target_turn", it["target_turn"]), ("distance", it["distance"]),
                ("interference", it["interference"]), ("gold_index", it["gold_index"]),
                ("n_turns", it["n_turns"]), ("fact_turn", it["fact_turn"])]
        if it.get("distractor_window"):
            keys.append(("distractor_window", it["distractor_window"]))
        if it.get("correction_kind"):
            keys.append(("correction_kind", it["correction_kind"]))
        small = sum(1 for k in it["turn_kinds"][:it["target_turn"] - 1]
                    if k in ("small_talk", "digression", "opener"))
        keys.append(("small_talk_turns_before_target", small))
        for k, v in keys:
            f.setdefault(k, {})
            f[k][str(v)] = f[k].get(str(v), 0) + 1
    return {"items": len(items), "families": by}


def cmd_build(a):
    data = dump(build_items())
    sha = hashlib.sha256(data).hexdigest()
    with open(a.out, "wb") as f:
        f.write(data)
    summary = summarise([json.loads(x) for x in data.decode("utf-8").splitlines()])
    print(json.dumps({"items": summary["items"], "by_family": {k: v["items"] for k, v in summary["families"].items()},
                      "sha256": sha, "out": a.out, "draft": True}, indent=1))
    return 0


def cmd_summary(a):
    items = [json.loads(x) for x in open(a.set, encoding="utf-8") if x.strip()]
    rep = summarise(items) | {"set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"),
                              "sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(),
                              "rebuild_equals_file": dump(build_items()) == open(a.set, "rb").read()}
    text = json.dumps(rep, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    print(text)
    return 0


# ---------------------------------------------------------------- context budget
EOT, USER, ASSISTANT = "<|endoftext|>", "<|user|>", "<|assistant|>"


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def dist(xs, limits=(512, 1024, 2048)):
    return {"n": len(xs), "min": min(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "p99": pct(xs, 0.99),
            "max": max(xs), "mean": round(sum(xs) / len(xs), 1),
            **{f"over_{t}": sum(1 for x in xs if x > t) for t in limits}}


def cmd_budget(a):
    import llama3_bpe
    tok = llama3_bpe.from_gguf(a.vocab)

    def n(text):
        return len(tok.encode(text, add_special_tokens=False).ids)

    items = [json.loads(x) for x in open(a.set, encoding="utf-8") if x.strip()]
    reply_lens = [int(x) for x in a.reply_tokens.split(",")]
    cache = {}

    def nc(text):
        if text not in cache:
            cache[text] = n(text)
        return cache[text]

    rows = []
    check_whole = 0
    for idx, it in enumerate(items):
        tt = it["target_turn"]
        turns = it["turns"]
        users = [nc(t["user"]) for t in turns]
        golds = [nc(t["gold"]) for t in turns]
        # One exchange = <|user|> U <|assistant|> A <|endoftext|>: 3 special tokens; specials are split out
        # before the pre-tokeniser, so a rendering's length is the sum of its pieces (checked below).
        target_gold = sum(users[:tt - 1]) + sum(golds[:tt - 1]) + 3 * (tt - 1) + users[tt - 1] + 2 + golds[tt - 1] + 1
        r = {"id": it["id"], "family": it["family"], "n_turns": it["n_turns"], "target_turn": tt,
             "gold": target_gold,
             "twin": 2 + nc(it["twin"]) + golds[tt - 1] + 1,
             "noctx": 2 + nc(it["noctx"]) + golds[tt - 1] + 1,
             "near": sum(nc(t["user"]) + nc(t["gold"]) + 3 for t in it["near"]),
             "full_gold": sum(users) + sum(golds) + 3 * len(turns),
             "user_tokens": sum(users), "gold_reply_tokens": sum(golds)}
        for L in reply_lens:
            r[f"self{L}"] = sum(users[:tt - 1]) + L * (tt - 1) + 3 * (tt - 1) + users[tt - 1] + 2 + golds[tt - 1] + 1
            r[f"full_self{L}"] = sum(users) + L * len(turns) + 3 * len(turns)
        if idx % 97 == 0:  # the whole rendering, tokenised in one piece, equals the sum of pieces
            whole = "".join(f"{USER}{t['user']}{ASSISTANT}{t['gold']}{EOT}" for t in turns[:tt])
            assert n(whole) == target_gold, (it["id"], n(whole), target_gold)
            check_whole += 1
        rows.append(r)
    keys = ["gold", "twin", "noctx", "near", "full_gold"] + [f"self{L}" for L in reply_lens] + \
        [f"full_self{L}" for L in reply_lens]
    rep = {"set": os.path.relpath(a.set, ROOT).replace(os.sep, "/"),
           "set_sha256": hashlib.sha256(open(a.set, "rb").read()).hexdigest(),
           "vocab": a.vocab, "vocab_sha256": hashlib.sha256(open(a.vocab, "rb").read()).hexdigest(),
           "template": "<|user|>U<|assistant|>A<|endoftext|> per exchange (LOCAL_LLM_STACK.md section 3 point 3)",
           "definitions": {
               "gold": "the gold-history prompt up to the target turn plus the gold reply and <|endoftext|>: the "
                       "longest sequence forced choice scores (candidates share the frame)",
               "selfN": "the same with every earlier assistant reply assumed to be N tokens (self history; "
                        "carry5_eval caps generated replies at --max-new)",
               "full_selfN": "every turn of the item with replies of N tokens: the longest context the self "
                             "mode reaches while it plays the whole conversation",
               "twin": "the single-turn twin prompt plus the gold reply", "near": "the near control rendering",
               "full_gold": "every turn with the gold replies"},
           "whole_rendering_checks": check_whole,
           "all": {k: dist([r[k] for r in rows]) for k in keys},
           "by_family": {}, "by_target_turn": {}}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        rep["by_family"][fam] = {k: dist([r[k] for r in rs]) for k in ("gold", "self24", "self48", "full_self48")
                                 if k in rs[0]}
    for tt in sorted({r["target_turn"] for r in rows}):
        rs = [r for r in rows if r["target_turn"] == tt]
        rep["by_target_turn"][str(tt)] = {k: dist([r[k] for r in rs]) for k in ("gold", "self48") if k in rs[0]}
    reply_tok = [nc(t["gold"]) for it in items for t in it["turns"]]
    user_tok = [nc(t["user"]) for it in items for t in it["turns"]]
    rep["per_turn"] = {"gold_reply_tokens": dist(reply_tok, ()), "user_tokens": dist(user_tok, ())}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
        f.write("\n")
    print(json.dumps({k: rep["all"][k] for k in keys}, indent=1))
    return 0


# ---------------------------------------------------------------- contamination
def cmd_decontam(a):
    """carry5_build's scan (decontam.py rules plus template sentences, fact sentences and item leaks) over
    this set. Its fact-sentence rule reads families recall, correction and topic; load items are scanned
    as recall (same templates), so every value-bearing fact is checked."""
    items = [json.loads(x) for x in open(a.set, encoding="utf-8") if x.strip()]
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with tmp:
        for it in items:
            if it["family"] == "load":
                it = dict(it, family="recall")
            tmp.write(json.dumps(it, ensure_ascii=False) + "\n")
    argv = ["decontam", "--out", a.out, "--set", tmp.name] + (["--state", a.state] if a.state else []) + a.corpora
    try:
        return C5.main(argv)
    finally:
        os.remove(tmp.name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default=OUT)
    p = sub.add_parser("summary")
    p.add_argument("--set", default=OUT)
    p.add_argument("--out", default="")
    p = sub.add_parser("budget")
    p.add_argument("--vocab", required=True)
    p.add_argument("--set", default=OUT)
    p.add_argument("--reply-tokens", default="24,48")
    p.add_argument("--out", required=True)
    p = sub.add_parser("decontam")
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--state", default="")
    p.add_argument("--set", default=OUT)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "summary": cmd_summary, "budget": cmd_budget, "decontam": cmd_decontam}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
