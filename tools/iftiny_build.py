#!/usr/bin/env python3
"""IF-tiny v1 and the turn-distance set v1: instruction following that code can check, sized for tiny models
(LOCAL_LLM_STACK.md section 11, item 3).

    python tools/iftiny_build.py            # writes corpus/eval/iftiny_v1.jsonl and ifdist_v1.jsonl, prints sha256s

Both sets are generated from the templates below (no randomness): rebuilding gives the same bytes, and
tests/test_iftiny_checks.py holds the frozen files equal to the builder.

IF-tiny (single turn). 15 families, each an instruction whose compliance tools/iftiny_checks.py decides:
one word, three words or fewer, lower case, capitals, start with a word, end with a word, end with a full stop,
use a word, avoid a word, double quotation marks, yes or no, a number only, a numbered list of N, repeat a word
N times, repeat a phrase. Per item: `prompt` (the instruction, then the question when the family has one) and
`bare` (the same content without the instruction: the base rate of the format), `check`, the content `answer`
where there is one, and a compliant and a violating example reply (the tests check the checker against both).

Turn distance. A persistent instruction ("For the rest of this conversation, ...", 7 families that apply to any
question) is given at turn 1 and checked at turn 1 + d, d in {0, 1, 2, 4}; the earlier turns are teacher-forced
(`history`: the filler questions with canned replies). The target question of a sequence is the same at every
distance. Conditions:
  instr_user    the instruction opens the first user message (history replies comply);
  instr_system  the instruction is the system message, which conversation schema v1 folds into the first user
                message (section 3, point 3; the engines do the folding);
  nohint        no instruction, history replies comply (what the format alone carries over);
  bare          no instruction, plain history replies (the base rate).
At d = 0 there is no history, so nohint and bare are the same prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import iftiny_checks  # noqa: E402

# (question, plain answer sentence, one-word answer)
QA = [
    ("What colour is grass?", "Grass is green.", "Green"),
    ("What colour is the sky on a clear day?", "The sky is blue.", "Blue"),
    ("What is the capital of France?", "The capital of France is Paris.", "Paris"),
    ("What animal says moo?", "A cow says moo.", "Cow"),
    ("What do bees make?", "Bees make honey.", "Honey"),
    ("What is frozen water called?", "Frozen water is called ice.", "Ice"),
    ("What do you use to write on paper?", "You use a pen.", "Pen"),
    ("Which season comes after winter?", "Spring comes after winter.", "Spring"),
    ("What is the opposite of hot?", "The opposite of hot is cold.", "Cold"),
    ("What shines in the sky during the day?", "The sun shines during the day.", "Sun"),
    ("What do you wear on your feet?", "You wear shoes on your feet.", "Shoes"),
    ("Which fruit is long and yellow?", "A banana is long and yellow.", "Banana"),
    ("Where do fish live?", "Fish live in water.", "Water"),
    ("What is the opposite of up?", "The opposite of up is down.", "Down"),
    ("Which animal barks?", "A dog barks.", "Dog"),
    ("What do you sleep on?", "You sleep on a bed.", "Bed"),
    ("What is the largest ocean?", "The largest ocean is the Pacific.", "Pacific"),
    ("Which planet do we live on?", "We live on Earth.", "Earth"),
    ("What colour is snow?", "Snow is white.", "White"),
    ("What is the opposite of big?", "The opposite of big is small.", "Small"),
    ("Which meal do people eat in the morning?", "People eat breakfast in the morning.", "Breakfast"),
    ("What do birds use to fly?", "Birds use their wings.", "Wings"),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo.", "Tokyo"),
    ("What do you call a baby cat?", "A baby cat is a kitten.", "Kitten"),
    ("What do plants need from the sun?", "Plants need light.", "Light"),
    ("What is the opposite of day?", "The opposite of day is night.", "Night"),
    ("What do you use to cut paper?", "You use scissors.", "Scissors"),
    ("Which animal has a very long neck?", "A giraffe has a very long neck.", "Giraffe"),
    ("What falls from clouds on a wet day?", "Rain falls from clouds.", "Rain"),
    ("What colour is a ripe tomato?", "A ripe tomato is red.", "Red"),
    ("Which month comes after June?", "July comes after June.", "July"),
    ("What do you open to go into a room?", "You open a door.", "Door"),
    ("What is the opposite of wet?", "The opposite of wet is dry.", "Dry"),
    ("Which instrument has black and white keys?", "A piano has black and white keys.", "Piano"),
    ("What is the capital of Italy?", "The capital of Italy is Rome.", "Rome"),
    ("Which gas do people need to breathe?", "People need oxygen.", "Oxygen"),
    ("What do you call a baby dog?", "A baby dog is a puppy.", "Puppy"),
    ("What do cows give us to drink?", "Cows give us milk.", "Milk"),
    ("Which day comes after Monday?", "Tuesday comes after Monday.", "Tuesday"),
    ("What do you use to unlock a door?", "You use a key.", "Key"),
]
YES_NO = [
    ("Is the sky blue on a clear day?", "yes"), ("Is fire cold?", "no"), ("Can fish swim?", "yes"),
    ("Is snow hot?", "no"), ("Do cows say moo?", "yes"), ("Is the sun dark?", "no"), ("Can birds fly?", "yes"),
    ("Is ice made of water?", "yes"), ("Do dogs have wings?", "no"), ("Is Paris in France?", "yes"),
    ("Is a mouse bigger than an elephant?", "no"), ("Is grass green?", "yes"), ("Do people need water?", "yes"),
    ("Is the moon made of cheese?", "no"), ("Can a car fly like a bird?", "no"), ("Do bees make honey?", "yes"),
    ("Is a week seven days long?", "yes"), ("Is night brighter than day?", "no"), ("Are bananas yellow?", "yes"),
    ("Is the ocean full of sand?", "no"), ("Do trees have leaves?", "yes"), ("Is a cat a kind of fish?", "no"),
    ("Is winter colder than summer?", "yes"), ("Can people breathe under water without help?", "no"),
    ("Is Tokyo in Japan?", "yes"), ("Do apples grow on trees?", "yes"), ("Is two bigger than five?", "no"),
    ("Is milk white?", "yes"), ("Do spiders have six legs?", "no"), ("Is the Earth round?", "yes"),
    ("Can a dog read books?", "no"), ("Does rain fall from clouds?", "yes"), ("Is a lemon sweet?", "no"),
    ("Do people sleep at night?", "yes"), ("Is a tomato blue?", "no"), ("Are kittens baby cats?", "yes"),
    ("Is water dry?", "no"), ("Do ducks swim?", "yes"), ("Is a rock soft?", "no"), ("Do horses eat grass?", "yes"),
]
NUMBERS = [
    ("How many legs does a dog have?", "4"), ("How many days are in a week?", "7"), ("What is two plus two?", "4"),
    ("How many wheels does a bicycle have?", "2"), ("How many months are in a year?", "12"),
    ("What is three plus five?", "8"), ("How many legs does a spider have?", "8"), ("What is ten minus one?", "9"),
    ("How many hours are in a day?", "24"), ("How many sides does a triangle have?", "3"),
    ("What is five plus five?", "10"), ("How many fingers are on one hand?", "5"),
    ("How many minutes are in an hour?", "60"), ("What is six minus four?", "2"),
    ("How many sides does a square have?", "4"), ("How many legs does a bird have?", "2"),
    ("What is one plus one?", "2"), ("How many eyes do people have?", "2"), ("What is seven plus three?", "10"),
    ("How many seasons are in a year?", "4"), ("What is nine minus three?", "6"),
    ("How many days are in a weekend?", "2"), ("What is four plus four?", "8"), ("How many wheels does a car have?", "4"),
]
LIST_THINGS = [
    ("fruits", ["apple", "banana", "pear", "orange", "grape"]),
    ("animals", ["dog", "cat", "horse", "cow", "sheep"]),
    ("colours", ["red", "blue", "green", "yellow", "purple"]),
    ("countries", ["France", "Japan", "Italy", "Brazil", "Canada"]),
    ("sports", ["football", "tennis", "cricket", "swimming", "golf"]),
    ("vegetables", ["carrot", "pea", "potato", "onion", "cabbage"]),
    ("cities", ["Paris", "Tokyo", "Rome", "London", "Sydney"]),
    ("tools", ["hammer", "saw", "drill", "spanner", "screwdriver"]),
    ("drinks", ["water", "milk", "tea", "coffee", "juice"]),
    ("birds", ["robin", "eagle", "owl", "parrot", "duck"]),
]
REPEAT_WORDS = ["apple", "hello", "blue", "cat", "yes", "sun", "happy", "tree", "music", "ball"]
NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}
PHRASES = [
    "The red ball is round.", "I like green apples.", "The cat sat on the mat.", "We go to school.",
    "The sun is hot.", "My dog runs fast.", "Birds sing in the morning.", "The book is on the table.",
    "I drink water every day.", "The sky is very blue.", "She has a small bag.", "We play in the park.",
    "The tree is tall.", "He likes to read.", "The milk is cold.", "It is a sunny day.", "The door is open.",
    "I can see the moon.", "The fish swims in the pond.", "Please close the window.", "The bus is late today.",
    "I have two hands.", "The bread is fresh.", "Snow is cold and white.", "We eat lunch at noon.",
    "The car is red.", "My hat is black.", "The baby is sleeping.", "Rain makes the grass wet.",
    "The shop opens at nine.", "I see a big ship.", "The flowers are pretty.", "He kicks the ball.",
    "The cake is sweet.", "We walk to the beach.", "The clock is on the wall.", "Cows eat grass.",
    "The lamp is bright.", "I like to sing.", "The river is long.",
]
START_WORDS = ["Sure", "Well", "Okay", "Yes"]
END_WORDS = ["done", "thanks", "friend", "today"]
INCLUDE_WORDS = ["today", "really", "always", "indeed"]
AVOID_WORDS = ["the", "is"]


def lower_first(s):
    return s[:1].lower() + s[1:]


def comply(fam, plain, one, w=None):
    """A canned reply to a QA question that satisfies family `fam` (also the turn-distance history replies)."""
    return {"one_word": one + ".", "max_words": one + ".", "lowercase": plain.lower(), "uppercase": plain.upper(),
            "start_with": f"{w}, {lower_first(plain)}", "end_with_word": f"{plain[:-1]}, {w}.",
            "end_full_stop": plain, "include_word": f"{plain[:-1]} {w}.", "avoid_word": one + ".",
            "quoted": f'"{plain}"'}[fam]


def violate(fam, plain, one, w=None):
    return {"one_word": plain, "max_words": "The answer is " + plain[0].lower() + plain[1:], "lowercase": plain,
            "uppercase": plain, "start_with": plain, "end_with_word": plain, "end_full_stop": plain[:-1],
            "include_word": plain, "avoid_word": f"The answer is {one.lower()}.", "quoted": plain}[fam]


QA_FAMILIES = [
    ("one_word", "Answer in one word.", lambda w: {"kind": "one_word"}),
    ("max_words", "Answer in three words or fewer.", lambda w: {"kind": "max_words", "n": 3}),
    ("lowercase", "Reply in lower case letters only.", lambda w: {"kind": "lowercase"}),
    ("uppercase", "Reply in capital letters only.", lambda w: {"kind": "uppercase"}),
    ("start_with", 'Start your reply with the word "{w}".', lambda w: {"kind": "start_with", "word": w}),
    ("end_with_word", 'End your reply with the word "{w}".', lambda w: {"kind": "end_with_word", "word": w}),
    ("end_full_stop", "End your reply with a full stop.", lambda w: {"kind": "end_full_stop"}),
    ("include_word", 'Use the word "{w}" in your reply.', lambda w: {"kind": "include_word", "word": w}),
    ("avoid_word", 'Do not use the word "{w}" in your reply.', lambda w: {"kind": "avoid_word", "word": w}),
    ("quoted", "Put your whole reply in double quotation marks.", lambda w: {"kind": "quoted"}),
]
FAMILY_WORDS = {"start_with": START_WORDS, "end_with_word": END_WORDS, "include_word": INCLUDE_WORDS,
                "avoid_word": AVOID_WORDS}


def single_turn():
    items = []
    for fam, instr, mk in QA_FAMILIES:
        for k, (q, plain, one) in enumerate(QA):
            ws = FAMILY_WORDS.get(fam)
            w = ws[k % len(ws)] if ws else None
            ins = instr.format(w=w)
            items.append({"id": f"if/{fam}/{k}", "family": fam, "check": mk(w), "instruction": ins, "question": q,
                          "prompt": f"{ins} {q}", "bare": q, "answer": [one.lower()],
                          "comply_example": comply(fam, plain, one, w), "violate_example": violate(fam, plain, one, w)})
    for k, (q, a) in enumerate(YES_NO):
        ins = "Answer with yes or no only."
        items.append({"id": f"if/yes_no/{k}", "family": "yes_no", "check": {"kind": "yes_no"}, "instruction": ins,
                      "question": q, "prompt": f"{ins} {q}", "bare": q, "answer": [a],
                      "comply_example": a.capitalize() + ".",
                      "violate_example": a.capitalize() + (", it is." if a == "yes" else ", it is not.")})
    for k, (q, a) in enumerate(NUMBERS):
        ins = "Reply with a number only."
        items.append({"id": f"if/number_only/{k}", "family": "number_only", "check": {"kind": "number_only"},
                      "instruction": ins, "question": q, "prompt": f"{ins} {q}", "bare": q, "answer": [a],
                      "comply_example": a, "violate_example": f"The answer is {a}."})
    k = 0
    for n in (2, 3, 4, 5):
        for things, ex in LIST_THINGS:
            ins = f"Give a numbered list of {NUM_WORD[n]} {things}."
            items.append({"id": f"if/numbered_list/{k}", "family": "numbered_list", "check": {"kind": "numbered_list",
                          "n": n}, "instruction": ins, "question": "", "prompt": ins, "bare": f"Tell me about {things}.",
                          "comply_example": "\n".join(f"{i + 1}. {x}" for i, x in enumerate(ex[:n])),
                          "violate_example": ", ".join(ex[:n])})
            k += 1
    k = 0
    for n in (2, 3, 4):
        for w in REPEAT_WORDS:
            ins = f'Repeat the word "{w}" {NUM_WORD[n]} times.'
            items.append({"id": f"if/repeat_word/{k}", "family": "repeat_word",
                          "check": {"kind": "repeat_word", "word": w, "n": n}, "instruction": ins, "question": "",
                          "prompt": ins, "bare": f'What does the word "{w}" mean?',
                          "comply_example": " ".join([w] * n), "violate_example": w})
            k += 1
    for k, ph in enumerate(PHRASES):
        ins = f"Repeat after me: {ph}"
        items.append({"id": f"if/repeat_phrase/{k}", "family": "repeat_phrase",
                      "check": {"kind": "repeat_phrase", "phrase": ph}, "instruction": ins, "question": "",
                      "prompt": ins, "bare": ph, "comply_example": ph, "violate_example": "Okay."})
    return items


PERSISTENT = [
    ("one_word", "For the rest of this conversation, answer every question in one word."),
    ("max_words", "For the rest of this conversation, answer every question in three words or fewer."),
    ("lowercase", "For the rest of this conversation, reply in lower case letters only."),
    ("uppercase", "For the rest of this conversation, reply in capital letters only."),
    ("start_with", 'For the rest of this conversation, start every reply with the word "{w}".'),
    ("end_with_word", 'For the rest of this conversation, end every reply with the word "{w}".'),
    ("quoted", "For the rest of this conversation, put every reply in double quotation marks."),
]
CHECKS = {f: mk for f, _, mk in QA_FAMILIES}
DISTANCES = (0, 1, 2, 4)
SEQUENCES = 12
CONDITIONS = ("instr_user", "instr_system", "nohint", "bare")


def distance():
    items = []
    for fi, (fam, text) in enumerate(PERSISTENT):
        for s in range(SEQUENCES):
            ws = FAMILY_WORDS.get(fam)
            w = ws[s % len(ws)] if ws else None
            ins = text.format(w=w)
            # five distinct questions per sequence: four fillers, then the target (the same at every distance)
            base = (fi * 7 + s * 5) % len(QA)
            qs = [QA[(base + j) % len(QA)] for j in range(5)]
            target, fillers = qs[4], qs[:4]
            for d in DISTANCES:
                for cond in CONDITIONS:
                    turns = fillers[:d]
                    good = cond in ("instr_user", "instr_system", "nohint")
                    hist = [[q, comply(fam, plain, one, w) if good else plain] for q, plain, one in turns]
                    user = target[0]
                    if cond == "instr_user":
                        if hist:
                            hist[0][0] = f"{ins} {hist[0][0]}"
                        else:
                            user = f"{ins} {user}"
                    items.append({"id": f"ifd/{fam}/{s}/d{d}/{cond}", "family": fam, "sequence": s, "distance": d,
                                  "condition": cond, "check": CHECKS[fam](w), "instruction": ins,
                                  "system": ins if cond == "instr_system" else None,
                                  "history": hist, "user": user, "answer": [target[2].lower()]})
    return items


def write(path, rows):
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    st, dist = single_turn(), distance()
    for it in st:
        assert iftiny_checks.check(it["check"], it["comply_example"]), it["id"]
        assert not iftiny_checks.check(it["check"], it["violate_example"]), it["id"]
    out = {}
    for name, rows in (("iftiny_v1.jsonl", st), ("ifdist_v1.jsonl", dist)):
        out[name] = {"items": len(rows), "sha256": write(os.path.join(ROOT, "corpus", "eval", name), rows)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
