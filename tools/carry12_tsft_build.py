#!/usr/bin/env python3
"""Carry-12's template-SFT set (cell T): hand-written templated dialogues whose later turn needs a fact, a
correction or a standing instruction from N = 1-12 turns back. TRAINING data for one registered cell, with
templates, entities and values disjoint from Carry-12 v1's (and Carry-5's).

    python tools/carry12_tsft_build.py build [--out FILE]        # the dialogues, {"messages", "uri", "meta"} per row
    python tools/carry12_tsft_build.py check                     # rebuild in memory; compare with the pinned sha256
    python tools/carry12_tsft_build.py ledger --out FILE.json [--vocab V.gguf]   # counts and hashes
    python tools/carry12_tsft_build.py disjoint --out FILE.json  # the disjointness check against Carry-12 v1 and Carry-5

experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md sections 0, 4 (option 5) and 6. The data rule keeps
our templates out of training; Jonathan's ruling of 27 Oct 2026 (an outside reviewer's suggestion: SFT with
explicit recall targets) makes this set the one registered exception, for cell T only, under option 5's
conditions: written by us as templates (no model text), sharing no template, entity or value with the
evaluation set, and T scored on Carry-12 v1, which is built from other templates.

Families (one carry target per dialogue, at turn target = fact turn + N, then one closing turn):
  recall       a fact about the user (six slots: boat, street, locker, car, team, language) at turn 1-3, asked
               N turns later; 0-3 same-slot distractor facts about other people before the target.
  correction   the fact planted, then corrected ("... rather than OLD"); the question N turns after the
               correction needs the corrected value; 0-2 further distractors.
  instruction  a standing format instruction with a question at turn 1-3 (opener: start every reply with
               Indeed / Certainly / Gladly / Absolutely; closer: end it with "no doubt" / "as ever" / "in fact" /
               "all in all"); every reply from then on complies, the target question N turns later.
Carry-12's topic and load families have no counterpart here (one user fact per dialogue, no digression), so
the analysis can read T's gain on the trained families (recall, correction) against the untrained ones (topic,
load). Turns without an event are hand-written small talk (SMALL_TALK, ARCS), none naming a value of any list
here or in Carry-12/Carry-5 or raising one of this set's slot subjects. No turn comes from the human dialogue
pool: splicing a human exchange into a templated conversation would present separate exchanges as one
conversation, which section 4's rule forbids, and the pool is not scrubbed of Carry-12's values.

Counts: recall 6 slots x N 1-12 x fact turn 1-3 x 10 (2,160), correction 6 x 12 x 20 (1,440), instruction
2 kinds x 12 x 26 (624): 4,224 dialogues, 20% of T's SFT rows beside the T 1,024 data job's 16,932 human rows.
Seed 20261027; byte-reproducible; pinned in corpus/hashes/carry12_tsft_v1.sha256. The data job
(tools/colab_chat/tiny_chat_data.py --template-sft) rebuilds it and refuses bytes other than the pin.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

VERSION = "carry12_tsft_v1"
SEED = 20261027
HASH = os.path.join(ROOT, "corpus", "hashes", "carry12_tsft_v1.sha256")
CARRY12 = os.path.join(ROOT, "corpus", "eval", "carry12_v1.jsonl")
CARRY5 = os.path.join(ROOT, "corpus", "eval", "carry5_v1.jsonl")
N_RANGE = tuple(range(1, 13))
FACT_TURNS = (1, 2, 3)
MAX_TARGET = 15                       # target = fact + N <= 15; one closing turn after it: at most 16 turns
REPS = {"recall": 10, "correction": 20, "instruction": 26}

# ---------------------------------------------------------------- value lists (none in Carry-12 or Carry-5)
BOATS = ["Kittiwake", "Seaspray", "Driftwood", "Halcyon", "Sandpiper", "Merrow", "Tidewater", "Saltmarsh",
         "Moonraker", "Starling", "Kelpie", "Gannet", "Cormorant", "Petrel", "Shearwater", "Seafoam", "Brightwater",
         "Windrush", "Fairwind", "Whimbrel", "Curlew", "Plover", "Dunlin", "Skerry", "Selkie", "Wavecrest",
         "Albatross", "Osprey", "Lapwing", "Moorhen"]
STREETS = ["Larch", "Quarry", "Tannery", "Chapel", "Foundry", "Wharf", "Kiln", "Glebe", "Rookery", "Bellfield",
           "Abbey", "Priory", "Ferry", "Ropewalk", "Malthouse", "Granary", "Spinney", "Copse", "Fenwick", "Ashgrove",
           "Elmwood", "Saddler", "Mercer", "Cooper", "Hatter", "Pottery", "Tollgate", "Lockhart", "Bridgend", "Cobble"]
LOCKERS = [str(x) for x in sorted(random.Random(f"{SEED}:lockers").sample(range(101, 990), 30))]
CARS = ["hatchback", "convertible", "campervan", "minivan", "sedan", "coupe", "roadster", "wagon", "ute", "jeep",
        "limousine", "pickup", "minibus", "buggy"]
TEAMS = ["Herons", "Otters", "Badgers", "Falcons", "Wolves", "Stags", "Ravens", "Lynxes", "Pumas", "Hornets",
         "Panthers", "Jaguars", "Kestrels", "Stingrays", "Mustangs", "Cobras", "Vipers", "Bison", "Coyotes", "Marlins"]
LANGUAGES = ["Portuguese", "Swahili", "Welsh", "Tagalog", "Finnish", "Gaelic", "Dutch", "Greek", "Hungarian",
             "Icelandic", "Polish", "Turkish", "Korean", "Malay", "Danish", "Czech", "Catalan", "Basque", "Romanian",
             "Estonian"]
RELATIONS = ["flatmate", "grandad", "nephew", "niece", "boss", "landlord", "classmate", "teammate"]

SLOTS = {
    "boat": {"values": BOATS,
             "plant": ["My boat is named {v}.", "I own a little boat named {v}."],
             "distractor": ["My {rel}'s boat is named {v}.", "My {rel} owns a boat named {v}."],
             "question": ["What is my boat named?", "Remind me, what did I name my boat?"],
             "correction": ["Let me fix something: my boat is named {new}, and {old} was a slip.",
                            "I mixed that up earlier. My boat is named {new} rather than {old}."],
             "frame": "Your boat is named {}."},
    "street": {"values": STREETS,
               "plant": ["My flat is on {v} Street.", "I rent a flat on {v} Street."],
               "distractor": ["My {rel}'s flat is on {v} Street.", "My {rel} rents a flat on {v} Street."],
               "question": ["Which street is my flat on?", "Remind me, which street did I say my flat is on?"],
               "correction": ["Let me fix something: my flat is on {new} Street, and {old} Street was a slip.",
                              "I mixed that up earlier. My flat is on {new} Street rather than {old} Street."],
               "frame": "Your flat is on {} Street."},
    "locker": {"values": LOCKERS,
               "plant": ["My gym locker is number {v}.", "At the gym I always use locker {v}."],
               "distractor": ["My {rel}'s gym locker is number {v}.", "My {rel} uses locker {v} at the gym."],
               "question": ["What is my locker number?", "Remind me, which locker do I use at the gym?"],
               "correction": ["Let me fix something: my gym locker is number {new}, and {old} was a slip.",
                              "I mixed that up earlier. My gym locker is number {new} rather than {old}."],
               "frame": "Your locker is number {}."},
    "car": {"values": CARS,
            "plant": ["My car is a {v}.", "I drive a {v} these days."],
            "distractor": ["My {rel}'s car is a {v}.", "My {rel} drives a {v}."],
            "question": ["What kind of car do I have?", "Remind me, what sort of car do I drive?"],
            "correction": ["Let me fix something: my car is a {new}, and the {old} was a slip.",
                           "I mixed that up earlier. My car is a {new} rather than a {old}."],
            "frame": "Your car is a {}."},
    "team": {"values": TEAMS,
             "plant": ["The team I cheer for is the {v}.", "I always cheer for the {v}."],
             "distractor": ["My {rel} cheers for the {v}.", "The team my {rel} cheers for is the {v}."],
             "question": ["Which team do I cheer for?", "Remind me, which team did I say I cheer for?"],
             "correction": ["Let me fix something: I cheer for the {new}, and the {old} was a slip.",
                            "I mixed that up earlier. I cheer for the {new} rather than the {old}."],
             "frame": "You cheer for the {}."},
    "language": {"values": LANGUAGES,
                 "plant": ["At home I speak {v}.", "The language I speak at home is {v}."],
                 "distractor": ["My {rel} speaks {v} at home.", "At home my {rel} speaks {v}."],
                 "question": ["Which language do I speak at home?", "Remind me, what language do I speak at home?"],
                 "correction": ["Let me fix something: at home I speak {new}, and {old} was a slip.",
                                "I mixed that up earlier. At home I speak {new} rather than {old}."],
                 "frame": "You speak {} at home."},
}
# Words that raise a slot's subject: small talk holds none of them (an unlabelled cue otherwise).
SUBJECT_WORDS = {"boat", "boats", "sail", "named", "street", "flat", "flats", "road", "locker", "gym", "number", "car",
                 "cars", "drive", "drives", "driving", "team", "teams", "cheer", "support", "language", "languages",
                 "speak", "speaks", "rent", "rents", "own", "owns"} | set(RELATIONS)

# ---------------------------------------------------------------- standing instructions (forms not Carry-12's)
INSTRUCTIONS = {
    "opener": {"say": ["For the rest of our chat, open each reply with '{W}'.",
                       "Until we finish, every reply of yours should begin with '{W}'.",
                       "Here is a rule for this chat: put '{W}' at the start of every reply."],
               "forms": ["Indeed", "Certainly", "Gladly", "Absolutely"]},
    "closer": {"say": ["For the rest of our chat, close each reply with '{W}'.",
                       "Until we finish, every reply of yours should end with '{W}'.",
                       "Here is a rule for this chat: put '{W}' at the end of every reply."],
               "forms": ["no doubt", "as ever", "in fact", "all in all"]},
}
# (question, statement, answer): everyday questions whose answers are in no Carry-12 or Carry-5 list.
QA = [("What do you use to sweep a floor?", "you sweep a floor with a broom", "broom"),
      ("What keeps you dry in a downpour?", "an umbrella keeps you dry in a downpour", "umbrella"),
      ("What forms on a pond in a hard frost?", "ice forms on a pond in a hard frost", "ice"),
      ("Which way does a compass needle point?", "a compass needle points north", "north"),
      ("What do you light to read in a power cut?", "a torch helps you read in a power cut", "torch"),
      ("What makes bread dough rise?", "yeast makes bread dough rise", "yeast"),
      ("What fastens sheets of paper together?", "a stapler fastens sheets of paper together", "stapler"),
      ("What shows your reflection on a wall?", "a mirror shows your reflection", "mirror"),
      ("What keeps the rain off your body on a walk?", "a raincoat keeps the rain off you", "raincoat"),
      ("Which tool turns a screw?", "a screwdriver turns a screw", "screwdriver"),
      ("What do you dig a hole with?", "you dig a hole with a spade", "spade"),
      ("Which bird spreads a big tail fan?", "a peacock spreads a big tail fan", "peacock"),
      ("What do you stir a big pot of soup with?", "you stir a pot of soup with a ladle", "ladle"),
      ("What powder is ground from wheat for bread?", "flour is ground from wheat", "flour"),
      ("What turns its sails in the wind on a hill?", "a windmill turns its sails in the wind", "windmill"),
      ("What do you cut a plank of wood with?", "you cut a plank of wood with a saw", "saw"),
      ("What do you call a baby frog?", "a baby frog is a tadpole", "tadpole"),
      ("What do you call a group of stars that makes a shape?", "a group of stars like that is a constellation",
       "constellation"),
      ("Where does the sky seem to meet the land?", "the sky meets the land at the horizon", "horizon"),
      ("What do you call a person who writes poems?", "a person who writes poems is a poet", "poet"),
      ("Which creature spins a silk cocoon?", "a silkworm spins a silk cocoon", "silkworm"),
      ("What do you call the very top of a mountain?", "the top of a mountain is the summit", "summit"),
      ("What do you use to look at distant planets?", "you look at distant planets through a telescope",
       "telescope"),
      ("What rumbles during a thunderstorm?", "thunder rumbles during a storm", "thunder"),
      ("Where do you peg out washing to dry?", "you peg out washing on a line", "line"),
      ("What arc of light can appear after a shower of rain?", "a rainbow can appear after a shower", "rainbow"),
      ("What do you call a very young tree?", "a very young tree is a sapling", "sapling"),
      ("What do you call a long seat in a park?", "a long seat in a park is a bench", "bench"),
      ("What soft toy is shaped like a bear?", "a soft toy bear is a teddy", "teddy"),
      ("What frozen treat comes on a stick?", "an ice lolly comes on a stick", "lolly"),
      ("Where does a bird lay its eggs?", "a bird lays its eggs in a nest", "nest"),
      ("What covers the trunk of a tree?", "bark covers the trunk of a tree", "bark"),
      ("What do you strike to light a candle?", "you strike a match to light a candle", "match"),
      ("What do you call the glass you drink juice from?", "you drink juice from a tumbler", "tumbler"),
      ("What holds a painting on the wall?", "a painting sits in a frame", "frame"),
      ("What do you call a person who mends shoes?", "a person who mends shoes is a cobbler", "cobbler"),
      ("What lights the way for ships at night?", "a lighthouse lights the way for ships", "lighthouse"),
      ("What do you call a small stream?", "a small stream is a brook", "brook")]

# ---------------------------------------------------------------- conversation filler (hand-written)
OPENERS = [("Hello, are you free to talk for a while?", "I am free. What would you like to discuss?"),
           ("Good evening! I could use some company.", "Good evening! I am glad to keep you company."),
           ("Hiya, got time for a natter?", "Plenty of time. Go ahead."),
           ("Morning! Mind if I ramble a little?", "Not at all, ramble away."),
           ("Hello there, it is me again.", "Hello! Nice to hear from you again.")]
CLOSINGS = [("That is everything for today, thank you.", "My pleasure. Enjoy the rest of your day."),
            ("I will leave it there for now.", "No problem. Talk again soon."),
            ("Time for me to head off.", "Take care, and come back any time.")]
ACK_FACT = ["Noted, thank you.", "Good to hear, I have that now.", "Thank you, I will keep that in mind."]
ACK_FIX = ["Understood, I have updated that.", "No worries, I have changed that now.",
           "Thanks, the earlier one is replaced."]
ACK_OTHER = ["That is a nice detail.", "Interesting, thank you.", "Good to hear about them."]
SMALL_TALK = [
    ("Do you have a tip for a calmer commute?", "Leave a few minutes early and listen to something gentle."),
    ("How can I keep houseplants alive?", "Water them only when the soil feels dry."),
    ("What is a nice way to spend a slow morning?", "A long breakfast and a stroll are hard to beat."),
    ("How do I stop my tea going cold so fast?", "Warm the mug first and keep a saucer on top."),
    ("Any advice for a stiff neck?", "Gentle stretches and a warm flannel often ease it."),
    ("How do I get better at waking up?", "Open the curtains straight away and let the daylight in."),
    ("What is a good way to clear my head?", "A brisk walk without your phone usually does it."),
    ("How do I keep a notebook tidy?", "Date each page and leave a margin for notes."),
    ("What can I do with stale bread?", "Toast it into crumbs or make a bread pudding."),
    ("How do I clean a burnt saucepan?", "Soak it in hot soapy water and scrub it gently."),
    ("Is it worth making a weekly plan?", "Yes, even a short list can make the week calmer."),
    ("How can I sit better at a desk?", "Keep both feet on the floor and the screen at eye height."),
    ("What should I do when I cannot concentrate?", "Take a short break, then start with the smallest task."),
    ("How do I stop a cut apple going dark?", "A squeeze of juice over it slows that down."),
    ("Any tips for a long queue?", "Bring something to read or a puzzle to pass the time."),
    ("How can I stay cool at night?", "Use a light sheet and keep a window open a crack."),
    ("What is a simple lunch for a busy day?", "A wrap with cheese and salad is quick to make."),
    ("How do I stop misplacing my glasses?", "Always set them down in the same spot."),
    ("What helps with a dry throat?", "Warm drinks with honey can soothe it."),
    ("How can I make a small room feel bigger?", "Pale walls and a mirror opposite the window help."),
    ("How often should I water a cactus?", "Every few weeks is plenty for most of them."),
    ("What is a good way to unwind after a long day?", "A hot shower and a quiet hour can help."),
    ("How do I keep my desk from getting cluttered?", "Clear it for two minutes at the end of each day."),
    ("Do you have a trick for remembering errands?", "Group them by place and write them on one list."),
    ("What should I pack for a picnic in the park?", "A blanket, some sandwiches and a flask of tea."),
    ("How can I tell if bread dough has proved enough?", "Press it gently, and the dent should spring back slowly."),
    ("What is a good stretch after sitting all day?", "Reach both arms up high and lean slowly to each side."),
    ("How do I keep fresh herbs from wilting?", "Wrap them in a damp paper towel and keep them cool."),
    ("Any tips for writing a short speech?", "Start with one clear point and finish with a thank you."),
    ("How do I make scrambled eggs softer?", "Cook them slowly over a low heat and stir often."),
    ("What do I do if I miss the last bus?", "Check for a night service or share a taxi."),
    ("How can I be more patient?", "Pause for a breath before you answer."),
    ("Is it good to keep a diary?", "Many people find it helps them sort out their thoughts."),
    ("How do I get candle wax off a tablecloth?", "Lay paper over it and press it with a warm cloth."),
    ("What should I do before a long walk?", "Check the weather and take a light jacket."),
    ("How do I stop the kettle furring up?", "Descale it now and then with a little vinegar."),
    ("Any ideas for a damp weekend indoors?", "Bake some bread, sort some photos or watch an old film."),
    ("How can I fall asleep faster?", "Keep the room dark and try slow, steady breathing."),
    ("What is a polite way to decline a second helping?", "Say it was lovely and that you are full."),
    ("How do I make a to-do list I will actually use?", "Keep it short and put the hardest thing first."),
    ("How do I stop my bike chain squeaking?", "A few drops of oil on the chain should fix it."),
    ("What is a good way to start a conversation?", "Ask an open question and listen to the answer."),
    ("How can I keep my hands warm outside?", "Keep them in your pockets and wear a thick scarf."),
    ("How do I keep a kitchen sponge from smelling?", "Rinse it thoroughly and let it dry out between uses."),
    ("What can I do with leftover pasta?", "Bake it with some cheese on top for an easy supper."),
    ("How do I get a stain out of a shirt?", "Rinse it in cold water quickly, then wash it as usual."),
    ("Is it fine to skip a workout now and then?", "Yes, rest days help your body recover."),
    ("How can I make my room smell fresh?", "Open a window for ten minutes each morning."),
    ("What helps when I feel nervous?", "Slow breathing and a few kind words to yourself."),
    ("How do I keep track of my spending?", "Write down every purchase for a week and look back."),
    ("What is an easy pudding to make?", "Baked apples with a spoon of cream are simple."),
    ("How can I tidy a messy drawer?", "Empty it, sort it into piles and put back only what you need."),
    ("What should I do on a blustery day?", "Tie down anything loose and enjoy a warm drink indoors."),
    ("How do I stop my phone battery draining?", "Lower the screen brightness and close apps you are not using."),
    ("What is a nice way to welcome a guest?", "Offer them a drink and a comfy seat."),
    ("How can I make the most of a short lunch break?", "Step outside for a few minutes, even just around the block."),
    ("How do I stop my window steaming up?", "Open it a little while you cook and dry the glass."),
    ("Any tips for keeping a plant on a windowsill?", "Turn the pot now and then so it leans evenly."),
    ("How can I relax my shoulders?", "Roll them back slowly a few times and breathe out."),
    ("What do I do with old tea towels?", "They make good rags for cleaning."),
    ("How do I make a mug of cocoa?", "Warm some milk and stir in the cocoa with a little sugar."),
    ("How do I make plain rice tastier?", "Cook it in stock and add a pinch of salt."),
    ("What is a kind thing to do for someone?", "Send a short message to say you are thinking of them."),
    ("How do I stop a rug from slipping?", "Put a grip mat underneath it."),
    ("Any advice for a long video call?", "Sit near a window and keep a glass of something nearby."),
    ("How can I sleep on a noisy night?", "A fan or a soft background sound can mask the noise."),
    ("What is a good snack before a hike?", "A handful of oats or a cereal bar gives steady energy."),
]
ARCS = [
    [("The heating came on for the first time today.", "A sign the colder months are here. Is it cosy now?"),
     ("Very cosy, I might not leave the sofa.", "A good plan for a chilly evening.")],
    [("I finally fixed the wobbly table.", "Nice work. What was wrong with it?"),
     ("One leg was a little short, so I added a pad.", "Simple and clever.")],
    [("I spent the morning sorting old photos.", "That can bring back lots of memories. Did you find any good ones?"),
     ("A few from school that made me laugh.", "Those are worth keeping somewhere safe.")],
    [("There was a glorious sunset this evening.", "Were the clouds lit up?"),
     ("They glowed for about ten minutes.", "What a nice way to end the day.")],
    [("I tried a new route on my walk today.", "How was it? Did you find anything interesting?"),
     ("A small path along a stream I never knew about.", "What a nice discovery.")],
    [("The power went out for an hour last night.", "What a nuisance. Did you have candles?"),
     ("Yes, and it was strangely peaceful.", "Sometimes a quiet hour like that is nice.")],
    [("I baked a loaf for the first time.", "How did it turn out?"),
     ("A bit dense, but it tasted good.", "The next loaf will be even better.")],
    [("I have started doing a crossword each morning.", "A nice habit. Are they getting easier?"),
     ("Slowly, and today I finished one without help.", "That is a real achievement.")],
    [("It snowed a little overnight.", "How pretty. Did it settle?"),
     ("Only on the grass, the paths were clear.", "The best kind of snow.")],
    [("I repainted the front door yesterday.", "That must look smart. How long did it take?"),
     ("Most of the day, with two coats.", "It will look fresh for a long time.")],
    [("I went to a quiz night yesterday.", "Fun. How did you do?"),
     ("We came third, which was a surprise.", "Third is a great result.")],
    [("The washing machine has started making a noise.", "Oh dear. Is it a loud one?"),
     ("A sort of rattle when it spins.", "It might be a coin caught somewhere inside.")],
    [("I sorted out the shed today.", "Good job. Was it very full?"),
     ("Full of old pots and a broken stool.", "It must feel much roomier now.")],
    [("The library had a sale of old paperbacks.", "Did you pick any up?"),
     ("Three of them for a few coins.", "A bargain for a damp afternoon.")],
    [("I had a long chat on the phone with an old schoolmate.", "How nice. Had it been a while?"),
     ("Nearly a decade, but it felt like no time at all.", "Old friendships are like that.")],
    [("My hands were freezing after the walk back.", "Did you warm up with a hot drink?"),
     ("A mug of soup did the trick.", "Soup is perfect after a cold walk.")],
]


def words(text):
    """carry5_build's normalisation: lower-case alphanumeric runs."""
    out, w = [], []
    for ch in text.lower():
        if ch.isascii() and ch.isalnum():
            w.append(ch)
        elif w:
            out.append("".join(w))
            w = []
    if w:
        out.append("".join(w))
    return out


def forms(w):
    """A word and its plural / singular forms (carry12_build's rule)."""
    out = {w, w + "s", w + "es"}
    if w.endswith("es"):
        out.add(w[:-2])
    if w.endswith("s"):
        out.add(w[:-1])
    return out


def lower_first(text):
    first = text.split()[0]
    if first in ("I", "I'm") or first[:2] == "I'":
        return text
    return text[0].lower() + text[1:]


def comply(kind, form, reply):
    """A reply in the instruction's form: opener = 'Form, reply...'; closer = 'reply..., form.'"""
    if kind == "opener":
        return f"{form}, {lower_first(reply)}"
    body = reply.rstrip(".!")
    return f"{body}, {form}."


# ---------------------------------------------------------------- building
class Builder:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.cycles = {}

    def cycle(self, key, values):
        st = self.cycles.setdefault(key, [])
        if not st:
            st.extend(values)
            self.rng.shuffle(st)
        return st.pop()

    def others(self, key, values, exclude, k):
        """k distinct values not in EXCLUDE from a per-key shuffled cycle (skipped values go back on top)."""
        st = self.cycles.setdefault(key, [])
        out, skipped = [], []
        while len(out) < k:
            if not st:
                st.extend(values)
                self.rng.shuffle(st)
            v = st.pop()
            (skipped if v in exclude or v in out else out).append(v)
        st.extend(reversed(skipped))
        return out


def compose(b, events, n, target, instruction=None):
    """[(user, assistant)] for turns 1..n. Event turns hold their text; empty turns before the target take small
    talk (a two-turn arc where two empty turns follow each other, half the time; never in instruction
    dialogues, whose arcs' questions would not take the closer form), turn 1 an opener, turn n (after the
    target) a closing. INSTRUCTION = (kind, form, turn): every reply from that turn on complies."""
    rng = b.rng
    singles = list(SMALL_TALK)
    rng.shuffle(singles)
    arcs = [list(a) for a in ARCS] if instruction is None else []
    rng.shuffle(arcs)
    out = [None] * n
    t = 1
    while t <= n:
        if t in events:
            out[t - 1] = events[t]
            t += 1
            continue
        if t == n and t > target:
            q, a = rng.choice(CLOSINGS)
            out[t - 1] = {"user": q, "gold": a, "kind": "closing"}
            t += 1
            continue
        if t == 1:
            q, a = rng.choice(OPENERS)
            out[0] = {"user": q, "gold": a, "kind": "opener"}
            t += 1
            continue
        nxt_empty = t + 1 < target and (t + 1) not in events
        if nxt_empty and arcs and rng.random() < 0.5:
            for k, (q, a) in enumerate(arcs.pop()):
                out[t - 1 + k] = {"user": q, "gold": a, "kind": "small_talk"}
            t += 2
            continue
        q, a = singles.pop()
        out[t - 1] = {"user": q, "gold": a, "kind": "small_talk"}
        t += 1
    if instruction:
        kind, form, at = instruction
        for k in range(at - 1, n):
            if out[k]["kind"] not in ("instruction", "question_inst"):
                out[k] = dict(out[k], gold=comply(kind, form, out[k]["gold"]))
    return out


def free_turns(b, target, taken, k):
    free = [t for t in range(1, target) if t not in taken]
    return sorted(b.rng.sample(free, min(k, len(free))))


def row(fam, idx, conv, meta):
    return {"messages": [m for t in conv for m in ({"role": "user", "content": t["user"]},
                                                   {"role": "assistant", "content": t["gold"]})],
            "uri": f"{VERSION}/{fam}/{idx:04d}", "meta": {"family": fam, **meta,
                                                           "turn_kinds": [t["kind"] for t in conv]}}


def build_rows(seed=SEED):
    b = Builder(seed)
    rng = b.rng
    rows, seen = [], set()
    count = {"recall": 0, "correction": 0, "instruction": 0}

    def unique(fam, make):
        for _ in range(200):
            conv, meta = make()
            key = tuple(t["user"] for t in conv)
            if key not in seen:
                seen.add(key)
                count[fam] += 1
                rows.append(row(fam, count[fam], conv, meta))
                return
        raise ValueError(f"no unique {fam} dialogue after 200 draws")

    def slot_fact(slot, key):
        spec = SLOTS[slot]
        return spec, b.cycle((key, slot), spec["values"])

    def distractors(spec, slot, key, gold, turns, exclude=()):
        vals = b.others((key, "others", slot), spec["values"], {gold, *exclude}, len(turns))
        rels = rng.sample(RELATIONS, len(turns))
        ev = {}
        for t, v, r in zip(turns, vals, rels):
            ev[t] = {"user": rng.choice(spec["distractor"]).format(rel=r, v=v), "gold": rng.choice(ACK_OTHER),
                     "kind": "distractor"}
        return ev, vals

    # ---- recall: slot x N x fact turn x 10 (distractors 0-3 by repetition)
    def recall(slot, N, f, rep):
        spec, gold = slot_fact(slot, "recall")
        target = f + N
        n = target + 1
        ev = {f: {"user": rng.choice(spec["plant"]).format(v=gold), "gold": rng.choice(ACK_FACT), "kind": "plant"}}
        dturns = free_turns(b, target, {f}, rep % 4)
        dev, dvals = distractors(spec, slot, "recall", gold, dturns)
        ev.update(dev)
        ev[target] = {"user": rng.choice(spec["question"]), "gold": spec["frame"].format(gold), "kind": "question"}
        conv = compose(b, ev, n, target)
        return conv, {"slot": slot, "N": N, "fact_turn": f, "target_turn": target, "n_turns": n, "value": gold,
                      "distractor_turns": dturns, "distractor_values": dvals}

    for slot, N, f, rep in itertools.product(SLOTS, N_RANGE, FACT_TURNS, range(REPS["recall"])):
        unique("recall", lambda: recall(slot, N, f, rep))

    # ---- correction: slot x N (from the correction) x 20; plant 1-3, correction 1-4 turns later
    def correction(slot, N, rep):
        spec, new = slot_fact(slot, "fix")
        old = b.others(("fix", "old", slot), spec["values"], {new}, 1)[0]
        pairs = [(p, c) for p in FACT_TURNS for c in range(p + 1, p + 5) if c + N <= MAX_TARGET]
        p, c = pairs[rng.randrange(len(pairs))]
        target = c + N
        n = target + 1
        ev = {p: {"user": rng.choice(spec["plant"]).format(v=old), "gold": rng.choice(ACK_FACT), "kind": "plant"},
              c: {"user": rng.choice(spec["correction"]).format(new=new, old=old), "gold": rng.choice(ACK_FIX),
                  "kind": "correction"}}
        dturns = free_turns(b, target, {p, c}, rep % 3)
        dev, dvals = distractors(spec, slot, "fix", new, dturns, exclude=(old,))
        ev.update(dev)
        ev[target] = {"user": rng.choice(spec["question"]), "gold": spec["frame"].format(new), "kind": "question"}
        conv = compose(b, ev, n, target)
        return conv, {"slot": slot, "N": N, "fact_turn": c, "plant_turn": p, "target_turn": target, "n_turns": n,
                      "value": new, "old_value": old, "distractor_turns": dturns, "distractor_values": dvals}

    for slot, N, rep in itertools.product(SLOTS, N_RANGE, range(REPS["correction"])):
        unique("correction", lambda: correction(slot, N, rep))

    # ---- instruction: kind x N x 26; the instruction (with a question) at turn 1-3
    def instruction(kind, N, rep):
        spec = INSTRUCTIONS[kind]
        form = b.cycle(("inst", kind), spec["forms"])
        at = FACT_TURNS[rep % 3]
        target = at + N
        n = target + 1
        qa = rng.sample(QA, 2)
        say = rng.choice(spec["say"]).format(W=form)

        def answer(s):
            return comply(kind, form, s[0].upper() + s[1:] + ".")
        ev = {at: {"user": f"{say} {qa[0][0]}", "gold": answer(qa[0][1]), "kind": "instruction"},
              target: {"user": qa[1][0], "gold": answer(qa[1][1]), "kind": "question_inst"}}
        conv = compose(b, ev, n, target, instruction=(kind, form, at))
        return conv, {"slot": kind, "N": N, "fact_turn": at, "target_turn": target, "n_turns": n, "form": form,
                      "distractor_turns": [], "distractor_values": []}

    for kind, N, rep in itertools.product(INSTRUCTIONS, N_RANGE, range(REPS["instruction"])):
        unique("instruction", lambda: instruction(kind, N, rep))
    validate(rows)
    return rows


# ---------------------------------------------------------------- validation (build time)
def own_values():
    """This set's value strings: slot values, instruction forms and QA answers' content words."""
    vals = [v for s in SLOTS.values() for v in s["values"]]
    vals += [f for k in INSTRUCTIONS.values() for f in k["forms"]]
    vals += [a for _, _, a in QA]
    return vals


def banned_small_talk_words():
    """Small talk must hold no slot subject, no relation word and no value word of this set."""
    out = {w for s in SUBJECT_WORDS for w in forms(s)}
    for v in [v for s in SLOTS.values() for v in s["values"]]:
        for x in words(v):
            out |= forms(x)
    return out


def validate(rows):
    """Structure: one target per dialogue at fact turn + N, N 1-12, at most 16 turns, the gold reply the frame
    filled with the gold (recall, correction) or the form applied (instruction: every reply from the instruction
    on complies, none before it); small talk holds no subject, relation or value word of this set."""
    ban = banned_small_talk_words()
    for text in [t for p in SMALL_TALK + OPENERS + CLOSINGS for t in p] + \
            [t for arc in ARCS for p in arc for t in p] + ACK_FACT + ACK_FIX + ACK_OTHER:
        bad = sorted({x for x in words(text) if x in ban})
        if bad:
            raise ValueError(f"filler {text!r} raises {bad}")
    if len({q for q, _ in SMALL_TALK} | {q for a in ARCS for q, _ in a}) != len(SMALL_TALK) + 2 * len(ARCS):
        raise ValueError("small talk questions repeat")
    for r in rows:
        m, meta = r["messages"], r["meta"]
        n, tt, f, N = meta["n_turns"], meta["target_turn"], meta["fact_turn"], meta["N"]
        assert len(m) == 2 * n and n == tt + 1 <= MAX_TARGET + 1 and tt - f == N and 1 <= N <= 12, r["uri"]
        assert [x["role"] for x in m] == ["user", "assistant"] * n, r["uri"]
        gold = m[2 * tt - 1]["content"]
        kinds = meta["turn_kinds"]
        if meta["family"] in ("recall", "correction"):
            spec = SLOTS[meta["slot"]]
            assert gold == spec["frame"].format(meta["value"]), r["uri"]
            assert kinds[f - 1] == ("plant" if meta["family"] == "recall" else "correction"), r["uri"]
            assert all(t < tt for t in meta["distractor_turns"]), r["uri"]
            if meta["family"] == "correction":
                assert meta["plant_turn"] < f and kinds[meta["plant_turn"] - 1] == "plant", r["uri"]
                assert meta["old_value"] != meta["value"], r["uri"]
            assert meta["value"] not in meta["distractor_values"], r["uri"]
        else:
            kind, form = meta["slot"], meta["form"]
            for k in range(n):
                a = m[2 * k + 1]["content"]
                ok = a.startswith(form + ", ") if kind == "opener" else a.endswith(", " + form + ".")
                assert ok == (k >= f - 1), (r["uri"], k, a)
        for t, k in zip(m[0::2], kinds):
            if k in ("small_talk", "opener", "closing"):
                assert not any(x in ban for x in words(t["content"])), (r["uri"], t)


def dump(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def pinned():
    if not os.path.exists(HASH):
        return None
    for line in open(HASH, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            return line.split()[0]
    return None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def build_checked():
    """(bytes, sha256) of the set, refused unless the sha256 is the pin (the data job's path)."""
    data = dump(build_rows())
    sha = sha256_bytes(data)
    pin = pinned()
    if pin != sha:
        raise SystemExit(f"{VERSION}: this build's sha256 {sha} is not the pinned {pin} ({HASH})")
    return data, sha


# ---------------------------------------------------------------- ledger
def ledger(rows, data, vocab=None):
    def hist(key, fam=None):
        h = {}
        for r in rows:
            if fam and r["meta"]["family"] != fam:
                continue
            k = str(r["meta"][key])
            h[k] = h.get(k, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: (len(kv[0]), kv[0])))
    fams = sorted({r["meta"]["family"] for r in rows})
    out = {"version": VERSION, "seed": SEED, "rows": len(rows), "sha256": sha256_bytes(data), "bytes": len(data),
           "pinned": pinned(), "generator_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "families": {f: {"rows": sum(1 for r in rows if r["meta"]["family"] == f), "N": hist("N", f),
                            "slot": hist("slot", f), "fact_turn": hist("fact_turn", f), "n_turns": hist("n_turns", f),
                            "distractors": {str(k): sum(1 for r in rows if r["meta"]["family"] == f and
                                                        len(r["meta"]["distractor_turns"]) == k) for k in range(4)}}
                        for f in fams},
           "N_all": hist("N"),
           "values": {s: len(v["values"]) for s, v in SLOTS.items()},
           "instruction_forms": {k: v["forms"] for k, v in INSTRUCTIONS.items()},
           "fillers": {"small_talk": len(SMALL_TALK), "arcs": len(ARCS), "openers": len(OPENERS),
                       "closings": len(CLOSINGS), "qa": len(QA)},
           "turns": sum(r["meta"]["n_turns"] for r in rows),
           "small_talk_turns": sum(sum(1 for k in r["meta"]["turn_kinds"] if k == "small_talk") for r in rows)}
    if vocab:
        import chat_template as CT
        import llama3_bpe
        tok = llama3_bpe.from_gguf(vocab)

        def n(text):
            return len(tok.encode(text, add_special_tokens=False).ids)
        lens, trained = [], []
        for r in rows:
            m = r["messages"]
            total = sum(n(x["content"]) for x in m) + 3 * (len(m) // 2)
            lens.append(total)
            trained.append(sum(n(x["content"]) + 1 for x in m[1::2]))
            if len(lens) % 500 == 1:  # the rendering in one piece equals the sum of its pieces
                pairs = [(m[i]["content"], m[i + 1]["content"]) for i in range(0, len(m), 2)]
                assert n(CT.render(CT.from_pairs(pairs))) == total, r["uri"]
        lens.sort()
        out["tokens"] = {"vocab_sha256": hashlib.sha256(open(vocab, "rb").read()).hexdigest(),
                         "per_row": {"min": lens[0], "p50": lens[len(lens) // 2], "max": lens[-1],
                                     "mean": round(sum(lens) / len(lens), 1)},
                         "total": sum(lens), "trained_total": sum(trained),
                         "over_1024": sum(1 for x in lens if x > 1025)}
    return out


# ---------------------------------------------------------------- the disjointness check
def _items(path):
    return [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]


def eval_values(items, which):
    """The evaluation set's values: every list its items draw from (carry5_build's names, pet names, cities, jobs,
    favourites, topic values, ages 18-79, Carry-12's instruction forms) and every item's candidates, answers,
    distractor and old values (instruction candidates: their forms)."""
    import carry5_build as C5
    import carry12_build as C12
    vals = set()
    for vs in [C5.NAMES, C5.PET_NAMES, C5.CITIES, C5.JOBS, *C5.FAVOURITES.values(),
               *(t["values"] for t in C5.TOPICS.values())]:
        vals |= set(vs)
    vals |= {str(a) for a in range(18, 80)} | {w for a in range(18, 80) for w in C5.number_words(a)}
    if which == "carry12":
        vals |= {f for k in C12.INSTRUCTIONS.values() for f in k["forms"]}
    for it in items:
        if it["family"] == "instruction":
            vals |= set((it.get("check") or {}).get("forms") or [])
            vals |= set(it.get("answers") or [])
        else:
            vals |= set(it["candidates"])
            vals |= set(it.get("answers") or [])
        vals |= set(it.get("distractor_values") or [])
        if it.get("old_value"):
            vals.add(it["old_value"])
    return vals


def number_forms(w):
    """A value word with its plural, and its singular when the value is plural (whales, tomatoes); unlike forms(),
    never a stem of three letters or fewer, which would turn Ines into 'in' and bees into 'be'."""
    out = {w, w + "s", w + "es"}
    if w.endswith("es") and len(w) > 5:
        out.add(w[:-2])
    if w.endswith("s") and len(w) > 4:
        out.add(w[:-1])
    return out


def entity_words(which):
    """Words of the evaluation set's named entities and slot values (the lists its memory families ask about):
    the text of this set may hold none of them."""
    import carry5_build as C5
    import carry12_build as C12
    ws = set()
    for vs in [C5.NAMES, C5.PET_NAMES, C5.CITIES, C5.JOBS, *C5.FAVOURITES.values(),
               *(t["values"] for t in C5.TOPICS.values()), C5.ANIMALS]:
        for v in vs:
            ws |= {f for x in words(v) for f in number_forms(x)}
    ws |= {str(a) for a in range(18, 80)}
    ws |= {"eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy"}
    forms_ = C12.INSTRUCTIONS if which == "carry12" else {}
    single = {f.lower() for k in forms_.values() for f in k["forms"] if len(f.split()) == 1}
    phrases = [f.lower() for k in forms_.values() for f in k["forms"] if len(f.split()) > 1]
    return ws | single, phrases


STOP_VALUE = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "is", "it", "as", "all", "no"}


def disjoint_report(rows, sets=(("carry12", CARRY12), ("carry5", CARRY5))):
    """Our checks of this set against each evaluation set: (1) corpus/decontam.py's rules (word 13-grams, a user
    turn equal to a 4+ word eval prompt, an 8-12 word eval prompt as a span) with the data job's targets (the
    frozen eval sets, Carry-5 and Carry-12, built by corpus/fetchers/human_dialogue.decontaminator): rows it would
    drop; (2) carry5_build's scan (Carry-12's decontam): 13-grams, prompts, spans, every 3+ word template sentence
    as a span, fact sentences and item leaks; (3) exact value overlap: this set's values (slot values, instruction
    forms, QA answers' content words) against the evaluation set's (eval_values), as whole strings and as content
    words with plural forms; (4) entities: any word of the evaluation set's entity and slot-value lists (or an
    instruction form) anywhere in this set's text; (5) exact equality of any user turn or reply, and of any raw
    template string."""
    sys.path.insert(0, os.path.join(ROOT, "corpus"))
    sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
    import decontam
    import carry5_build as C5
    import carry12_build as C12
    import human_dialogue
    dc, _ = human_dialogue.decontaminator(carry5=True, carry12=True)
    drops, flags = [], 0
    for r in rows:
        d, f = dc.check([m["content"] for m in r["messages"] if m["role"] == "user"],
                        "\n".join(m["content"] for m in r["messages"]))
        if d:
            drops.append({"uri": r["uri"], "matches": d})
        flags += len(f)
    rep = {"rows": len(rows), "decontam_py": {"targets": "simple_qa_v1, tinychat_hf_prompts_v1, chat_prompts_v1, "
                                                         "Carry-5 v1 and Carry-12 v1 (turns, golds, twins)",
                                              "rows_dropped": len(drops), "drops": drops[:20], "answer_flags": flags,
                                              "rules": {"ngram": decontam.N, "span_min_words": decontam.SPAN_MIN,
                                                        "prompt_min_words": decontam.PROMPT_MIN}}}
    ours_vals = own_values()
    ours_val_words = {f for v in ours_vals for x in words(v) if x not in STOP_VALUE for f in forms(x)}
    our_turns = {m["content"] for r in rows for m in r["messages"]}
    our_templates = {t for s in SLOTS.values() for k in ("plant", "distractor", "question", "correction")
                     for t in s[k]} | {s["frame"] for s in SLOTS.values()} | \
        {t for k in INSTRUCTIONS.values() for t in k["say"]}
    for name, path in sets:
        items = _items(path)
        if name == "carry12":
            items_scan = [dict(it, family="recall") if it["family"] == "load" else it for it in items]
        else:
            items_scan = items
        pats = C5.Patterns(items_scan, decontam)
        hits = {}
        for r in rows:
            users = [m["content"] for m in r["messages"] if m["role"] == "user"]
            for rule, pid, ctx in pats.check(users, "\n".join(m["content"] for m in r["messages"])):
                h = hits.setdefault(rule, {"rows": 0, "examples": []})
                h["rows"] += 1
                if len(h["examples"]) < 5:
                    h["examples"].append({"uri": r["uri"], "pattern": pid, "context": ctx})
        ev = eval_values(items, name)
        ev_words = {f for v in ev for x in words(v) if x not in STOP_VALUE for f in forms(x)}
        whole = sorted(set(v.lower() for v in ours_vals) & {v.lower() for v in ev})
        word_overlap = sorted(ours_val_words & ev_words)
        ent, phrases = entity_words(name)
        text_hits = {}
        for r in rows:
            for m in r["messages"]:
                ws = words(m["content"])
                bad = sorted({x for x in ws if x in ent})
                low = " " + " ".join(ws) + " "
                bad += [p for p in phrases if " " + " ".join(words(p)) + " " in low]
                for x in bad:
                    t = text_hits.setdefault(x, {"rows": 0, "example": r["uri"]})
                    t["rows"] += 1
        eval_turns = {t[k] for it in items for t in it["turns"] for k in ("user", "gold")}
        eval_turns |= {it["twin"] for it in items if it.get("twin")}
        eval_templates = set()
        for s in C5.SLOTS.values():
            for k in ("plant", "distractor", "question", "correction"):
                eval_templates |= set(s[k])
            eval_templates.add(s["frame"])
        if name == "carry12":
            eval_templates |= {t for k in C12.INSTRUCTIONS.values() for t in k["say"]}
        rep[name] = {"set": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                     "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "items": len(items),
                     "carry5_build_scan": {k: v for k, v in sorted(hits.items())},
                     "value_overlap": {"eval_values": len(ev), "ours": len(ours_vals), "whole_strings": whole,
                                       "content_words": word_overlap},
                     "entity_words_in_text": text_hits,
                     "equal_turns": sorted(our_turns & eval_turns)[:20],
                     "equal_turns_n": len(our_turns & eval_turns),
                     "equal_templates": sorted(our_templates & eval_templates)}
        rep[name]["disjoint"] = (not hits and not whole and not word_overlap and not text_hits
                                 and not rep[name]["equal_turns_n"] and not rep[name]["equal_templates"])
    rep["disjoint"] = rep["decontam_py"]["rows_dropped"] == 0 and all(rep[n]["disjoint"] for n, _ in sets)
    return rep


# ---------------------------------------------------------------- commands
def cmd_build(a):
    data = dump(build_rows())
    sha = sha256_bytes(data)
    pin = pinned()
    if pin and pin != sha:
        sys.exit(f"this build's sha256 {sha} differs from the pinned {pin}")
    with open(a.out, "wb") as f:
        f.write(data)
    if not pin:
        with open(HASH, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Carry-12 template-SFT set v1 (cell T's template rows; tools/carry12_tsft_build.py, seed 20261027):"
                    " TRAINING data for cell T only, disjoint from Carry-12 v1 and Carry-5 v1\n")
            f.write(f"{sha}  carry12_tsft_v1.jsonl\n")
    print(json.dumps({"rows": data.count(b"\n"), "sha256": sha, "out": a.out}))
    return 0


def cmd_check(a):
    data = dump(build_rows())
    sha = sha256_bytes(data)
    ok = sha == pinned()
    print(json.dumps({"sha256": sha, "pinned": pinned(), "ok": ok}))
    return 0 if ok else 1


def cmd_ledger(a):
    rows = build_rows()
    rep = ledger(rows, dump(rows), a.vocab or None)
    text = json.dumps(rep, indent=1)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print(json.dumps({k: rep[k] for k in ("rows", "sha256", "pinned")}))
    return 0


def cmd_disjoint(a):
    rows = build_rows()
    rep = {"version": VERSION, "sha256": sha256_bytes(dump(rows)), **disjoint_report(rows)}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"disjoint": rep["disjoint"], "decontam_py_dropped": rep["decontam_py"]["rows_dropped"],
                      **{n: rep[n]["disjoint"] for n in ("carry12", "carry5")}}))
    return 0 if rep["disjoint"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default="carry12_tsft_v1.jsonl")
    sub.add_parser("check")
    p = sub.add_parser("ledger")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", default="")
    p = sub.add_parser("disjoint")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check, "ledger": cmd_ledger, "disjoint": cmd_disjoint}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
