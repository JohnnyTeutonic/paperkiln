#!/usr/bin/env python3
"""Carry-5: a fixed 5-turn chat benchmark for tiny language models (5M-150M parameters).

    python tools/carry5_build.py build [--out corpus/eval/carry5_v1.jsonl]
    python tools/carry5_build.py check            # regenerate in memory; compare bytes and the pinned sha256
    python tools/carry5_build.py decontam --out REPORT.json [--state STATE.json] CORPUS [CORPUS ...]

Every item is a scripted 5-turn conversation: five user turns, a canned short assistant reply per
turn (the GOLD history used for teacher forcing), one TARGET turn (4 or 5) whose reply is scored,
its accepted answers, and four forced-choice candidates written into an answer frame shared by all
four ("Your dog is called {}."). Families (experiments/tiny_chat_litcheck_2026-10-23.md section 3):

  recall       a fact about the user (name, pet, city, job, age, a favourite) planted at turn 1-3,
               asked at turn 4 or 5; 0-2 same-type distractor facts about other people.
  correction   the fact is planted, then corrected ("..., not X"); the question needs the corrected
               value; the old value is always a candidate; 0-2 further distractors.
  instruction  a standing instruction (one word / five words or fewer / yes or no / a number only)
               given with a question at turn 1-3; the target question at turn 4 or 5 should still
               respect it (a machine check, tools/carry5_eval.py); candidates are one compliant and
               three non-compliant but correct replies.
  topic        a topic with a detail (a trip to a city, a cake flavour ...) at turn 1-3, a
               digression ("By the way, ..."), then "Anyway, back to ..." at turn 4 or 5; 0-2
               same-type distractors inside the digression.
  knowledge    a common-knowledge question at turn 4 or 5 after small talk (few items: the objective
               is chat, knowledge is secondary); none repeats a simple_qa_v1 question.

Controls carried by every item: `interference` (0-2 distractors), `distance` (target turn minus the
turn holding the needed fact or instruction), a single-turn `twin` (the same statements and the
question in one user turn: reading without carry-over) and `noctx` (the question alone: the model's
prior over the candidates, chance by design because every value is gold and distractor equally
often across the set).

Items come from the hand-written templates and value lists below with a fixed seed, so the file is
byte-reproducible; it is frozen at first generation (sha256 in corpus/hashes/carry5_v1.sha256) and a
change is a new version. It is an evaluation set: nothing here may enter training data.

`decontam` checks the set against training corpora with corpus/decontam.py's normalisation and
rules (word 13-grams, a user turn equal to an item's user turn of 4+ words, an item's user turn of
8-12 words as a contiguous span) plus two Carry-5 rules: every template sentence of 3+ words as a
span (expected: templates are everyday English), and ITEM LEAKS, a training row holding an item's
value-bearing fact sentence and its question both (the contamination that matters). It reads
.jsonl rows with "messages", "prompt"/"completion" (SFT rows) or "text", and .txt files as
documents separated by <|endoftext|>; it checkpoints its position every ~2 minutes and resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "corpus", "eval", "carry5_v1.jsonl")
HASH = os.path.join(ROOT, "corpus", "hashes", "carry5_v1.sha256")
VERSION = "carry5_v1"
SEED = 20261023
# Cells are repeated with fresh draws: recall 108 cells x 4, correction 108 x 2, instruction 24 x 6,
# topic 90 x 2, knowledge 60 questions once: 1,032 items (the pre-registration's item-level power).
RECALL_REPS, CORRECTION_REPS, INSTRUCTION_REPS, TOPIC_REPS = 4, 2, 6, 2

# ---------------------------------------------------------------- hand-written value lists
NAMES = ["Priya", "Tomas", "Aisha", "Mateo", "Ingrid", "Kofi", "Mei", "Rafael", "Freya", "Dmitri", "Leila",
         "Oscar", "Nadia", "Hugo", "Saanvi", "Callum", "Yuki", "Elena", "Tariq", "Maeve", "Bruno", "Anika",
         "Felix", "Zara", "Ravi", "Greta", "Omar", "Isla", "Kenji", "Lucia", "Arjun", "Sienna", "Malik", "Hana",
         "Declan", "Esme", "Niko", "Amara", "Viktor", "Rosa", "Ines", "Joaquin", "Ayla", "Soren", "Femi", "Lena",
         "Pavel", "Ruby", "Tenzin", "Noor"]
PET_NAMES = ["Biscuit", "Pepper", "Milo", "Luna", "Noodle", "Ziggy", "Pickle", "Juniper", "Waffles", "Bramble",
             "Toffee", "Olive", "Pumpkin", "Socks", "Rocket", "Clover", "Dumpling", "Nutmeg", "Pebble", "Jasper",
             "Hazel", "Rusty", "Maple", "Bean", "Gizmo", "Truffle", "Willow", "Sprout", "Comet", "Fudge",
             "Marbles", "Poppy", "Ginger", "Scout", "Nacho", "Domino", "Peanut", "Button", "Tango", "Lulu"]
ANIMALS = ["dog", "cat", "rabbit", "hamster", "parrot", "tortoise"]
CITIES = ["Leeds", "Porto", "Hobart", "Bergen", "Lyon", "Seville", "Krakow", "Adelaide", "Galway", "Utrecht",
          "Bilbao", "Malmo", "Dundee", "Geelong", "Tampere", "Graz", "Bologna", "Ghent", "Aarhus", "Cork",
          "Halifax", "Nagoya", "Busan", "Cusco", "Mombasa", "Darwin", "Toulouse", "Bristol", "Cairns", "Valencia",
          "Bremen", "Nantes", "Gdansk", "Turku", "Antwerp", "Granada", "Kyoto", "Leipzig", "Limerick", "Brisbane"]
JOBS = ["nurse", "baker", "plumber", "teacher", "pilot", "chef", "dentist", "librarian", "gardener", "carpenter",
        "electrician", "architect", "engineer", "farmer", "firefighter", "journalist", "lawyer", "mechanic",
        "pharmacist", "photographer", "accountant", "vet", "tailor", "optician", "butcher", "barber", "florist",
        "surveyor", "welder", "translator", "paramedic"]
FAVOURITES = {"colour": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey", "silver"],
              "fruit": ["mango", "banana", "peach", "cherry", "kiwi", "pear", "plum", "pineapple", "apricot", "lime"],
              "sport": ["tennis", "football", "swimming", "cricket", "golf", "rugby", "basketball", "cycling",
                        "hockey", "netball"]}
RELATIVES = ["brother", "sister", "cousin", "friend", "neighbour", "aunt", "uncle", "colleague"]
NEUTRAL = ["cousin", "friend", "neighbour", "colleague"]  # name distractors: no gendered relative

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n):
    """Accepted spellings of n (0-99): ["34", "thirty-four", "thirty four"]."""
    if n < 20:
        return [str(n), ONES[n]]
    t, o = divmod(n, 10)
    if o == 0:
        return [str(n), TENS[t]]
    return [str(n), f"{TENS[t]}-{ONES[o]}", f"{TENS[t]} {ONES[o]}"]


def article(noun):
    return ("an " if noun[0] in "aeiou" else "a ") + noun


# ---------------------------------------------------------------- conversation filler
OPENERS = [("Hi there! How are you today?", "I am well, thank you. How can I help?"),
           ("Hello! Can we chat for a bit?", "Of course! What would you like to talk about?"),
           ("Good morning! I have a few things to ask you.", "Good morning! Ask away."),
           ("Hey, have you got a minute?", "Yes, of course. What is on your mind?")]
FILLERS = [("What is a good way to relax after work?", "A short walk or a warm bath can help you relax."),
           ("Can you suggest something quick to cook for dinner?", "A simple stir fry with vegetables is quick and tasty."),
           ("How can I sleep better at night?", "Try to go to bed at the same time each night and avoid screens."),
           ("What should I take on a picnic?", "Sandwiches, water and a blanket are a good start."),
           ("Do you have a tip for remembering things?", "Writing things down in a small notebook helps a lot."),
           ("What is a nice thing to do on a rainy day?", "You could read a book, bake something or watch a film."),
           ("How do I keep my houseplants healthy?", "Give them enough light, and do not water them too much."),
           ("Can you tell me a fun fact?", "Sea otters hold hands while they sleep."),
           ("What is a good way to start the morning?", "A glass of water and a short stretch are a good start."),
           ("How can I save a little money each week?", "Plan your meals and make a shopping list before you shop."),
           ("What is a polite way to say no to an invitation?", "Thank them warmly and say you cannot make it this time."),
           ("Any ideas for a small gift for a friend?", "A candle, a book or homemade biscuits make lovely gifts."),
           ("How do I make a cup of tea?", "Boil water, pour it over a tea bag, wait a few minutes, then add milk if you like."),
           ("What can I do when I feel bored?", "Try a short walk, a puzzle or calling someone you like."),
           ("How can I be more organised?", "Keep a simple to-do list and tidy up for a few minutes each day."),
           ("What is a good exercise for beginners?", "Walking is a great place to start."),
           ("Can you suggest a game to play at a party?", "Charades is fun and needs no equipment."),
           ("How do I stay warm in winter?", "Wear layers, a hat and a warm coat."),
           ("What makes a good breakfast?", "Porridge or eggs on toast will keep you going."),
           ("How do I stop my glasses from fogging up?", "Wash them with a drop of soap and let them dry."),
           ("What should I do if I cannot fall asleep?", "Get up for a few minutes, read something calm, then try again."),
           ("Is it fine to take a short nap in the afternoon?", "Yes, a short nap can help, but keep it under half an hour.")]
CLOSINGS = [("Thanks, that is all for now.", "You are welcome! Have a lovely day."),
            ("Great, thank you for chatting.", "It was a pleasure. Talk soon!"),
            ("Okay, I have to go now. Bye!", "Goodbye! Take care.")]
ACK_FACT = ["Thanks for telling me.", "That is good to know.", "Okay, I will remember that.", "Thanks for sharing that."]
ACK_NAME = ["Nice to meet you!", "Lovely to meet you.", "Hello, nice to meet you."]
ACK_OTHER = ["That sounds nice.", "How lovely.", "Oh, that is nice."]
ACK_FIX = ["Thanks for correcting that.", "No problem, thanks for letting me know.", "Got it, thanks for the correction."]
ACK_TOPIC = ["That sounds lovely!", "How exciting!", "What a nice plan."]

# ---------------------------------------------------------------- recall / correction slots
# plant, distractor, question and correction templates; frame = the forced-choice answer.
SLOTS = {
    "name": {"plant": ["My name is {v}.", "By the way, my name is {v}.", "You can call me {v}."],
             "distractor": ["My {rel} is called {v}.", "My {rel}'s name is {v}."],
             "question": ["What is my name?", "Do you remember my name?", "Can you tell me my name?"],
             "correction": ["Sorry, I said that wrong. My name is {new}, not {old}.",
                            "Actually, my name is {new}, not {old}."],
             "frame": "Your name is {}."},
    "pet": {"plant": ["I have a {a} called {v}.", "My {a} is called {v}."],
            "distractor": ["My {rel} has a {a} called {v}."],
            "question": ["What is my {a} called?", "What is the name of my {a}?"],
            "correction": ["Sorry, I got that wrong. My {a} is called {new}, not {old}.",
                           "Actually, my {a} is called {new}, not {old}."],
            "frame": "Your {a} is called {}."},
    "city": {"plant": ["I live in {v}.", "My home is in {v}."],
             "distractor": ["My {rel} lives in {v}."],
             "question": ["Which city do I live in?", "Where do I live?"],
             "correction": ["Sorry, my mistake. I live in {new}, not {old}.",
                            "Actually, I live in {new}, not {old}."],
             "frame": "You live in {}."},
    "job": {"plant": ["I work as {v}.", "I have a job as {v}."],
            "distractor": ["My {rel} works as {v}."],
            "question": ["What is my job?", "What do I do for work?"],
            "correction": ["Sorry, I said that wrong. I work as {new}, not {old}.",
                           "Actually, I work as {new}, not {old}."],
            "frame": "You work as {}."},
    "age": {"plant": ["I am {v} years old.", "I turned {v} this year."],
            "distractor": ["My {rel} is {v} years old."],
            "question": ["How old am I?", "What is my age?"],
            "correction": ["Sorry, I made a mistake. I am {new} years old, not {old}.",
                           "Actually, I am {new} years old, not {old}."],
            "frame": "You are {} years old."},
    "favourite": {"plant": ["My favourite {c} is {v}."],
                  "distractor": ["My {rel}'s favourite {c} is {v}."],
                  "question": ["What is my favourite {c}?", "Which {c} do I like best?"],
                  "correction": ["Sorry, I got that wrong. My favourite {c} is {new}, not {old}.",
                                 "Actually, my favourite {c} is {new}, not {old}."],
                  "frame": "Your favourite {c} is {}."},
}

# ---------------------------------------------------------------- topic return
TOPICS = {
    "trip": {"values": CITIES,
             "plant": ["I am going on a trip to {v} next month.", "Next month I am travelling to {v} for a holiday."],
             "distractor": ["My {rel} has just come back from {v}.", "My {rel} went to {v} last summer."],
             "question": "Anyway, back to my trip. Where am I going?",
             "twin_q": ["Where am I going on my trip?", "Which city am I travelling to?"],
             "frame": "You are going to {}."},
    "cake": {"values": ["lemon", "chocolate", "carrot", "coconut", "orange", "coffee", "almond", "strawberry",
                        "ginger", "vanilla"],
             "plant": ["I am baking a {v} cake for my mum's birthday.", "For my dad's birthday I am making a {v} cake."],
             "distractor": ["Last year my {rel} made a {v} cake.", "My {rel} prefers {v} cake."],
             "question": "Anyway, back to the cake. What flavour is it?",
             "twin_q": ["What flavour is the cake I am baking?", "Which flavour of cake am I making?"],
             "frame": "It is a {} cake."},
    "book": {"values": ["whales", "volcanoes", "trains", "castles", "bees", "pirates", "rivers", "robots", "deserts",
                        "glaciers"],
             "plant": ["I am reading a book about {v}.", "I just started a library book about {v}."],
             "distractor": ["My {rel} is reading a book about {v}.", "My {rel} wrote a story about {v}."],
             "question": "Anyway, back to my book. What is it about?",
             "twin_q": ["What is my book about?", "What is the subject of the book I am reading?"],
             "frame": "Your book is about {}."},
    "garden": {"values": ["tomatoes", "roses", "carrots", "sunflowers", "potatoes", "tulips", "beans", "lettuce",
                          "strawberries", "onions"],
               "plant": ["This weekend I am planting {v} in my garden.", "I want to grow {v} in my garden this year."],
               "distractor": ["My {rel} is planting {v}.", "My {rel} grows {v} on a balcony."],
               "question": "Anyway, back to my garden. What am I planting?",
               "twin_q": ["What am I planting in my garden?", "What am I going to grow in my garden?"],
               "frame": "You are planting {}."},
    "party": {"values": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
              "plant": ["I am having a party on {v}.", "My birthday party will be on {v}."],
              "distractor": ["My {rel}'s party is on {v}.", "My {rel} has a dinner on {v}."],
              "question": "Anyway, back to my party. Which day is it on?",
              "twin_q": ["Which day is my party on?", "On which day is my party?"],
              "frame": "Your party is on {}."},
    "music": {"values": ["piano", "guitar", "violin", "drums", "flute", "trumpet", "cello", "harp", "clarinet",
                         "ukulele"],
              "plant": ["I have just started learning the {v}.", "I am taking lessons to play the {v}."],
              "distractor": ["My {rel} plays the {v}.", "My {rel} once tried the {v}."],
              "question": "Anyway, back to my music lessons. Which instrument am I learning?",
              "twin_q": ["Which instrument am I learning?", "What instrument am I having lessons on?"],
              "frame": "You are learning the {}."},
}

# ---------------------------------------------------------------- persistent instructions
# (question, compliant short answer, content statement, accepted content answers)
ONE_WORD_QA = [("What colour is snow?", "White.", "snow is white", ["white"]),
               ("Which animal barks?", "Dog.", "a dog barks", ["dog"]),
               ("What do cows give us to drink?", "Milk.", "cows give us milk", ["milk"]),
               ("Which meal do people eat in the morning?", "Breakfast.", "people eat breakfast in the morning",
                ["breakfast"]),
               ("What do you wear on your feet?", "Shoes.", "you wear shoes on your feet", ["shoe", "sock", "boot"]),
               ("Which animal has a trunk?", "Elephant.", "an elephant has a trunk", ["elephant"]),
               ("What do you sleep in at night?", "Bed.", "you sleep in a bed at night", ["bed"]),
               ("What colour is a ripe tomato?", "Red.", "a ripe tomato is red", ["red"]),
               ("Which season is the hottest?", "Summer.", "summer is the hottest season", ["summer"]),
               ("What do birds use to fly?", "Wings.", "birds use their wings to fly", ["wing"]),
               ("What do you use to open a locked door?", "Key.", "you use a key to open a locked door", ["key"]),
               ("Which animal says quack?", "Duck.", "a duck says quack", ["duck"]),
               ("What colour is a lemon?", "Yellow.", "a lemon is yellow", ["yellow"]),
               ("Which animal has a mane and roars?", "Lion.", "a lion has a mane and roars", ["lion"]),
               ("What do you drink when you are thirsty?", "Water.", "you drink water when you are thirsty",
                ["water"]),
               ("Which room do people cook in?", "Kitchen.", "people cook in the kitchen", ["kitchen"]),
               ("What do children write with at school?", "Pencils.", "children write with pencils", ["pencil", "pen"]),
               ("Which bird lays the eggs we eat?", "Hen.", "a hen lays the eggs we eat", ["hen", "chicken"]),
               ("What do you wear on your hands in the cold?", "Gloves.", "you wear gloves on your hands in the cold",
                ["glove", "mitten"]),
               ("Which planet do we live on?", "Earth.", "we live on Earth", ["earth"]),
               ("What colour is fresh grass?", "Green.", "fresh grass is green", ["green"]),
               ("Which bird hoots at night?", "Owl.", "an owl hoots at night", ["owl"])]
SHORT_QA = [(q, s[0].upper() + s[1:] + ".", s, a) for q, _, s, a in ONE_WORD_QA if len(s.split()) <= 5]
YES_NO_QA = [("Is snow cold to touch?", "Yes.", "snow is cold to touch", ["yes"]),
             ("Can a fish climb a tree?", "No.", "a fish cannot climb a tree", ["no"]),
             ("Is the sea salty?", "Yes.", "the sea is salty", ["yes"]),
             ("Do dogs have feathers?", "No.", "dogs do not have feathers", ["no"]),
             ("Can a candle flame burn you?", "Yes.", "a candle flame can burn you", ["yes"]),
             ("Can cats fly?", "No.", "cats cannot fly", ["no"]),
             ("Do plants need water?", "Yes.", "plants need water", ["yes"]),
             ("Is a mouse bigger than a horse?", "No.", "a mouse is smaller than a horse", ["no"]),
             ("Is grass usually green?", "Yes.", "grass is usually green", ["yes"]),
             ("Do people breathe underwater without help?", "No.", "people cannot breathe underwater without help",
              ["no"]),
             ("Is a lemon sour?", "Yes.", "a lemon is sour", ["yes"]),
             ("Can a car swim?", "No.", "a car cannot swim", ["no"]),
             ("Is a snail faster than a car?", "No.", "a snail is slower than a car", ["no"]),
             ("Do birds have wings?", "Yes.", "birds have wings", ["yes"]),
             ("Can a stone float on water?", "No.", "a stone cannot float on water", ["no"]),
             ("Is chocolate sweet?", "Yes.", "chocolate is sweet", ["yes"]),
             ("Do fish have fur?", "No.", "fish do not have fur", ["no"]),
             ("Can you see the moon at night?", "Yes.", "you can see the moon at night", ["yes"]),
             ("Can a baby drive a car?", "No.", "a baby cannot drive a car", ["no"]),
             ("Does rain make things wet?", "Yes.", "rain makes things wet", ["yes"])]
NUMBER_QA = [("How many legs does a cat have?", "4.", "a cat has four legs", ["4", "four"]),
             ("How many wheels does a wheelbarrow usually have?", "1.", "a wheelbarrow usually has one wheel", ["1", "one"]),
             ("How many sides does a triangle have?", "3.", "a triangle has three sides", ["3", "three"]),
             ("How many eyes does a person have?", "2.", "a person has two eyes", ["2", "two"]),
             ("How many wheels does a car have?", "4.", "a car has four wheels", ["4", "four"]),
             ("How many fingers are on one hand?", "5.", "there are five fingers on one hand", ["5", "five"]),
             ("How many wheels does a unicycle have?", "1.", "a unicycle has one wheel", ["1", "one"]),
             ("How many legs does a bird have?", "2.", "a bird has two legs", ["2", "two"]),
             ("How many minutes are in an hour?", "60.", "there are sixty minutes in an hour", ["60", "sixty"]),
             ("How many days are there in a weekend?", "2.", "there are two days in a weekend", ["2", "two"]),
             ("How many toes are on one foot?", "5.", "there are five toes on one foot", ["5", "five"]),
             ("How many letters are in the word cat?", "3.", "the word cat has three letters", ["3", "three"]),
             ("How many legs does a duck have?", "2.", "a duck has two legs", ["2", "two"]),
             ("How many ears does a cat have?", "2.", "a cat has two ears", ["2", "two"]),
             ("How many noses does a person have?", "1.", "a person has one nose", ["1", "one"]),
             ("How many wings does a bird have?", "2.", "a bird has two wings", ["2", "two"]),
             ("How many legs does a stool usually have?", "3.", "a stool usually has three legs", ["3", "three"]),
             ("How many hours are in half a day?", "12.", "there are twelve hours in half a day", ["12", "twelve"]),
             ("How many letters are in the word apple?", "5.", "the word apple has five letters", ["5", "five"]),
             ("How many wheels does a motorbike have?", "2.", "a motorbike has two wheels", ["2", "two"])]
INSTRUCTIONS = {
    "one_word": {"say": ["From now on, please answer in one word.", "Please reply with just one word from now on.",
                         "Answer every question in a single word from now on."],
                 "qa": ONE_WORD_QA},
    "short": {"say": ["From now on, keep your answers to five words or fewer.",
                      "Please keep every answer very short from now on, five words at most.",
                      "From now on, use five words or fewer in each answer."],
              "qa": SHORT_QA},
    "yes_no": {"say": ["From now on, please answer only with yes or no.",
                       "Please reply with just yes or no from now on.",
                       "From now on, just say yes or no to my questions."],
               "qa": YES_NO_QA},
    "number": {"say": ["From now on, please answer with a number only.",
                       "Please reply with just a number from now on.",
                       "From now on, answer my questions with just a number."],
               "qa": NUMBER_QA},
}
VERBOSE = ["{S}.", "I think {s}.", "Well, {s}, as far as I know."]
LONG = ["Well, {s}, as far as I know.", "That is a good question, and the answer is that {s}.",
        "I am happy to help with that: {s}, which most people know."]

# ---------------------------------------------------------------- knowledge (none repeats simple_qa_v1)
# (question, frame, gold, three distractors, accepted answers for the gold)
KNOWLEDGE = [
    ("How many wheels does a tricycle have?", "A tricycle has {} wheels.", "three", ["two", "four", "six"],
     ["three", "3"]),
    ("What colour is a ripe banana?", "A ripe banana is {}.", "yellow", ["blue", "purple", "grey"], ["yellow"]),
    ("Which animal says moo?", "A {} says moo.", "cow", ["cat", "duck", "sheep"], ["cow"]),
    ("What do you call a baby cat?", "A baby cat is called a {}.", "kitten", ["puppy", "calf", "foal"],
     ["kitten"]),
    ("What do you call a baby dog?", "A baby dog is called a {}.", "puppy", ["kitten", "lamb", "chick"],
     ["puppy", "pup"]),
    ("What is the first month of the year?", "The first month of the year is {}.", "January",
     ["March", "June", "October"], ["january"]),
    ("Which day comes after Monday?", "The day after Monday is {}.", "Tuesday", ["Thursday", "Saturday", "Sunday"],
     ["tuesday"]),
    ("How many arms does a starfish usually have?", "A starfish usually has {} arms.", "five",
     ["three", "four", "eight"], ["five", "5"]),
    ("What colour is coal?", "Coal is {}.", "black", ["white", "pink", "yellow"], ["black"]),
    ("Which animal has black and white stripes?", "A {} has black and white stripes.", "zebra",
     ["lion", "camel", "bear"], ["zebra"]),
    ("Where do fish live?", "Fish live in {}.", "water", ["trees", "the desert", "the sky"], ["water", "sea",
                                                                                          "river", "ocean"]),
    ("What do we use to cut paper?", "We use {} to cut paper.", "scissors", ["spoons", "pillows", "cups"],
     ["scissors"]),
    ("Which season comes after winter?", "After winter comes {}.", "spring", ["summer", "autumn", "winter"],
     ["spring"]),
    ("What is the opposite of big?", "The opposite of big is {}.", "small", ["tall", "fast", "loud"],
     ["small", "little", "tiny"]),
    ("What is the opposite of up?", "The opposite of up is {}.", "down", ["left", "over", "back"], ["down"]),
    ("Which meal do people eat in the evening?", "People eat {} in the evening.", "dinner",
     ["breakfast", "brunch", "elevenses"], ["dinner", "supper", "tea"]),
    ("What do you call the yellow part of an egg?", "The yellow part of an egg is the {}.", "yolk",
     ["shell", "crust", "peel"], ["yolk"]),
    ("What do we call a person who flies a plane?", "A person who flies a plane is a {}.", "pilot",
     ["farmer", "baker", "dentist"], ["pilot"]),
    ("Who teaches children at school?", "Children are taught by a {}.", "teacher", ["plumber", "chef", "sailor"],
     ["teacher"]),
    ("Which number comes after nineteen?", "After nineteen comes {}.", "twenty", ["eighteen", "thirty", "twelve"],
     ["twenty", "20"]),
    ("How many minutes are in half an hour?", "There are {} minutes in half an hour.", "thirty",
     ["fifteen", "sixty", "forty"], ["thirty", "30"]),
    ("What do you use to brush your teeth?", "You use a {} to brush your teeth.", "toothbrush",
     ["fork", "comb", "sponge"], ["toothbrush", "brush"]),
    ("What falls from the clouds when it rains?", "When it rains, {} falls from the clouds.", "water",
     ["sand", "sugar", "dust"], ["water", "rain"]),
    ("What is ketchup usually made from?", "Ketchup is made from {}.", "tomatoes", ["potatoes", "carrots", "onions"],
     ["tomato"]),
    ("What do you call a house made of snow?", "It is called {}.", "an igloo", ["a tent", "a castle", "a cabin"],
     ["igloo"]),
    ("What sound does a cat make?", "A cat says {}.", "meow", ["moo", "quack", "woof"], ["meow", "miaow", "mew"]),
    ("Which part of a plant grows under the ground?", "The {} grow under the ground.", "roots",
     ["petals", "leaves", "flowers"], ["root"]),
    ("What do you call the frozen rain that falls as small white flakes?", "It is called {}.", "snow",
     ["fog", "dew", "steam"], ["snow"]),
    ("Which animal is famous for being very slow and carrying its shell?", "The {} is very slow.", "snail",
     ["cheetah", "rabbit", "horse"], ["snail", "tortoise", "turtle"]),
    ("What do you call the person who brings letters to your house?", "That person is the {}.", "postman",
     ["chef", "pilot", "doctor"], ["postman", "postwoman", "postie", "mail carrier", "mailman", "postal worker"]),
    ("What do you call a baby sheep?", "A baby sheep is called a {}.", "lamb", ["kitten", "foal", "chick"], ["lamb"]),
    ("What do you call a baby horse?", "A baby horse is called a {}.", "foal", ["puppy", "lamb", "calf"],
     ["foal", "colt", "filly", "pony"]),
    ("Which month comes after March?", "The month after March is {}.", "April", ["June", "January", "November"],
     ["april"]),
    ("Which day comes before Saturday?", "The day before Saturday is {}.", "Friday",
     ["Monday", "Wednesday", "Sunday"], ["friday"]),
    ("What colour are most fire engines?", "Most fire engines are {}.", "red", ["purple", "brown", "grey"],
     ["red", "yellow"]),
    ("Which animal hops and carries its baby in a pouch?", "The {} carries its baby in a pouch.", "kangaroo",
     ["giraffe", "penguin", "owl"], ["kangaroo", "wallaby"]),
    ("Which bird cannot fly but swims well in icy seas?", "The {} swims in icy seas.", "penguin",
     ["eagle", "parrot", "pigeon"], ["penguin"]),
    ("What do you use to dry yourself after a bath?", "You use a {} to dry yourself.", "towel",
     ["spoon", "pencil", "ladder"], ["towel"]),
    ("What do you wear on your head to keep the sun off?", "You wear a {}.", "hat", ["glove", "sock", "belt"],
     ["hat", "cap"]),
    ("What is the opposite of open?", "The opposite of open is {}.", "closed", ["early", "bright", "wide"],
     ["closed", "shut"]),
    ("What is the opposite of happy?", "The opposite of happy is {}.", "sad", ["fast", "tall", "green"],
     ["sad", "unhappy"]),
    ("What is the opposite of wet?", "The opposite of wet is {}.", "dry", ["loud", "soft", "early"], ["dry"]),
    ("How many legs does a cow have?", "A cow has {} legs.", "four", ["two", "six", "eight"], ["four", "4"]),
    ("How many hands does a person have?", "A person has {} hands.", "two", ["one", "three", "five"], ["two", "2"]),
    ("Which animal gives us wool?", "Wool comes from {}.", "sheep", ["cats", "fish", "snakes"], ["sheep"]),
    ("Which insect lives in a hive?", "A {} lives in a hive.", "bee", ["horse", "rabbit", "whale"], ["bee"]),
    ("What do you call the meal in the middle of the day?", "It is called {}.", "lunch",
     ["breakfast", "supper", "dessert"], ["lunch"]),
    ("Which tool do you use to hit a nail?", "You use a {} to hit a nail.", "hammer", ["spoon", "brush", "sponge"],
     ["hammer"]),
    ("Who fixes people's teeth?", "Teeth are fixed by a {}.", "dentist", ["pilot", "farmer", "baker"], ["dentist"]),
    ("Who grows crops on a farm?", "Crops are grown by a {}.", "farmer", ["dentist", "sailor", "tailor"],
     ["farmer"]),
    ("Which season comes after summer?", "After summer comes {}.", "autumn", ["spring", "winter", "summer"],
     ["autumn", "fall"]),
    ("How many days are in a fortnight?", "There are {} days in a fortnight.", "fourteen",
     ["seven", "ten", "thirty"], ["fourteen", "14"]),
    ("Which number comes before ten?", "The number before ten is {}.", "nine", ["eleven", "five", "seven"],
     ["nine", "9"]),
    ("What shape is a football?", "A football is {}.", "round", ["square", "flat", "triangular"],
     ["round", "sphere", "circle"]),
    ("What do you call small balls of ice that fall from the sky?", "It is called {}.", "hail",
     ["mist", "dew", "steam"], ["hail", "sleet"]),
    ("What do you look at to see the time?", "You look at a {}.", "clock", ["cup", "shoe", "pillow"],
     ["clock", "watch"]),
    ("What does a thermometer measure?", "A thermometer measures {}.", "temperature",
     ["weight", "distance", "speed"], ["temperature", "heat"]),
    ("Who drives a bus?", "A bus is driven by a {}.", "driver", ["pilot", "chef", "nurse"], ["driver"]),
    ("Which farm animal says oink?", "A {} says oink.", "pig", ["cow", "duck", "goat"], ["pig"]),
    ("What do you stick on a letter before you post it?", "You stick on a {}.", "stamp",
     ["button", "feather", "ribbon"], ["stamp"]),
]
KNOWLEDGE_ASK = ["{q}", "Quick question: {q}", "My little cousin asked me this. {q}", "Can you help me with this? {q}"]


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Builder:
    def __init__(self, seed=SEED):
        self.rng = random.Random(seed)
        self.pools = {}

    def cycle(self, key, values):
        """The next value from a per-key shuffled cycle: across the set every value is gold about equally
        often, so the noctx prior averages out to chance."""
        st = self.pools.get(key)
        if not st:
            st = list(values)
            self.rng.shuffle(st)
            self.pools[key] = st
        return st.pop()

    def others(self, values, exclude, k):
        pool = [v for v in values if v not in exclude]
        return self.rng.sample(pool, k)

    def layout(self, n_turns, fixed, free_events, before):
        """Turn contents 1..n_turns: `fixed` {turn: event}; `free_events` are placed into the remaining
        turns before turn `before` (the target), shared when there are fewer free turns than events."""
        turns = {t: [] for t in range(1, n_turns + 1)}
        for t, ev in fixed.items():
            turns[t].append(ev)
        free = [t for t in range(1, before) if t not in fixed]
        if free_events:
            self.rng.shuffle(free)
            slots = free[:len(free_events)] if len(free) >= len(free_events) else \
                [free[i % len(free)] for i in range(len(free_events))]
            for t, ev in zip(slots, free_events):
                turns[t].append(ev)
        return turns


def plant_templates(spec, turn):
    """A conversation does not open with "By the way"."""
    return [t for t in spec["plant"] if turn > 1 or not t.startswith("By the way")]


def compose(turns, target, closing, rng, used_fillers):
    """User text and gold reply per turn from the events; empty turns get an opener (turn 1) or a filler."""
    out = []
    for t in sorted(turns):
        evs = turns[t]
        if t > target:
            out.append({"user": closing[0], "gold": closing[1]})
            continue
        if not evs:
            pool = OPENERS if t == 1 else [f for f in FILLERS if f[0] not in used_fillers]
            q, a = rng.choice(pool)
            used_fillers.add(q)
            out.append({"user": q, "gold": a})
            continue
        text = " ".join(e["text"] for e in evs)
        gold = next((e["gold"] for e in evs if e.get("main")), evs[-1]["gold"])
        out.append({"user": text, "gold": gold})
    return out


def shuffle_candidates(rng, gold, others, accepted):
    cands = [gold] + list(others)
    order = list(range(4))
    rng.shuffle(order)
    return [cands[i] for i in order], order.index(0), [accepted[i] for i in order]


def build_items(seed=SEED):
    b = Builder(seed)
    rng = b.rng
    items = []
    seen = set()   # twins and user-turn sequences already used: no two items share either

    def slot_values(slot, item_ctx):
        if slot == "name":
            return NAMES, lambda v: [v.lower()], lambda v: v
        if slot == "pet":
            return PET_NAMES, lambda v: [v.lower()], lambda v: v
        if slot == "city":
            return CITIES, lambda v: [v.lower()], lambda v: v
        if slot == "job":
            return JOBS, lambda v: [v], article
        if slot == "age":
            return list(range(18, 80)), lambda v: number_words(v), str
        return FAVOURITES[item_ctx["c"]], lambda v: [v], lambda v: v

    def fill(tpl, ctx, **kw):
        return tpl.format(**{**ctx, **kw})

    def unique(make):
        """make() until its item's twin and user turns are new (a redraw consumes the seeded stream, so
        the set stays reproducible)."""
        for _ in range(200):
            it = make()
            keys = (("twin", it["twin"]), ("turns", tuple(t["user"] for t in it["turns"])))
            if not any(k in seen for k in keys):
                seen.update(keys)
                items.append(it)
                return
        raise ValueError(f"no unique {it['family']} item after 200 draws")

    # ---- recall: slot x plant turn x target turn x interference, each cell RECALL_REPS times
    def recall(n, slot, plant, target, inter):
        spec = SLOTS[slot]
        ctx = {"a": rng.choice(ANIMALS), "c": list(FAVOURITES)[n % 3]}
        values, acc, show = slot_values(slot, ctx)
        gold = b.cycle((slot, ctx.get("c") if slot == "favourite" else ""), values)
        dis = b.others(values, {gold}, inter)
        fill_k = b.others(values, {gold, *dis}, 3 - inter)
        rels = rng.sample(NEUTRAL if slot == "name" else RELATIVES, inter)
        plant_txt = fill(rng.choice(plant_templates(spec, plant)), ctx, v=show(gold))
        ack = rng.choice(ACK_NAME if slot == "name" else ACK_FACT)
        events = {plant: {"text": plant_txt, "gold": ack, "main": True, "kind": "plant"}}
        dis_ev = [{"text": fill(rng.choice(spec["distractor"]), ctx, rel=r, v=show(d)), "gold": rng.choice(ACK_OTHER),
                   "kind": "distractor"} for d, r in zip(dis, rels)]
        q = fill(rng.choice(spec["question"]), ctx)
        frame = spec["frame"].replace("{a}", ctx["a"]).replace("{c}", ctx["c"])
        events[target] = {"text": q, "gold": frame.format(show(gold)), "main": True, "kind": "question"}
        turns = b.layout(5, events, dis_ev, target)
        conv = compose(turns, target, rng.choice(CLOSINGS), rng, set())
        statements = [e["text"] for t in sorted(turns) if t < target for e in turns[t]]
        cands, gi, cacc = shuffle_candidates(rng, show(gold), [show(v) for v in dis + fill_k],
                                             [acc(v) for v in [gold] + dis + fill_k])
        dturns = sorted(t for t in turns if any(e["kind"] == "distractor" for e in turns[t]))
        return {"family": "recall", "slot": slot, "turns": conv, "target_turn": target, "fact_turn": plant,
                "distance": target - plant, "interference": inter, "distractor_turns": dturns,
                "distractor_values": [show(v) for v in dis],
                "answers": acc(gold), "frame": frame, "candidates": cands, "gold_index": gi,
                "candidate_answers": cacc, "check": None,
                "twin": " ".join(statements + [q]), "noctx": q}

    cells = list(itertools.product(SLOTS, (1, 2, 3), (4, 5), (0, 1, 2)))
    for rep in range(RECALL_REPS):
        for n, (slot, plant, target, inter) in enumerate(cells):
            unique(lambda: recall(n + rep, slot, plant, target, inter))

    # ---- correction: slot x (plant, correction) x target x interference, each cell CORRECTION_REPS times
    def correction(n, slot, plant, corr, target, inter):
        spec = SLOTS[slot]
        ctx = {"a": rng.choice(ANIMALS), "c": list(FAVOURITES)[n % 3]}
        values, acc, show = slot_values(slot, ctx)
        new = b.cycle(("fix", slot, ctx.get("c") if slot == "favourite" else ""), values)
        old = b.others(values, {new}, 1)[0]
        dis = b.others(values, {new, old}, inter)
        fill_k = b.others(values, {new, old, *dis}, 2 - inter)
        rels = rng.sample(NEUTRAL if slot == "name" else RELATIVES, inter)
        events = {plant: {"text": fill(rng.choice(plant_templates(spec, plant)), ctx, v=show(old)),
                          "gold": rng.choice(ACK_NAME if slot == "name" else ACK_FACT), "main": True, "kind": "plant"},
                  corr: {"text": fill(rng.choice(spec["correction"]), ctx, new=show(new), old=show(old)),
                         "gold": rng.choice(ACK_FIX), "main": True, "kind": "correction"}}
        dis_ev = [{"text": fill(rng.choice(spec["distractor"]), ctx, rel=r, v=show(d)), "gold": rng.choice(ACK_OTHER),
                   "kind": "distractor"} for d, r in zip(dis, rels)]
        q = fill(rng.choice(spec["question"]), ctx)
        frame = spec["frame"].replace("{a}", ctx["a"]).replace("{c}", ctx["c"])
        events[target] = {"text": q, "gold": frame.format(show(new)), "main": True, "kind": "question"}
        turns = b.layout(5, events, dis_ev, target)
        conv = compose(turns, target, rng.choice(CLOSINGS), rng, set())
        statements = [e["text"] for t in sorted(turns) if t < target for e in turns[t]]
        cands, gi, cacc = shuffle_candidates(rng, show(new), [show(v) for v in [old] + dis + fill_k],
                                             [acc(v) for v in [new, old] + dis + fill_k])
        dturns = sorted(t for t in turns if any(e["kind"] == "distractor" for e in turns[t]))
        return {"family": "correction", "slot": slot, "turns": conv, "target_turn": target, "fact_turn": corr,
                "plant_turn": plant, "distance": target - corr, "interference": inter,
                "distractor_turns": dturns, "distractor_values": [show(v) for v in dis], "old_value": show(old),
                "answers": acc(new), "frame": frame, "candidates": cands, "gold_index": gi,
                "candidate_answers": cacc, "check": None, "twin": " ".join(statements + [q]), "noctx": q}

    cells = list(itertools.product(SLOTS, ((1, 2), (1, 3), (2, 3)), (4, 5), (0, 1, 2)))
    for rep in range(CORRECTION_REPS):
        for n, (slot, (plant, corr), target, inter) in enumerate(cells):
            unique(lambda: correction(n + rep, slot, plant, corr, target, inter))

    # ---- instruction: kind x instruction turn x target, each cell INSTRUCTION_REPS times
    def instruction(kind, at, target):
        spec = INSTRUCTIONS[kind]
        qa = list(spec["qa"])
        rng.shuffle(qa)
        tq = b.cycle(("inst", kind), spec["qa"])
        rest = [x for x in qa if x[0] != tq[0]]
        say = rng.choice(spec["say"])
        events = {at: {"text": f"{say} {rest[0][0]}", "gold": rest[0][1], "main": True, "kind": "instruction"}}
        for k, t in enumerate(range(at + 1, target), start=1):
            events[t] = {"text": rest[k][0], "gold": rest[k][1], "main": True, "kind": "compliant"}
        q, short, s, ans = tq
        events[target] = {"text": q, "gold": short, "main": True, "kind": "question"}
        turns = b.layout(5, events, [], target)
        conv = compose(turns, target, rng.choice(CLOSINGS), rng, set())
        bad = [t.format(S=s[0].upper() + s[1:], s=s) for t in (LONG if kind == "short" else VERBOSE)]
        cands, gi, _ = shuffle_candidates(rng, short, bad, [ans] * 4)
        return {"family": "instruction", "slot": kind, "turns": conv, "target_turn": target, "fact_turn": at,
                "distance": target - at, "interference": 0, "distractor_turns": [], "distractor_values": [],
                "answers": ans, "frame": "{}", "candidates": cands, "gold_index": gi, "candidate_answers": None,
                "check": {"kind": kind}, "twin": f"{say} {q}", "noctx": q}

    cells = list(itertools.product(INSTRUCTIONS, (1, 2, 3), (4, 5)))
    for _rep in range(INSTRUCTION_REPS):
        for kind, at, target in cells:
            unique(lambda: instruction(kind, at, target))

    # ---- topic return: kind x (plant, target) x interference, each cell TOPIC_REPS times
    def topic(kind, plant, target, inter):
        spec = TOPICS[kind]
        values = spec["values"]
        gold = b.cycle(("topic", kind), values)
        dis = b.others(values, {gold}, inter)
        fill_k = b.others(values, {gold, *dis}, 3 - inter)
        rels = rng.sample(RELATIVES, inter)
        plant_txt = rng.choice(spec["plant"]).format(v=gold)
        events = {plant: {"text": plant_txt, "gold": rng.choice(ACK_TOPIC), "main": True, "kind": "plant"}}
        dig = [t for t in range(plant + 1, target)]
        fillers = rng.sample(FILLERS, len(dig))
        dis_slots = [dig[i % len(dig)] for i in range(inter)]
        dist_sent = [rng.choice(spec["distractor"]).format(rel=r, v=d) for d, r in zip(dis, rels)]
        for t, (fq, fa) in zip(dig, fillers):
            ds = [x for x, s2 in zip(dist_sent, dis_slots) if s2 == t]
            if t == dig[0]:
                fq = "By the way, " + fq[0].lower() + fq[1:]
            events[t] = {"text": " ".join(([ds[0]] if ds and t != dig[0] else []) +
                                          ([fq] if not ds or t == dig[0] else []) +
                                          (ds if t == dig[0] else ds[1:])),
                         "gold": fa if (not ds or t == dig[0]) else rng.choice(ACK_OTHER), "main": True,
                         "kind": "digression"}
        q = spec["question"]
        frame = spec["frame"]
        events[target] = {"text": q, "gold": frame.format(gold), "main": True, "kind": "question"}
        turns = b.layout(5, events, [], target)
        conv = compose(turns, target, rng.choice(CLOSINGS), rng, set(f[0] for f in fillers))
        twin_q = rng.choice(spec["twin_q"])
        cands, gi, cacc = shuffle_candidates(rng, gold, dis + fill_k, [[v.lower()] for v in [gold] + dis + fill_k])
        return {"family": "topic", "slot": kind, "turns": conv, "target_turn": target, "fact_turn": plant,
                "distance": target - plant, "interference": inter, "distractor_turns": sorted(set(dis_slots)),
                "distractor_values": list(dis), "answers": [gold.lower()], "frame": frame, "candidates": cands,
                "gold_index": gi, "candidate_answers": cacc, "check": None,
                "twin": " ".join([plant_txt] + dist_sent + [twin_q]), "noctx": twin_q}

    cells = list(itertools.product(TOPICS, ((1, 4), (2, 4), (1, 5), (2, 5), (3, 5)), (0, 1, 2)))
    for _rep in range(TOPIC_REPS):
        for kind, (plant, target), inter in cells:
            unique(lambda: topic(kind, plant, target, inter))

    # ---- knowledge: each question once, target 4 or 5 alternating
    def knowledge(n, q, frame, gold, dis, acc):
        target = 4 + n % 2
        ask = KNOWLEDGE_ASK[n % len(KNOWLEDGE_ASK)].format(q=q)
        events = {target: {"text": ask, "gold": frame.format(gold), "main": True, "kind": "question"}}
        turns = b.layout(5, events, [], target)
        conv = compose(turns, target, rng.choice(CLOSINGS), rng, set())
        cands, gi, cacc = shuffle_candidates(rng, gold, dis, [acc] + [[d.lower()] for d in dis])
        return {"family": "knowledge", "slot": "knowledge", "turns": conv, "target_turn": target,
                "fact_turn": None, "distance": None, "interference": 0, "distractor_turns": [],
                "distractor_values": [], "answers": acc, "frame": frame, "candidates": cands, "gold_index": gi,
                "candidate_answers": cacc, "check": None, "twin": ask, "noctx": q}

    for n, k in enumerate(KNOWLEDGE):
        unique(lambda: knowledge(n, *k))

    counts = {}
    out = []
    for it in items:
        fam = it["family"]
        counts[fam] = counts.get(fam, 0) + 1
        out.append({"id": f"{VERSION}:{fam}:{counts[fam]:04d}", "version": VERSION, **it})
    validate(out)
    return out


def validate(items):
    """Build-time checks: five turns, a target of 4 or 5, four distinct candidates, the gold candidate's
    accepted answers present in the gold reply, no filler/opener/acknowledgement text naming a value
    from any list (it would act as an unlabelled distractor), and the needed fact before the target."""
    values = set()
    for vs in [NAMES, PET_NAMES, CITIES, JOBS, *FAVOURITES.values(), *(t["values"] for t in TOPICS.values())]:
        values |= {w for v in vs for w in words(v)}
    canned = [t for pair in OPENERS + FILLERS + CLOSINGS for t in pair] + ACK_FACT + ACK_NAME + ACK_OTHER + \
        ACK_FIX + ACK_TOPIC
    for text in canned:
        clash = values & set(words(text))
        if clash:
            raise ValueError(f"canned text {text!r} names list values {sorted(clash)}")
    simple_qa = os.path.join(ROOT, "corpus", "eval", "simple_qa_v1.jsonl")
    if os.path.exists(simple_qa):
        sq = {" ".join(words(json.loads(x)["prompt"])) for x in open(simple_qa, encoding="utf-8") if x.strip()}
        for it in items:
            if " ".join(words(it["noctx"])) in sq:
                raise ValueError(f"{it['id']}: question repeats simple_qa_v1")
    for it in items:
        assert len(it["turns"]) == 5 and it["target_turn"] in (4, 5), it["id"]
        assert len(set(it["candidates"])) == 4, it["id"]
        gold = it["turns"][it["target_turn"] - 1]["gold"]
        assert gold == it["frame"].format(it["candidates"][it["gold_index"]]), it["id"]
        if it["fact_turn"]:
            assert it["fact_turn"] < it["target_turn"], it["id"]
        if it["family"] != "instruction":
            g = gold.lower()
            assert any(a in g for a in it["answers"]), it["id"]
        assert len(it["distractor_values"]) == it["interference"], it["id"]
        assert all(t < it["target_turn"] for t in it["distractor_turns"]), it["id"]
        before = " ".join(t["user"] for t in it["turns"][:it["target_turn"] - 1])
        for v in it["distractor_values"] + ([it["old_value"]] if "old_value" in it else []):
            assert re.search(r"\b" + re.escape(v) + r"\b", before), (it["id"], v)
            assert v in it["candidates"], (it["id"], v)
            assert re.search(r"\b" + re.escape(v) + r"\b", it["twin"]), (it["id"], v)
        assert it["candidates"][it["gold_index"]] not in it["distractor_values"], it["id"]


def dump(items):
    return "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode("utf-8")


def pinned():
    if not os.path.exists(HASH):
        return None
    for line in open(HASH, encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            return line.split()[0]
    return None


def cmd_build(a):
    data = dump(build_items())
    sha = hashlib.sha256(data).hexdigest()
    pin = pinned()
    if os.path.exists(a.out):
        old = open(a.out, "rb").read()
        if old != data:
            sys.exit(f"{a.out} exists and differs from this build (frozen set; a change is a new version)")
    elif pin and pin != sha:
        sys.exit(f"this build's sha256 {sha} differs from the pinned {pin}")
    else:
        with open(a.out, "wb") as f:
            f.write(data)
    if not pin and os.path.abspath(a.out) == os.path.abspath(OUT):
        with open(HASH, "w", encoding="utf-8", newline="\n") as f:
            f.write("# FROZEN Carry-5 v1: 5-turn chat carry-over benchmark for tiny models "
                    "(tools/carry5_build.py, seed 20261023); an evaluation set, never training data\n")
            f.write(f"{sha}  carry5_v1.jsonl\n")
    print(json.dumps(summarise(build_items()) | {"sha256": sha, "out": a.out}, indent=1))
    return 0


def summarise(items):
    by = {}
    for it in items:
        f = by.setdefault(it["family"], {"items": 0, "slots": {}, "target_turn": {}, "distance": {},
                                         "interference": {}, "gold_index": {}})
        f["items"] += 1
        for k, v in (("slots", it["slot"]), ("target_turn", it["target_turn"]), ("distance", it["distance"]),
                     ("interference", it["interference"]), ("gold_index", it["gold_index"])):
            f[k][str(v)] = f[k].get(str(v), 0) + 1
    return {"items": len(items), "families": by}


def cmd_check(a):
    data = dump(build_items())
    sha = hashlib.sha256(data).hexdigest()
    disk = open(OUT, "rb").read()
    ok = data == disk and sha == pinned()
    print(json.dumps({"rebuild_equals_file": data == disk, "sha256": sha, "pinned": pinned(), "ok": ok}))
    return 0 if ok else 1


# ---------------------------------------------------------------- contamination
def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class Patterns:
    """Every carry5 text the scan looks for, indexed by its first three normalised words."""

    def __init__(self, items, decontam):
        self.w = decontam.words
        self.N, self.SPAN_MIN, self.PROMPT_MIN = decontam.N, decontam.SPAN_MIN, decontam.PROMPT_MIN
        self.index = {}
        self.prompts = {}
        self.item_need = {}       # item id -> (fact pattern, question pattern)
        self.desc = {}
        texts = []                # (pattern id, text, is_user_turn)
        for it in items:
            for k, t in enumerate(it["turns"], 1):
                texts.append((f"{it['id']}:t{k}:user", t["user"], True))
                texts.append((f"{it['id']}:t{k}:gold", t["gold"], False))
            texts.append((f"{it['id']}:twin", it["twin"], True))
        seen_sent = set()
        for pid, text, is_user in texts:
            w = self.w(text)
            if is_user and len(w) >= self.PROMPT_MIN:
                self.prompts.setdefault(" ".join(w), pid)
            for i in range(len(w) - self.N + 1):
                self._add(tuple(w[i:i + self.N]), ("ngram", pid))
            if is_user and self.SPAN_MIN <= len(w) < self.N:
                self._add(tuple(w), ("span", pid))
            for s in sentences(text):
                sw = tuple(self.w(s))
                if len(sw) >= 3 and sw not in seen_sent:
                    seen_sent.add(sw)
                    self._add(sw, ("sentence", " ".join(sw)))
        for it in items:
            if it["family"] in ("recall", "correction", "topic"):
                ft = it["turns"][it["fact_turn"] - 1]["user"]
                gold = it["candidates"][it["gold_index"]]
                fact = next((s for s in sentences(ft) if gold.lower() in s.lower()), None)
                q = it["turns"][it["target_turn"] - 1]["user"]
                if fact:
                    fw, qw = tuple(self.w(fact)), tuple(self.w(sentences(q)[-1]))
                    self.item_need[it["id"]] = (fw, qw)
                    self._add(fw, ("fact", it["id"]))
                    self._add(qw, ("question", it["id"]))
        self.keys = set(self.index)

    def _add(self, words_t, tag):
        if len(words_t) < 3:
            return
        self.index.setdefault(words_t[:3], []).append((words_t, tag))

    def check(self, user_turns, text):
        hits = []
        for u in user_turns:
            pid = self.prompts.get(" ".join(self.w(u)))
            if pid:
                hits.append(("prompt", pid, u[:120]))
        w = self.w(text)
        tri = list(zip(w, w[1:], w[2:]))
        hit = self.keys.intersection(tri)
        found = set()
        facts, questions = set(), set()
        if hit:
            for i, k in enumerate(tri):
                if k not in hit:
                    continue
                for pat, (rule, pid) in self.index[k]:
                    if (rule, pid) in found or tuple(w[i:i + len(pat)]) != pat:
                        continue
                    found.add((rule, pid))
                    if rule == "fact":
                        facts.add(pid)
                    elif rule == "question":
                        questions.add(pid)
                    else:
                        hits.append((rule, pid, " ".join(w[max(0, i - 8):i + len(pat) + 8])))
        for iid in facts & questions:
            hits.append(("item_leak", iid, ""))
        for iid in facts:
            hits.append(("fact_sentence", iid, ""))
        return hits


def iter_rows(path, start=0):
    """(byte offset after the row, user turns, all text) per training row. Chat-template markers
    become line breaks, so they never join words across turns."""
    turn_re = re.compile(r"<\|user\|>(.*?)<\|assistant\|>", re.S)
    marks = re.compile(r"<\|(?:user|assistant|endoftext)\|>")
    if path.endswith(".txt"):
        # A row ends at a line holding <|endoftext|> (the whole line is kept: the human_* corpora put a
        # dialogue on one line with <|endoftext|> between its turns), or after 200 lines of prose.
        with open(path, "rb") as f:
            f.seek(start)
            buf = []
            for line in f:
                buf.append(line.decode("utf-8", errors="replace"))
                if "<|endoftext|>" in buf[-1] or len(buf) >= 200:
                    text = "".join(buf)
                    yield f.tell(), turn_re.findall(text), marks.sub("\n", text)
                    buf = []
            if buf:
                text = "".join(buf)
                yield f.tell(), turn_re.findall(text), marks.sub("\n", text)
        return
    with open(path, "rb") as f:
        f.seek(start)
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "messages" in r:
                users = [m["content"] for m in r["messages"] if m.get("role") == "user"]
                text = "\n".join(m["content"] for m in r["messages"])
            elif "prompt" in r:
                users = turn_re.findall(r["prompt"])
                text = marks.sub("\n", r["prompt"] + "\n" + r.get("completion", ""))
            else:
                users, text = [], r.get("text", "")
            yield f.tell(), users, text


def cmd_decontam(a):
    sys.path.insert(0, os.path.join(ROOT, "corpus"))
    import decontam
    items = [json.loads(x) for x in open(a.set, encoding="utf-8") if x.strip()]
    pats = Patterns(items, decontam)
    state_path = a.state or a.out + ".state.json"
    state = json.load(open(state_path, encoding="utf-8")) if os.path.exists(state_path) else {}

    def save():
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, state_path)

    for path in a.corpora:
        st = state.setdefault(path, {"offset": 0, "done": False, "rows": 0, "bytes": os.path.getsize(path),
                                     "hits": {}, "examples": {}})
        if st["done"]:
            continue
        last = time.time()
        for off, users, text in iter_rows(path, st["offset"]):
            st["rows"] += 1
            for rule, pid, ctx in pats.check(users, text):
                key = f"{rule}|{pid}"
                st["hits"][key] = st["hits"].get(key, 0) + 1
                ex = st["examples"].setdefault(key, [])
                if len(ex) < 2 and ctx:
                    ex.append({"row": st["rows"], "context": ctx})
            st["offset"] = off
            if time.time() - last > 120:
                save()
                last = time.time()
                print(f"{os.path.basename(path)}: {off / st['bytes']:.1%} {st['rows']} rows", flush=True)
        st["done"] = True
        save()
        print(f"{path}: done, {st['rows']} rows", flush=True)
    report = summarise_decontam(items, state, a.corpora)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["summary"], indent=1))
    return 0


def summarise_decontam(items, state, corpora):
    n_items = len(items)
    total = {}
    per = {}
    leaks = {}
    for path in corpora:
        st = state[path]
        c = {}
        for key, n in st["hits"].items():
            rule, pid = key.split("|", 1)
            c.setdefault(rule, {"rows": 0, "patterns": 0})
            c[rule]["rows"] += n
            c[rule]["patterns"] += 1
            total.setdefault(rule, set()).add(pid)
            if rule == "item_leak":
                leaks.setdefault(pid, {})[os.path.basename(os.path.dirname(path)) + "/" +
                                       os.path.basename(path)] = n
        per[path] = {"rows": st["rows"], "bytes": st["bytes"], "done": st["done"], "by_rule": c}
    top_sent = {}
    for path in corpora:
        for key, n in state[path]["hits"].items():
            if key.startswith("sentence|"):
                top_sent[key[9:]] = top_sent.get(key[9:], 0) + n
    examples = {}
    for path in corpora:
        for key, ex in state[path]["examples"].items():
            if key.split("|")[0] in ("ngram", "span", "prompt") and key not in examples:
                examples[key] = {"corpus": path, "examples": ex}
    summary = {"items": n_items, "corpora": len(corpora),
               "rows_scanned": sum(per[p]["rows"] for p in corpora),
               "item_leaks": len(leaks),
               "items_with_fact_sentence_in_training": len(total.get("fact_sentence", set())),
               "eval_texts_hit": {r: len(v) for r, v in total.items() if r not in ("fact_sentence", "item_leak")},
               "rules": "decontam.py ngram (13 words), prompt (user turn equal, 4+ words), span (8-12 word user "
                        "turn); sentence: any 3+ word template sentence as a span; fact_sentence: an item's "
                        "value-bearing fact sentence; item_leak: fact sentence and question in one row"}
    return {"summary": summary, "per_corpus": per, "item_leaks": leaks,
            "top_sentences": dict(sorted(top_sent.items(), key=lambda x: -x[1])[:60]),
            "drop_rule_examples": examples}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default=OUT)
    sub.add_parser("check")
    p = sub.add_parser("decontam")
    p.add_argument("corpora", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--state", default="")
    p.add_argument("--set", default=OUT)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "check": cmd_check, "decontam": cmd_decontam}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
