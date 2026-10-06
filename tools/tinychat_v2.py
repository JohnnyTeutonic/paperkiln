"""TinyChat v2: the same small world as TinyChat, said in many more ways.

TinyChat v1 (tools/get_tinychat_data.py, verbatim from transformer_cpp) gives
each question one or two fixed phrasings, so a model trained on it answers
"what is your hobby ?" well and "what do you enjoy doing ?" badly, treats
"hello , how are you ?" as two unrelated things, and answers a question it
has never seen with whatever topic is nearest. v2 keeps the world (food,
drinks, outings, weather, hobbies, pets, invitations, how are you) and its
answer semantics, and changes how it is talked about:

- every question has many phrasings, contractions included ("what's",
  "how's"), so binding has to go through meaning rather than one string;
- a user turn can open with a greeting and carry a question in the same
  message;
- the assistant sometimes asks back ("what about you ?") and acknowledges
  the user's answer, and answers "ask me something" with a question;
- it can say its name and what it is;
- "favourite" and "favorite" both appear;
- questions outside its world (facts, sums, poems, news) get one polite
  fallback that says what it can talk about, instead of a random topic.

The vocabulary stays a few hundred words. Seeded and deterministic.
"""
from __future__ import annotations

import random

FOODS = ["pizza", "pasta", "rice", "soup", "salad", "eggs", "bread",
         "cheese", "apples", "fish", "chicken", "pancakes", "noodles"]
DRINKS = ["tea", "coffee", "juice", "water", "milk"]
PLACES = ["the park", "the beach", "the market", "the library",
          "the museum", "the garden", "the city", "the lake"]
ACTIVITIES = ["reading", "cooking", "running", "painting", "swimming",
              "gardening", "playing chess", "watching movies", "hiking"]
WEATHER = ["sunny", "rainy", "cold", "warm", "windy", "cloudy"]
GOOD = ["great", "very good", "happy", "fine", "wonderful"]
BAD = ["tired", "a little sad", "busy", "sleepy", "not so good"]
DAYS = ["today", "yesterday", "this morning", "last night", "this week", "on the weekend"]
PETS = ["dog", "cat", "bird", "rabbit", "fish"]

GREETINGS = ["hi", "hello", "hey", "hey there", "hi there", "good morning", "good evening", "hello there"]
GREET_REPLIES = ["hi ! nice to see you .", "hello ! it is nice to talk to you .",
                 "hey ! good to see you .", "hello ! i am glad you are here ."]
FAREWELLS = ["goodbye !", "bye !", "see you later !", "i have to go now . goodbye !",
             "talk to you soon !", "good night !", "i need to go , bye !"]
FAREWELL_REPLIES = ["goodbye ! have a nice day .", "see you soon ! take care .",
                    "bye ! it was nice talking to you .", "good night ! sleep well ."]
ACKS = ["that sounds nice !", "oh , that is lovely .", "nice !", "that is great to hear .", "how lovely !"]
ACKS_SAD = ["oh no , i hope you feel better soon .", "i am sorry to hear that .", "take it easy today ."]
ASK_BACK = ["what about you ?", "and you ?", "how about you ?"]
FALLBACK = ["sorry , i only know small talk . ask me about food , pets or the weather !",
            "i am not sure . i can talk about food , drinks , hobbies , pets and the weather .",
            "i do not know that one . i am a tiny model that only knows small talk ."]
OUT_OF_SCOPE = [
    "what is the capital of {c} ?", "what is {a} plus {b} ?", "what is {a} times {b} ?",
    "can you write a poem ?", "who is the president ?", "what is the news today ?",
    "can you help me with my homework ?", "what is the meaning of life ?",
    "how far away is the moon ?", "can you write some code ?", "what year is it ?",
    "tell me a fact about history .", "how do computers work ?", "translate this into french .",
]
COUNTRIES = ["france", "japan", "italy", "spain", "china", "brazil", "canada", "india"]


def _pick(rng, xs):
    return rng.choice(xs)


def _how_are_you(rng):
    q = _pick(rng, ["how are you ?", "how are you doing ?", "how's it going ?", "how do you feel ?",
                    "how are you feeling " + _pick(rng, DAYS) + " ?", "are you ok ?",
                    "how have you been ?", "how is your day going ?"])
    f = _pick(rng, GOOD) if rng.random() < 0.7 else _pick(rng, BAD)
    a = _pick(rng, ["i am " + f + " , thank you .", "i feel " + f + " " + _pick(rng, DAYS) + " .",
                    "i'm " + f + " , thanks for asking ."])
    good = f in GOOD
    return q, a, ("i am " + _pick(rng, GOOD) + " too ." if good else "i am " + _pick(rng, GOOD) + " .",
                  _pick(rng, ACKS))


def _food(rng):
    food = _pick(rng, FOODS)
    q = _pick(rng, ["what do you like to eat ?", "what is your favorite food ?", "what's your favorite food ?",
                    "what food do you like ?", "what do you love to eat ?", "what did you have for dinner ?",
                    "are you hungry ? what would you eat ?", "what is the best food ?"])
    a = _pick(rng, ["i like " + food + " very much .", "my favorite food is " + food + " .",
                    "i love " + food + " .", "i had some " + food + " , it was very good ."])
    mine = _pick(rng, FOODS)
    return q, a, ("i like " + mine + " .", _pick(rng, ["yum , " + mine + " is good too !", _pick(rng, ACKS)]))


def _drink(rng):
    d = _pick(rng, DRINKS)
    q = _pick(rng, ["would you like some " + d + " ?", "do you want " + d + " or " + _pick(rng, DRINKS) + " ?",
                    "can i get you a drink ?", "what do you like to drink ?", "do you drink " + d + " ?",
                    "shall i make some " + d + " ?"])
    a = _pick(rng, ["yes please , i would love some " + d + " .",
                    "no thank you , i just had some " + _pick(rng, DRINKS) + " .",
                    "i like " + d + " best .", "some " + d + " would be lovely , thank you ."])
    return q, a, None


def _outing(rng):
    p, act, day = _pick(rng, PLACES), _pick(rng, ACTIVITIES), _pick(rng, DAYS)
    q = _pick(rng, ["where did you go " + day + " ?", "what did you do " + day + " ?",
                    "did you go anywhere " + day + " ?", "what have you been up to ?",
                    "how was your " + _pick(rng, ["day", "weekend", "week"]) + " ?",
                    "anything fun " + day + " ?"])
    a = _pick(rng, ["i went to " + p + " .", "i spent the day " + act + " .",
                    "i went to " + p + " and enjoyed " + act + " .", "it was nice , i went to " + p + " ."])
    return q, a, ("i went to " + _pick(rng, PLACES) + " .", _pick(rng, ACKS))


def _weather(rng):
    w, day = _pick(rng, WEATHER), _pick(rng, DAYS)
    q = _pick(rng, ["how is the weather " + day + " ?", "what is the weather like ?", "what's the weather like ?",
                    "is it " + _pick(rng, WEATHER) + " outside ?", "how was the weather " + day + " ?",
                    "is it nice outside ?"])
    a = _pick(rng, ["it is very " + w + " .", "it was " + w + " all day .", "it is " + w + " today .",
                    "a bit " + w + " , but nice ."])
    return q, a, None


def _hobby(rng):
    act = _pick(rng, ACTIVITIES)
    q = _pick(rng, ["what is your hobby ?", "what do you like to do ?", "what do you enjoy doing ?",
                    "what do you do for fun ?", "what's your hobby ?", "do you have any hobbies ?",
                    "what do you enjoy ?", "what do you like doing on the weekend ?"])
    a = _pick(rng, ["i really enjoy " + act + " .", "my hobby is " + act + " .", "i love " + act + " .",
                    "for fun i like " + act + " ."])
    mine = _pick(rng, ACTIVITIES)
    return q, a, ("i like " + mine + " .", _pick(rng, ["oh , " + mine + " is fun !", _pick(rng, ACKS)]))


def _pet(rng):
    pet = _pick(rng, PETS)
    q = _pick(rng, ["do you have a pet ?", "do you like animals ?", "do you have any pets ?",
                    "what is your favorite animal ?", "do you like dogs or cats ?", "tell me about your pet ."])
    a = _pick(rng, ["yes , i have a little " + pet + " .", "yes , my " + pet + " is very sweet .",
                    "i love animals , i have a " + pet + " .", "my favorite animal is the " + pet + " ."])
    mine = _pick(rng, PETS)
    return q, a, ("i have a " + mine + " .", _pick(rng, ["a " + mine + " ! how sweet .", _pick(rng, ACKS)]))


def _invite(rng):
    p = _pick(rng, PLACES)
    q = _pick(rng, ["do you want to go to " + p + " with me ?", "shall we visit " + p + " ?",
                    "would you like to go to " + p + " ?", "let's go to " + p + " !",
                    "do you want to come " + _pick(rng, ACTIVITIES) + " with me ?"])
    a = _pick(rng, ["yes , that sounds lovely .", "sure , i would like that .",
                    "sorry , i am too busy " + _pick(rng, DAYS) + " .", "yes please , when shall we go ?"])
    return q, a, None


def _name(rng):
    q = _pick(rng, ["what is your name ?", "what's your name ?", "who are you ?", "what are you ?",
                    "are you a robot ?", "tell me about yourself ."])
    a = _pick(rng, ["my name is kiln . i am a tiny chat model .",
                    "i am kiln , a very small language model .",
                    "i am kiln . i like small talk about food , pets and the weather ."])
    return q, a, None


TOPICS = [_how_are_you, _food, _drink, _outing, _weather, _hobby, _pet, _invite]


def _ask_me(rng):
    """'ask me a question' -> the assistant asks one of the world's questions."""
    q = _pick(rng, ["ask me a question !", "ask me something .", "you ask me something .",
                    "what do you want to know ?", "your turn to ask ."])
    topic = _pick(rng, TOPICS)
    their_q, their_a, _ = topic(rng)
    return [("user", q), ("assistant", their_q), ("user", their_a), ("assistant", _pick(rng, ACKS))]


def _out_of_scope(rng):
    q = _pick(rng, OUT_OF_SCOPE).format(c=_pick(rng, COUNTRIES), a=rng.randrange(2, 10), b=rng.randrange(2, 10))
    return [("user", q), ("assistant", _pick(rng, FALLBACK))]


def _exchange(rng, greet_prefix=None):
    r = rng.random()
    if r < 0.07:
        return _ask_me(rng)
    if r < 0.14:
        return _out_of_scope(rng)
    if r < 0.20:
        q, a, _ = _name(rng)
        return [("user", (greet_prefix + " , " if greet_prefix else "") + q), ("assistant", a)]
    q, a, back = _pick(rng, TOPICS)(rng)
    if greet_prefix:
        q = greet_prefix + " , " + q
    turns = [("user", q)]
    if back and rng.random() < 0.35:
        my_answer, ack = back
        turns += [("assistant", a + " " + _pick(rng, ASK_BACK)), ("user", my_answer), ("assistant", ack)]
    else:
        turns.append(("assistant", a))
    return turns


def tinychat_v2_rows(n_dialogues=150_000, seed=11):
    rng = random.Random(seed)
    for _ in range(n_dialogues):
        turns = []
        g = _pick(rng, GREETINGS)
        if rng.random() < 0.35:
            # greeting and first question in one message
            turns += _exchange(rng, greet_prefix=g)
        else:
            if rng.random() < 0.8:
                turns += [("user", g + " !"), ("assistant", _pick(rng, GREET_REPLIES))]
            turns += _exchange(rng)
        for _ in range(rng.randrange(0, 3)):
            turns += _exchange(rng)
        if rng.random() < 0.5:
            turns += [("user", _pick(rng, FAREWELLS)), ("assistant", _pick(rng, FAREWELL_REPLIES))]
        line = " ".join(f"{who}: {text}" for who, text in turns)
        # both spellings, so "favourite" and "favorite" are the same word to the model
        yield line.replace("favorite", "favourite") if rng.random() < 0.5 else line
