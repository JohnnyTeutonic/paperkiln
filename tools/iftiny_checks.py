"""Code-checkable instruction compliance for IF-tiny and the turn-distance set (tools/iftiny_build.py).

    from iftiny_checks import check
    check({"kind": "one_word"}, "Paris.")            -> True

Every check takes the reply as the engine returned it (role markers already dropped by the server) and
strips surrounding white space first; an empty reply complies with nothing. Words are runs of letters, digits
and apostrophes ([A-Za-z0-9']+), so punctuation never counts as a word. Comparisons of words ignore case
unless the check is about case. The kinds:

  one_word                exactly one word
  max_words n             1 to n words
  lowercase               at least one letter and no upper-case letter
  uppercase               at least one letter and no lower-case letter
  start_with word         the first word is `word`
  end_with_word word      the last word is `word`
  end_full_stop           the last character is a full stop (not an ellipsis)
  numbered_list n         the lines that begin with a number ("1." or "1)") are numbered 1..n in order,
                          each with text after the marker, and there are exactly n of them
  repeat_word word n      the words are `word` n times and nothing else
  repeat_phrase phrase    the reply's words equal the phrase's words
  yes_no                  exactly one word, "yes" or "no"
  number_only             exactly one word, a numeral or a number word
  include_word word       `word` occurs as a word
  avoid_word word         `word` does not occur as a word
  quoted                  the reply opens with a double quotation mark and closes with one (a final full stop,
                          question or exclamation mark after the closing mark is allowed); straight or curly
"""
from __future__ import annotations

import re

WORD = re.compile(r"[A-Za-z0-9']+")
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                   "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                   "hundred thousand".split())
ITEM = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*)$")
OPEN_Q = ('"', "“")
CLOSE_Q = ('"', "”")


def words(text):
    return [w.strip("'") for w in WORD.findall(text) if w.strip("'")]


def is_number(w):
    return w.isdigit() or w.lower() in NUMBER_WORDS


def check(spec, reply):
    """True when `reply` complies with the instruction `spec` ({"kind", ...params})."""
    r = (reply or "").strip()
    if not r:
        return False
    k = spec["kind"]
    w = words(r)
    lw = [x.lower() for x in w]
    if k == "one_word":
        return len(w) == 1
    if k == "max_words":
        return 1 <= len(w) <= spec["n"]
    if k == "lowercase":
        return any(c.isalpha() for c in r) and not any(c.isupper() for c in r)
    if k == "uppercase":
        return any(c.isalpha() for c in r) and not any(c.islower() for c in r)
    if k == "start_with":
        return bool(lw) and lw[0] == spec["word"].lower()
    if k == "end_with_word":
        return bool(lw) and lw[-1] == spec["word"].lower()
    if k == "end_full_stop":
        return r.endswith(".") and not r.endswith("..")
    if k == "numbered_list":
        nums = []
        for line in r.splitlines():
            m = ITEM.match(line)
            if m:
                if not m.group(2).strip():
                    return False
                nums.append(int(m.group(1)))
        return nums == list(range(1, spec["n"] + 1))
    if k == "repeat_word":
        return lw == [spec["word"].lower()] * spec["n"]
    if k == "repeat_phrase":
        return lw == [x.lower() for x in words(spec["phrase"])]
    if k == "yes_no":
        return len(lw) == 1 and lw[0] in ("yes", "no")
    if k == "number_only":
        return len(w) == 1 and is_number(w[0])
    if k == "include_word":
        return spec["word"].lower() in lw
    if k == "avoid_word":
        return spec["word"].lower() not in lw
    if k == "quoted":
        body = r[:-1] if len(r) > 2 and r[-1] in ".?!" and r[-2] in CLOSE_Q else r
        return len(body) >= 3 and body[0] in OPEN_Q and body[-1] in CLOSE_Q and bool(words(body))
    raise ValueError(f"unknown check kind {k!r}")


KINDS = ("one_word", "max_words", "lowercase", "uppercase", "start_with", "end_with_word", "end_full_stop",
         "numbered_list", "repeat_word", "repeat_phrase", "yes_no", "number_only", "include_word", "avoid_word",
         "quoted")
