"""Per-turn state tokens on ids: the Python twin of nn::insert_state_ids, nn::state_ids_error and
ops::turn_state_codes (include/microtorch/llama.hpp, ops.hpp; experiments/tiny_chat_long_2026-10/
IR_SPEC_DRAFT.md sections 1.2, 1.3 and 1.5). Evaluation tools (Carry-12's state-token template) and
the tests build prompts with it; mtstudio applies the same rules itself.

    insert(ids, vocab, k, eot, user, assistant) -> (ids with state blocks, blocks, closed_early)
    error(ids, vocab, k, eot, user, assistant)  -> "" or the first violation (raw-ids validation)
    codes(ids, vocab, k)                        -> [2 * segment + is_state] per position

The state ids are vocab .. vocab + k - 1 (S_0 .. S_{k-1}). An exchange opens at <|user|>, has a
reply once <|assistant|> follows, and closes at the next <|endoftext|>, which gets S_0 .. S_{k-1}
after it; a <|user|> that arrives while a reply is open first gets <|endoftext|> and the block (a
reply that stopped early). Every <|endoftext|> ends the exchange state.
"""
from __future__ import annotations

NONE, USER, REPLY = 0, 1, 2


def insert(ids, vocab, k, eot, user, assistant):
    out, phase, blocks, closed = [], NONE, 0, 0
    block = list(range(vocab, vocab + k))
    for t in ids:
        if not 0 <= t < vocab:
            raise ValueError(f"state_tokens.insert: id {t} is not a text id")
        if t == user:
            if phase == REPLY:
                out += [eot] + block
                blocks += 1
                closed += 1
            phase = USER
            out.append(t)
        elif t == assistant:
            if phase == USER:
                phase = REPLY
            out.append(t)
        elif t == eot:
            out.append(t)
            if phase == REPLY:
                out += block
                blocks += 1
            phase = NONE
        else:
            out.append(t)
    return out, blocks, closed


def error(ids, vocab, k, eot, user, assistant):
    phase, closed, p = NONE, False, 0
    while p < len(ids):
        t = ids[p]
        if not 0 <= t < vocab + k:
            return f"id {t} at position {p} is out of range"
        if t >= vocab:
            if not closed or t != vocab:
                return f"state id {t} at position {p} does not start a block after an <|endoftext|> that ends a reply"
            if ids[p:p + k] != list(range(vocab, vocab + k)):
                return f"the state block at position {p} is not the ids {vocab} .. {vocab + k - 1} in order"
            p += k
            closed = False
            continue
        closed = False
        if t == user:
            phase = USER
        elif t == assistant:
            if phase == USER:
                phase = REPLY
        elif t == eot:
            closed = phase == REPLY
            phase = NONE
        p += 1
    return ""


def codes(ids, vocab, k):
    """ops::turn_state_codes for one sequence (ir.hpp attention mask turn_state)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec", "tools"))
    import ir_numpy
    return ir_numpy.turn_state_codes(ids, vocab, k)
