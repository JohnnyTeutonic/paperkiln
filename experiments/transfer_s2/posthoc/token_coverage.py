"""POST-HOC, descriptive (9 Oct 2026): how often the studies' tokenizer maps text to ID 0.

Reimplements, byte for byte, the vocabulary reader and word tokenizer of
tools/mtstudio.cpp at the Study 2 licence commit (90791ed): the first 4096
entries of the GGUF's tokenizer.ggml.tokens array; ASCII letter, digit and
apostrophe runs lowercased and looked up whole; every other non-space byte
looked up as a one-byte string; anything missing maps to ID 0; stop at
400000 ids (the final flush can add one); the last floor(N/20) ids are the
validation split. Changes no frozen script or receipt.

usage: python token_coverage.py CORPUS.txt VOCAB.gguf
"""
import collections
import hashlib
import struct
import sys

EXPECTED = {
    'corpus': '672622b80c9052da17adaefe93a9f34bb51bdfaffdd70f4fc485fd593c16036b',
    'vocab': '9c4b2719a7b9443a27f198ebe5845a9c00f4dd7b7ab69343d30f72421ee8453c',
}


def read_gguf_tokens(buf):
    p = 0

    def rd(fmt):
        nonlocal p
        v = struct.unpack_from('<' + fmt, buf, p)[0]
        p += struct.calcsize(fmt)
        return v

    def rd_str():
        nonlocal p
        n = rd('Q')
        s = buf[p:p + n]
        p += n
        return s

    if rd('I') != 0x46554747:
        raise ValueError('not GGUF')
    rd('I')
    rd('Q')
    tokens = []
    for _ in range(rd('Q')):
        key = rd_str()
        vt = rd('I')
        if vt == 4:
            rd('I')
        elif vt == 5:
            rd('i')
        elif vt == 6:
            rd('f')
        elif vt == 8:
            rd_str()
        elif vt == 9:
            et, n = rd('I'), rd('Q')
            for _ in range(n):
                if et == 8:
                    t = rd_str()
                    if key == b'tokenizer.ggml.tokens':
                        tokens.append(t)
                elif et == 6:
                    rd('f')
                else:
                    raise ValueError('bad array')
        else:
            raise ValueError('bad meta')
    return tokens


def is_alpha(c):
    return 65 <= c <= 90 or 97 <= c <= 122


def is_digit(c):
    return 48 <= c <= 57


def is_space(c):
    return c in (32, 9, 10, 11, 12, 13)


def tokenize(text, vocab, max_tokens):
    ids, words, cur = [], [], bytearray()

    def flush():
        if cur:
            w = bytes(cur)
            ids.append(vocab.get(w, 0))
            words.append(w)
            cur.clear()

    for c in text:
        if len(ids) >= max_tokens:
            break
        if is_alpha(c) or c == 39 or is_digit(c):
            cur.append(c + 32 if 65 <= c <= 90 else c)
        else:
            flush()
            if not is_space(c):
                w = bytes([c])
                ids.append(vocab.get(w, 0))
                words.append(w)
    flush()
    return ids, words


def main(corpus_path, vocab_path):
    corpus = open(corpus_path, 'rb').read()
    vbuf = open(vocab_path, 'rb').read()
    for name, data in (('corpus', corpus), ('vocab', vbuf)):
        got = hashlib.sha256(data).hexdigest()
        print(f'{name} sha256 {got} ({"matches" if got == EXPECTED[name] else "DIFFERS FROM"} the recorded input)')
    tokens = read_gguf_tokens(vbuf)[:4096]
    vocab = {}
    for i, t in enumerate(tokens):
        vocab.setdefault(t, i)
    ids, words = tokenize(corpus, vocab, 400000)
    n = len(ids)
    val_start = n - n // 20
    print(f'tokens {n}; validation tokens {n - val_start}; vocabulary {len(tokens)}')
    print(f'id 0 is the vocabulary entry {tokens[0]!r}')
    for label, lo, hi in (('all', 0, n), ('train', 0, val_start), ('validation', val_start, n)):
        zero = sum(1 for i in ids[lo:hi] if i == 0)
        unknown = sum(1 for w in words[lo:hi] if w not in vocab)
        print(f'{label}: id-0 share {zero / (hi - lo):.4f} ({zero} of {hi - lo}); '
              f'out-of-vocabulary share {unknown / (hi - lo):.4f} ({unknown})')
    distinct = collections.Counter(words)
    oov = collections.Counter(w for w in words if w not in vocab)
    used = len({i for i in ids})
    print(f'distinct surface strings {len(distinct)}; distinct ids used {used}; '
          f'distinct out-of-vocabulary strings {len(oov)}')
    def kind(w):
        if len(w) == 1 and w[0] >= 128:
            return 'non-ASCII byte'
        if is_alpha(w[0]) or is_digit(w[0]) or w[0] == 39:
            return 'word'
        return 'ASCII punctuation'
    by_kind = collections.Counter(kind(w) for w in words if w not in vocab)
    words_total = sum(1 for w in words if kind(w) == 'word')
    for k in ('word', 'ASCII punctuation', 'non-ASCII byte'):
        print(f'out-of-vocabulary {k}: {by_kind[k]} ({by_kind[k] / n:.4f} of all tokens)')
    print(f'word tokens {words_total}; share of word tokens out of vocabulary '
          f'{by_kind["word"] / words_total:.4f}')
    print('most frequent out-of-vocabulary strings:',
          ', '.join(f'{w.decode(errors="replace")}:{c}' for w, c in oov.most_common(15)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
