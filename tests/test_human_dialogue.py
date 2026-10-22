"""Human-written dialogue corpora (corpus/fetchers/human_dialogue.py): role mapping, merging,
multi-party drops, detokenisation, the incremental JSON reader, ConvAI2's perspective
de-duplication, the oasst human-only branch, the LLM-tell filter, licence gates and whole builds on
fixture rows, without downloads."""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "corpus", "fetchers"))
sys.path.insert(0, os.path.join(ROOT, "corpus"))
import decontam  # noqa: E402
import human_dialogue as H  # noqa: E402
from collections import Counter  # noqa: E402


def test_symmetric_first_speaker_is_user_and_runs_merge():
    c = Counter()
    m, why = H.to_messages([("bo", "Hi"), ("bo", "you there?"), ("ann", "Yes"), ("bo", "ok")], None, c)
    assert why is None and [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[0]["content"] == "Hi\nyou there?" and c["turns_merged_same_speaker"] == 1


def test_fixed_roles_trim_a_leading_assistant_and_drop_three_speakers():
    c = Counter()
    m, _ = H.to_messages([("W", "Hello, I know Elvis."), ("A", "Tell me!"), ("W", "He sang.")],
                         {"A": "user", "W": "assistant"}, c)
    assert [x["content"] for x in m] == ["Tell me!", "He sang."] and c["leading_assistant_turns_trimmed"] == 1
    assert H.to_messages([("a", "x"), ("b", "y"), ("c", "z")], None, c) == (None, "multi_party")
    assert H.to_messages([("a", "x"), ("q", "y")], {"a": "user"}, c) == (None, "unknown_speaker")
    m, _ = H.to_messages([("a", "__SILENCE__"), ("b", "hi"), ("a", " "), ("b", "anyone?"), ("a", "yes")], None, c)
    assert m == [{"role": "user", "content": "hi\nanyone?"}, {"role": "assistant", "content": "yes"}]


def test_detok():
    assert H.detok("i 'm fine , thanks . what do you do ?", case=True) == "I'm fine, thanks. What do you do?"
    assert H.detok("Say , Jim , how about it ? ") == "Say, Jim, how about it?"
    assert H.detok("i do n't know") == "i don't know"
    assert H.detok("I don ’ t know . Let ' s go . I ’ Ve") == "I don’t know. Let's go. I’ve"


def test_iter_json_array_small_chunks():
    data = [{"a": i, "s": "x" * (i * 7)} for i in range(50)]
    got = list(H.iter_json_array(io.BytesIO(json.dumps(data, indent=1).encode()), chunk=13))
    assert got == data
    assert list(H.iter_json_array(io.BytesIO(b" [ ] "))) == []


def test_convai2_keeps_the_longest_perspective():
    a = [("partner", "__SILENCE__"), ("self", "hi , how are you ?"), ("partner", "good . you ?"),
         ("self", "great"), ("partner", "cool"), ("self", "bye")]
    b = [("partner", "hi , how are you ?"), ("self", "good . you ?"), ("partner", "great"), ("self", "cool"),
         ("partner", "bye"), ("self", "see you")]
    other = [("partner", "hello"), ("self", "hey")]
    out = list(H.convai2_dialogues([a, b, other], "t.txt", Counter()))
    assert [r for _, _, r in out] == ["perspective_duplicate", None, None]
    assert out[1][1][0] == {"role": "user", "content": "Hi, how are you?"} and len(out[1][1]) == 6


def _msg(mid, parent, role, text, rank=None, synthetic=False):
    return {"message_id": mid, "parent_id": parent, "role": role, "text": text, "rank": rank, "lang": "en",
            "deleted": False, "review_result": True, "created_date": "2023", "synthetic": synthetic,
            "model_name": None}


def test_oasst_branch_skips_a_tell_for_the_next_ranked_reply():
    msgs = {m["message_id"]: m for m in [
        _msg("r", None, "prompter", "Q1"),
        _msg("a0", "r", "assistant", "As an AI language model, I cannot.", rank=0),
        _msg("a1", "r", "assistant", "Sure, here it is.", rank=1),
        _msg("a2", "r", "assistant", "generated", rank=2, synthetic=True)]}
    path = H.oasst_branch("r", {"r": ["a0", "a1", "a2"]}, msgs)
    assert [p["content"] for p in path] == ["Q1", "Sure, here it is."]


def test_build_tells_duplicates_and_decontamination(tmp_path):
    dc = decontam.Decontaminator(chat_sets=[])
    sq = json.loads(open(decontam.SIMPLE_QA, encoding="utf-8").readline())["prompt"]
    u = lambda t: {"role": "user", "content": t}  # noqa: E731
    a = lambda t: {"role": "assistant", "content": t}  # noqa: E731
    rows = [("r1", [u("How do I bake bread?"), a("Flour, water, salt and time.")], None),
            ("r2", [u("how do I bake bread"), a("flour water salt and time")], None),
            ("r3", [u("Write a poem"), a("As a large language model I love poems.")], None),
            ("r4", [u(sq), a("x")], None),
            ("r5", None, "not_english")]
    rep = H.build("dolly", dc=dc, rows=rows, out_dir=str(tmp_path))
    c = rep["counts"]
    assert c["written"] == 1 and c["duplicate"] == 1 and c["llm_tell"] == 1 and c["not_english"] == 1
    assert c["decontaminated"] == (1 if len(decontam.words(sq)) >= decontam.PROMPT_MIN else 0)
    text = open(tmp_path / "human_dolly_v1" / "text.txt", encoding="utf-8").read()
    assert text == "<|user|>How do I bake bread?<|assistant|>Flour, water, salt and time.<|endoftext|>\n"
    # a pre-ChatGPT source keeps the phrase: it cannot be ChatGPT's
    rep = H.build("bst", dc=dc, rows=[rows[2]], out_dir=str(tmp_path))
    assert rep["counts"]["written"] == 1


def test_simplewiki_documents(tmp_path):
    dc = decontam.Decontaminator(chat_sets=[])
    long = "The kangaroo is a marsupial from Australia. " * 5
    rows = [("w1", {"title": "Kangaroo", "text": long}), ("w2", {"title": "Stub", "text": "Too short."}),
            ("w3", {"title": "Kangaroo again", "text": long})]
    rep = H.build("simplewiki", dc=dc, rows=rows, out_dir=str(tmp_path))
    assert rep["counts"] == {"duplicate": 1, "rows_in": 3, "too_short": 1, "written": 1}
    assert open(tmp_path / "human_simplewiki_v1" / "text.txt", encoding="utf-8").read() == long.strip() + "\n<|endoftext|>\n"


def test_licence_gates_and_sets():
    class A:
        allow_sa = allow_nc = allow_llm_risk = False
    assert all(H.gate_ok(s, A) for s in H.DEFAULT_SET) and not any(H.gate_ok(s, A) for s in H.OPTIN_SET)
    assert set(H.DEFAULT_SET) == {"oasst", "convai2", "wow", "bst", "taskmaster", "multiwoz", "self_dialogue"}
    assert set(H.RAW) | {"simplewiki"} == set(H.SOURCES)


def _tgz(path, members):
    import tarfile
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return str(path)


def test_wow_and_convai2_readers_on_synthetic_archives(tmp_path, monkeypatch):
    wow = [{"chosen_topic": "Elvis", "dialog": [
        {"speaker": "0_Wizard", "text": "Elvis was a singer."}, {"speaker": "1_Apprentice", "text": "Tell me more!"},
        {"speaker": "0_Wizard", "text": "He was born in 1935."}]},
        {"dialog": [{"speaker": "1_Apprentice", "text": "I like jazz."}, {"speaker": "0_Wizard", "text": "Jazz began in New Orleans."}]}]
    wpath = _tgz(tmp_path / "wow.tgz", [("./data.json", b"[]"), ("./train.json", json.dumps(wow).encode())])
    lines = ("1 __SILENCE__\thi , i 'm sam .\n2 hello sam .\thow are you ?\n3 good , thanks .\tgreat .\n"
             "1 hi , i 'm sam .\thello sam .\n2 how are you ?\tgood , thanks .\n3 great .\tbye !\n").encode()
    cpath = _tgz(tmp_path / "c2.tgz", [("./train_none_original_no_cands.txt", lines), ("./valid.txt", b"")])
    monkeypatch.setattr(H, "fetch_url", lambda url, raw, want=None: wpath if "wizard" in url else cpath)
    ctx = {"raw": str(tmp_path), "counts": Counter()}
    out = list(H.raw_wow(ctx))
    assert [m[1] for m in out] == [[{"role": "user", "content": "Tell me more!"}, {"role": "assistant", "content": "He was born in 1935."}],
                                   [{"role": "user", "content": "I like jazz."}, {"role": "assistant", "content": "Jazz began in New Orleans."}]]
    out = list(H.raw_convai2(ctx))
    assert [r for _, _, r in out] == ["perspective_duplicate", None]
    assert [t["content"] for t in out[1][1]] == ["Hi, I'm sam.", "Hello sam.", "How are you?", "Good, thanks.", "Great.", "Bye!"]


def test_taskmaster_and_multiwoz_readers(tmp_path, monkeypatch):
    conv = lambda cid: {"conversation_id": cid, "utterances": [  # noqa: E731
        {"speaker": "USER", "text": "Book a table."}, {"speaker": "USER", "text": "For two."},
        {"speaker": "ASSISTANT", "text": "Where?"}]}
    files = {}
    for p in H.TM_FILES:
        f = tmp_path / p.replace("/", "_")
        if p.endswith("dev.csv"):
            f.write_text("dlg-dev,\n")
        elif p.endswith(".csv"):
            f.write_text("")
        elif p.endswith("self-dialogs.json"):
            f.write_text(json.dumps([conv("dlg-dev"), conv("dlg-1")]))
        else:
            f.write_text("[]")
        files[p] = str(f)
    for p in H.MWOZ_FILES:
        f = tmp_path / p.replace("/", "_")
        f.write_text(json.dumps([{"dialogue_id": "x", "turns": [{"speaker": "USER", "utterance": "Hi"},
                                                                  {"speaker": "SYSTEM", "utterance": "Hello"}]}]
                                if p.endswith("001.json") else []))
        files[p] = str(f)
    monkeypatch.setattr(H, "github_file", lambda rc, p, raw: files[p])
    ctx = {"raw": str(tmp_path), "counts": Counter()}
    out = list(H.raw_taskmaster(ctx))
    assert [r for _, _, r in out] == ["held_out_split", None]
    assert out[1][1][0] == {"role": "user", "content": "Book a table.\nFor two."}
    out = list(H.raw_multiwoz(ctx))
    assert out == [("multiwoz:data/MultiWOZ_2.2/train/dialogues_001.json#x",
                    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}], None)]
