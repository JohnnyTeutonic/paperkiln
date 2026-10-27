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


# ---- long-dialogue sources (24 Oct 2026) ---------------------------------------------------------
V1_ORDER = ["oasst", "convai2", "wow", "bst", "taskmaster", "multiwoz", "aya_en", "self_dialogue", "topical_chat",
            "dolly", "quac", "simplewiki", "empathetic", "dailydialog", "no_robots"]


def test_long_sources_leave_the_v1_groups_and_gates_unchanged():
    """The pinned v1 corpora are built by default/optin/all: the long sources join none of them."""
    assert [s for s in H.SOURCES if s not in H.LONG_SET] == V1_ORDER
    assert H.OPTIN_SET == ["aya_en", "topical_chat", "dolly", "quac", "simplewiki", "empathetic", "dailydialog",
                           "no_robots"]
    assert not set(H.LONG_SET) & (set(H.DEFAULT_SET) | set(H.OPTIN_SET))
    assert H.SOURCES["taskmaster"][:3] == (H.DEFAULT, "CC BY 4.0", "USER=user, ASSISTANT=assistant")

    class A:
        allow_sa = allow_nc = allow_llm_risk = False
    assert all(H.gate_ok(s, A) for s in ("taskmaster3", "abcd", "ccpe", "p4g")) and not H.gate_ok("msc", A)
    A.allow_unstated_licence = True
    assert H.gate_ok("msc", A)


def _tm3_files(tmp_path, convs, held_id):
    files = {}
    for p in H.TM3_HELD:
        f = tmp_path / p.replace("/", "_")
        f.write_text(f"<U>hi<A>hello\t[A]bye\t{held_id}\n" if p.endswith("dev-aa.tsv") else "", encoding="utf-8")
        files[p] = str(f)
    for p in H.TM3_TRAIN:  # per-turn splits (c989091e): an id held out only when no train file names it
        f = tmp_path / p.replace("/", "_")
        f.write_text("", encoding="utf-8")
        files[p] = str(f)
    for p in H.TM_FILES + H.TM3_FILES:
        f = tmp_path / p.replace("/", "_")
        f.write_text(json.dumps(convs if p == "TM-3-2020/data/data_00.json" else []), encoding="utf-8")
        files[p] = str(f)
    return files


def test_taskmaster3_reads_lower_case_labels_and_drops_templates_and_held_ids(tmp_path, monkeypatch):
    def conv(cid, scenario):
        return {"conversation_id": cid, "scenario": scenario, "utterances": [
            {"speaker": "assistant", "text": "Welcome."}, {"speaker": "user", "text": "Two tickets, please."},
            {"speaker": "user", "text": "For tonight."}, {"speaker": "assistant", "text": "Which film?"}]}
    convs = [conv("dlg-free", "Successful Purchase"), conv("dlg-auto", "Auto Template 13 movie name resolution"),
             conv("dlg-auto2", "Auto template 1"), conv("dlg-held", "Successful Purchase")]
    files = _tm3_files(tmp_path, convs, "dlg-held")
    monkeypatch.setattr(H, "github_file", lambda rc, p, raw, want=None: files[p])
    ctx = {"raw": str(tmp_path), "counts": Counter()}
    out = list(H.raw_taskmaster3(ctx))
    assert [r for _, _, r in out] == [None, "auto_template", "auto_template", "held_out_split"]
    assert out[0][0] == "taskmaster3:TM-3-2020/data/data_00.json#dlg-free"
    assert out[0][1] == [{"role": "user", "content": "Two tickets, please.\nFor tonight."},
                         {"role": "assistant", "content": "Which film?"}]
    # the v1 reader is unchanged: TM-3's lower-case labels still drop there
    out = list(H.raw_taskmaster(ctx))
    assert [r for _, _, r in out] == ["unknown_speaker"] * 4


def test_taskmaster3_build_writes_its_own_corpus_id(tmp_path):
    dc = decontam.Decontaminator(chat_sets=[])
    rows = [("t1", [{"role": "user", "content": "Two tickets."}, {"role": "assistant", "content": "Which film?"}], None),
            ("t2", None, "auto_template")]
    rep = H.build("taskmaster3", dc=dc, rows=rows, out_dir=str(tmp_path))
    assert rep["id"] == "human_taskmaster3_v1" and rep["counts"]["auto_template"] == 1 and rep["counts"]["written"] == 1
    assert (tmp_path / "human_taskmaster3_v1" / "dialogues.jsonl").exists()
    assert not (tmp_path / "human_taskmaster_v1").exists()


def test_msc_joins_sessions_with_positional_speakers():
    s1 = [{"text": "Hi, I'm Sam."}, {"text": "Hello Sam."}]
    s2 = [{"text": "How was the hike?", "id": "Speaker 1"}, {"text": "Great!", "id": "Speaker 2"},
          {"text": "Glad.", "id": "Speaker 1"}]                                   # odd: ends on Speaker 1
    s3 = [{"text": "Back again!", "id": "4711"}, {"text": "Welcome back.", "id": "Speaker 2"}]  # numeric opener id
    rec2 = {"dialog": s2, "previous_dialogs": [{"dialog": s1}], "metadata": {"initial_data_id": "train:ordered_1"}}
    rec3 = {"dialog": s3, "previous_dialogs": [{"dialog": s1}, {"dialog": s2, "time_num": 2, "time_unit": "days"}],
            "metadata": {"initial_data_id": "train:ordered_1"}}
    lone = {"dialog": [{"text": "Hey"}, {"text": "Hey you"}], "previous_dialogs": [{"dialog": [{"text": "A"}, {"text": "B"}]}],
            "metadata": {"initial_data_id": "train:ordered_2"}}
    c = Counter()
    out = list(H.msc_episodes({"train:ordered_1": {2: rec2, 3: rec3}, "train:ordered_2": {2: lone}}, c))
    assert [u for u, _, _ in out] == ["msc:session_3/train.txt#train:ordered_1", "msc:session_2/train.txt#train:ordered_2"]
    assert out[0][1] == [{"role": "user", "content": "Hi, I'm Sam."}, {"role": "assistant", "content": "Hello Sam."},
                         {"role": "user", "content": "How was the hike?"}, {"role": "assistant", "content": "Great!"},
                         {"role": "user", "content": "Glad.\nBack again!"}, {"role": "assistant", "content": "Welcome back."}]
    assert c["msc_session_joins_same_speaker"] == 1
    assert c["msc_episodes_3_sessions"] == 1 and c["msc_episodes_2_sessions"] == 1
    assert c["msc_session_text_differs_from_its_own_record"] == 0
    rec3["previous_dialogs"][1]["dialog"] = s2[:2] + [{"text": "Glad!"}]
    list(H.msc_episodes({"x": {2: rec2, 3: rec3}}, c))
    assert c["msc_session_text_differs_from_its_own_record"] == 1


def test_msc_reader_on_a_synthetic_archive(tmp_path, monkeypatch):
    rec = {"dialog": [{"text": "Back!", "id": "Speaker 1"}, {"text": "Hi again.", "id": "Speaker 2"}],
           "previous_dialogs": [{"dialog": [{"text": "Hello"}, {"text": "Hi"}]}],
           "metadata": {"initial_data_id": "train:ordered_9"}}
    line = (json.dumps(rec) + "\n").encode()
    path = _tgz(tmp_path / "msc.tgz", [("msc/msc_dialogue/session_2/train.txt", line),
                                       ("msc/msc_dialogue/session_2/valid.txt", line)])
    monkeypatch.setattr(H, "fetch_url", lambda url, raw, want=None: path)
    out = list(H.raw_msc({"raw": str(tmp_path), "counts": Counter()}))
    assert len(out) == 1 and [m["content"] for m in out[0][1]] == ["Hello", "Hi", "Back!", "Hi again."]


def test_abcd_ccpe_p4g_readers(tmp_path, monkeypatch):
    import gzip
    abcd = {"train": [{"convo_id": 7, "original": [
        ["agent", "Hi!"], ["customer", "I need a refund."], ["agent", "Sure."],
        ["action", "Account has been pulled up for X."], ["agent", "Your name?"], ["customer", "Ann"]]}],
        "dev": [], "test": []}
    ap = tmp_path / "abcd.json.gz"
    with gzip.open(ap, "wt", encoding="utf-8") as f:
        json.dump(abcd, f)
    cp = tmp_path / "ccpe.json"
    cp.write_text(json.dumps([{"conversationId": "c1", "utterances": [
        {"index": 1, "speaker": "USER", "text": "Thrillers."},
        {"index": 0, "speaker": "ASSISTANT", "text": "What do you watch?"},
        {"index": 2, "speaker": "ASSISTANT", "text": "Why?"}, {"index": 3, "speaker": "USER", "text": "Suspense."}]}]),
        encoding="utf-8")
    pp = tmp_path / "p4g.csv"
    pp.write_text(",Unit,Turn,B4,B2\n0,Good morning.,0,0,d1\n1,How are you?,0,0,d1\n2,Fine.,0,1,d1\n"
                  "3,Ever heard of Save the Children?,1,0,d1\n4,No.,1,1,d1\n", encoding="utf-8")
    monkeypatch.setattr(H, "github_file", lambda rc, p, raw, want=None: str(ap) if p.endswith(".gz") else str(cp))
    monkeypatch.setattr(H, "fetch_url", lambda url, raw, want=None: str(pp))
    ctx = {"raw": str(tmp_path), "counts": Counter()}
    out = list(H.raw_abcd(ctx))
    assert out[0][0] == "abcd:train#7" and out[0][1] == [
        {"role": "user", "content": "I need a refund."}, {"role": "assistant", "content": "Sure.\nYour name?"},
        {"role": "user", "content": "Ann"}]
    assert ctx["counts"]["action_lines_removed"] == 1 and ctx["counts"]["leading_assistant_turns_trimmed"] == 1
    out = list(H.raw_ccpe(ctx))
    assert out[0][1] == [{"role": "user", "content": "Thrillers."}, {"role": "assistant", "content": "Why?"},
                         {"role": "user", "content": "Suspense."}]
    out = list(H.raw_p4g(ctx))
    assert out == [("p4g:full_dialog.csv#d1", [{"role": "user", "content": "Good morning. How are you?"},
                                              {"role": "assistant", "content": "Fine."},
                                              {"role": "user", "content": "Ever heard of Save the Children?"},
                                              {"role": "assistant", "content": "No."}], None)]
