"""tools/model_card.py on fake run directories and a fake sampler.

No trained model, no mtstudio build: runs in seconds.

    python -m pytest tools/test_model_card.py -q
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_card as M  # noqa: E402
import pytest  # noqa: E402


def _write_run(d: Path, spec=None, result=None, events=None):
    d.mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (d / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    if result is not None:
        (d / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if events is not None:
        lines = [json.dumps(e) for e in events]
        (d / "events.jsonl").write_text("\n".join(lines) + "\n{\"event\":\"step\",", encoding="utf-8")
    return d


@pytest.fixture
def chat_run(tmp_path):
    spec = {"name": "chat9", "arch": {"preset": "gpt2-nano",
                                       "custom": {"attention": "swa", "window": 32, "sinks": 1}},
            "data": {"corpus": "data/tinychat-txt/train-0.txt", "vocab": "v.gguf",
                     "vocab_cap": 2048, "T": 128},
            "train": {"steps": 2000}, "out_dir": "/somewhere/else"}
    result = {"name": "chat9", "best_val": 1.234, "final_step": 1500,
              "steps_requested": 2000, "early_stopped": True, "params": 640000,
              "family": "gpt2", "attention": "swa", "wall_seconds": 600}
    events = [{"event": "start", "name": "chat9", "steps": 2000},
              {"event": "data", "tokens": 123456, "val_tokens": 5000, "vocab": 2048},
              {"event": "model", "family": "gpt2", "attention": "swa", "d": 128, "layers": 2,
               "heads": 4, "T": 128, "norm": "layernorm", "activation": "gelu",
               "position": "learned", "residual": "residual", "window": 32, "sinks": 1,
               "params": 640000},
              {"event": "eval", "step": 100, "val_loss": 2.0},
              {"event": "eval", "step": 1500, "val_loss": 1.234}]
    return _write_run(tmp_path / "chat9", spec, result, events)


class FakeBackend:
    """Canned replies keyed by the last user turn; records every call."""

    name = "fake"

    def __init__(self, replies, default="the the the ."):
        self.replies, self.default, self.calls = replies, default, []

    def reply(self, turns):
        self.calls.append(turns)
        return self.replies.get(turns[-1]["text"], self.default)


GOOD = {
    "hello !": "hi ! nice to see you .",
    "how are you ?": "i am great , thank you .",
    "what is your favorite food ?": "my favorite food is pizza . user: bye",
    "would you like some tea ?": "yes please , i would love some tea .",
    "where did you go today ?": "i went to the park .",
    "how is the weather today ?": "it is very sunny .",
    "what is your hobby ?": "my hobby is painting .",
    "do you have a pet ?": "yes , i have a little dog .",
    "do you want to go to the park with me ?": "yes , that sounds lovely .",
    "see you later !": "goodbye ! have a nice day .",
}


def test_probe_set_shape():
    p = M.load_probes()
    assert 20 <= len(p["probes"]) <= 30
    ids = [x["id"] for x in p["probes"]]
    assert len(ids) == len(set(ids))
    for x in p["probes"]:
        assert x["tier"] in ("A", "B", "off")
        if x["tier"] == "off":
            assert "expect" in x and "why" in x
        else:
            assert x["intent"] in p["intents"]
    # all eight TinyChat question kinds plus greeting and goodbye
    assert {x["intent"] for x in p["probes"] if x["tier"] != "off"} == set(p["intents"])
    assert any(len(x["turns"]) > 1 for x in p["probes"])  # multi-turn present


def test_classify_matches_generator_phrasing():
    intents = M.load_probes()["intents"]
    assert M.classify("i am fine , thank you .", intents)[0] == "wellbeing"
    assert M.classify("i went to the museum and enjoyed reading .", intents)[0] == "day_activity"
    assert M.classify("yes , my cat is very sweet .", intents)[0] == "pet"
    assert M.classify("sorry , i am too busy today .", intents)[0] == "invitation"
    # graded on the first sentence: a trailing goodbye does not count
    assert M.classify("it was warm . goodbye ! take care .", intents)[0] == "weather"
    # a drink named outside a proper reply is not credited (chat8b's 'no thank milk .')
    assert M.classify("no thank milk .", intents)[0] == "unrelated"
    assert M.classify("i ! assistant:", intents)[0] == "unrelated"


def test_cut_reply_stops_at_invented_turn():
    assert M.cut_reply("i like rice . user: what ?") == ("i like rice .", True)
    assert M.cut_reply("i like rice . <|endoftext|> x") == ("i like rice .", True)
    assert M.cut_reply("i like rice .") == ("i like rice .", False)


def test_architecture_prefers_events_and_canonicalises(chat_run):
    run = M.load_run(chat_run)
    a = M.architecture(run["spec"], run["result"], run["events"])
    assert a["attention"] == "swa" and a["window"] == 32 and a["T"] == 128
    assert a["optimizer"] == "adamw" and a["family"] == "gpt2"


def test_llama_preset_fills_its_block(tmp_path):
    d = _write_run(tmp_path / "ll", spec={"arch": {"preset": "llama-tiny"},
                                          "data": {"corpus": "tinystories/train.txt"}})
    run = M.load_run(d)
    a = M.architecture(run["spec"], run["result"], run["events"])
    assert (a["position"], a["norm"], a["activation"]) == ("rope", "rmsnorm", "swiglu")
    card = M.build_card(run)
    comps = {c["slot"]: c for c in card["what_it_is"]["components"]}
    assert comps["position"]["part_of_family"] and not comps["position"]["default"]
    assert card["what_to_expect"]["source"] == "heuristic"
    assert "children's-story" in card["what_to_expect"]["summary"].lower()


def test_heuristic_card_without_probe(chat_run):
    card = M.build_card(M.load_run(chat_run))
    st = card["size_and_training"]
    assert st["params"] == 640000 and st["corpus"]["kind"] == "tinychat"
    assert st["corpus"]["tokens"] == 123456 and st["early_stopped"] is True
    assert st["perplexity"] == pytest.approx(3.4346, rel=1e-3)
    swa = next(c for c in card["what_it_is"]["components"] if c["slot"] == "attention")
    assert not swa["default"] and "fixed distance (32 words)" in swa["plain"]
    assert "forgets earlier context" in swa["plain"]
    assert swa["technical"]  # from tools/synthesis/vocab.py COMPONENTS
    ex = card["what_to_expect"]
    assert ex["source"] == "heuristic" and card["probe"] is None
    assert "knows no facts" in ex["summary"]
    assert "do arithmetic" in ex["cannot"]
    md = M.render_md(card)
    for head in ("## What it is", "## Size and training", "## What to expect"):
        assert head in md
    assert "Probe results" not in md
    assert "1,500 steps of 2,000 requested, stopped early" in md


def test_probe_drives_wording_and_is_deterministic(chat_run):
    probes = M.load_probes()
    fake = FakeBackend(GOOD)
    pr = M.run_probes(fake, probes)
    assert pr == M.run_probes(FakeBackend(GOOD), probes)  # deterministic
    # every tier-A prompt answered on topic; tier B falls to the default
    assert pr["tier_a"]["passed"] == pr["tier_a"]["total"]
    assert pr["tier_b"]["passed"] == 0
    assert pr["off_distribution"]["passed"] == 0
    assert set(pr["per_intent"]) == set(probes["intents"])
    assert len(pr["examples"]) == 3
    assert [e["passed"] for e in pr["examples"][:2]] == [True, False]
    assert pr["examples"][2]["tier"] == "off"
    food = next(r for r in pr["results"] if r["id"] == "food_a")
    assert food["reply"] == "my favorite food is pizza ." and food["rambled"]
    # multi-turn probes feed the model's own earlier reply back in
    mem = [c for c in fake.calls if c[0]["text"] == "my name is tom ." and len(c) == 3]
    assert mem and mem[0][1] == {"role": "assistant", "text": "the the the ."}

    card = M.build_card(M.load_run(chat_run), pr)
    ex = card["what_to_expect"]
    assert ex["source"] == "probe"
    assert "phrased as in training" in ex["summary"] and "when reworded" in ex["summary"]
    assert "Rewording a question makes it noticeably worse" in ex["summary"]
    assert "failed all 6 questions outside its training" in ex["summary"]
    md = M.render_md(card)
    assert "## Probe results" in md and "| outside training (expected to fail) | 0/6 |" in md


def test_off_distribution_pass_is_not_overclaimed(chat_run):
    replies = dict(GOOD, **{"what is the capital of france ?": "paris ."})
    pr = M.run_probes(FakeBackend(replies), M.load_probes())
    assert pr["off_distribution"]["passed"] == 1
    ex = M.build_card(M.load_run(chat_run), pr)["what_to_expect"]
    assert "luck or a memorised phrase" in ex["summary"]


def test_paper_provenance_from_emit_spec(tmp_path):
    spec = {"_comment": "Generated by papers/fetch.py from arXiv:2004.05150 (Longformer). "
                        "house dims (d=128, L=2, H=4, T=256); paper mechanism only. "
                        "not applied: activation=gelu (contested); norm=x (no engine support). "
                        "Omitted fields fall to engine defaults, never guessed.",
            "base": {"arch": {"preset": "gpt2-nano",
                              "custom": {"d": 128, "layers": 2, "heads": 4,
                                         "attention": "swa", "window": 64, "sinks": 0}},
                     "data": {"corpus": "<PATH TO CORPUS .txt>", "T": 256}},
            "factors": {}, "design": "grid"}
    d = _write_run(tmp_path / "paper", spec=spec)
    card = M.build_card(M.load_run(d))
    paper = card["what_it_is"]["paper"]
    assert paper["arxiv_id"] == "2004.05150" and paper["title"] == "Longformer"
    assert paper["mechanisms"] == {"attention": "swa", "window": 64}
    assert paper["house_dims"] and len(paper["not_applied"]) == 2
    assert "arXiv:2004.05150 (Longformer)" in paper["text"]
    swa = next(c for c in card["what_it_is"]["components"] if c["slot"] == "attention")
    assert swa["from_paper"]
    assert "From the paper." in M.render_md(card)


def test_cli_writes_both_files(chat_run, capsys):
    assert M.main([str(chat_run)]) == 0
    card = json.loads((chat_run / "card.json").read_text(encoding="utf-8"))
    assert card["schema"] == M.CARD_SCHEMA and card["name"] == "chat9"
    assert (chat_run / "card.md").read_text(encoding="utf-8").startswith("# Model card: chat9")


def test_mtstudio_backend_strips_prompt_and_rewrites_spec(chat_run, tmp_path):
    (chat_run / "chat9.safetensors").write_bytes(b"")
    run = M.load_run(chat_run)
    spec_path = M.sampling_spec(run, str(tmp_path))
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    assert spec["out_dir"] == str(chat_run.resolve()) and spec["name"] == "chat9"
    fake = tmp_path / "fake_mtstudio.py"
    fake.write_text(
        "import sys\n"
        "a = sys.argv\n"
        "assert a[1] == 'sample' and a[a.index('--temp') + 1] == '0'\n"
        "assert a[a.index('--topk') + 1] == '1'\n"
        "print('loading ...')\n"
        "print(a[a.index('--prompt') + 1] + ' i am fine , thank you .')\n",
        encoding="utf-8")
    be = M.MtstudioSampleBackend([sys.executable, str(fake)], spec_path, tokens=8)
    turns = [{"role": "user", "text": "how are you ?"}]
    assert M.MtstudioSampleBackend.format_prompt(turns) == "user: how are you ? assistant:"
    assert be.reply(turns) == "i am fine , thank you ."


def test_missing_run_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        M.load_run(tmp_path)


def test_http_chat_backend_round_trip():
    import http.server
    import threading

    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.update(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"reply": " i am fine . "}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = M.HttpChatBackend(f"http://127.0.0.1:{srv.server_port}/chat", tokens=8)
        turns = [{"role": "user", "text": "hi !"}, {"role": "assistant", "text": "hello !"},
                 {"role": "user", "text": "how are you ?"}]
        assert be.reply(turns) == "i am fine ."
        assert seen["user_input"] == "how are you ?" and len(seen["history"]) == 2
        assert seen["temp"] == 0 and seen["topk"] == 1
    finally:
        srv.shutdown()
