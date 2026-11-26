"""SYMW's chat campaign is the token ladder's N1B campaign with the base's name changed, and nothing else.
Claude Fable 5.2, 26 Nov 2026. CPU only."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "colab_chat"))
import b2cue_jobs as B  # noqa: E402
import ladder_chat_jobs as J  # noqa: E402
import symw_chat_jobs as S  # noqa: E402


def test_the_file_on_disk_is_the_rendering_and_the_ladders_files_are_untouched():
    assert open(S.OUT).read() == S.text()
    for b in J.BASES:
        assert open(J.out_path(b)).read() == J.text(b)
    assert "SYMW" not in J.BASES


def test_the_campaign_is_n1bs_but_for_the_name():
    got, ref = S.render(), J.render("N1B")
    got.pop("_comment"), ref.pop("_comment")
    assert json.loads(json.dumps(got).replace("SYMW", "N1B")) == ref
    assert set(got) == set(ref) and {k: got[k] for k in got if k != "jobs"} == {k: ref[k] for k in ref if k != "jobs"}
    want = ["repro_A", "gate", "sft_tlSYMW", "watch_sft_tlSYMW"]
    for sd in (1, 2):
        want += ["TL_SYMW_%d" % sd] + ["early_TL_SYMW_%d_%d" % (sd, s) for s in (128, 256, 512)]
    want += ["score_TL_SYMW_1", "score_TL_SYMW_2"]
    assert [j["name"] for j in got["jobs"]] == want


def test_specs_base_shape_packet_and_seeds():
    by = {j["name"]: j for j in S.render()["jobs"]}
    assert by["repro_A"]["spec"]["arch"] == B.A_ARCH
    sft = by["sft_tlSYMW"]["spec"]
    assert sft["arch"] == B.D1_ARCH and sft["train"]["steps"] == 1500
    assert sft["train"]["init_from"] == "/content/chat/upload/base_tlSYMW.safetensors"
    for sd in (1, 2):
        s = by["TL_SYMW_%d" % sd]["spec"]
        assert s["arch"] == B.D1_ARCH and s["train"]["seed"] == sd and s["train"]["steps"] == 1024 and s["train"]["lr"] == 0.0001
        assert s["data"]["sft_examples"] == "/content/chat/upload/S71.jsonl"
        assert s["train"]["init_from"] == "/content/chat/sft_tlSYMW/sft_tlSYMW.safetensors"
        assert "twin_loss" not in s["train"] and "twin_examples" not in s["data"]


def test_no_other_campaigns_name_or_file_is_used():
    t = S.text()
    for word in ("N250", "S250", "N2B", "S2B", "S1B", "TL_NT", "tlNT", "train_twins", "twin_loss", "answer_mask", "confirmation",
                 "selection_req", "S74", "S73", "readout_req", "base_C.safetensors"):
        assert word not in t, word
    assert t.count("N1B") == 1          # the comment names the ladder base it copies, once; no job or file does
    assert "N1B" not in json.dumps(S.render()["jobs"])
    ladder_names = {j["name"] for b in J.BASES for j in J.render(b)["jobs"]} - {"repro_A", "gate"}
    assert not ladder_names & {j["name"] for j in S.render()["jobs"]}


def test_a_stray_occurrence_of_the_ladder_bases_name_refuses(monkeypatch):
    real = J.render

    def bad(base):
        d = real(base)
        d["jobs"][0]["note"] = "see N1B"
        return d
    monkeypatch.setattr(J, "render", bad)
    try:
        S.render()
    except ValueError:
        return
    raise AssertionError("a stray N1B was rendered")
