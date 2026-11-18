"""E12's jobs file against the token ladder's N250 chat campaign: only the start and the packet differ."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import ladder_chat_jobs as L  # noqa: E402
import selfcorr_chat_jobs as E  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(E.OUT).read() == E.text()


def test_only_start_and_packet_differ():
    lad = by_name(L.render("N250")["jobs"])
    e12 = by_name(E.render()["jobs"])
    assert "sft_tlN250" in lad and "sft_tlN250" not in e12 and "watch_sft_tlN250" not in e12
    for sd in (1, 2):
        a, b = lad[f"TL_N250_{sd}"], e12[f"E12_N250_{sd}"]
        sa, sb = json.loads(json.dumps(a["spec"])), json.loads(json.dumps(b["spec"]))
        assert sa["train"]["init_from"] == "/content/chat/sft_tlN250/sft_tlN250.safetensors"
        assert sb["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
        assert sa["data"]["sft_examples"].endswith("/S71.jsonl") and sb["data"]["sft_examples"].endswith("/S71SC.jsonl")
        for s in (sa, sb):
            s["train"].pop("init_from")
            s["data"].pop("sft_examples")
        assert sa == sb
        assert a.get("init_job") == "sft_tlN250" and "init_job" not in b
        assert b["after"] == ["gate"]
    for k in ("repro_A", "gate"):
        assert lad[k] == e12[k]


def test_names_are_own():
    names = set(by_name(E.render()["jobs"]))
    assert {"E12_N250_1", "E12_N250_2", "score_E12_N250_1", "score_E12_N250_2"} <= names
    assert not any(n.startswith("TL_") or n.startswith("B2C") for n in names)
