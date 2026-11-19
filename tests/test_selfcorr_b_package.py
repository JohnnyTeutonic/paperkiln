"""E12b's jobs file against E12's: only the packet and the names differ."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import selfcorr_chat_jobs as E12  # noqa: E402
import selfcorr_b_chat_jobs as B  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(B.OUT).read() == B.text()


def test_rendering_leaves_e12_unchanged():
    before = E12.text()
    B.render()
    assert E12.text() == before and E12.PACKET == "S71SC.jsonl" and E12.arm_name(1) == "E12_N250_1"


def test_only_packet_and_names_differ():
    e12 = E12.render()["jobs"]
    b = B.render()["jobs"]
    assert len(e12) == len(b)
    for x, y in zip(e12, b):
        tx = json.dumps(x, sort_keys=True).replace("/S71SC.jsonl", "/S71SCb.jsonl").replace("E12_N250_", "E12b_N250_")
        assert tx == json.dumps(y, sort_keys=True)
    jb = by_name(b)
    for sd in (1, 2):
        s = jb[f"E12b_N250_{sd}"]["spec"]
        assert s["data"]["sft_examples"].endswith("/S71SCb.jsonl")
        assert s["train"]["steps"] == E12.STEPS == 1289
        assert s["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
        assert s["train"]["seed"] == sd


def test_names_are_own():
    names = set(by_name(B.render()["jobs"]))
    assert {"E12b_N250_1", "E12b_N250_2", "score_E12b_N250_1", "score_E12b_N250_2"} <= names
    assert not any(n.startswith(("TL_", "B2C", "E12_")) for n in names)
