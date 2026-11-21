"""E12c's jobs file against E12b's: only the packet and the names differ. Claude Opus 5.5 (a fork of the coordinating session),
22 Nov 2026."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import selfcorr_b_chat_jobs as B  # noqa: E402
import selfcorr_c_chat_jobs as C  # noqa: E402
import selfcorr_chat_jobs as E12  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(C.OUT).read() == C.text()


def test_rendering_leaves_e12_unchanged():
    before = E12.text()
    C.render()
    assert E12.text() == before and E12.PACKET == "S71SC.jsonl" and E12.arm_name(1) == "E12_N250_1"


def test_only_packet_and_names_differ_from_e12b():
    b = B.render()["jobs"]
    c = C.render()["jobs"]
    assert len(b) == len(c)
    for x, y in zip(b, c):
        tx = json.dumps(x, sort_keys=True).replace("/S71SCb.jsonl", "/S71SCc.jsonl").replace("E12b_N250_", "E12c_N250_")
        assert tx == json.dumps(y, sort_keys=True)
    jc = by_name(c)
    for sd in (1, 2):
        s = jc[f"E12c_N250_{sd}"]["spec"]
        assert s["data"]["sft_examples"].endswith("/S71SCc.jsonl")
        assert s["train"]["steps"] == E12.STEPS == 1289
        assert s["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
        assert s["train"]["seed"] == sd


def test_names_are_own():
    names = set(by_name(C.render()["jobs"]))
    assert {"E12c_N250_1", "E12c_N250_2", "score_E12c_N250_1", "score_E12c_N250_2"} <= names
    assert not any(n.startswith(("TL_", "B2C", "E12_", "E12b_")) for n in names)
