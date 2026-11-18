"""The relevance screens' jobs files against the token ladder's N250 chat campaign: only the start and the packet differ."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import ladder_chat_jobs as L  # noqa: E402
import relevance_chat_jobs as R  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_files_match_rendering():
    for s in R.SCREENS:
        assert open(R.out_path(s)).read() == R.text(s)


def test_only_start_and_packet_differ():
    lad = by_name(L.render("N250")["jobs"])
    for s, packet in R.SCREENS.items():
        rel = by_name(R.render(s)["jobs"])
        assert "sft_tlN250" in lad and "sft_tlN250" not in rel and "watch_sft_tlN250" not in rel
        for sd in (1, 2):
            a, b = lad[f"TL_N250_{sd}"], rel[f"{s}_N250_{sd}"]
            sa, sb = json.loads(json.dumps(a["spec"])), json.loads(json.dumps(b["spec"]))
            assert sa["train"]["init_from"] == "/content/chat/sft_tlN250/sft_tlN250.safetensors"
            assert sb["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
            assert sa["data"]["sft_examples"].endswith("/S71.jsonl") and sb["data"]["sft_examples"].endswith("/" + packet)
            for x in (sa, sb):
                x["train"].pop("init_from")
                x["data"].pop("sft_examples")
            assert sa == sb
            assert a.get("init_job") == "sft_tlN250" and "init_job" not in b
            assert b["after"] == ["gate"]
        for k in ("repro_A", "gate"):
            assert lad[k] == rel[k]


def test_names_are_own():
    for s in R.SCREENS:
        names = set(by_name(R.render(s)["jobs"]))
        assert {f"{s}_N250_1", f"{s}_N250_2", f"score_{s}_N250_1", f"score_{s}_N250_2"} <= names
        assert not any(n.startswith(("TL_", "B2C", "E12_")) for n in names)
