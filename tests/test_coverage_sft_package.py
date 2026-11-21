"""The coverage-SFT campaign A's jobs file against R2's screen and the token ladder's N250 chat campaign: only the start and the packet
differ from the ladder's S71 spec; three arms at two seeds in the registered order; no early readouts; R2's reproduction check and
gate unchanged. Claude Opus 5.5 (a fork of the coordinating session), 22 Nov 2026."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import coverage_sft_chat_jobs as C  # noqa: E402
import ladder_chat_jobs as L  # noqa: E402
import relevance_chat_jobs as R  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(C.out_path()).read() == C.text()


def test_only_start_and_packet_differ_from_the_ladder_and_r2():
    lad = by_name(L.render("N250")["jobs"])
    r2 = by_name(R.render("R2")["jobs"])
    cov = by_name(C.render()["jobs"])
    assert "sft_tlN250" in lad and "sft_tlN250" not in cov and "watch_sft_tlN250" not in cov
    for arm, packet in C.ARMS.items():
        for sd in (1, 2):
            a, b, r = lad[f"TL_N250_{sd}"], cov[f"{arm}_N250_{sd}"], r2[f"R2_N250_{sd}"]
            sa, sb, sr = (json.loads(json.dumps(x["spec"])) for x in (a, b, r))
            assert sb["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors" == sr["train"]["init_from"]
            assert sb["data"]["sft_examples"] == PACKET_PATHS[arm] == sr["data"]["sft_examples"].replace("S71BD.jsonl", f"S71{arm}.jsonl")
            for x in (sa, sb, sr):
                x["train"].pop("init_from")
                x["data"].pop("sft_examples")
            assert sa == sb == sr                                   # everything else is S71's spec, as R2's
            assert "init_job" not in b and b["after"] == ["gate"]
            # the job beside its spec is R2's but for the rung snapshots, which it does not ask for
            assert r["snapshot_steps"] == [128, 256, 512] and "snapshot_steps" not in b
            assert {k: v for k, v in b.items() if k not in ("name", "spec")} == \
                {k: v for k, v in r.items() if k not in ("name", "spec", "snapshot_steps")}
            # its score job is R2's with the names changed
            sc, sr2 = cov[f"score_{arm}_N250_{sd}"], r2[f"score_R2_N250_{sd}"]
            assert json.dumps(sc, sort_keys=True) == json.dumps(sr2, sort_keys=True).replace(f"R2_N250_{sd}", f"{arm}_N250_{sd}")
    for k in ("repro_A", "gate"):
        assert cov[k] == r2[k] == lad[k]


# the registered packets, written out here rather than taken from the renderer
PACKET_PATHS = {"COV": "/content/chat/upload/S71COV.jsonl", "PERS": "/content/chat/upload/S71PERS.jsonl",
                "STATE": "/content/chat/upload/S71STATE.jsonl"}


def test_the_registered_order_and_nothing_brought_home_early():
    jobs = C.render()["jobs"]
    names = [j["name"] for j in jobs]
    train = [n for n in names if n.split("_")[0] in C.ARMS and not n.startswith("score_")]
    assert train == ["COV_N250_1", "PERS_N250_1", "STATE_N250_1", "COV_N250_2", "PERS_N250_2", "STATE_N250_2"]
    assert names[:2] == ["repro_A", "gate"] and names[2:8] == train
    assert names[8:] == [f"score_{n}" for n in train]
    assert not any(n.startswith("early_") for n in names)       # no early readouts (the lead's ruling, 21 Nov 2026)
    assert not any("snapshot_steps" in j or "rung_of" in j for j in jobs)     # and no rung snapshots asked for or scored
    assert "snapshot_steps" not in C.text() and "rung_" not in C.text()
    assert C.render()["max_parallel"] == 1                       # one pod, one training job at a time, in that order


def test_names_are_own():
    names = set(by_name(C.render()["jobs"]))
    assert not any(n.startswith(("TL_", "B2C", "E12", "R1_", "R2_")) for n in names)
