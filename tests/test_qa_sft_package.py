"""The QA-SFT arm's jobs file against the coverage-SFT campaign's, R2's screen and the token ladder's N250 chat campaign: only the start
and the packet differ from the ladder's S71 spec; one arm at two seeds in the registered order; no early readouts; R2's reproduction
check and gate unchanged. Claude Opus 5.5 (subagent), 25 Nov 2026 (the coverage campaign's test with the names changed)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import coverage_sft_chat_jobs as C  # noqa: E402
import ladder_chat_jobs as L  # noqa: E402
import qa_sft_chat_jobs as Q  # noqa: E402
import relevance_chat_jobs as R  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


# the registered packet, written out here rather than taken from the renderer
PACKET_PATH = "/content/chat/upload/S71QA.jsonl"


def test_file_matches_rendering():
    assert open(Q.out_path()).read() == Q.text()


def test_only_start_and_packet_differ_from_the_ladder_r2_and_coverage():
    lad = by_name(L.render("N250")["jobs"])
    r2 = by_name(R.render("R2")["jobs"])
    cov = by_name(C.render()["jobs"])
    qa = by_name(Q.render()["jobs"])
    assert "sft_tlN250" in lad and "sft_tlN250" not in qa and "watch_sft_tlN250" not in qa
    assert Q.ARMS == {"QA": "S71QA.jsonl"}
    for sd in (1, 2):
        a, b, r, c = lad[f"TL_N250_{sd}"], qa[f"QA_N250_{sd}"], r2[f"R2_N250_{sd}"], cov[f"COV_N250_{sd}"]
        sa, sb, sr, sc = (json.loads(json.dumps(x["spec"])) for x in (a, b, r, c))
        assert sb["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors" == sr["train"]["init_from"]
        assert sb["data"]["sft_examples"] == PACKET_PATH
        for x in (sa, sb, sr, sc):
            x["train"].pop("init_from")
            x["data"].pop("sft_examples")
        assert sa == sb == sr == sc                              # everything else is S71's spec, as R2's and COV's
        assert "init_job" not in b and b["after"] == ["gate"]
        assert "snapshot_steps" not in b
        assert {k: v for k, v in b.items() if k not in ("name", "spec")} == \
            {k: v for k, v in c.items() if k not in ("name", "spec")}
        # its score job is the coverage campaign's with the names changed
        sq, scv = qa[f"score_QA_N250_{sd}"], cov[f"score_COV_N250_{sd}"]
        assert json.dumps(sq, sort_keys=True) == json.dumps(scv, sort_keys=True).replace(f"COV_N250_{sd}", f"QA_N250_{sd}")
    for k in ("repro_A", "gate"):
        assert qa[k] == cov[k] == r2[k] == lad[k]


def test_the_registered_order_and_nothing_brought_home_early():
    jobs = Q.render()["jobs"]
    names = [j["name"] for j in jobs]
    assert names == ["repro_A", "gate", "QA_N250_1", "QA_N250_2", "score_QA_N250_1", "score_QA_N250_2"]
    assert not any("snapshot_steps" in j or "rung_of" in j for j in jobs)
    assert "snapshot_steps" not in Q.text() and "rung_" not in Q.text()
    assert Q.render()["max_parallel"] == 1
    top = {k: v for k, v in Q.render().items() if k not in ("jobs", "_comment")}
    assert top == {k: v for k, v in C.render().items() if k not in ("jobs", "_comment")}


def test_names_are_own():
    names = set(by_name(Q.render()["jobs"]))
    assert not any(n.startswith(("TL_", "B2C", "E12", "R1_", "R2_", "COV_", "PERS_", "STATE_")) for n in names)
