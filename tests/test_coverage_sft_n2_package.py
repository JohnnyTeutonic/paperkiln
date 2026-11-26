"""The coverage-SFT campaign C's jobs file (COV at N2) against campaign A's and the token ladder's N250s2 chat campaign: COV_N250s2_<s>
is campaign A's COV_N250_<s> with only the start changed (sft_tlN250 -> sft_tlN250s2), and the ladder's TL_N250s2_<s> (the
controls) with only the start's upload path and the packet changed; two jobs in seed order; the reproduction check and gate
unchanged. Claude Opus 5.5, 26 Nov 2026."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import coverage_sft_chat_jobs as A  # noqa: E402
import coverage_sft_n2_chat_jobs as C  # noqa: E402
import ladder_chat_jobs as L  # noqa: E402


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(C.out_path()).read() == C.text()


def test_campaign_a_unchanged():
    assert open(A.out_path()).read() == A.text()


def test_only_the_start_differs_from_campaign_a_and_start_and_packet_from_the_controls():
    a = by_name(A.render()["jobs"])
    c = by_name(C.render()["jobs"])
    lad = by_name(L.render("N250s2")["jobs"])
    for sd in (1, 2):
        ja, jc, jl = a[f"COV_N250_{sd}"], c[f"COV_N250s2_{sd}"], lad[f"TL_N250s2_{sd}"]
        sa, sc, sl = (json.loads(json.dumps(x["spec"])) for x in (ja, jc, jl))
        assert sc["train"]["init_from"] == "/content/chat/upload/sft_tlN250s2.safetensors"
        assert sa["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
        assert sl["train"]["init_from"] == "/content/chat/sft_tlN250s2/sft_tlN250s2.safetensors"
        assert sc["data"]["sft_examples"] == "/content/chat/upload/S71COV.jsonl" == sa["data"]["sft_examples"]
        assert sl["data"]["sft_examples"] == "/content/chat/upload/S71.jsonl"
        assert sc["train"]["seed"] == sd == sl["train"]["seed"]
        for x in (sa, sc, sl):
            x["train"].pop("init_from")
            x["data"].pop("sft_examples")
        assert sa == sc == sl                                   # everything else is S71's spec at D1
        assert {k: v for k, v in jc.items() if k not in ("name", "spec")} == \
            {k: v for k, v in ja.items() if k not in ("name", "spec")}
        sca, scc = a[f"score_COV_N250_{sd}"], c[f"score_COV_N250s2_{sd}"]
        assert json.dumps(scc, sort_keys=True) == json.dumps(sca, sort_keys=True).replace(f"COV_N250_{sd}", f"COV_N250s2_{sd}")
    for k in ("repro_A", "gate"):
        assert c[k] == a[k] == lad[k]


def test_order_and_nothing_early():
    jobs = C.render()["jobs"]
    names = [j["name"] for j in jobs]
    assert names == ["repro_A", "gate", "COV_N250s2_1", "COV_N250s2_2", "score_COV_N250s2_1", "score_COV_N250s2_2"]
    assert not any("snapshot_steps" in j or "rung_of" in j or j["name"].startswith("early_") for j in jobs)
    assert C.render()["max_parallel"] == 1


def test_names_are_own():
    names = set(by_name(C.render()["jobs"])) - {"repro_A", "gate"}
    assert not any(n.startswith(("TL_", "B2C", "E12", "R1_", "R2_", "PERS_", "STATE_")) for n in names)
    assert not (names & set(by_name(A.render()["jobs"])))
