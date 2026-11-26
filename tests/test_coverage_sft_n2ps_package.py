"""The coverage-SFT campaign D's jobs file (PERS and STATE at N2) against campaign A's, campaign C's and the token ladder's N250s2
chat campaign: <ARM>_N250s2_<s> is campaign A's <ARM>_N250_<s> with only the start changed (sft_tlN250 -> sft_tlN250s2), campaign
C's COV_N250s2_<s> with only the packet changed, and the ladder's TL_N250s2_<s> (the controls) with only the start's upload path and
the packet changed; four jobs in the order PERS_1, STATE_1, PERS_2, STATE_2; the reproduction check and gate unchanged.
Claude Opus 5.5, 26 Nov 2026."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import coverage_sft_chat_jobs as A  # noqa: E402
import coverage_sft_n2_chat_jobs as C  # noqa: E402
import coverage_sft_n2ps_chat_jobs as D  # noqa: E402
import ladder_chat_jobs as L  # noqa: E402

PACKETS = {"PERS": "S71PERS.jsonl", "STATE": "S71STATE.jsonl"}


def by_name(jobs):
    return {j["name"]: j for j in jobs}


def test_file_matches_rendering():
    assert open(D.out_path()).read() == D.text()


def test_campaigns_a_and_c_unchanged():
    assert open(A.out_path()).read() == A.text()
    assert open(C.out_path()).read() == C.text()


def test_only_the_start_differs_from_campaign_a_the_packet_from_c_and_start_and_packet_from_the_controls():
    a = by_name(A.render()["jobs"])
    c = by_name(C.render()["jobs"])
    d = by_name(D.render()["jobs"])
    lad = by_name(L.render("N250s2")["jobs"])
    for arm, pkt in PACKETS.items():
        for sd in (1, 2):
            ja, jc, jd, jl = a[f"{arm}_N250_{sd}"], c[f"COV_N250s2_{sd}"], d[f"{arm}_N250s2_{sd}"], lad[f"TL_N250s2_{sd}"]
            sa, sc, sd_, sl = (json.loads(json.dumps(x["spec"])) for x in (ja, jc, jd, jl))
            assert sd_["train"]["init_from"] == "/content/chat/upload/sft_tlN250s2.safetensors" == sc["train"]["init_from"]
            assert sa["train"]["init_from"] == "/content/chat/upload/sft_tlN250.safetensors"
            assert sl["train"]["init_from"] == "/content/chat/sft_tlN250s2/sft_tlN250s2.safetensors"
            assert sd_["data"]["sft_examples"] == f"/content/chat/upload/{pkt}" == sa["data"]["sft_examples"]
            assert sc["data"]["sft_examples"] == "/content/chat/upload/S71COV.jsonl"
            assert sl["data"]["sft_examples"] == "/content/chat/upload/S71.jsonl"
            assert sd_["train"]["seed"] == sd == sl["train"]["seed"] == sc["train"]["seed"]
            for x in (sa, sc, sd_, sl):
                x["train"].pop("init_from")
                x["data"].pop("sft_examples")
            assert sa == sc == sd_ == sl                              # everything else is S71's spec at D1
            assert {k: v for k, v in jd.items() if k not in ("name", "spec")} == \
                {k: v for k, v in ja.items() if k not in ("name", "spec")} == \
                {k: v for k, v in jc.items() if k not in ("name", "spec")}
            sca, scd = a[f"score_{arm}_N250_{sd}"], d[f"score_{arm}_N250s2_{sd}"]
            assert json.dumps(scd, sort_keys=True) == \
                json.dumps(sca, sort_keys=True).replace(f"{arm}_N250_{sd}", f"{arm}_N250s2_{sd}")
    for k in ("repro_A", "gate"):
        assert d[k] == a[k] == c[k] == lad[k]


def test_order_and_nothing_early():
    jobs = D.render()["jobs"]
    names = [j["name"] for j in jobs]
    assert names == ["repro_A", "gate", "PERS_N250s2_1", "STATE_N250s2_1", "PERS_N250s2_2", "STATE_N250s2_2",
                     "score_PERS_N250s2_1", "score_STATE_N250s2_1", "score_PERS_N250s2_2", "score_STATE_N250s2_2"]
    assert not any("snapshot_steps" in j or "rung_of" in j or j["name"].startswith("early_") for j in jobs)
    assert D.render()["max_parallel"] == 1
    top = {k: v for k, v in D.render().items() if k not in ("jobs", "_comment")}
    assert top == {k: v for k, v in C.render().items() if k not in ("jobs", "_comment")}


def test_names_are_own():
    names = set(by_name(D.render()["jobs"])) - {"repro_A", "gate"}
    assert not any(n.startswith(("TL_", "B2C", "E12", "R1_", "R2_", "COV_")) for n in names)
    assert not (names & set(by_name(A.render()["jobs"])))
    assert not (names & set(by_name(C.render()["jobs"])))
