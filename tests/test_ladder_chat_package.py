"""The token ladder's chat campaigns: A35's jobs file with the base, the shape and the names changed, and nothing else.
Claude Fable 5.2, 17 Nov 2026. CPU only."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "colab_chat"))
import a35_jobs as A35  # noqa: E402
import b2cue_jobs as B  # noqa: E402
import ladder_chat_jobs as J  # noqa: E402


def test_files_on_disk_are_the_renderings_and_a35s_is_untouched():
    for b in J.BASES:
        assert open(J.out_path(b)).read() == J.text(b)
    assert open(A35.OUT).read() == json.dumps(A35.render(), indent=1) + "\n"
    assert J.BASES == ("N1B", "S1B", "N2B", "S2B", "N250", "S250", "NT", "ST", "N250s2", "S250s2") and J.SEEDS == (1, 2)


def back_to_a35(obj, base):
    """The ladder file with its names put back to A35's and D1's architecture put back to A's."""
    t = json.dumps(obj)
    for sd in J.SEEDS:
        t = t.replace(J.arm_name(base, sd), A35.arm_name(sd))
    t = t.replace(J.tag(base), "C")
    d = json.loads(t)
    for j in d["jobs"]:
        if "spec" in j and j["spec"]["arch"] == B.D1_ARCH:
            j["spec"]["arch"] = json.loads(json.dumps(B.A_ARCH))
    return d


@pytest.mark.parametrize("base", J.BASES)
def test_a_campaign_is_a35s_but_for_the_base_the_shape_and_the_names(base):
    got, ref = J.render(base), A35.render()
    assert [j["name"] for j in got["jobs"]] == ["repro_A", "gate", f"sft_tl{base}", f"watch_sft_tl{base}"] + \
        [x for sd in (1, 2) for x in [f"TL_{base}_{sd}"] + [f"early_TL_{base}_{sd}_{s}" for s in (128, 256, 512)]] + \
        [f"score_TL_{base}_1", f"score_TL_{base}_2"]
    back = back_to_a35(got, base)
    gj, rj = {j["name"]: j for j in back["jobs"]}, {j["name"]: j for j in ref["jobs"]}
    assert list(gj) == list(rj)
    for n in rj:
        g, r = json.loads(json.dumps(gj[n])), json.loads(json.dumps(rj[n]))
        assert g == r, n
    by = {j["name"]: j for j in got["jobs"]}
    assert by["repro_A"]["spec"]["arch"] == B.A_ARCH          # the reproduction is A's, at A's shape, as in A35
    assert by[f"sft_tl{base}"]["spec"]["arch"] == B.D1_ARCH
    assert by[f"sft_tl{base}"]["spec"]["train"]["init_from"] == f"/content/chat/upload/base_tl{base}.safetensors"
    for sd in (1, 2):
        s = by[f"TL_{base}_{sd}"]["spec"]
        assert s["arch"] == B.D1_ARCH and s["train"]["seed"] == sd and s["train"]["steps"] == 1024 and s["train"]["lr"] == 0.0001
        assert s["data"]["sft_examples"] == "/content/chat/upload/S71.jsonl"
        assert s["train"]["init_from"] == f"/content/chat/sft_tl{base}/sft_tl{base}.safetensors"
        assert "twin_loss" not in s["train"] and "twin_examples" not in s["data"]


def test_no_campaign_names_a_twin_file_a_sealed_request_file_or_another_campaigns_model():
    for base in J.BASES:
        t = J.text(base)
        for word in ("train_twins", "twin_loss", "answer_mask", "confirmation", "selection_req", "S74", "S73", "readout_req",
                     "sft_N250/", "sft_S/", "sft_N2", "sft_S2", "base_N2", "base_S2", "sft_C", "S75_", "base_N250.safetensors", "base_S.safetensors", "base_C.safetensors"):
            assert word not in t, (base, word)
    names = [j["name"] for b in J.BASES for j in J.render(b)["jobs"] if j["name"] not in ("repro_A", "gate")]
    assert len(names) == len(set(names))
