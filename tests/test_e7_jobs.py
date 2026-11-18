"""E7A's jobs file and rung spec against the token ladder's N1B and base 2's D1 rung spec (make_e7.py)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CC = os.path.join(os.path.dirname(HERE), "tools", "colab_chat")
sys.path.insert(0, CC)
import make_e7 as E  # noqa: E402
import make_ladder as L  # noqa: E402


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, path + (i,))
    else:
        yield path, x


def test_files_written_equal_builder():
    assert E.main(["--check"]) == 0


def test_only_shape_names_and_rung_spec_differ_from_n1b():
    ref = L.arm("N1B")
    ref.pop("_comment"), ref.pop("_base2")
    got = E.arm()
    got.pop("_comment"), got.pop("_base2")
    a, b = dict(leaves(ref)), dict(leaves(got))
    assert set(a) == set(b)
    arch = {("d",): 768, ("layers",): 8, ("heads",): 12, ("d_ff",): 2240}
    for p, v in a.items():
        w = b[p]
        if v == w:
            continue
        if p[-3:-1] == ("arch", "custom") and (p[-1],) in arch:
            assert w == arch[(p[-1],)], (p, v, w)
            continue
        if v == E.RUNG_D1:
            assert w == E.RUNG_A, (p, v, w)
            continue
        assert isinstance(v, str) and "N1B" in v and w == v.replace("N1B", "E7A"), (p, v, w)


def test_training_keys_are_n1b_s():
    pt = [j for j in E.arm()["jobs"] if j["name"] == "pt_base2_E7A"][0]["spec"]
    tr, da = pt["train"], pt["data"]
    assert (tr["steps"], tr["batch"], tr["accum"], tr["lr"], tr["seed"]) == (60000, 8, 2, 0.0005, 1)
    assert tr["lr_schedule"] == {"warmup_steps": 4000, "decay": "cosine", "min_lr_frac": 0.1}
    assert (tr["precision"], tr["attention_kernel"]) == ("bf16", "flash")
    assert da["corpus"].endswith("/N.main.ids") and da["anneal"] == {"corpus": "/content/chat/data/N.tail.ids",
                                                                     "from_fraction": 0.85}
    assert (da["T"], da["vocab_cap"]) == (1024, 16384)
    assert pt["arch"]["state_tokens"] == {"k": 4, "mask": "causal"}


def test_rung_spec_is_d1_but_the_arch():
    d1 = json.load(open(os.path.join(CC, "base2_rung_D1.json"), encoding="utf-8"))
    a = E.rung_spec()
    assert a["arch"]["custom"] == E.A_CUSTOM
    for k in ("data", "train", "export"):
        assert a[k] == d1[k]
    assert {k: v for k, v in a["arch"].items() if k != "custom"} == {k: v for k, v in d1["arch"].items() if k != "custom"}


def test_a_shape_matches_arm_c_and_meta():
    c = json.load(open(os.path.join(CC, "base2_C.json"), encoding="utf-8"))
    pt = [j for j in c["jobs"] if j["name"] == "pt_base2_C"][0]["spec"]
    assert pt["arch"]["custom"] == E.A_CUSTOM and c["_base2"]["params"] == E.A_PARAMS
    meta = E.arm()["_base2"]
    assert (meta["arm"], meta["shape"], meta["steps"], meta["positions"], meta["slot"]) == ("E7A", "A", 60000,
                                                                                            983040000, None)
