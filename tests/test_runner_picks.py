"""runner_multi.py on a replacement VM: which home post jobs a pick can never choose (spared_posts), in process.
tests/runner_multi_equiv/run_picks.py covers the same end to end on the fake colab CLI, with tiny_chat_run.py pick
run on the fake VM."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import runner_multi as rm  # noqa: E402
import tiny_chat_run  # noqa: E402

TOOL = "tools/colab_chat/tiny_chat_run.py"


def post(name, pt):
    return {"name": name, "after": pt, "init_job": pt, "cmd": ["python3", TOOL, "post", "--out", "{out}"],
            "outputs": ["summary.json"]}


def pick(name, posts, **kw):
    cmd = ["python3", TOOL, "pick", "--out", "{out}"]
    for p in posts:
        cmd += ["--run", f"/content/chat/{p}"]
    return {"name": name, "after": list(posts), "cmd": cmd, "outputs": ["best.json"], **kw}


def campaign():
    js = [{"name": "data", "cmd": ["x"], "outputs": []}]
    js += [{"name": f"pt_{x}", "after": "data", "spec": {}} for x in "abcd"]
    js += [post(f"post_{x}", f"pt_{x}") for x in "abcd"]
    js += [pick("pick_abc", ["post_a", "post_b", "post_c"]), pick("pick_d", ["post_d"])]
    return js


def setup(tmp_path, monkeypatch, home, summaries, status=None):
    """home: job names home; summaries: {post: (seed, primary)} written to ART/<post>/summary.json."""
    art = tmp_path / "art"
    for p, (seed, prim) in summaries.items():
        (art / p).mkdir(parents=True)
        (art / p / "summary.json").write_text(json.dumps({"seed": seed, "primary": prim}))
    monkeypatch.setattr(rm, "ART", str(art))
    monkeypatch.setattr(rm, "HOME", str(tmp_path))
    st = rm.State()
    for n in home:
        st.job(n).update(home=True, status=(status or {}).get(n, "FINISHED"))
    return st


def spare(st, jobs):
    by = {j["name"]: j for j in jobs}
    needed, todo = set(), [n for j in jobs if not st.job(j["name"])["home"] for n in rm.job_needs(j)]
    while todo:  # prepare()'s closure
        n = todo.pop()
        if n in by and n not in needed:
            needed.add(n)
            todo.extend(rm.job_needs(by[n]))
    failed = [j["name"] for j in jobs if st.job(j["name"])["home"] and st.job(j["name"]).get("status") == "FAILED"]
    return rm.spared_posts(st, jobs, by, needed, failed)


def test_all_home_one_rerun(tmp_path, monkeypatch):
    jobs = campaign()
    home = ["data", "pt_a", "pt_b", "pt_c", "pt_d", "post_a", "post_b", "post_c", "post_d"]
    st = setup(tmp_path, monkeypatch, home, {"post_a": (1, 0.1), "post_b": (2, 0.3), "post_c": (3, 0.3),
                                             "post_d": (1, 0.0)})
    s = spare(st, jobs)
    assert sorted(s) == ["post_a", "post_c"]  # b and c tie: the lower seed (b) can still be chosen
    assert s["post_a"] == str(tmp_path / "art" / "post_a" / "summary.json")
    # pick_d's only post is its best: rerun. The rule is pick's own function.
    rows = [tiny_chat_run.pick_row(str(tmp_path / "art" / p)) for p in ("post_a", "post_b", "post_c")]
    assert tiny_chat_run.pick_best(rows)["run"].endswith("post_b")


def test_partial_home_reruns_the_best_home_post(tmp_path, monkeypatch):
    jobs = campaign()
    st = setup(tmp_path, monkeypatch, ["data", "pt_a", "pt_b", "post_a", "post_b"],
               {"post_a": (1, 0.4), "post_b": (2, 0.2)})
    assert sorted(spare(st, jobs)) == ["post_b"]  # post_c may still win; post_a reruns in case it does not


def test_no_primary_and_missing_summary(tmp_path, monkeypatch):
    jobs = campaign()
    st = setup(tmp_path, monkeypatch, ["data", "pt_a", "pt_b", "pt_c", "post_a", "post_b", "post_c"],
               {"post_a": (1, None), "post_b": (2, 0.2)})  # post_c home without a summary at home
    assert sorted(spare(st, jobs)) == ["post_a"]  # post_c reruns (its value is unknown), and so does post_b


def test_failed_input_spares_the_rest(tmp_path, monkeypatch):
    jobs = campaign()
    st = setup(tmp_path, monkeypatch, ["data", "pt_a", "pt_b", "pt_c", "post_a", "post_b", "post_c"],
               {"post_a": (1, 0.1), "post_b": (2, 0.3)}, status={"post_c": "FAILED"})
    assert sorted(spare(st, jobs)) == ["post_a", "post_b"]  # the pick fails on the VM whatever runs


def test_other_dependant_keeps_the_rerun(tmp_path, monkeypatch):
    jobs = campaign() + [{"name": "extra", "after": "post_a", "cmd": ["e"], "outputs": []}]
    home = ["data", "pt_a", "pt_b", "pt_c", "post_a", "post_b", "post_c"]
    st = setup(tmp_path, monkeypatch, home, {"post_a": (1, 0.1), "post_b": (2, 0.3), "post_c": (3, 0.0)})
    assert sorted(spare(st, jobs)) == ["post_c"]  # post_a also feeds "extra", which may read its files


def test_home_pick_and_no_picks(tmp_path, monkeypatch):
    jobs = campaign()
    home = [j["name"] for j in jobs if j["name"] != "pick_d"]
    st = setup(tmp_path, monkeypatch, home, {"post_a": (1, 0.1), "post_b": (2, 0.3), "post_c": (3, 0.0),
                                             "post_d": (1, 0.0)})
    assert spare(st, jobs) == {}  # pick_abc is home; pick_d's single post is its best
    plain = [j for j in jobs if "pick" not in j["name"]] + [{"name": "after_all", "after": "post_a", "cmd": ["z"]}]
    assert spare(st, plain) == {}


def test_e5_jobs_file_parses():
    jobs = json.load(open(ROOT / "tools" / "colab_chat" / "tiny_chat_e5.json"))["jobs"]
    by = {j["name"]: j for j in jobs}
    picks = {j["name"]: rm.pick_inputs(j, by) for j in jobs if rm.pick_inputs(j, by)}
    assert sorted(picks) == ["pick_e5_s0", "pick_e5_s17", "pick_e5_s35", "pick_e5_s70"]
    assert picks["pick_e5_s17"] == ["post_e5_s17_r1", "post_e5_s17_r2", "post_e5_s17_r3"]
    assert all(rm.pick_inputs(j, by) is None for j in jobs if not j["name"].startswith("pick_"))
