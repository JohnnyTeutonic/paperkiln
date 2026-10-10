"""Small synthetic tests only. Never load completed experiments or train."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import atlas_designer as D  # noqa: E402
from atlas_designer_demo import example, synthetic_block  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def model(payload):
    return D.Model(D.Space(payload["space"]), payload["observations"], **payload["model"])


def test_shared_anchor_information_matches_independent_matrix_calculation():
    x = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    noise = .3
    r = noise * (np.eye(3) + np.ones((3, 3)))
    np.testing.assert_allclose(D.block_information(x, noise), x.T @ np.linalg.solve(r, x), atol=1e-12)
    # Treating pairs as independent WOULD produce a different result.
    assert not np.allclose(D.block_information(x, noise), x.T @ x / (2*noise))


def test_planted_interaction_and_seed_offsets():
    p = example()
    space = D.Space(p["space"])
    p["observations"]["blocks"] = [synthetic_block(s, space.candidates) for s in (1, 2, 3, 4)]
    m = model(p)
    contrast = space.contrast(p["design"]["targets"][0]["terms"])
    assert m.describe(contrast)["mean"] == pytest.approx(-.8, abs=.015)
    shifted = copy.deepcopy(p)
    for block in shifted["observations"]["blocks"]:
        for run in block["runs"]:
            run["value"] += block["seed"] * 1000.
    np.testing.assert_allclose(model(shifted).mean, m.mean, atol=1e-10)


def test_interaction_predicts_unobserved_combination():
    p = example()
    space = D.Space(p["space"])
    held = space.candidates[-1]
    observed = space.candidates[:-1]
    p["observations"]["blocks"] = [synthetic_block(s, observed) for s in (1, 2, 3, 4)]
    m = model(p)
    base = p["design"]["baseline"]
    delta = m.describe(space.vector(held) - space.vector(base))["mean"]
    from atlas_designer_demo import synthetic_value
    truth = synthetic_value(held, 0) - synthetic_value(base, 0)
    assert delta == pytest.approx(truth, abs=.025)


def test_batch_expected_covariance_matches_actual_refit():
    p = example()
    old_covariance = model(p).covariance
    before = D.run(p)
    for seed in before["seeds"]:
        p["observations"]["blocks"].append(synthetic_block(seed, [before["baseline"]] + before["selected"]))
    m = model(p)
    assert np.linalg.eigvalsh(old_covariance - m.covariance).min() >= -1e-12
    target = m.space.contrast(p["design"]["targets"][0]["terms"])
    assert m.describe(target)["variance"] == pytest.approx(before["targets"][0]["planned_variance"])
    for entry in before["acquisition"]:
        assert np.all(np.array(entry["target_variance_after"]) <= np.array(entry["target_variance_before"]) + 1e-12)


def test_variance_acquisition_is_outcome_independent_but_means_change():
    p = example()
    q = copy.deepcopy(p)
    q["observations"]["blocks"][0]["runs"][1]["value"] += 10.
    a, b = D.run(p), D.run(q)
    assert a["acquisition"] == b["acquisition"]
    assert a["conditional_fingerprint"][0]["mean"] != b["conditional_fingerprint"][0]["mean"]


def test_prior_dominance_and_empty_observations():
    p = example()
    m = model(p)
    c = m.space.contrast(p["design"]["targets"][0]["terms"])
    assert m.describe(c)["has_unidentified_component"]
    assert m.describe(c)["posterior_to_prior_variance"] == pytest.approx(1.)
    p["observations"]["blocks"] = []
    assert np.all(np.isfinite(model(p).covariance))


def test_conditional_fingerprint_changes_with_context():
    p = example()
    space = D.Space(p["space"])
    p["observations"]["blocks"] = [synthetic_block(s, space.candidates) for s in (1, 2, 3, 4)]
    m = model(p)
    baseline = p["design"]["baseline"]
    base_effect = next(r for r in m.fingerprint(baseline) if r["slot"] == "norm")
    changed_effect = next(r for r in m.fingerprint(dict(baseline, activation="relu")) if r["slot"] == "norm")
    assert base_effect["mean"] > 0
    assert changed_effect["mean"] < 0
    assert changed_effect["baseline"]["activation"] == "relu"
    assert changed_effect["target_observed"]


def test_deterministic_plan_and_no_redundant_batch_entries():
    p = example()
    assert D.run(p) == D.run(copy.deepcopy(p))
    report = D.run(p)
    first = report["acquisition"][0]
    assert first["incremental_cost"] == 4.  # two new seeds, candidate plus anchor
    assert all(r["incremental_cost"] == 2. for r in report["acquisition"][1:])


def test_budget_and_no_duplicate_candidates():
    p = example()
    report = D.run(p)
    assert report["spent"] <= p["design"]["budget"]
    assert report["planned_runs"] == (1 + len(report["selected"])) * 2
    assert len({D.canonical(s) for s in report["selected"]}) == len(report["selected"])
    p["design"]["budget"] = .5
    report = D.run(p)
    assert report["selected"] == [] and report["planned_runs"] == 0


def test_cost_sensitive_choice():
    p = example()
    p["design"]["max_candidates"] = 1
    original = D.run(p)["selected"][0]
    q = copy.deepcopy(p)
    for cost in q["design"]["costs"]:
        if cost["settings"] == original:
            cost["per_seed"] = 1000.
    assert D.run(q)["selected"][0] != original


def test_manifest_round_trip(tmp_path):
    import mtsweep
    report = D.run(example())
    sweep = report["sweep"]
    _, combos, runs = mtsweep.expand(sweep)
    assert len(combos) == 1 + len(report["selected"])
    assert len(runs) == report["planned_runs"]
    paths = mtsweep.materialise(sweep, runs, str(tmp_path))
    actual = []
    for path, _ in paths:
        spec = json.loads(Path(path).read_text())
        actual.append({s: spec["arch"]["custom"][s] for s in example()["space"]["factors"]})
    expected = [s for s in [report["baseline"]] + report["selected"] for _ in report["seeds"]]
    assert actual == expected


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_cost(bad):
    p = example()
    p["design"]["costs"][0]["per_seed"] = bad
    with pytest.raises(ValueError):
        D.run(p)


@pytest.mark.parametrize("change", ["scope", "seed_reuse", "category", "duplicate", "nonfinite", "weights"])
def test_fail_closed(change):
    p = example()
    if change == "scope":
        p["space"]["scope"]["corpus_id"] = "different-corpus"
    elif change == "seed_reuse":
        p["design"]["seeds"] = [1, 2]
    elif change == "category":
        p["design"]["baseline"]["norm"] = "unknown"
    elif change == "duplicate":
        p["observations"]["blocks"].append(copy.deepcopy(p["observations"]["blocks"][0]))
    elif change == "nonfinite":
        p["observations"]["blocks"][0]["runs"][0]["value"] = float("nan")
    else:
        p["design"]["targets"][0]["terms"][0]["weight"] = 2.
    with pytest.raises(ValueError):
        D.run(p)


def test_illegal_candidate_rejected_and_resource_bound():
    p = example()
    p["space"]["base_spec"]["arch"]["custom"]["d"] = 65
    with pytest.raises(ValueError, match="no legal"):
        D.Space(p["space"])
    p = example()
    p["space"]["factors"]["d"] = list(range(64, 2048, 8))
    with pytest.raises(ValueError, match="resource bound"):
        D.Space(p["space"])
