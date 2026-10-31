import json
import pytest
import finalize_sfull_operational_v2 as f


def save(tmp_path, seed=1):
    pt, sft = f.expected_specs(seed)
    base = tmp_path / f"c12_Sfull_r{seed}"
    a = base / f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json"
    b = base / f"post_c12_Sfull_r{seed}/sft_spec.json"
    for p, spec in [(a, pt), (b, sft)]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec))
    return a, b


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_amended_specs(seed, tmp_path):
    save(tmp_path, seed)
    f.validate_runtime_specs(tmp_path)
    pt, sft = f.expected_specs(seed)
    assert pt["train"]["checkpoint_every"] == sft["train"]["checkpoint_every"] == 25
    assert pt["train"]["seed"] == sft["train"]["seed"] == seed


@pytest.mark.parametrize("phase", [0, 1])
@pytest.mark.parametrize(
    "field", ["checkpoint_every", "steps", "lr", "seed", "batch", "accum", "extra"]
)
def test_spec_mutation_refused(tmp_path, phase, field):
    path = save(tmp_path)[phase]
    s = json.loads(path.read_text())
    s["train"][field] = 999
    path.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="runtime spec"):
        f.validate_runtime_specs(tmp_path)


def test_old_calculation_and_original_paths_remain_delegated():
    assert f.old.calculate.__code__.co_filename == str(
        f.EXPERIMENT / "finalize_sfull.py"
    )
    assert (
        f.old.REFERENCE.parent == f.EXPERIMENT / "receipts/sfull_prospective_20261031"
    )
    assert f.old.LAUNCH == f.PROOF / "launch_pin_v2.json"
    assert f.original_launch in f.sources()
