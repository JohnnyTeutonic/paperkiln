import importlib.util
import os
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "selective_replay", Path(__file__).parents[1] / "tools/replay_selective_scan.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
# Pass a receipt argument so tests bind to the actual produced fixture, not an invented expected row.

RAW = Path(
    os.environ.get(
        "SELECTIVE_SCAN_RAW",
        str(
            Path(__file__).parents[1]
            / "docs/receipts/selective_scan_20261031/cpu.jsonl"
        ),
    )
)


class Contracts(unittest.TestCase):
    def mutate(self, fn, gpu=False):
        rows = [
            json.loads(x) for x in RAW.read_text().splitlines() if x.startswith("{")
        ]
        fn(rows)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tamper.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            with self.assertRaises((ValueError, TypeError, KeyError, IndexError)):
                replay.replay(p, gpu)

    def test_actual_cpu(self):
        self.assertEqual(replay.replay(RAW)["cases"], 220)

    def test_cpu_is_not_gpu(self):
        with self.assertRaises(ValueError):
            replay.replay(RAW, True)

    def test_missing_devcheck(self):
        self.mutate(lambda r: r[-1].update(devcheck=False))

    def test_missing_case(self):
        self.mutate(lambda r: r.pop(0))

    def test_duplicate(self):
        self.mutate(lambda r: r.insert(0, r[0]))

    def test_nonfinite(self):
        self.mutate(lambda r: r[0]["y"][0].__setitem__(0, float("nan")))

    def test_wrong_output(self):
        self.mutate(lambda r: r[0]["y"][0].__setitem__(0, 99))

    def test_wrong_shape(self):
        self.mutate(lambda r: r[0]["x"][0].append(0))

    def test_fixture_substitution(self):
        self.mutate(lambda r: r[0]["x"][0].__setitem__(0, 0))

    def test_wrong_kernel_count(self):
        self.mutate(lambda r: r[0].update(device_fwd=1))

    def test_wrong_mask(self):
        self.mutate(lambda r: r[0].update(mask=128))

    def test_missing_gradient(self):
        self.mutate(lambda r: next(x for x in r if "dx" in x).pop("dx"))

    def test_unrequested_gradient(self):
        self.mutate(lambda r: r[0].update(dx=r[0]["x"]))

    def test_wrong_case_device(self):
        self.mutate(lambda r: r[0].update(actual_cuda=True))

    def test_wrong_mode(self):
        self.mutate(lambda r: r[0].update(mode="invented"))


if __name__ == "__main__":
    unittest.main()
