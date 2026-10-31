import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "attention_replay",
    Path(__file__).parents[1] / "tools/replay_attention_precision.py",
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
RAW = Path(__file__).parents[1] / "docs/receipts/attention_precision_20261031/cpu.jsonl"


class ReplayContracts(unittest.TestCase):
    def mutate(self, fn, gpu=False):
        rows = [
            json.loads(x) for x in RAW.read_text().splitlines() if x.startswith("{")
        ]
        fn(rows)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            with self.assertRaises((ValueError, TypeError, KeyError, IndexError)):
                replay.replay(p, gpu)

    def test_actual_cpu_receipt(self):
        self.assertEqual(replay.replay(RAW)["cases"], 44)

    def test_cpu_is_not_gpu(self):
        with self.assertRaises(ValueError):
            replay.replay(RAW, True)

    def test_missing_devcheck(self):
        self.mutate(lambda r: r[-1].update(devcheck=False))

    def test_missing_case(self):
        self.mutate(lambda r: r.pop(0))

    def test_duplicate_case(self):
        self.mutate(lambda r: r.insert(0, r[0]))

    def test_wrong_output(self):
        self.mutate(
            lambda r: next(x for x in r if x.get("raw_case"))["y"][0].__setitem__(0, 99)
        )

    def test_nonfinite(self):
        self.mutate(
            lambda r: next(x for x in r if x.get("raw_case"))["y"][0].__setitem__(
                0, float("nan")
            )
        )

    def test_fake_device_count(self):
        self.mutate(
            lambda r: next(x for x in r if x.get("raw_case")).update(
                measured_device_gemms=6
            )
        )

    def test_wrong_gradient_mask(self):
        self.mutate(
            lambda r: next(x for x in r if x.get("raw_case")).update(grad_mask=8)
        )

    def test_wrong_shapes(self):
        self.mutate(lambda r: next(x for x in r if x.get("raw_case"))["q"].append([0]))

    def test_false_metric(self):
        self.mutate(
            lambda r: next(x for x in r if x.get("case")).update(worst=float("inf"))
        )

    def test_independent_rounding_ties(self):
        import numpy as np

        vals = np.array(
            [0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000], dtype=np.uint32
        ).view(np.float32)
        expected = np.array(
            [0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000], dtype=np.uint32
        ).view(np.float32)
        np.testing.assert_array_equal(replay.quantized(vals, True), expected)


if __name__ == "__main__":
    unittest.main()
