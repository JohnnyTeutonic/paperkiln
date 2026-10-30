"""The post-hoc readiness audit does not reinterpret the registered selection."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("readiness", Path(__file__).with_name("check_readiness.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadinessTests(unittest.TestCase):
    def test_extension_winner_does_not_repick(self):
        rows = [{"seed": 1, "primary": -.07}, {"seed": 2, "primary": -.06},
                {"seed": 3, "primary": -.05}, {"seed": 4, "primary": .5},
                {"seed": 5, "primary": .8}]
        self.assertEqual(module.registered_winner(rows)["seed"], 3)

    def test_tie_prefers_lower_seed(self):
        rows = [{"seed": 3, "primary": .1}, {"seed": 2, "primary": .1},
                {"seed": 1, "primary": .1}]
        self.assertEqual(module.registered_winner(rows)["seed"], 1)

    def test_incomplete_or_duplicate_initial_seed_refused(self):
        for rows in ([{"seed": 1, "primary": .1}],
                     [{"seed": 1, "primary": .1}] * 3,
                     [{"seed": r, "primary": None} for r in (1, 2, 3)]):
            with self.assertRaises(ValueError):
                module.registered_winner(rows)

    def test_only_checkout_line_endings_are_ignored(self):
        self.assertTrue(module.same_source_bytes(b'{\n "x": 1\n}\n', b'{\r\n "x": 1\r\n}\r\n'))
        self.assertFalse(module.same_source_bytes(b'{"x": 1}\n', b'{"x": 2}\n'))
        self.assertFalse(module.same_source_bytes(b'{"x": 1}\n', b'{"x":1}\n'))


if __name__ == "__main__":
    unittest.main()
