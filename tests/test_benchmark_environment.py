import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/benchmark_environment.py"
spec = importlib.util.spec_from_file_location("benchmark_environment", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"os": {"id": "ubuntu", "version": "24.04"},
                         "compiler": {"version": "g++13", "target": "x86_64-linux-gnu"},
                         "cpu": {"architecture": "x86_64", "model": "cpuA", "flags": ["avx2"],
                                 "logical_cores": 4, "usable_cores": 4},
                         "threads": {key: None for key in m.THREAD_KEYS}, "build_type": "Release",
                         "effective_flags": {"attention": "-O3 -fopenmp"}, "libc": ["glibc", "2.39"]}

    def test_canonical_order(self):
        reverse = dict(reversed(list(self.identity.items())))
        self.assertEqual(m.report(reverse), m.report(self.identity))

    def test_actual_environment_changes_partition_history(self):
        for path, value in [(('os', 'version'), '26.04'), (('compiler', 'version'), 'g++15'),
                            (('compiler', 'target'), 'arm-linux'), (('cpu', 'model'), 'cpuB'),
                            (('cpu', 'flags'), ['avx512']), (('cpu', 'logical_cores'), 8),
                            (('cpu', 'usable_cores'), 2), (('threads', 'OMP_NUM_THREADS'), '1'),
                            (('threads', 'OMP_THREAD_LIMIT'), '1'), (('threads', 'OMP_MAX_ACTIVE_LEVELS'), '1'),
                            (('threads', 'GOMP_CPU_AFFINITY'), '0'), (('effective_flags', 'attention'), '-O0')]:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.identity)
                changed[path[0]][path[1]] = value
                self.assertNotEqual(m.report(changed)['cohort'], m.report(self.identity)['cohort'])

    def test_fresh_and_matching_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = m.report(self.identity)
            m.validate_cache(tmp, current)
            Path(tmp, 'benchmark-data.json').write_text('[]')
            m.validate_cache(tmp, current)
            self.assertEqual(json.loads(Path(tmp, 'environment.json').read_text()), current)

    def test_unknown_history_refused_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp, 'benchmark-data.json')
            history.write_text('["original"]')
            with self.assertRaisesRegex(ValueError, 'no environment'):
                m.validate_cache(tmp, m.report(self.identity))
            self.assertEqual(history.read_text(), '["original"]')

    def test_foreign_cache_refused_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = m.report(self.identity)
            m.validate_cache(tmp, current)
            original = Path(tmp, 'environment.json').read_bytes()
            other = copy.deepcopy(self.identity)
            other['compiler']['version'] = 'g++15'
            with self.assertRaisesRegex(ValueError, 'differs'):
                m.validate_cache(tmp, m.report(other))
            self.assertEqual(Path(tmp, 'environment.json').read_bytes(), original)

    def test_workflow_cohort_and_threshold_contract(self):
        text = (SCRIPT.parents[1] / 'workflows/benchmarks.yml').read_text()
        self.assertEqual(text.count('key: benchmark-data-v2-${{ steps.benchmark_env.outputs.cohort }}-${{ github.run_id }}'), 2)
        self.assertIn('restore-keys: benchmark-data-v2-${{ steps.benchmark_env.outputs.cohort }}-', text)
        self.assertNotIn('restore-keys: benchmark-data-\n', text)
        self.assertIn("alert-threshold: '250%'", text)
        self.assertIn('fail-on-alert: true', text)
        self.assertIn('--validate-cache bench-cache', text)


if __name__ == '__main__':
    unittest.main()
