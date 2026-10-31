"""Offline proof-parser negatives: transport success cannot stand in for device/numeric proof."""
import importlib.util
import json
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('fallback_gate', Path(__file__).parents[1] / 'tools/fallback_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

def fixture(case, passing):
    items = [{'case': case, 'actual_cuda': True, 'stale_operand_proved': True,
              'host_poison': True, 'ops_disabled_before_fallback': True}]
    wanted = list(gate.quantities(case))
    if not passing:
        wanted = wanted[:1 if '_forward_' in case else 2]
    for q in wanted:
        good = passing or q != wanted[-1]
        items.append({'case': case, 'quantity': q, 'normalized_max_error': 0.0 if good else .1,
                      'numeric_reference_pass': good})
    if passing:
        items += [{'pass': True, 'case': case, 'actual_cuda': True, 'independent_forward_backward': True},
                  {'pass': True, 'actual_cuda': True, 'devcheck_enabled': True, 'cases': 1}]
    text = '\n'.join(map(json.dumps, items)) + '\n'
    if not passing:
        text += 'CHECK failed /content/source/tests/test_fallback_materialization.cpp:77: maximum < 2e-5\n'
    return text

class GateTests(unittest.TestCase):
    def test_registered_matrix(self):
        self.assertEqual(len(gate.NEGATIVE), 13)
        self.assertEqual(len(gate.CONTROLS), 3)
        self.assertEqual(len(set(gate.CASES)), 16)
        for case in gate.NEGATIVE:
            gate.check_baseline(1, fixture(case, False), case)
        for case in gate.CASES:
            gate.check_fixed(0, fixture(case, True), case)

    def test_baseline_wrong_status_or_case(self):
        for status, case in [(0, 'gelu_forward_x'), (2, 'gelu_forward_x'), (1, 'group_forward_x')]:
            with self.assertRaises(ValueError):
                gate.check_baseline(status, fixture(case, False), case)

    def test_baseline_wrong_assertion(self):
        with self.assertRaises(ValueError):
            gate.check_baseline(1, fixture('gelu_backward_x', False).replace('maximum < 2e-5', 'cuda_ready'), 'gelu_backward_x')

    def test_baseline_wrong_numeric_stage(self):
        with self.assertRaises(ValueError):
            gate.check_baseline(1, fixture('gelu_forward_x', False).replace('"forward"', '"input_gradient"'), 'gelu_forward_x')

    def test_baseline_wrong_threshold(self):
        with self.assertRaises(ValueError):
            gate.check_baseline(1, fixture('gelu_forward_x', False).replace('0.1', '0.000001'), 'gelu_forward_x')

    def test_actual_device_and_staleness_are_required(self):
        for key in ('actual_cuda', 'stale_operand_proved', 'host_poison', 'ops_disabled_before_fallback'):
            for passing in (True, False):
                text = fixture('gelu_forward_x', passing).replace('"' + key + '": true', '"' + key + '": false')
                with self.assertRaises(ValueError):
                    if passing:
                        gate.check_fixed(0, text, 'gelu_forward_x')
                    else:
                        gate.check_baseline(1, text, 'gelu_forward_x')

    def test_nonfinite_and_missing_diagnostics_refused(self):
        for value in ('NaN', 'Infinity', 'null', 'true', '-0.1'):
            text = fixture('gelu_forward_x', True).replace('"normalized_max_error": 0.0', '"normalized_max_error": ' + value)
            with self.assertRaises(ValueError):
                gate.check_fixed(0, text, 'gelu_forward_x')

    def test_fixed_needs_all_weight_gradient_checks(self):
        text = fixture('rmsnorm_backward_w', True)
        text = '\n'.join(x for x in text.splitlines() if 'weight_gradient' not in x)
        with self.assertRaises(ValueError):
            gate.check_fixed(0, text, 'rmsnorm_backward_w')

    def test_fixed_duplicate_check_refused(self):
        text = fixture('gelu_forward_x', True)
        duplicate = next(x for x in text.splitlines() if 'normalized_max_error' in x)
        with self.assertRaises(ValueError):
            gate.check_fixed(0, duplicate + '\n' + text, 'gelu_forward_x')

    def test_fixed_requires_devcheck_and_single_case(self):
        for old, new in [('"devcheck_enabled": true', '"devcheck_enabled": false'), ('"cases": 1', '"cases": 2')]:
            with self.assertRaises(ValueError):
                gate.check_fixed(0, fixture('gelu_forward_x', True).replace(old, new), 'gelu_forward_x')

    def test_transport_exit_zero_is_not_pass(self):
        with self.assertRaises((ValueError, IndexError)):
            gate.check_fixed(0, '{"pass": false}\n', 'gelu_forward_x')

if __name__ == '__main__':
    unittest.main()
