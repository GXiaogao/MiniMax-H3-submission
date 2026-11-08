"""Diagnostic observers must preserve calls, signatures and returned tensors."""
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'artifacts/infinifabric-build-v1/diagnose_minimax_h3_native_chain.py'


class NativeChainTraceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), 'Native chain diagnostic is missing')
        spec = importlib.util.spec_from_file_location('native_chain_test', MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.trace = self.mod.ChainTrace(self.root)

    def events(self):
        return [json.loads(line) for line in (self.root / 'native-chain.jsonl').read_text().splitlines()]

    def test_tensor_summary_bounded_sampling_and_nonfinite_outside_sample(self):
        value = np.arange(200000, dtype=np.float32)
        report = self.mod.tensor_summary(value)
        self.assertLessEqual(report['sample_count'], 65536)
        self.assertEqual(report['numel'], 200000)
        self.assertTrue(report['all_finite'])
        value[1] = np.nan  # not a sampled index when stride > 1
        report = self.mod.tensor_summary(value)
        self.assertFalse(report['all_finite'])
        json.dumps(report, allow_nan=False)

    def test_wrapper_preserves_signature_identity_and_values(self):
        output = np.ones((2, 3), dtype=np.float32)
        def forward(hidden_states, *, position_ids, return_dict=False):
            self.assertIs(hidden_states, inputs)
            self.assertIs(position_ids, positions)
            return (output,)
        owner = SimpleNamespace(forward=forward)
        signature = inspect.signature(owner.forward)
        inputs = np.arange(6).reshape(2, 3)
        positions = np.array([1, 2])
        self.trace.wrap(owner, 'forward', 'transformer')
        self.assertEqual(inspect.signature(owner.forward), signature)
        actual = owner.forward(inputs, position_ids=positions)
        self.assertIs(actual[0], output)
        np.testing.assert_array_equal(inputs, np.arange(6).reshape(2, 3))
        events = self.events()
        self.assertEqual([x['status'] for x in events], ['START', 'DONE'])
        self.assertEqual(events[0]['inputs']['position_ids']['shape'], [2])
        self.trace.close()
        self.assertIs(owner.forward, forward)

    def test_exception_is_retained_and_propagated_without_retry(self):
        error = ValueError('upstream failure')
        calls = []
        def bad(value):
            calls.append(value)
            raise error
        owner = SimpleNamespace(forward=bad)
        self.trace.wrap(owner, 'forward', 'transformer')
        with self.assertRaises(ValueError) as caught:
            owner.forward(np.zeros(1))
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.events()[-1]['status'], 'FAILED')
        self.assertIn('upstream failure', self.events()[-1]['error'])

    def test_scheduler_delta_uses_snapshot_even_if_input_mutated(self):
        def step(model_output, timestep, sample, return_dict=False):
            sample -= model_output
            return (sample,)
        owner = SimpleNamespace(step=step)
        self.trace.wrap(owner, 'step', 'scheduler.video', track_update=True)
        result = owner.step(np.ones(4), 1.0, np.full(4, 5.0))
        np.testing.assert_array_equal(result[0], np.full(4, 4.0))
        change = self.events()[-1]['update']
        self.assertEqual(change['max_abs_delta'], 1.0)
        self.assertEqual(change['rms_delta'], 1.0)

    def test_qwen_records_selected_hidden_layer_and_not_every_layer(self):
        hidden = tuple(np.full((1, 2, 3), i, dtype=float) for i in range(52))
        result = SimpleNamespace(hidden_states=hidden, last_hidden_state=hidden[-1])
        summary = self.mod.summarize_output(result)
        self.assertEqual(summary['hidden_state_50']['sample_mean'], 50)
        self.assertEqual(summary['hidden_state_count'], 52)
        self.assertNotIn('hidden_state_49', summary)

    def test_observer_failure_never_silently_claims_completion(self):
        def unsupported():
            return np.array([np.inf])
        owner = SimpleNamespace(forward=unsupported)
        self.trace.wrap(owner, 'forward', 'transformer')
        owner.forward()
        self.trace.finish(0)
        report = json.loads((self.root / 'native-chain-summary.json').read_text())
        self.assertEqual(report['quality_status'], 'UNASSESSED')
        self.assertTrue(report['nonfinite_observations'])
        self.assertFalse(report['performance_measurement_valid'])

    def test_snapshots_keep_exact_values_and_incomplete_run_is_not_complete(self):
        value = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 2, 2)
        self.trace.save_tensor('decoder-input-001', value)
        with np.load(self.root / 'decoder-input-001.npz') as saved:
            np.testing.assert_array_equal(saved['tensor'], value)
        self.assertEqual(self.trace.finish(0), 2)
        report = json.loads((self.root / 'native-chain-summary.json').read_text())
        self.assertEqual(report['result'], 'INCOMPLETE')
        self.assertIn('transformer', report['missing_stages'])

    def test_official_invocation_restores_argv_and_loader_after_failure(self):
        pipe, encoder, transformer = object(), object(), object()
        calls = []
        def load(*args, **kwargs):
            calls.append('load')
            return pipe, encoder, transformer
        module = SimpleNamespace(load_pipeline=load)
        def run():
            actual = module.load_pipeline(legacy=False)
            self.assertEqual(actual, (pipe, encoder, transformer))
            raise RuntimeError('inference failure')
        module.main = run
        common = SimpleNamespace(load_official_example=lambda path: module,
                                 install_official_stage_tracing=lambda module: None)
        self.trace.attach_pipeline = lambda *items: calls.append(items)
        original_argv = sys.argv[:]
        with self.assertRaisesRegex(RuntimeError, 'inference failure'):
            self.mod.invoke_observed_example(common, self.trace, Path('example.py'), ['fl2va'])
        self.assertEqual(calls, ['load', (pipe, encoder, transformer)])
        self.assertEqual(sys.argv, original_argv)
        self.assertIs(module.load_pipeline, load)

    def test_legacy_never_reaches_the_model_loader(self):
        calls = []
        module = SimpleNamespace(load_pipeline=lambda **kwargs: calls.append(kwargs))
        module.main = lambda: module.load_pipeline(legacy=True)
        common = SimpleNamespace(load_official_example=lambda path: module,
                                 install_official_stage_tracing=lambda module: None)
        with self.assertRaisesRegex(RuntimeError, 'refuses the legacy backend'):
            self.mod.invoke_observed_example(common, self.trace, Path('example.py'), ['fl2va', '--legacy'])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
