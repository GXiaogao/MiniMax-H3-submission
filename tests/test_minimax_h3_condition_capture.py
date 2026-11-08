import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'


def load(name):
    spec = importlib.util.spec_from_file_location('condition_' + name, FOLDER / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Node:
    def __init__(self, operation=lambda x: x):
        self.operation = operation
        self.pre = []
        self.post = []

    def register_forward_pre_hook(self, hook):
        self.pre.append(hook)
        return type('Handle', (), {'remove': lambda _: self.pre.remove(hook)})()

    def register_forward_hook(self, hook):
        self.post.append(hook)
        return type('Handle', (), {'remove': lambda _: self.post.remove(hook)})()

    def __call__(self, value):
        for hook in self.pre:
            hook(self, (value,))
        output = self.operation(value)
        for hook in self.post:
            hook(self, (value,), output)
        return output


class ConditionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((FOLDER / 'diagnose_minimax_h3_conditions.py').exists(), 'condition capture missing')
        self.m = load('diagnose_minimax_h3_conditions.py')

    def fixture(self, work):
        extension = work / self.m.EXTENSION_RELATIVE
        extension.parent.mkdir(parents=True)
        extension.write_bytes(b'fixture-extension')
        official = work / 'Projects/InfiniLM/python/infinilm'
        official.mkdir(parents=True)
        (official / '__init__.py').write_text('# identical\n')
        (extension.parents[1] / '__init__.py').write_text('# identical\n')
        run = work / self.m.BUILD_LOG_RELATIVE
        run.mkdir(parents=True)
        (run / 'artifacts.sha256').write_text(self.m.base.sha256(extension) + '  ' + str(extension) + '\n')
        patch = load('prepare_minimax_h3_condition_trace.py')
        source = ROOT / 'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'
        text, meta = patch.instrument_text(source.read_bytes())
        (run / 'minimax_h3_transformer.cpp').write_text(text, encoding='utf-8', newline='\n')
        (run / 'minimax_h3_transformer.cpp.trace.json').write_text(json.dumps(meta))
        marker = {'build_version': '1.0-isolated-h3-condition-trace', 'cxx11_abi': '1',
                  'extension': str(extension), 'run_dir': str(run), 'artifacts_sha256': str(run / 'artifacts.sha256')}
        (work / 'state').mkdir()
        (work / 'state/infinilm-condition.success').write_text(''.join(f'{k}={v}\n' for k, v in marker.items()))
        return extension, run

    def test_preflight_reads_digest_and_checks_source_and_python_copy(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            extension, run = self.fixture(work)
            actual, tree, evidence = self.m.validate_diagnostic(work)
            self.assertEqual(actual, extension.resolve())
            self.assertEqual(tree, extension.parents[2].resolve())
            self.assertEqual(evidence['extension_sha256'], self.m.base.sha256(extension))
            self.assertEqual(evidence['python_files_checked'], 1)
            (extension.parents[1] / '__init__.py').write_text('# changed')
            with self.assertRaisesRegex(ValueError, 'Python copy'):
                self.m.validate_diagnostic(work)

    def test_extension_and_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            extension, run = self.fixture(work)
            extension.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'extension.*hash'):
                self.m.validate_diagnostic(work)
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            extension, run = self.fixture(work)
            (run / 'minimax_h3_transformer.cpp').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source'):
                self.m.validate_diagnostic(work)

    def test_wrong_build_and_missing_or_duplicate_digest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            extension, run = self.fixture(work)
            manifest = run / 'artifacts.sha256'
            original = manifest.read_text()
            for content in ('', original + original):
                manifest.write_text(content)
                with self.assertRaises(ValueError):
                    self.m.validate_diagnostic(work)
            manifest.write_text(original)
            marker = work / 'state/infinilm-condition.success'
            marker.write_text(marker.read_text().replace('071847Z-74362', 'other-build'))
            with self.assertRaises(ValueError):
                self.m.validate_diagnostic(work)

    def test_refiner_hooks_record_distinct_blocks_and_correct_residual_stage(self):
        model = type('Model', (), {})()
        model.context_embedder = Node(lambda x: x * 5)
        model.token_refiner = type('Refiner', (), {})()
        blocks = []
        for i in range(2):
            block = Node()
            block.norm1 = Node(lambda x: x / 2)
            block.attn = Node(lambda x: x + 2)
            block.norm2 = Node(lambda x: x / 4)
            block.ff = Node(lambda x: x * 3)
            block.operation = lambda x, b=block: (lambda y: y + b.ff(b.norm2(y)))(x + b.attn(b.norm1(x)))
            blocks.append(block)
        model.token_refiner.refiner_blocks = blocks
        model.token_refiner.final_norm = Node(lambda x: x / 10)
        main = Node()
        main.adaln_proj = type('Projection', (), {'linear': Node()})()
        main.attn = Node()
        main.norm2 = Node()
        main.ff = Node()
        model.transformer_blocks = [main]
        model.time_embedder = Node()
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            handles = self.m.attach_reference(model, folder)
            x = np.ones((1, 2, 3), dtype=np.float32)
            y = model.context_embedder(x)
            for block in blocks:
                y = block(y)
            model.token_refiner.final_norm(y)
            a, _ = self.m.read_snapshot(folder, 'condition_0_after_attention')
            b, _ = self.m.read_snapshot(folder, 'condition_1_norm1')
            np.testing.assert_array_equal(a, np.full(6, 9.5))
            np.testing.assert_array_equal(b, np.full(6, 8.3125))
            self.assertEqual(len(list(folder.glob('*.json'))), 15)
            for handle in handles:
                handle.remove()
            self.assertFalse(blocks[0].norm2.pre)

    def test_failed_preflight_never_reports_comparison_complete(self):
        with tempfile.TemporaryDirectory() as d:
            result = self.m.main(['--run-dir', d, '--work-home', d, '--local-target', d + '/local', '--model-path', d + '/model'])
            self.assertEqual(result, 1)
            report = json.loads((Path(d) / 'condition-parity.json').read_text())
            self.assertEqual(report['result'], 'FAILED')
            self.assertEqual(report['completed_workers'], [])
