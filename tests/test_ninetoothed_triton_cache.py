import errno
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'
WHEEL = Path('E:/projects/minimax-h3-match/相关代码文件/wheel/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl')


class TritonCachePatchTests(unittest.TestCase):
    def setUp(self):
        s = importlib.util.spec_from_file_location('cache_installer_test', FOLDER/'install_ninetoothed_gfx936.py')
        self.m = importlib.util.module_from_spec(s);s.loader.exec_module(self.m)
        self.assertTrue(hasattr(self.m, 'patch_triton_cache'), 'cache compatibility patch missing')

    def source(self):
        if not WHEEL.is_file():self.skipTest('pinned vendor wheel unavailable')
        with zipfile.ZipFile(WHEEL) as z:return z.read('triton/runtime/cache.py')

    def test_real_vendor_put_uses_same_directory_rename_and_preserves_groups(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = root/'cache.py';source.write_bytes(self.source())
            evidence = self.m.patch_triton_cache(source)
            self.assertEqual(evidence['patched_sha256'], self.m.sha(source))
            stub = ModuleType('triton');stub.__version__ = 'test'
            stub.knobs = SimpleNamespace(cache=SimpleNamespace(dir=str(root/'files')))
            spec = importlib.util.spec_from_file_location('vendor_cache_test', source)
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {'triton': stub}):spec.loader.exec_module(module)
            manager = module.FileCacheManager('key')
            real_replace = os.replace
            def forbid_cross_directory(src, dst):
                if Path(src).parent != Path(dst).parent:raise OSError(errno.EXDEV, 'nested rename rejected')
                return real_replace(src, dst)
            with patch.object(module.os, 'replace', side_effect=forbid_cross_directory):
                binary = manager.put(b'compiled-bytes', 'hip_utils.so')
                text = manager.put('metadata', 'kernel.json', binary=False)
                manager.put_group('group', {'binary': binary, 'text': text})
                self.assertEqual(manager.get_group('group'), {'binary': binary, 'text': text})
                manager.put(b'replaced', 'hip_utils.so')
            self.assertEqual(Path(binary).read_bytes(), b'replaced')
            self.assertEqual(Path(text).read_text(), 'metadata')
            self.assertEqual(list(Path(manager.cache_dir).glob('.tmp.pid_*')), [])

    def test_source_drift_and_repeat_patch_rejected_with_original_backup(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/'cache.py';data = self.source();source.write_bytes(data)
            self.m.patch_triton_cache(source)
            self.assertEqual(source.with_suffix('.py.before-cache-patch').read_bytes(), data)
            with self.assertRaises(ValueError):self.m.patch_triton_cache(source)
            source.write_bytes(data+b'\n# drift')
            with self.assertRaises(ValueError):self.m.patch_triton_cache(source)
