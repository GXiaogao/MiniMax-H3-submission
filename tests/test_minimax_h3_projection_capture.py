import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/infinifabric-build-v1'


class ProjectionCaptureTests(unittest.TestCase):
    def setUp(self):
        path = FOLDER / 'diagnose_minimax_h3_projection.py'
        self.assertTrue(path.exists(), 'projection capture missing')
        spec = importlib.util.spec_from_file_location('projection_capture_test', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_weight_shards_slice_columns_before_transpose(self):
        weight = np.arange(4 * 16).reshape(4, 16)
        for rank in range(8):
            expected = weight[:, rank * 2:(rank + 1) * 2]
            np.testing.assert_array_equal(self.m.weight_shard(weight, 8, rank), expected)
            np.testing.assert_array_equal(self.m.weight_shard(weight, 8, rank, packed=True), expected.T)
        with self.assertRaises(ValueError):
            self.m.weight_shard(weight, 8, 8)

    def test_projection_roundtrip_layout_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            for packed in (False, True):
                stage = 'out_weight_packed' if packed else 'out_weight'
                shape = [896, 5376] if packed else [5376, 896]
                x = np.ones(shape, dtype=np.float32)
                self.m.save_projection(folder, stage, 8, 3, x, 'bfloat16')
                samples, meta = self.m.read_projection(folder, stage, 8, 3)
                self.assertEqual(meta['shape'], shape)
                np.testing.assert_array_equal(samples, np.ones(65536))
            path = folder / 'refiner0_out_weight.rank3.json'
            data = json.loads(path.read_text()); data['tp_size'] = 1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): self.m.read_projection(folder, 'out_weight', 8, 3)

    def test_output_samples_and_host_sum_use_same_coordinates(self):
        arrays = [np.full(65536, rank + .25, dtype=np.float32) for rank in range(8)]
        result = self.m.sum_local_samples(arrays)
        self.assertEqual(result.dtype, np.dtype('float64'))
        np.testing.assert_array_equal(result, np.full(65536, 30.))
        with self.assertRaises(ValueError): self.m.sum_local_samples(arrays[:-1])
        with self.assertRaises(ValueError): self.m.sum_local_samples(arrays[:-1] + [np.zeros(3)])

    def test_projection_invalid_dtype_nonfinite_and_missing_rank(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            x = np.zeros((1, 228, 5376), dtype=np.float32)
            self.m.save_projection(folder, 'out_local', 8, 0, x, 'bfloat16')
            with self.assertRaises(FileExistsError): self.m.save_projection(folder, 'out_local', 8, 0, x, 'bfloat16')
            x[:] = np.nan
            with self.assertRaises(ValueError): self.m.save_projection(folder, 'out_local', 8, 1, x, 'bfloat16')
            with self.assertRaises(FileNotFoundError): self.m.read_projection(folder, 'out_local', 8, 7)
            (folder / 'refiner0_out_local.rank0.f32').write_bytes(b'bad')
            with self.assertRaises(ValueError): self.m.read_projection(folder, 'out_local', 8, 0)

    def test_returned_binary_identity_and_failed_report(self):
        self.assertEqual(self.m.EXPECTED_EXTENSION, 'cbe5cad7446f359a39196bcae3fde17d08d818b5ba5c1c77778d75cc45595065')
        self.assertIn('20261106T162326Z-2351670', str(self.m.EXTENSION_RELATIVE))
        with tempfile.TemporaryDirectory() as d:
            rc = self.m.main(['--run-dir', d, '--work-home', d, '--local-target', d+'/local', '--model-path', d+'/model'])
            self.assertEqual(rc, 1)
            report = json.loads((Path(d)/'projection-parity.json').read_text())
            self.assertEqual(report['result'], 'FAILED')
            self.assertFalse(report['performance_measurement_valid'])
            self.assertEqual(report['completed_workers'], [])

    def test_full_reference_hook_multiplies_matching_unsampled_channel_shards(self):
        class Tensor:
            dtype = 'torch.float32'
            def __init__(self, array): self.array = np.asarray(array); self.shape = self.array.shape
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self.array
            def contiguous(self): return self
            def to(self, dtype): return self
            def __getitem__(self, key): return Tensor(self.array[key])
            def __add__(self, other): return Tensor(self.array+other.array)
        channels = np.arange(7168, dtype=np.float32)
        value = Tensor(np.broadcast_to(channels, (1, 228, 7168)))
        weight = Tensor(np.broadcast_to(channels, (5376, 7168)))
        calls = []
        def linear(x, w, bias):
            np.testing.assert_array_equal(x.array[0, 0], w.array[0])
            calls.append(x.shape[-1])
            scalar = np.dot(x.array[0, 0].astype('float64'), w.array[0].astype('float64'))
            return Tensor(np.broadcast_to(np.float32(scalar), (1, 228, 5376)))
        fake = types.SimpleNamespace(no_grad=contextlib.nullcontext,
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(linear=linear)))
        module = types.SimpleNamespace(weight=weight, bias=None)
        saved = []
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', {'torch': fake}), \
                patch.object(self.m, 'save_projection', side_effect=lambda *a: saved.append((a[1], a[2], a[3], a[4].shape))), \
                patch.object(self.m, 'save_projection_tensor', side_effect=lambda *a: saved.append((a[1], a[2], a[3], a[4].shape))):
            self.assertIsNone(self.m.reference_projection_before(Path(d))(module, (value,)))
            self.assertIsNone(self.m.reference_projection_after(Path(d))(module, (value,), value))
        self.assertEqual(calls, [7168]+[896]*8)
        self.assertIn(('out_weight_packed', 8, 7, (896, 5376)), saved)
        self.assertEqual(sum(stage=='out_local' for stage, *_ in saved), 9)
        self.assertEqual(sum(stage=='out_control_sum' for stage, *_ in saved), 9)

    def test_pinned_sources_quant_helpers_and_python_copy_checked(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);ext=work/self.m.EXTENSION_RELATIVE
            ext.parent.mkdir(parents=True);ext.write_bytes(b'fixture')
            run=work/self.m.BUILD_LOG_RELATIVE;run.mkdir(parents=True)
            (run/'artifacts.sha256').write_text(self.m.EXPECTED_EXTENSION+'  '+str(ext)+'\n')
            spec=importlib.util.spec_from_file_location('projection_patcher_test',FOLDER/'prepare_minimax_h3_projection_trace.py')
            patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
            source_root=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc'
            h3,meta=patcher.instrument_text((source_root/'models/minimax_h3/minimax_h3_transformer.cpp').read_bytes())
            quant=patcher.instrument_quant((source_root/'layers/quantization/none_quantization.cpp').read_bytes())
            (run/'minimax_h3_transformer.cpp').write_text(h3,encoding='utf-8',newline='\n')
            (run/'none_quantization.cpp').write_text(quant,encoding='utf-8',newline='\n')
            meta['quant_patched_sha256']=patcher.sha(quant.encode());meta['helper_hashes']={}
            for helper in patcher.HELPERS:
                data=(FOLDER/helper).read_bytes();(run/helper).write_bytes(data)
                meta['helper_hashes'][helper]=patcher.sha(data)
            (run/'minimax_h3_transformer.cpp.trace.json').write_text(json.dumps(meta))
            official=work/'Projects/InfiniLM/python/infinilm';official.mkdir(parents=True)
            (official/'__init__.py').write_text('# same');(ext.parents[1]/'__init__.py').write_text('# same')
            (work/'state').mkdir()
            marker=dict(build_version='1.0-isolated-h3-projection-trace',cxx11_abi='1',extension=str(ext),run_dir=str(run),artifacts_sha256=str(run/'artifacts.sha256'))
            (work/'state/infinilm-projection.success').write_text(''.join(f'{k}={v}\n' for k,v in marker.items()))
            real=self.m.base.sha256
            with patch.object(self.m.base,'sha256',side_effect=lambda p:self.m.EXPECTED_EXTENSION if Path(p).resolve()==ext.resolve() else real(p)):
                _,_,evidence=self.m.validate_diagnostic(work)
                self.assertEqual(evidence['python_files_checked'],1)
                self.assertEqual(evidence['extension_sha256'],self.m.EXPECTED_EXTENSION,
                    'worker must receive verified binary digest, never a helper header digest')
                (run/'none_quantization.cpp').write_text('drift')
                with self.assertRaisesRegex(ValueError,'quantization'): self.m.validate_diagnostic(work)

    def test_projection_comparison_reports_partial_sum_original_and_layout_separately(self):
        def read(folder,stage,tp,rank):
            value=30. if stage!='out_local' else rank+.25
            shape=[896,5376] if stage=='out_weight' and tp==8 else [5376,7168//tp] if stage.startswith('out_weight') else [1,228,5376]
            if stage=='out_weight_packed': shape=[7168//tp,5376]
            return np.full(65536,value,dtype=np.float32),dict(shape=shape,data_sha256='fixture')
        with patch.object(self.m,'read_projection',side_effect=read), \
                patch.object(self.m,'read_rank',return_value=(np.full(65536,30.),{})):
            result=self.m.compare_projection_run(Path('fixture'))
        rank=result['tp8']['7']
        self.assertTrue(rank['host_fp64_sum_vs_control_sum']['exact_equal'])
        self.assertTrue(rank['original_before_vs_after_control']['exact_equal'])
        self.assertEqual(rank['out_weight_vs_reference']['shape'],[896,5376])
        self.assertNotEqual(rank['host_fp64_sum_vs_original']['scope'],'all-elements')
