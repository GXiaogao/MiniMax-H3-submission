import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'


class FullForwardLoadSyncTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'diagnose_minimax_h3_full_forward_load_sync.py'
        self.assertTrue(path.exists(),'full-forward capture missing')
        spec=importlib.util.spec_from_file_location('full_forward_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_full_engine_never_truncates_layers_and_disables_graphs(self):
        calls=[]
        factory=self.m.full_engine_factory(lambda *a,**kw:calls.append(kw))
        factory('model',hf_config_overrides={'skip_sampling':True},enable_graph_compiling=True)
        self.assertFalse(calls[0]['enable_graph_compiling'])
        self.assertNotIn('num_layers',calls[0]['hf_config_overrides'])
        with self.assertRaises(ValueError):factory('model',hf_config_overrides={'num_layers':1})

    def test_native_full_keys_and_trace_disabling(self):
        keys=[f'transformer_blocks.{i}.attn.to_q.weight' for i in range(50)]
        self.m.validate_native_full(keys,{'num_layers':50})
        with self.assertRaises(ValueError):self.m.validate_native_full(keys[:-1],{'num_layers':50})
        with self.assertRaises(ValueError):self.m.validate_native_full(keys,{'num_layers':1})
        with patch.dict('os.environ',{'H3_BOUNDARY_TRACE_DIR':'old'}):
            self.m.disable_trace()
            import os
            self.assertNotIn('H3_BOUNDARY_TRACE_DIR',os.environ)

    def test_full_outputs_require_both_complete_shapes(self):
        arrays=dict(video=np.zeros((1,8208,96),dtype='float32'),audio=np.zeros((1,414,32),dtype='float32'))
        self.m.validate_outputs(arrays)
        for bad in ({'video':arrays['video']},{**arrays,'audio':arrays['audio'][:,:1]}):
            with self.assertRaises(ValueError):self.m.validate_outputs(bad)
        arrays['video'][0,0,0]=np.nan
        with self.assertRaises(ValueError):self.m.validate_outputs(arrays)

    def test_actual_optional_trace_gate_allows_full_config_when_disabled(self):
        spec=importlib.util.spec_from_file_location('full_candidate_patch',FOLDER/'prepare_minimax_h3_projection_trace.py')
        patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
        source=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc/models/minimax_h3/minimax_h3_transformer.cpp'
        text,_=patcher.instrument_text(source.read_bytes())
        self.assertIn('if (boundary_trace::enabled() && transformer_blocks_.size()!=1)',text)
        self.assertIn('if (!active) return;', (FOLDER/'h3_projection_trace.hpp').read_text())
        self.assertIn('active = first_refiner && enabled();',(FOLDER/'h3_attention_trace.hpp').read_text())

    def test_returned_prefix_provenance_and_failure_report(self):
        archived=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/load-sync-result-20261107T034425Z-2750046/load-sync-parity.json'
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);path=work/self.m.PREFIX_RELATIVE;path.parent.mkdir(parents=True);path.write_bytes(archived.read_bytes())
            evidence=self.m.validate_prefix_result(work)
            self.assertEqual(evidence['report_sha256'],self.m.PREFIX_SHA)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):self.m.validate_prefix_result(work)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.m.main(['--run-dir',d,'--work-home',d,'--local-target',d+'/local','--model-path',d+'/model']),1)
            report=json.loads((Path(d)/'full-forward-parity.json').read_text())
            self.assertEqual(report['result'],'FAILED')
            self.assertEqual(report['completed_workers'],[])
            self.assertFalse(report['performance_measurement_valid'])

    def test_reference_factory_preserves_full_loading_and_checks_implementation(self):
        model=SimpleNamespace(transformer_blocks=list(range(50)),config=SimpleNamespace(num_layers=50))
        calls=[]
        def load(*a,**kw):calls.append(kw);return model,{}
        factory=self.m.full_reference_factory(load,lambda model:'pinned',expected_sha='pinned')
        self.assertEqual(factory('model',device_map='balanced')[0],model)
        self.assertEqual(calls[0],{'device_map':'balanced'})
        model.transformer_blocks=model.transformer_blocks[:1]
        with self.assertRaises(ValueError):factory('model')

    def test_loaded_extension_requires_exact_candidate_path_and_digest(self):
        with tempfile.TemporaryDirectory() as d:
            binary=Path(d)/'candidate.so';binary.write_bytes(b'fixture')
            args=SimpleNamespace(diag_extension=binary,extension_sha256=self.m.capture.EXPECTED_EXTENSION)
            with patch.object(self.m.base,'sha256',return_value=self.m.capture.EXPECTED_EXTENSION):
                self.m.validate_loaded_extension(binary,args)
                with self.assertRaises(RuntimeError):self.m.validate_loaded_extension(binary.with_name('other.so'),args)
                args.extension_sha256='wrong'
                with self.assertRaises(RuntimeError):self.m.validate_loaded_extension(binary,args)

    def test_worker_not_complete_until_full_model_metadata_and_outputs_validate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'native';folder.mkdir()
            model=dict(main_blocks=50,tp_size=8,expected_keys=[f'transformer_blocks.{i}.weight' for i in range(50)],
                config={'num_layers':50},trace_enabled=False,graphs_enabled=False,extension_sha256=self.m.capture.EXPECTED_EXTENSION)
            path=folder/'native-full-model.json';path.write_text(json.dumps(model))
            report=dict(completed_workers=[])
            with patch.object(self.m.base,'read_bundle',return_value=({'video':np.zeros((1,8208,96),dtype='float32')},{'sha256':'fixture'})):
                with self.assertRaises(ValueError):self.m.complete_worker(root,'native',report,lambda:None)
            self.assertEqual(report['completed_workers'],[])
            arrays=dict(video=np.zeros((1,8208,96),dtype='float32'),audio=np.zeros((1,414,32),dtype='float32'))
            with patch.object(self.m.base,'read_bundle',return_value=(arrays,{'sha256':'fixture'})):
                self.m.complete_worker(root,'native',report,lambda:None)
            self.assertEqual(report['completed_workers'],['native'])
            self.assertEqual(report['worker_models']['native']['main_blocks'],50)
