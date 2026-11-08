import importlib.util
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'artifacts/infinifabric-build-v1/diagnose_minimax_h3_prefix_parity.py'

class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), 'prefix diagnostic missing')
        spec=importlib.util.spec_from_file_location('prefix_test',SCRIPT)
        self.m=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)

    def test_small_reference_prefix_keeps_scatter_inputs_on_one_gpu(self):
        # Automatic balancing split token_refiner from indices in the actual run.
        with tempfile.TemporaryDirectory() as d:
            for count in (0,1):
                args=SimpleNamespace(prefix=count,run_dir=Path(d))
                calls=[]
                model=SimpleNamespace(transformer_blocks=[object() for _ in range(count)])
                def factory(*a,**kw):
                    calls.append((a,kw))
                    return model,{'missing_keys':[],'unexpected_keys':['transformer_blocks.49.norm1.weight']}
                got,info=self.m.load_reference_prefix(factory,args,'checkpoint',device_map='balanced',dtype='bfloat16',output_loading_info=True)
                self.assertIs(got,model)
                self.assertEqual(calls[0][1]['device_map'],{'':'cuda:0'})
                self.assertEqual(calls[0][1]['num_layers'],count)
                self.assertEqual(calls[0][1]['dtype'],'bfloat16')
                self.assertTrue(calls[0][1]['output_loading_info'])
                self.assertEqual(info['unexpected_keys'],[])
                raw=json.loads((Path(d)/'reference-prefix-loading.json').read_text())
                self.assertEqual(raw['unfiltered_loading_info']['unexpected_keys'],['transformer_blocks.49.norm1.weight'])

    def test_invalid_reference_prefix_cannot_load_full_model_on_one_gpu(self):
        called=[]
        with self.assertRaises(ValueError):
            self.m.load_reference_prefix(lambda **kw:called.append(kw),SimpleNamespace(prefix=50),'checkpoint')
        self.assertEqual(called,[])

    def test_only_excluded_blocks_are_allowed_as_unexpected(self):
        info={'unexpected_keys':['transformer_blocks.1.attn.to_q.weight','transformer_blocks.49.ff.net.2.weight'], 'missing_keys':[]}
        cleaned=self.m.validate_prefix_loading(info,1,50)
        self.assertEqual(cleaned['unexpected_keys'],[])
        self.assertEqual(len(info['unexpected_keys']),2)
        for key in ['proj_in.bias','transformer_blocks.0.attn.to_q.weight','transformer_blocks.50.attn.to_q.weight']:
            with self.assertRaises(ValueError):
                self.m.validate_prefix_loading({'unexpected_keys':[key]},1,50)

    def test_missing_or_mismatched_active_weights_fail(self):
        for field in ['missing_keys','mismatched_keys','error_msgs']:
            with self.assertRaises(ValueError):
                self.m.validate_prefix_loading({field:['bad']},0,50)

    def test_runtime_override_preserves_architecture_and_enables_filter(self):
        calls=[]
        def factory(*a,**kw): calls.append((a,kw)); return 'engine'
        wrapped=self.m.prefix_engine_factory(factory,1)
        self.assertEqual(wrapped('checkpoint',hf_config_overrides={'hidden_size':5376}),'engine')
        self.assertEqual(calls[0][1]['hf_config_overrides'],{'hidden_size':5376,'num_layers':1,'load_only_model_keys':True})

    def test_bad_prefix_rejected(self):
        for count in [-1,2,50]:
            with self.assertRaises(ValueError): self.m.prefix_engine_factory(lambda:None,count)

    def test_inputs_are_copied_only_after_manifest_validation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'src'; dst=Path(d)/'dst';src.mkdir();dst.mkdir()
            self.m.base.save_bundle(src,'first-inputs',{'a':np.array([2.0])})
            self.m.copy_inputs(src,dst)
            self.assertEqual((src/'first-inputs.npz').read_bytes(),(dst/'first-inputs.npz').read_bytes())
            (src/'first-inputs.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError): self.m.copy_inputs(src,dst)

    def test_failure_retains_failed_result(self):
        with tempfile.TemporaryDirectory() as d:
            args=['--run-dir',d,'--work-home',d,'--local-target',d+'/local','--model-path',d+'/model','--first-frame',d+'/image','--source-run',d+'/missing']
            self.assertEqual(self.m.main(args),1)
            result=json.loads((Path(d)/'prefix-parity.json').read_text())
            self.assertEqual(result['result'],'FAILED')
            self.assertEqual(result['completed_workers'],[])

