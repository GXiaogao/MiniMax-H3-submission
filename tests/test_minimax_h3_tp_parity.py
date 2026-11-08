import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'artifacts/infinifabric-build-v1/diagnose_minimax_h3_prefix_parity.py'

class TPTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('tp_test_prefix',SCRIPT)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_only_supported_tp_counts_can_select_gpus(self):
        self.assertEqual(self.m.diagnostic_devices(1),[0])
        self.assertEqual(self.m.diagnostic_devices(8),list(range(8)))
        for count in (0,2,50):
            with self.assertRaises(ValueError):self.m.diagnostic_devices(count)

    def test_triplet_metrics_keep_reference_denominator_and_output_names(self):
        ref={k:np.array([1.,2.]) for k in ('video','audio')}
        eight={k:np.array([1.,3.]) for k in ref}
        result=self.m.compare_tp_outputs(ref,eight,ref)
        self.assertEqual(result['tp1_vs_reference']['video']['relative_l2'],0)
        self.assertAlmostEqual(result['tp8_vs_reference']['audio']['relative_l2'],1/np.sqrt(5))
        self.assertAlmostEqual(result['tp1_vs_tp8']['video']['relative_l2'],1/np.sqrt(10))
        with self.assertRaises(ValueError):self.m.compare_tp_outputs(ref,{},ref)

    def fixture(self,root):
        out=root/'prefix-1';out.mkdir()
        meta=self.m.base.save_bundle(out,'first-inputs',{'a':np.array([1.])})
        for name in ('first-native-outputs','first-reference-outputs'):
            self.m.base.save_bundle(out,name,{k:np.array([1.,2.]) for k in ('video','audio')})
        self.m.base.write_json(root/'prefix-parity.json',{'result':'COMPARISON_COMPLETE',
            'completed_workers':['prefix-0-native','prefix-0-reference','prefix-1-native','prefix-1-reference'],
            'input_sha256':meta['sha256'],'extension_sha256':'extension'})
        return meta['sha256']

    def test_baseline_validates_bytes_and_run_provenance_before_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.fixture(root)
            result=self.m.load_tp_baseline(root,sha,'extension')
            self.assertEqual(result['input_sha256'],sha)
            self.assertEqual(result['reference']['video'].tolist(),[1.,2.])
            with self.assertRaises(ValueError):self.m.load_tp_baseline(root,sha,'changed')
            with self.assertRaises(ValueError):self.m.load_tp_baseline(root,'wrong','extension')
            (root/'prefix-1/first-native-outputs.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):self.m.load_tp_baseline(root,sha,'extension')

    def test_incomplete_baseline_is_not_usable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sha=self.fixture(root)
            path=root/'prefix-parity.json';report=json.loads(path.read_text())
            report['completed_workers'].pop();self.m.base.write_json(path,report)
            with self.assertRaises(ValueError):self.m.load_tp_baseline(root,sha,'extension')

    def test_single_worker_command_and_failure_do_not_claim_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();self.fixture(source)
            args=SimpleNamespace(run_dir=root/'new',model_path=root/'model',baseline_run=source)
            args.run_dir.mkdir()
            report={'completed_workers':[],'result':'FAILED'};calls=[]
            def fail(command,log,env):calls.append(command);raise RuntimeError('device failure')
            real=self.m.base.run_worker;self.m.base.run_worker=fail
            baseline={'reference':{},'native8':{}}
            try:
                with self.assertRaisesRegex(RuntimeError,'device failure'):
                    self.m.run_tp_comparison(args,report,lambda:None,baseline,root/'example')
            finally:self.m.base.run_worker=real
            self.assertEqual(len(calls),1)
            self.assertIn('--tp-size',calls[0])
            self.assertEqual(calls[0][calls[0].index('--tp-size')+1],'1')
            self.assertEqual(report['completed_workers'],[])
            self.assertEqual(report['result'],'FAILED')
