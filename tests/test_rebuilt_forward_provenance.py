import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'artifacts/infinifabric-build-v1/rebuild-forward-v1/diagnose_minimax_h3_full_forward_load_sync.py'

class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('rebuilt_test',FILE)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
    def test_historical_report_keeps_original_candidate_identity(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);p=work/m.PREFIX_RELATIVE;p.parent.mkdir(parents=True)
            report=dict(result='COMPARISON_COMPLETE',completed_workers=['native-tp1','native-tp8','reference'],input_sha256=m.INPUT_SHA,
              diagnostic_build=dict(extension_sha256='ee448772f8e03947f6f33fdaa670fd4079bdd88df31f4e2655ce405aa0b34047'),
              loading_comparison_summary=dict(weight_samples_exact_ranks_after=8))
            p.write_text(json.dumps(report),encoding='utf-8')
            with patch.object(m.base,'sha256',return_value=m.PREFIX_SHA):
                m.validate_prefix_result(work)
                report['diagnostic_build']['extension_sha256']=m.capture.EXPECTED_EXTENSION
                p.write_text(json.dumps(report),encoding='utf-8')
                with self.assertRaises(ValueError):m.validate_prefix_result(work)
    def test_worker_rejects_old_binary_identity(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'extension.so'
            args=SimpleNamespace(diag_extension=p,extension_sha256=m.capture.EXPECTED_EXTENSION)
            with patch.object(m.base,'sha256',return_value=m.capture.EXPECTED_EXTENSION):
                m.validate_loaded_extension(p,args)
                args.extension_sha256='ee448772f8e03947f6f33fdaa670fd4079bdd88df31f4e2655ce405aa0b34047'
                with self.assertRaises(RuntimeError):m.validate_loaded_extension(p,args)

if __name__=='__main__':unittest.main()
