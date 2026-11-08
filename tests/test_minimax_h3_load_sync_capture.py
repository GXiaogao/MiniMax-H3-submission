import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'


class LoadSyncCaptureTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'diagnose_minimax_h3_load_sync.py'
        self.assertTrue(path.exists(),'candidate capture missing')
        spec=importlib.util.spec_from_file_location('load_sync_capture_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def fixture(self,work):
        ext=work/self.m.EXTENSION_RELATIVE;ext.parent.mkdir(parents=True);ext.write_bytes(b'fixture binary')
        run=work/self.m.BUILD_LOG_RELATIVE;run.mkdir(parents=True)
        (run/'artifacts.sha256').write_text(self.m.EXPECTED_EXTENSION+'  '+str(ext)+'\n')
        spec=importlib.util.spec_from_file_location('candidate_patch',FOLDER/'prepare_minimax_h3_load_sync_trace.py')
        patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
        from tests.test_minimax_h3_load_sync_patch import LoadSyncPatchTests
        with contextlib.redirect_stdout(io.StringIO()):rank_data=LoadSyncPatchTests().post_base_rank()
        source=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/csrc'
        h3=run/'minimax_h3_transformer.cpp';h3.write_bytes((source/'models/minimax_h3/minimax_h3_transformer.cpp').read_bytes())
        quant=run/'none_quantization.cpp';quant.write_bytes((source/'layers/quantization/none_quantization.cpp').read_bytes())
        rank=run/'rank_worker.cpp';rank.write_bytes(rank_data)
        patcher.patch_files(h3,quant,rank)
        official=work/'Projects/InfiniLM/python/infinilm';official.mkdir(parents=True)
        (official/'__init__.py').write_text('# same');(ext.parents[1]/'__init__.py').write_text('# same')
        (work/'state').mkdir()
        marker=dict(build_version='1.0-isolated-h3-load-sync-projection-trace',cxx11_abi='1',extension=str(ext),run_dir=str(run),artifacts_sha256=str(run/'artifacts.sha256'))
        (work/'state/infinilm-load-sync.success').write_text(''.join(f'{k}={v}\n' for k,v in marker.items()))
        hashes={ext.resolve():self.m.EXPECTED_EXTENSION}
        for name,digest in self.m.EXPECTED_CORE_LIBRARIES.items():
            path=work/'.infini/lib'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture core')
            hashes[path.resolve()]=digest
        return ext,run,hashes

    def test_valid_candidate_returns_binary_digest_not_helper_digest(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);ext,run,hashes=self.fixture(work);real=self.m.base.sha256
            with patch.object(self.m.base,'sha256',side_effect=lambda p:hashes.get(Path(p).resolve(),real(p))):
                actual,_,evidence=self.m.validate_diagnostic(work)
            self.assertEqual(actual,ext.resolve())
            self.assertEqual(evidence['extension_sha256'],self.m.EXPECTED_EXTENSION)
            self.assertEqual(evidence['rank_trace_sha256'],self.m.RANK_SHA)
            self.assertEqual(evidence['core_library_file_hashes'],self.m.EXPECTED_CORE_LIBRARIES)

    def test_rank_helper_and_library_drift_rejected(self):
        for name in ('rank_worker.cpp','h3_weight_load_sync.hpp','libinfinicore_cpp_api.so'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                work=Path(d);ext,run,hashes=self.fixture(work);real=self.m.base.sha256
                target=(work/'.infini/lib'/name) if name.endswith('.so') else run/name
                target.write_bytes(b'drift');hashes.pop(target.resolve(),None)
                with patch.object(self.m.base,'sha256',side_effect=lambda p:hashes.get(Path(p).resolve(),real(p))):
                    with self.assertRaises(ValueError):self.m.validate_diagnostic(work)

    def test_pinned_baseline_and_summary_report_samples_only(self):
        archived=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/projection-result-20261107T015112Z-2822929/projection-parity.json'
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);path=work/self.m.BASELINE_RELATIVE;path.parent.mkdir(parents=True);path.write_bytes(archived.read_bytes())
            baseline,evidence=self.m.load_baseline(work)
            current=json.loads(json.dumps(baseline))
            for metric in current['projection_comparison']['tp8'].values():
                metric['out_weight_vs_reference'].update(exact_equal=True,relative_l2=0.)
            summary=self.m.loading_summary(current,baseline)
            self.assertEqual(summary['weight_samples_exact_ranks_before'],2)
            self.assertEqual(summary['weight_samples_exact_ranks_after'],8)
            self.assertEqual(summary['scope'],'projection-saved-samples-not-full-weight-or-model')
            self.assertEqual(evidence['report_sha256'],self.m.BASELINE_SHA)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):self.m.load_baseline(work)

    def test_prefix_outputs_use_all_arrays_and_missing_audio_rejected(self):
        arrays=dict(video=np.ones((1,2,3),dtype='float32'),audio=np.ones((1,2),dtype='float32'))
        def bundle(folder,name):return arrays,{'sha256':'fixture'}
        with patch.object(self.m.base,'read_bundle',side_effect=bundle):
            result=self.m.compare_prefix_outputs(Path('fixture'))
        self.assertTrue(result['tp8_vs_reference']['video']['exact_equal'])
        self.assertEqual(result['tp8_vs_reference']['video']['scope'],'one-main-block-prefix-all-output-elements')
        with patch.object(self.m.base,'read_bundle',return_value=({'video':arrays['video']},{})):
            with self.assertRaises(ValueError):self.m.compare_prefix_outputs(Path('fixture'))

    def test_failed_preflight_retains_candidate_report(self):
        with tempfile.TemporaryDirectory() as d:
            rc=self.m.main(['--run-dir',d,'--work-home',d,'--local-target',d+'/local','--model-path',d+'/model'])
            self.assertEqual(rc,1)
            report=json.loads((Path(d)/'load-sync-parity.json').read_text())
            self.assertEqual(report['result'],'FAILED')
            self.assertFalse(report['performance_measurement_valid'])

    def test_pins_match_returned_build_and_actual_copy_probe_receipts(self):
        root=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04'
        build=root/'load-sync-build-20261107T030125Z-2935713'
        receipt=json.loads((build/'summary.json').read_text())
        self.assertEqual(self.m.EXPECTED_EXTENSION,receipt['extension_sha256'])
        self.assertTrue(receipt['marker']['extension'].endswith(self.m.EXTENSION_RELATIVE.as_posix()))
        log=(build/'run.log').read_text()
        self.assertIn('"rank_patched_sha256": "'+self.m.RANK_SHA+'"',log)
        self.assertIn('"h3_weight_load_sync.hpp": "'+self.m.LOAD_HELPER_SHA+'"',log)
        copy=json.loads((root/'shard-copy-result-20261107T023515Z-2862675/shard-copy.json').read_text())
        self.assertEqual(self.m.EXPECTED_CORE_LIBRARIES,copy['groups']['mixed']['checkpoint']['library_hashes'])
