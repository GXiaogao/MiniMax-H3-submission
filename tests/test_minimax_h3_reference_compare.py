import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'

class ReferenceCompareTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'run_minimax_h3_reference_compare.py'
        self.assertTrue(path.is_file(),'reference runner missing')
        spec=importlib.util.spec_from_file_location('reference_compare_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_reference_argv_preserves_original_workload_and_explicit_reference_devices(self):
        argv=self.m.reference_argv(Path('/model'),Path('/image.png'),Path('/out.mp4'))
        for item in ('--legacy','--steps=50','--num-frames=124','--seed=42','--text=A dog falls asleep',
                '--legacy-text-gpus=0,1,2,3','--legacy-transformer-gpus=4,5,6,7','--legacy-memory-fraction=0.72'):
            self.assertIn(item,argv)
        self.assertFalse(any(a.startswith(('--height','--width')) for a in argv))

    def test_native_receipt_and_media_are_pinned(self):
        archived=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/media-recovery-result-20261107T090136Z-3046211/media-recovery.json'
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);path=work/self.m.RECEIPT_RELATIVE;path.parent.mkdir(parents=True);path.write_bytes(archived.read_bytes())
            native=work/self.m.NATIVE_RELATIVE;native.parent.mkdir(parents=True);native.write_bytes(b'changed video')
            with self.assertRaises(ValueError):self.m.validate_native(work)
            # Isolate the path/digest wiring without pretending these bytes are real video.
            real=self.m.g.base.sha256
            with patch.object(self.m.g.base,'sha256',side_effect=lambda p:self.m.NATIVE_SHA if Path(p)==native else real(p)):
                self.assertEqual(self.m.validate_native(work),native)
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):self.m.validate_native(work)

    def test_original_reference_source_is_not_replaced_by_precision_copy(self):
        original=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/examples/minimax_h3.py'
        self.m.g.validate_source(original,self.m.g.EXAMPLE_SHA,'baseline')
        with tempfile.TemporaryDirectory() as d:
            changed=Path(d)/'patched.py';changed.write_bytes(original.read_bytes().replace(b'pipe.vae.to(device=execution_device, dtype=DTYPE)',b'pipe.vae.to(device=execution_device)'))
            with self.assertRaises(ValueError):self.m.g.validate_source(changed,self.m.g.EXAMPLE_SHA,'baseline')

    def test_compare_command_provides_both_outputs_and_only_fl2va(self):
        cmd=self.m.compare_command(Path('/compare.py'),Path('/model'),Path('/native.mp4'),Path('/reference.mp4'),Path('/report'))
        for item in ('--tasks=fl2va',f'--reference-fl2va-video={Path("/reference.mp4")}',f'--infinilm-fl2va-video={Path("/native.mp4")}'):
            self.assertIn(item,cmd)
        self.assertFalse(any('ref2va' in a for a in cmd))

    def test_displayed_metrics_are_not_acceptance_threshold_or_speedup(self):
        text='Size: PASS (PyTorch 768x928, InfiniLM 768x928)\nLength: PASS (PyTorch 124 frames, InfiniLM 124 frames)\nAverage SSIM: 84.123%\n'
        result=self.m.parse_comparison(text)
        self.assertEqual(result['displayed_average_ssim_percent'],84.123)
        self.assertFalse(result['performance_measurement_valid'])
        self.assertEqual(result['quality_status'],'MEASURED_NO_ACCEPTANCE_THRESHOLD')
        for fragment in ('Size: PASS','Length: PASS','Average SSIM: 84.123%'):
            with self.assertRaises(ValueError):self.m.parse_comparison(text.replace(fragment,''))
        self.assertEqual(self.m.parse_comparison(text.replace('84.123%','-1.234%'))['displayed_average_ssim_percent'],-1.234)

    def test_missing_native_evidence_retains_failed_json_without_loading_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rc=self.m.main(['--work-home',str(root/'work'),'--run-dir',str(root/'run'),'--local-target',str(root/'site'),
                '--model-path',str(root/'model')])
            self.assertEqual(rc,1)
            result=json.loads((root/'run/reference-comparison.json').read_text())
            self.assertEqual(result['result'],'FAILED')
            self.assertFalse(result['performance_measurement_valid'])

    def test_actual_compare_provided_mode_never_calls_generation(self):
        path=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/examples/minimax_h3_compare.py'
        compare=self.m.g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','provided_common').load_official_example(path)
        video=dict(left_metadata=dict(width=768,height=928),right_metadata=dict(width=768,height=928),
            metadata_size_match=True,decoded_length_match=True,left_frames=124,right_frames=124,ssim_match_percent=90.0)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);native=root/'n.mp4';reference=root/'r.mp4';native.touch();reference.touch()
            argv=self.m.compare_command(path,root,native,reference,root/'report')
            with (patch('sys.argv',[str(path),*argv[3:]]),patch.object(compare,'run_task',side_effect=AssertionError('generation forbidden')),
                patch.object(compare,'compare_video_streams',return_value=video)):
                self.assertEqual(compare.main(),0)

    def test_comparison_dependency_failure_stops_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.dict(sys.modules,{'av':None}),patch.object(self.m,'reference_worker',side_effect=AssertionError('must not load model')):
                rc=self.m.main(['--worker','preflight','--run-dir',str(root),'--model-path',str(root/'model')])
            self.assertEqual(rc,1)
            result=json.loads((root/'comparison-preflight.json').read_text())
            self.assertEqual(result['result'],'FAILED')
            self.assertIn('ModuleNotFoundError',result['error'])

if __name__=='__main__':unittest.main()
