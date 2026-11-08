import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import json

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'artifacts/infinifabric-build-v1/recover_minimax_h3_generation_media.py'

class RecoverMediaTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(FILE.is_file(),'media recovery missing')
        spec=importlib.util.spec_from_file_location('recovery_test',FILE)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def fixture(self):
        g=self.m.g
        counts=dict(condition_encoder_calls=1,transformer_calls=49,video_scheduler_steps=49,audio_scheduler_steps=49)
        return dict(version='1.1-load-sync-template-Native-signature-preserved',result='FAILED',error=self.m.COUNT_ERROR,
            observed_calls=counts,extension_sha256=g.capture.EXPECTED_EXTENSION,
            models=dict(main_blocks=50,tp_size=8,trace_enabled=False,graphs_enabled=False),
            canvas=dict(width=768,height=928),
            workload=dict(task='fl2va',prompt='A dog falls asleep',steps=50,num_frames=124,seed=42,gpus=list(range(8)),canvas='pipeline-default'))

    def test_real_returned_log_supports_completed_schedule_only(self):
        source=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/generation-count-failure-20261107T082020Z-3236632/run.log'
        text=source.read_text(encoding='utf-8')
        self.assertEqual(self.m.validate_previous(self.fixture(),text),49)
        for fragment in ('call=transformer_calls index=49 status=DONE','call=audio_scheduler_steps index=49 status=DONE','[done] Saved to:'):
            with self.assertRaises(ValueError):self.m.validate_previous(self.fixture(),text.replace(fragment,'missing'))

    def test_bad_identity_workload_counts_or_failure_not_reclassified(self):
        source=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/generation-count-failure-20261107T082020Z-3236632/run.log'
        text=source.read_text(encoding='utf-8')
        for key,value in [('extension_sha256','wrong'),('error','OOM'),('version','old')]:
            with self.assertRaises(ValueError):self.m.validate_previous({**self.fixture(),key:value},text)
        bad=self.fixture();bad['workload']['steps']=49
        with self.assertRaises(ValueError):self.m.validate_previous(bad,text)
        bad=self.fixture();bad['observed_calls']['video_scheduler_steps']=48
        with self.assertRaises(ValueError):self.m.validate_previous(bad,text)

    def test_missing_prior_artifact_is_nonzero_and_no_generation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(self.m.main(['--work-home',str(root/'work'),'--run-dir',str(root/'run'),'--local-target',str(root/'site')]),1)
            report=json.loads((root/'run/media-recovery.json').read_text())
            self.assertEqual(report['result'],'FAILED')
            self.assertFalse(report['new_inference_executed'])
            self.assertFalse(report['performance_measurement_valid'])

if __name__=='__main__':unittest.main()
