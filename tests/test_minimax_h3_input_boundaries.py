import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'

class InputBoundaryTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'diagnose_minimax_h3_input_boundaries.py'
        self.assertTrue(path.is_file(),'input boundary runner missing')
        spec=importlib.util.spec_from_file_location('input_boundary_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_stop_preserves_h3_signature_and_never_executes_forward(self):
        calls=[]
        class H3:
            def forward(self,hidden_states,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,
                        token_tags,position_ids,video_indices,audio_indices,text_indices,attention_kwargs=None,return_dict=True):
                calls.append('should-not-run')
        owner=H3();original=owner.forward
        with tempfile.TemporaryDirectory() as d:
            capture=self.m.Boundaries(Path(d),{},lambda:None)
            with capture.stop_before_h3(owner):
                self.assertEqual(inspect.signature(owner.forward),inspect.signature(original))
                with self.assertRaises(self.m.InputsCaptured):
                    owner.forward(**{k:np.ones(1,dtype=np.float32) for k in self.m.g.base.INPUT_NAMES},return_dict=False)
            self.assertTrue(capture.stopped)
            arrays,_=self.m.g.base.read_bundle(Path(d),'first-inputs')
            self.assertEqual(set(arrays),set(self.m.g.base.INPUT_NAMES))
        self.assertEqual(owner.forward,original);self.assertEqual(calls,[])

    def test_qwen_observer_records_actual_inputs_and_selected_layer_without_changing_return(self):
        result=SimpleNamespace(hidden_states=(None,)*50+(np.ones((1,2,4),np.float32),))
        class Qwen:
            def forward(self,**kwargs):return result
        q=Qwen();original=q.forward
        with tempfile.TemporaryDirectory() as d:
            report={};capture=self.m.Boundaries(Path(d),report,lambda:None)
            with capture.qwen(q):
                self.assertIs(q.forward(input_ids=np.ones((1,2),np.int64),use_cache=False,output_hidden_states=True),result)
            arrays,_=self.m.g.base.read_bundle(Path(d),'qwen-layer50')
            self.assertEqual(arrays['hidden'].shape,(1,2,4))
            self.assertEqual(report['qwen_calls'],1)
        self.assertEqual(q.forward,original)

    def test_noise_observer_reads_rng_without_adding_draws_or_changing_values(self):
        class State:
            def __init__(self,value):self.value=value
            def cpu(self):return self
            def numpy(self):return np.array([self.value],np.uint8)
        class Generator:
            device='cpu'
            def __init__(self):self.draws=0
            def get_state(self):return State(self.draws)
        rng=Generator();out=np.ones((2,3),np.float32)
        def draw(shape,generator=None,device=None,dtype=None):generator.draws+=1;return out
        module=SimpleNamespace(randn_tensor=draw)
        with tempfile.TemporaryDirectory() as d:
            report={};capture=self.m.Boundaries(Path(d),report,lambda:None)
            with capture.noise(module):
                self.assertEqual(inspect.signature(module.randn_tensor),inspect.signature(draw))
                self.assertIs(module.randn_tensor((2,3),generator=rng,device='cuda:0',dtype='float32'),out)
            arrays,_=self.m.g.base.read_bundle(Path(d),'noise-000')
            self.assertEqual(arrays['rng_before'].tolist(),[0]);self.assertEqual(arrays['rng_after'].tolist(),[1])
            self.assertEqual(rng.draws,1)
        self.assertIs(module.randn_tensor,draw)

    def test_vae_function_keeps_arguments_signature_and_return(self):
        out=np.ones((1,2),np.float32)
        def encode(vae,pixels,pixel_mean,pixel_std,encode_seed=42):return out
        module=SimpleNamespace(encode_vae_condition=encode)
        with tempfile.TemporaryDirectory() as d:
            report={};capture=self.m.Boundaries(Path(d),report,lambda:None)
            with capture.vae(module):
                self.assertEqual(inspect.signature(module.encode_vae_condition),inspect.signature(encode))
                self.assertIs(module.encode_vae_condition(object(),np.ones((1,3),np.uint8),(0,0,0),(1,1,1)),out)
            self.assertEqual(report['vae_encoding']['encode_seed'],42)
        self.assertIs(module.encode_vae_condition,encode)

    def test_comparison_keeps_shape_missing_dtype_and_integer_differences_visible(self):
        cmp=self.m.compare_field
        self.assertEqual(cmp(np.ones(3,np.int64),np.array([1,2,1]))['different_elements'],1)
        self.assertNotIn('relative_l2',cmp(np.ones(3,np.uint8),np.zeros(3,np.uint8)))
        self.assertEqual(cmp(np.ones(3),np.ones(4))['status'],'SHAPE_MISMATCH')
        self.assertEqual(cmp(None,np.ones(4))['status'],'MISSING_NATIVE_FIELD')
        self.assertAlmostEqual(cmp(np.ones(3),np.ones(3))['relative_l2'],0)

    def test_incomplete_capture_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError,FileNotFoundError)):self.m.validate_capture(Path(d),{})

    def test_complete_bundle_receipt_is_verified_before_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            schedule=dict(requested_steps=50,evaluations=49,sigma_points=50,terminal_sigma=0.0,implementation_sha256=self.m.g.SCHEDULER_SHA)
            report=dict(first_h3_inputs_captured=True,H3_forward_executed=False,qwen_calls=1,vae_calls=1,
                noise_draws=1,noise_requests=[dict(shape=[2])],schedules=dict(video=schedule,audio=schedule))
            cap=self.m.Boundaries(folder,report,lambda:None)
            for name,values in [('qwen-inputs',dict(input_ids=np.ones(2,np.int64))),('qwen-layer50',dict(hidden=np.ones(2))),
                    ('vae-pixels',dict(pixels=np.ones(2,np.uint8))),('vae-condition',dict(latents=np.ones(2))),
                    ('noise-000',dict(values=np.ones(2),rng_before=np.zeros(2,np.uint8),rng_after=np.ones(2,np.uint8))),
                    ('first-inputs',{k:np.ones(2) for k in self.m.g.base.INPUT_NAMES})]:cap.store(name,values)
            self.assertEqual(len(self.m.validate_capture(folder,report)),6)
            report['bundles']['first-inputs']['sha256']='wrong'
            with self.assertRaises(ValueError):self.m.validate_capture(folder,report)

    def test_missing_previous_evidence_has_nonzero_and_failed_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rc=self.m.main(['--work-home',str(root/'work'),'--run-dir',str(root/'run'),
                '--local-target',str(root/'site'),'--model-path',str(root/'model')])
            self.assertEqual(rc,1)
            report=json.loads((root/'run/input-boundary-parity.json').read_text())
            self.assertEqual(report['result'],'FAILED')
            self.assertEqual(report['generation_status'],'NOT_EXECUTED')
            self.assertFalse(report['performance_measurement_valid'])

if __name__=='__main__':unittest.main()
