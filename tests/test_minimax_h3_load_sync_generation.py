import importlib.util
import ast
from contextlib import nullcontext
import functools
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'artifacts/infinifabric-build-v1'


class LoadSyncGenerationTests(unittest.TestCase):
    def setUp(self):
        path=FOLDER/'run_minimax_h3_load_sync_generation.py'
        self.assertTrue(path.is_file(),'generation runner missing')
        spec=importlib.util.spec_from_file_location('generation_test',path)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)

    def test_workload_matches_compare_without_canvas_override(self):
        argv=self.m.official_argv(Path('/model'),Path('/image.png'),Path('/output.mp4'))
        self.assertEqual(argv[0],'fl2va')
        for item in ('--steps=50','--num-frames=124','--seed=42','--text=A dog falls asleep','--gpus=0,1,2,3,4,5,6,7'):
            self.assertIn(item,argv)
        self.assertFalse(any(a.startswith(('--height','--width','--legacy')) for a in argv))

    def test_prior_evidence_is_pinned_and_not_a_quality_acceptance(self):
        source=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/full-forward-load-sync-result-20261107T065018Z-3136076/full-forward-parity.json'
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);target=work/self.m.FULL_REPORT_RELATIVE;target.parent.mkdir(parents=True)
            target.write_bytes(source.read_bytes())
            result=self.m.validate_full_report(work)
            self.assertEqual(result['report_sha256'],self.m.FULL_REPORT_SHA)
            target.write_bytes(target.read_bytes()+b'\n')
            with self.assertRaises(ValueError):self.m.validate_full_report(work)

    def test_observer_counts_only_completed_calls_and_restores_method(self):
        class Owner:
            def step(self,x):
                if x<0:raise RuntimeError('failed computation')
                return x+1
        owner=Owner();original=owner.step;counts={}
        with self.m.observe_calls(owner,'step','video_scheduler_steps',counts,lambda:None):
            self.assertEqual(owner.step(2),3)
            with self.assertRaises(RuntimeError):owner.step(-1)
        self.assertEqual(counts,{'video_scheduler_steps':1})
        self.assertEqual(owner.step,original)

    def test_actual_denoiser_routes_all_layout_fields_through_observer_and_official_profiler(self):
        # Execute the exact __call__ from the pinned, licensed wheel without Torch imports.
        # The real pipeline filters layout kwargs using inspect.signature(transformer.forward).
        source=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/generation-signature-failure-20261107T074948Z-1305412/denoise.py'
        self.assertEqual(self.m.base.sha256(source),'4c2bc8b9856c38ba1692b02e35aa68ec92beda4db273e2e71f0ebbf997e19c79')
        tree=ast.parse(source.read_text(encoding='utf-8'))
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3LoopDenoiser')
        call=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name=='__call__')
        call.decorator_list=[]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        namespace={'inspect':inspect}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,call],type_ignores=[])),str(source),'exec'),namespace)
        example=ROOT/'artifacts/infinifabric-migration-v3/verification-extract/InfiniLM/examples/minimax_h3.py'
        tree=ast.parse(example.read_text(encoding='utf-8'))
        profiler=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ComponentProfiler')
        scope={'functools':functools}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,profiler],type_ignores=[])),str(example),'exec'),scope)

        class Tensor:
            def __getitem__(self,key):return self
        seen=[]
        class Adapter:
            def forward(self,hidden_states,audio_hidden_states,encoder_hidden_states,timestep,timestep_indices,
                        token_tags,position_ids,video_indices,audio_indices,text_indices,attention_kwargs=None,return_dict=True):
                seen.append((token_tags,position_ids,video_indices,audio_indices,text_indices))
                return 'video','audio'
            def __call__(self,*a,**kw):return self.forward(*a,**kw)
        adapter=Adapter();original=adapter.forward;counts={}
        names=('token_tags','position_ids','video_indices','audio_indices','text_indices')
        fields={name:object() for name in names};fields['unused_extra']=object()
        state=SimpleNamespace(row_timestep_plan=[(1,2)],denoiser_input_fields=fields,
            latents=Tensor(),audio_latents=Tensor(),prompt_embeds=object(),attention_kwargs=None)
        components=SimpleNamespace(transformer=adapter)
        profile=scope['ComponentProfiler'](SimpleNamespace(measure=lambda *a,**kw:nullcontext()))
        with self.m.observe_calls(adapter,'forward','transformer_calls',counts,lambda:None):
            profile.wrap(adapter,'forward','compute.h3_transformer')
            try:
                # This raised the exact server TypeError before the signature-preservation fix.
                namespace['__call__'](SimpleNamespace(transformer_name='transformer'),components,state,i=0,t=0)
                self.assertEqual(inspect.signature(adapter.forward),inspect.signature(original))
            finally:profile.close()
        self.assertEqual(seen,[tuple(fields[name] for name in names)])
        self.assertEqual((state.noise_pred,state.audio_noise_pred),('video','audio'))
        self.assertEqual(counts,{'transformer_calls':1})
        self.assertEqual(adapter.forward,original)

    def test_completion_requires_actual_schedule_length_and_fresh_qwen(self):
        counts=dict(condition_encoder_calls=1,transformer_calls=49,video_scheduler_steps=49,audio_scheduler_steps=49)
        self.m.validate_counts(counts,49)
        for key in counts:
            with self.subTest(key=key):
                bad={**counts,key:0}
                with self.assertRaises(ValueError):self.m.validate_counts(bad,49)
        for value in (48,50):
            with self.assertRaises(ValueError):self.m.validate_counts({**counts,'video_scheduler_steps':value},49)
            with self.assertRaises(ValueError):self.m.validate_counts({**counts,'audio_scheduler_steps':value},49)
        with self.assertRaises(ValueError):self.m.validate_counts(counts,0)

    def test_real_scheduler_fifty_sigma_points_mean_fortynine_evaluations(self):
        import numpy as np
        source=ROOT/'artifacts/fl2va-quality-evidence-2026-11-04/generation-count-failure-20261107T082020Z-3236632/scheduling_minimax_h3.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MiniMaxH3Scheduler')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_timesteps')
        method.decorator_list=[]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        class Array(np.ndarray):
            def to(self,**kw):return self
            def numel(self):return self.size
        torch=SimpleNamespace(float32=np.float32,
            linspace=lambda a,b,n,dtype:np.linspace(a,b,n,dtype=dtype).view(Array),
            unique_consecutive=lambda a:a[np.r_[True,a[1:]!=a[:-1]]])
        scope={'torch':torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[])),str(source),'exec'),scope)
        scheduler=SimpleNamespace(_shift=12.0)
        scope['set_timesteps'](scheduler,50,device='cpu')
        self.assertEqual(len(scheduler.sigmas),50)
        self.assertEqual(len(scheduler.timesteps),49)
        self.m.validate_counts(dict(condition_encoder_calls=1,transformer_calls=49,video_scheduler_steps=49,audio_scheduler_steps=49),len(scheduler.timesteps))

    def test_schedule_observer_records_actual_length_and_preserves_signature(self):
        class Scalar:
            def item(self):return 0.0
        class Scheduler:
            def set_timesteps(self,num_inference_steps=None,device=None,sigmas=None):
                self.timesteps=list(range(49));self.sigmas=[Scalar()]*50
        scheduler=Scheduler();original=scheduler.set_timesteps;schedules={}
        with self.m.observe_schedule(scheduler,'video',schedules,lambda:None):
            self.assertEqual(inspect.signature(scheduler.set_timesteps),inspect.signature(original))
            scheduler.set_timesteps(50,device='cpu')
        self.assertEqual(self.m.validate_schedules(dict(video=schedules['video'],audio=schedules['video'])),49)
        self.assertEqual(scheduler.set_timesteps,original)
        with self.assertRaises(ValueError):self.m.validate_schedules(dict(video={**schedules['video'],'requested_steps':49},audio=schedules['video']))

    def test_engine_graphs_off_without_qwen_or_transformer_layer_changes(self):
        calls=[];factory=self.m.full.full_engine_factory(lambda *a,**kw:calls.append(kw))
        overrides={'condition_encoder_num_hidden_layers':50,'condition_encoder_mode':True}
        factory('model',hf_config_overrides=overrides)
        self.assertEqual(calls[0]['hf_config_overrides'],overrides)
        self.assertFalse(calls[0]['enable_graph_compiling'])
        with self.assertRaises(ValueError):factory('model',hf_config_overrides={'num_layers':1})

    def test_missing_assets_produce_failed_report_and_nonzero_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rc=self.m.main(['--work-home',str(root/'work'),'--run-dir',str(root/'run'),
                '--local-target',str(root/'site'),'--model-path',str(root/'model')])
            self.assertEqual(rc,1)
            report=json.loads((root/'run/native-generation.json').read_text())
            self.assertEqual(report['result'],'FAILED')
            self.assertEqual(report['quality_status'],'UNASSESSED')
            self.assertFalse(report['performance_measurement_valid'])

    def test_template_bytes_exact_and_expected_model_types(self):
        image=ROOT/'docs/source/preliminary-report-template-2026-11-07/image1.png'
        self.assertEqual(self.m.base.sha256(image),self.m.IMAGE_SHA)
        self.assertEqual(self.m.IMAGE_SHA,'b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54')
        fake=SimpleNamespace(MiniMaxH3ConditionEncoder=type('Encoder',(),{}),MiniMaxH3DiffusersAdapter=type('Adapter',(),{}))
        with self.assertRaises(ValueError):self.m.validate_native_models(object(),object(),fake)


if __name__=='__main__':unittest.main()
