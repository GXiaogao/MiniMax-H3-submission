#!/usr/bin/env python3
"""Capture actual upstream inputs on both profiles; stop BEFORE the first H3 call.

Component diagnostic only: no H3 evaluation, scheduler update, video or benchmark.
Original reference/baseline, actual random draws and return objects are preserved.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

spec=importlib.util.spec_from_file_location('boundary_reference_helpers',Path(__file__).with_name('run_minimax_h3_reference_compare.py'))
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
g=reference.g
PRIOR_RELATIVE=Path('logs/infinifabric-h3-reference-compare-20261107T124511Z-3527846/reference-comparison.json')
PRIOR_SHA='10c91f6e3efc0c1e13f1115d03604ad33c6562ab7a5ea998c5aed96289859847'
REF_VIDEO_SHA='ba5bc8cee538356719e909d914febb52446cd1806cf5f87d0c3055d54f5c3d59'
ENCODERS_SHA='594345d3619198c15fadf42b0fc8c13cec324e546ba10ee4c60c3d06c4b8e2fb'
NOISE_MODULE_SHA='c0c3484b238b7cf529fb494bcfd02c286db8b34ce3e23fd78a17eeda257764d1'


class InputsCaptured(BaseException):
    """Intentional diagnostic stop, never completion of a generation."""


def is_tensor(value):
    import numpy as np
    return isinstance(value,np.ndarray) or hasattr(value,'detach')


class Boundaries:
    def __init__(self,folder,report,checkpoint):
        self.folder=folder;self.report=report;self.checkpoint=checkpoint;self.stopped=False
        for name,value in (('bundles',{}),('qwen_calls',0),('vae_calls',0),('noise_draws',0),('noise_requests',[])):
            report.setdefault(name,value)

    def store(self,name,values):
        if name in self.report['bundles']:raise ValueError('duplicate boundary '+name)
        if not values:raise ValueError('empty boundary '+name)
        meta=g.base.save_bundle(self.folder,name,values)
        self.report['bundles'][name]=meta;self.checkpoint()

    @contextmanager
    def qwen(self,owner):
        original=owner.forward
        @wraps(original)
        def observed(*args,**kwargs):
            if args or self.report['qwen_calls']:raise ValueError('unexpected Qwen call signature/count')
            tensors={k:v for k,v in kwargs.items() if is_tensor(v)}
            options={k:v for k,v in kwargs.items() if k not in tensors}
            if any(not isinstance(v,(str,int,float,bool,type(None))) for v in options.values()):
                raise ValueError('unsupported Qwen scalar option')
            self.report['qwen_options']=options
            self.store('qwen-inputs',tensors)
            output=original(*args,**kwargs)
            self.store('qwen-layer50',{'hidden':output.hidden_states[50]})
            self.report['qwen_calls']+=1;self.checkpoint()
            return output
        owner.forward=observed
        try:yield
        finally:owner.forward=original

    @contextmanager
    def vae(self,module):
        original=module.encode_vae_condition;signature=inspect.signature(original)
        @wraps(original)
        def observed(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
            if self.report['vae_calls']:raise ValueError('unexpected multiple FL2VA keyframe conditions')
            self.report['vae_encoding']=dict(pixel_mean=list(map(float,values['pixel_mean'])),
                pixel_std=list(map(float,values['pixel_std'])),encode_seed=int(values['encode_seed']))
            self.store('vae-pixels',{'pixels':values['pixels']})
            output=original(*args,**kwargs)
            self.store('vae-condition',{'latents':output})
            self.report['vae_calls']+=1;self.checkpoint()
            return output
        module.encode_vae_condition=observed
        try:yield
        finally:module.encode_vae_condition=original

    @contextmanager
    def noise(self,module):
        original=module.randn_tensor;signature=inspect.signature(original)
        @wraps(original)
        def observed(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults();values=bound.arguments
            generator=values.get('generator')
            if generator is None or str(generator.device)!='cpu':raise ValueError('expected the unchanged request CPU generator')
            before=generator.get_state().cpu().numpy().copy()
            output=original(*args,**kwargs)
            after=generator.get_state().cpu().numpy().copy()
            index=self.report['noise_draws']
            self.store(f'noise-{index:03d}',dict(values=output,rng_before=before,rng_after=after))
            self.report['noise_requests'].append(dict(shape=list(map(int,values['shape'])),device=str(values.get('device')),dtype=str(values.get('dtype'))))
            self.report['noise_draws']+=1;self.checkpoint()
            return output
        module.randn_tensor=observed
        try:yield
        finally:module.randn_tensor=original

    @contextmanager
    def stop_before_h3(self,owner):
        original=owner.forward;signature=inspect.signature(original)
        @wraps(original)
        def observed(*args,**kwargs):
            if self.stopped:raise ValueError('duplicate first H3 boundary')
            bound=signature.bind(*args,**kwargs).arguments
            tensors={k:v for k,v in bound.items() if k in g.base.INPUT_NAMES}
            if set(tensors)!=set(g.base.INPUT_NAMES):raise ValueError('incomplete first H3 inputs')
            extras={k:v for k,v in bound.items() if k not in tensors}
            if extras.get('return_dict') is not False or extras.get('attention_kwargs') not in (None,{}):
                raise ValueError('unexpected H3 options')
            if set(extras)-{'return_dict','attention_kwargs'}:raise ValueError('unsupported H3 option')
            self.store('first-inputs',tensors)
            self.stopped=True;self.report['first_h3_inputs_captured']=True
            self.checkpoint()
            print('FIRST_H3_INPUTS_CAPTURED; intentional_stop=true; H3_forward_executed=false',flush=True)
            raise InputsCaptured()
        owner.forward=observed
        try:yield
        finally:owner.forward=original


def validate_capture(folder,report):
    if (report.get('first_h3_inputs_captured') is not True or report.get('H3_forward_executed') is not False
            or report.get('qwen_calls')!=1 or report.get('vae_calls')!=1 or report.get('noise_draws',0)<1):
        raise ValueError('upstream capture incomplete')
    g.validate_schedules(report['schedules'])
    n=report['noise_draws']
    if len(report.get('noise_requests',[]))!=n:raise ValueError('missing noise request metadata')
    expected={'qwen-inputs','qwen-layer50','vae-pixels','vae-condition','first-inputs'}|{f'noise-{i:03d}' for i in range(n)}
    if set(report.get('bundles',{}))!=expected:raise ValueError('boundary bundle set differs')
    arrays={}
    for name in sorted(expected):
        data,meta=g.base.read_bundle(folder,name)
        if meta!=report['bundles'][name]:raise ValueError('boundary receipt differs: '+name)
        arrays[name]=data
    if set(arrays['first-inputs'])!=set(g.base.INPUT_NAMES):raise ValueError('missing H3 argument fields')
    for name,fields in [('qwen-layer50',{'hidden'}),('vae-pixels',{'pixels'}),('vae-condition',{'latents'})]:
        if set(arrays[name])!=fields:raise ValueError('incomplete '+name)
    if 'input_ids' not in arrays['qwen-inputs']:raise ValueError('missing Qwen token IDs')
    for i in range(n):
        if set(arrays[f'noise-{i:03d}'])!={'values','rng_before','rng_after'}:raise ValueError('incomplete RNG capture')
    return arrays


def compare_field(native,ref):
    import numpy as np
    if native is None:return dict(status='MISSING_NATIVE_FIELD')
    if ref is None:return dict(status='MISSING_REFERENCE_FIELD')
    if native.shape!=ref.shape:return dict(status='SHAPE_MISMATCH',native_shape=list(native.shape),reference_shape=list(ref.shape))
    if native.dtype.kind in 'biu' or ref.dtype.kind in 'biu':
        return dict(status='COMPARED',shape=list(native.shape),exact_equal=bool(np.array_equal(native,ref)),
            different_elements=int(np.count_nonzero(native!=ref)),native_dtype=str(native.dtype),reference_dtype=str(ref.dtype))
    return dict(status='COMPARED',**g.base.compare_arrays(ref,native))


def compare_captures(native,ref,nmeta,rmeta):
    result={}
    for name in sorted(set(native)|set(ref)):
        left=native.get(name,{});right=ref.get(name,{})
        result[name]={key:compare_field(left.get(key),right.get(key)) for key in sorted(set(left)|set(right))}
        for key,value in result[name].items():
            value['original_torch_dtype_match']=(nmeta.get(name,{}).get('fields',{}).get(key,{}).get('torch_dtype')==rmeta.get(name,{}).get('fields',{}).get(key,{}).get('torch_dtype'))
    return result


def validate_prior(work):
    path=work/PRIOR_RELATIVE;g.validate_source(path,PRIOR_SHA,'returned 44.643 percent comparison')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('result')!='QUALITY_COMPARISON_COMPLETE' or data.get('comparison',{}).get('displayed_average_ssim_percent')!=44.643:
        raise ValueError('prior comparison scope differs')
    reference.validate_native(work)
    ref=work/'logs/infinifabric-h3-reference-compare-20261107T124511Z-3527846/reference/fl2va-reference-original-50steps.mp4'
    g.validate_source(ref,REF_VIDEO_SHA,'prior reference video')
    return dict(path=str(path),sha256=PRIOR_SHA,prior_SSIM_percent=44.643)


def worker(args,report,checkpoint):
    g.full.disable_trace()
    import torch
    import diffusers.modular_pipelines.minimax_h3.encoders as encoders
    import diffusers.modular_pipelines.minimax_h3.before_denoise as before_denoise
    from PIL import Image
    from infinilm.lib import _infinilm
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    native=args.worker=='native'
    g.validate_source(args.image,g.IMAGE_SHA,'template image')
    g.validate_source(inspect.getfile(encoders),ENCODERS_SHA,'encoder recipe')
    g.validate_source(inspect.getfile(before_denoise),NOISE_MODULE_SHA,'noise/layout recipe')
    common=g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','input_boundary_common')
    audit=g.base.load_sibling('diagnose_minimax_h3_vae_roundtrip.py','input_boundary_audit')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=8:raise ValueError('eight devices required')
    factory_owner=None;factory=None
    if native:
        import infinicore
        from infinicore.lib import _infinicore
        import infinilm.multimodal.minimax_h3 as bridge
        g.full.validate_loaded_extension(_infinilm.__file__,args)
        g.validate_source(inspect.getfile(bridge),g.BRIDGE_SHA,'Native bridge')
        common.validate_device_counts(list(range(8)),torch_count=8,native_count=_infinicore.get_device_count(_infinicore.Device.Type.HYGON))
        report['weight_staging']=common.configure_weight_staging(infinicore,list(range(8)),_infinicore.get_device_count(_infinicore.Device.Type.CPU))
        factory_owner=bridge;factory=bridge.InferEngine
        bridge.InferEngine=g.full.full_engine_factory(factory)
    else:g.validate_source(_infinilm.__file__,g.OFFICIAL_EXTENSION_SHA,'original extension')
    g.validate_source(args.example,args.example_sha256,'profile entry')
    report['extension_sha256']=g.base.sha256(_infinilm.__file__)
    module=common.load_official_example(args.example);original_load=module.load_pipeline
    capture=Boundaries(args.run_dir,report,checkpoint)
    with ExitStack() as observers:
        def load(*a,**kw):
            if bool(kw.get('legacy'))==native:raise ValueError('incorrect profile selected')
            report['last_stage']='model-load';checkpoint()
            pipe,encoder,transformer=original_load(*a,**kw)
            if native:g.validate_native_models(encoder,transformer,bridge)
            else:
                if type(encoder) is not module.Qwen3VLForConditionalGeneration or type(transformer) is not module.MiniMaxH3Transformer3DModel:
                    raise ValueError('reference must preserve original core classes')
                g.validate_source(inspect.getfile(type(encoder)),reference.QWEN_SHA,'reference Qwen')
                g.validate_source(inspect.getfile(type(transformer)),reference.H3_SHA,'reference H3')
                if len(transformer.transformer_blocks)!=50 or transformer.config.num_layers!=50:raise ValueError('incomplete reference model')
                g.base.validate_device_map(encoder.hf_device_map,list(range(4)))
                g.base.validate_device_map(transformer.hf_device_map,list(range(4,8)))
            if int(pipe.text_encoder_layer)!=50:raise ValueError('unexpected Qwen layer selection')
            with Image.open(args.image) as image:
                h,w=resolve_canvas_size(*image.size,pipe.canvas_multiple,pipe.config.canvas_short_edge,pipe.config.canvas_max_pixels)
            if (w,h)!=(768,928):raise ValueError('current template canvas changed')
            report['models']=dict(Qwen=type(encoder).__name__,H3=type(transformer).__name__,main_blocks=50,canvas=[w,h])
            report['parameter_audits']={k:audit.audit_parameters(v) for k,v in (('vae',pipe.vae),('audio_vae',pipe.audio_vae))}
            observers.enter_context(capture.qwen(encoder.model));observers.enter_context(capture.vae(encoders))
            observers.enter_context(capture.noise(before_denoise));observers.enter_context(capture.stop_before_h3(transformer))
            for label,owner in (('video',pipe.scheduler),('audio',pipe.audio_scheduler)):
                g.validate_source(inspect.getfile(type(owner)),g.SCHEDULER_SHA,'scheduler')
                observers.enter_context(g.observe_schedule(owner,label,report['schedules'],checkpoint))
                original=owner.step
                @wraps(original)
                def forbidden(*a,**kw):raise RuntimeError('diagnostic must stop before scheduler.step')
                observers.callback(setattr,owner,'step',original);owner.step=forbidden
            report['last_stage']='upstream-capture';checkpoint()
            return pipe,encoder,transformer
        module.load_pipeline=load;previous=sys.argv[:]
        no_video=args.run_dir/'DIAGNOSTIC-NO-VIDEO.mp4'
        command=(g.official_argv if native else reference.reference_argv)(args.model_path,args.image,no_video)
        report['official_argv']=command
        try:
            sys.argv=[str(args.example),*command]
            try:
                with common.StageHeartbeat(args.worker+'-upstream-capture'):module.main()
            except InputsCaptured:
                if not capture.stopped:raise RuntimeError('unexpected stop without saved H3 inputs')
            else:raise RuntimeError('diagnostic failed to stop before H3')
        finally:
            sys.argv=previous;module.load_pipeline=original_load
            if factory_owner is not None:factory_owner.InferEngine=factory
    if no_video.exists():raise ValueError('diagnostic unexpectedly produced video')
    validate_capture(args.run_dir,report)
    g.validate_source(args.example,args.example_sha256,'entry after capture')
    report.update(result='UPSTREAM_CAPTURE_COMPLETE',last_stage='complete');checkpoint()


def orchestrate(args,report,checkpoint):
    report['prior_comparison']=validate_prior(args.work_home)
    extension,python_root,evidence=g.capture.validate_diagnostic(args.work_home)
    report['diagnostic_build']=evidence
    source=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    g.validate_source(source,g.EXAMPLE_SHA,'protected original entry')
    image=args.work_home/'assets'/f'template-first-frame-{g.IMAGE_SHA}.png';g.validate_source(image,g.IMAGE_SHA,'template PNG')
    common=g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','input_boundary_parent')
    patcher=g.base.load_sibling('prepare_minimax_h3_vae_dtype_probe.py','input_boundary_patcher')
    native_example=args.local_target.parent/'native-example/minimax_h3.py'
    report['Native_example_patch']=patcher.patch_example(source,native_example)
    installer=g.base.load_sibling('install_infinilm_private_runtime.py','input_boundary_installer')
    hygon=g.base.load_sibling('prepare_minimax_h3_hygon_runtime.py','input_boundary_hygon')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest differs')
    report['last_stage']='runtime-staging';checkpoint()
    with common.StageHeartbeat('input-boundary-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),args.local_target)
        installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target));os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        report['hygon_runtime']=hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    captures={};receipts={}
    for mode in ('native','reference'):
        folder=args.run_dir/mode;folder.mkdir();report['last_stage']=mode+'-capture';checkpoint()
        example=native_example if mode=='native' else source
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',mode,'--run-dir',str(folder),'--model-path',str(args.model_path),
            '--example',str(example),'--example-sha256',g.base.sha256(example),'--image',str(image)]
        if mode=='native':cmd+=['--diag-extension',str(extension),'--extension-sha256',evidence['extension_sha256']]
        extra=[str(python_root)] if mode=='native' else []
        env={'PYTHONPATH':os.pathsep.join([str(args.local_target),str(args.local_target.parent/'hygon-site-packages'),*extra,os.environ.get('PYTHONPATH','')]),'H3_BOUNDARY_TRACE_DIR':''}
        g.base.run_worker(cmd,folder/'worker.log',env)
        data=json.loads((folder/'capture.json').read_text(encoding='utf-8'))
        if data.get('result')!='UPSTREAM_CAPTURE_COMPLETE':raise ValueError('worker capture incomplete')
        captures[mode]=validate_capture(folder,data);receipts[mode]=data
        report.setdefault('completed_workers',[]).append(mode);checkpoint()
    report['last_stage']='comparison';checkpoint()
    report['comparisons']=compare_captures(captures['native'],captures['reference'],receipts['native']['bundles'],receipts['reference']['bundles'])
    report['metadata_comparison']=dict(qwen_options_equal=receipts['native']['qwen_options']==receipts['reference']['qwen_options'],
        vae_recipe_equal=receipts['native']['vae_encoding']==receipts['reference']['vae_encoding'],
        noise_draw_count_native=receipts['native']['noise_draws'],noise_draw_count_reference=receipts['reference']['noise_draws'],
        noise_shapes_native=[v['shape'] for v in receipts['native']['noise_requests']],noise_shapes_reference=[v['shape'] for v in receipts['reference']['noise_requests']])
    report['worker_receipts']={k:dict(path=str(args.run_dir/k/'capture.json'),sha256=g.base.sha256(args.run_dir/k/'capture.json')) for k in receipts}
    g.validate_source(source,g.EXAMPLE_SHA,'protected original entry after diagnostic')
    validate_prior(args.work_home)
    report.update(result='INPUT_BOUNDARY_COMPARISON_COMPLETE',last_stage='complete');checkpoint()
    print('INPUT_BOUNDARY_COMPARISON_COMPLETE; H3_forward_executed=false; generation=NOT_EXECUTED; quality=UNASSESSED',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('native','reference'))
    for name in ('example-sha256','extension-sha256'):p.add_argument('--'+name)
    for name in ('work-home','run-dir','local-target','model-path','example','image','diag-extension'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv)
    required=('example','example_sha256','image') if args.worker else ('work_home','local_target')
    if args.worker=='native':required+=('diag_extension','extension_sha256')
    if any(getattr(args,k) is None for k in required):p.error('missing worker or parent arguments')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.0-current-template-upstream-boundaries',result='FAILED',last_stage='preflight',
        generation_status='NOT_EXECUTED',H3_forward_executed=False,quality_status='UNASSESSED',performance_measurement_valid=False,
        schedules={},scope='actual-Qwen-VAE-noise-and-first-H3-inputs-only',profile=args.worker)
    filename='capture.json' if args.worker else 'input-boundary-parity.json';start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds_including_setup_and_loading']=time.monotonic()-start
        g.base.write_json(args.run_dir/filename,report)
    checkpoint()
    try:(worker if args.worker else orchestrate)(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
