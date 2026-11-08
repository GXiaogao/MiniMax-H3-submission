#!/usr/bin/env python3
"""Full Native FL2VA with the returned load-sync candidate and template image.

Media completion is not quality acceptance or a controlled performance result.
The original entry/compare script, checkpoint and shared libraries stay intact.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
import importlib.util

spec=importlib.util.spec_from_file_location('generation_full_forward',Path(__file__).with_name('diagnose_minimax_h3_full_forward_load_sync.py'))
full=importlib.util.module_from_spec(spec);spec.loader.exec_module(full)
capture=full.capture
base=full.base
IMAGE_SHA='b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54'
IMAGE_NAME='minimax-h3-template-first-frame.png'
FULL_REPORT_RELATIVE=Path('logs/infinifabric-h3-full-forward-load-sync-20261107T065018Z-3136076/full-forward-parity.json')
FULL_REPORT_SHA='2d8b338462f918f57aff5af0a1e35aadd58fb0a0e9dc56683822274ee8a1fc78'
EXAMPLE_SHA='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
BRIDGE_SHA='5e5fc07bf0c02925c553603285fd163c44c973eb6c3de71349cf6c7677e3e4e5'
OFFICIAL_EXTENSION_SHA='e7d92053530cc1626dc112927788aa7f2116653a6803ac3cc7e23710610ed0dd'


def official_argv(model,image,output):
    # Match the fixed comparison script: canvas is resolved by the unmodified pipeline.
    return ['fl2va',f'--model-path={model}',f'--first-frame={image}',f'--output={output}',
            '--text=A dog falls asleep','--steps=50','--num-frames=124','--seed=42','--gpus=0,1,2,3,4,5,6,7']


def validate_full_report(work):
    path=work/FULL_REPORT_RELATIVE
    if base.sha256(path)!=FULL_REPORT_SHA:raise ValueError('returned full50 report changed')
    data=json.loads(path.read_text(encoding='utf-8'))
    if (data.get('result')!='COMPARISON_COMPLETE' or data.get('completed_workers')!=['native','reference']
            or data.get('diagnostic_build',{}).get('extension_sha256')!=capture.EXPECTED_EXTENSION
            or data.get('worker_models',{}).get('native',{}).get('main_blocks')!=50
            or data.get('trace_enabled') is not False):
        raise ValueError('returned full50 evidence incomplete')
    return dict(path=str(path),report_sha256=FULL_REPORT_SHA,scope='single-forward-not-generation-or-quality-acceptance')


@contextmanager
def observe_calls(owner,method,label,counts,checkpoint):
    original=getattr(owner,method)
    @wraps(original)
    def observed(*args,**kwargs):
        print(f'call={label} index={counts.get(label,0)+1} status=START',flush=True)
        output=original(*args,**kwargs)
        counts[label]=counts.get(label,0)+1
        checkpoint()
        print(f'call={label} index={counts[label]} status=DONE',flush=True)
        return output
    setattr(owner,method,observed)
    try:yield
    finally:setattr(owner,method,original)


def validate_counts(counts):
    if (counts.get('condition_encoder_calls',0)<1 or counts.get('transformer_calls',0)<50
            or counts.get('video_scheduler_steps')!=50 or counts.get('audio_scheduler_steps')!=50):
        raise ValueError('incomplete fresh Native encoding/50-step denoising schedules')


def validate_native_models(encoder,transformer,bridge):
    if type(encoder) is not bridge.MiniMaxH3ConditionEncoder or type(transformer) is not bridge.MiniMaxH3DiffusersAdapter:
        raise ValueError('generation must use Native Qwen and H3 adapters')
    full.validate_native_full(transformer.runner.engine.state_dict_keyname(),transformer.runner.engine.hf_config)
    config=encoder.engine.hf_config
    if config.get('condition_encoder_mode') is not True or config.get('condition_encoder_num_hidden_layers')!=50:
        raise ValueError('Native Qwen condition mode differs')


def validate_source(path,expected,label):
    if base.sha256(path)!=expected:raise ValueError(label+' changed')


def inspect_output(output,canvas,common,run):
    import av
    info=common.inspect_media(output,expected_width=canvas['width'],expected_height=canvas['height'],expected_frame_count=124)
    if info['average_frame_rate']!=24:raise ValueError('expected official 24fps output')
    with av.open(str(output)) as container:
        audio_samples=sum(frame.samples for frame in container.decode(audio=0))
    if audio_samples<1:raise ValueError('generated audio is empty or undecodable')
    stills=[]
    with av.open(str(output)) as container:
        for index,frame in enumerate(container.decode(video=0)):
            if index in (0,31,62,93,123):
                path=run/f'generated-{index:03d}.png';frame.to_image().save(path)
                stills.append(dict(frame=index,path=str(path),sha256=base.sha256(path)))
    if len(stills)!=5:raise ValueError('incomplete output frame samples')
    return dict(**info,audio_decoded_samples=audio_samples,sha256=base.sha256(output),stills=stills)


def worker(args,report,checkpoint):
    full.disable_trace()
    import torch
    import infinicore
    from infinicore.lib import _infinicore
    from infinilm.lib import _infinilm
    import infinilm.multimodal.minimax_h3 as bridge
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    full.validate_loaded_extension(_infinilm.__file__,args)
    validate_source(inspect.getfile(bridge),BRIDGE_SHA,'Native adapter')
    validate_source(args.image,IMAGE_SHA,'template PNG')
    validate_source(args.example,args.example_sha256,'job-local precision-preserving entry')
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','generation_common')
    audit=base.load_sibling('diagnose_minimax_h3_vae_roundtrip.py','generation_vae_audit')
    devices=list(range(8))
    common.validate_device_counts(devices,torch_count=torch.cuda.device_count(),native_count=_infinicore.get_device_count(_infinicore.Device.Type.HYGON))
    report['weight_staging']=common.configure_weight_staging(infinicore,devices,_infinicore.get_device_count(_infinicore.Device.Type.CPU))
    report['extension_sha256']=base.sha256(_infinilm.__file__)
    module=common.load_official_example(args.example)
    common.install_official_stage_tracing(module)
    counts=report['observed_calls']
    original_engine=bridge.InferEngine
    original_load=module.load_pipeline
    with ExitStack() as observers:
        def load(*a,**kw):
            if kw.get('legacy',False):raise ValueError('legacy core forbidden in Native generation')
            report['last_stage']='Native-pipeline-load';checkpoint()
            pipe,encoder,transformer=original_load(*a,**kw)
            validate_native_models(encoder,transformer,bridge)
            report['vae_precision']={name:audit.audit_parameters(getattr(pipe,name)) for name in ('vae','audio_vae')}
            if any(v['fp32_violations'] for v in report['vae_precision'].values()):raise ValueError('VAE FP32 modules were downcast')
            with Image.open(args.image) as image:
                height,width=resolve_canvas_size(*image.size,pipe.canvas_multiple,pipe.config.canvas_short_edge,pipe.config.canvas_max_pixels)
            report['canvas']=dict(height=height,width=width,source='unmodified-pipeline-default',
                canvas_short_edge=pipe.config.canvas_short_edge,canvas_max_pixels=pipe.config.canvas_max_pixels,canvas_multiple=pipe.canvas_multiple)
            report['models']=dict(condition_encoder=type(encoder).__name__,transformer=type(transformer).__name__,main_blocks=50,tp_size=8,
                trace_enabled=bool(os.environ.get('H3_BOUNDARY_TRACE_DIR')),graphs_enabled=False)
            if report['models']['trace_enabled']:raise ValueError('unexpected diagnostic trace enabled')
            for owner,method,label in ((encoder.model,'forward','condition_encoder_calls'),(transformer,'forward','transformer_calls'),
                    (pipe.scheduler,'step','video_scheduler_steps'),(pipe.audio_scheduler,'step','audio_scheduler_steps')):
                observers.enter_context(observe_calls(owner,method,label,counts,checkpoint))
            report['last_stage']='full-Native-FL2VA-generation';checkpoint()
            print('resolved_canvas='+json.dumps(report['canvas'])+'; steps=50; frames=124',flush=True)
            return pipe,encoder,transformer
        module.load_pipeline=load
        bridge.InferEngine=full.full_engine_factory(original_engine)
        previous=sys.argv[:]
        try:
            output=args.run_dir/'fl2va-native-load-sync-50steps.mp4'
            report['official_argv']=official_argv(args.model_path,args.image,output);checkpoint()
            sys.argv=[str(args.example),*report['official_argv']]
            with common.StageHeartbeat('Native-FL2VA-50steps'):module.main()
            for device in devices:torch.cuda.synchronize(device)
        finally:
            sys.argv=previous;module.load_pipeline=original_load;bridge.InferEngine=original_engine
    validate_counts(counts)
    report['last_stage']='media-decode-validation';checkpoint()
    report['media']=inspect_output(output,report['canvas'],common,args.run_dir)
    validate_source(_infinilm.__file__,capture.EXPECTED_EXTENSION,'loaded candidate')
    report['result']='MEDIA_COMPLETE';report['last_stage']='complete';checkpoint()
    print('NATIVE_FL2VA MEDIA_COMPLETE; quality=UNASSESSED; official_SSIM=NOT_TESTED; performance_valid=false',flush=True)


def orchestrate(args,report,checkpoint):
    full.disable_trace()
    report['full_forward_evidence']=validate_full_report(args.work_home)
    extension,python_root,evidence=capture.validate_diagnostic(args.work_home)
    report['diagnostic_build']=evidence
    marker=dict(line.split('=',1) for line in (args.work_home/'state/infinilm-extension.success').read_text().splitlines() if '=' in line)
    original_extension=Path(marker['extension'])
    validate_source(original_extension,OFFICIAL_EXTENSION_SHA,'official extension')
    validate_source(args.model_path/'diffusers/transformer/config.json','74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae','H3 checkpoint config')
    source=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    validate_source(source,EXAMPLE_SHA,'original entry')
    image=Path(__file__).with_name(IMAGE_NAME)
    validate_source(image,IMAGE_SHA,'template PNG')
    # Immutable uniquely named AFS input; do not overwrite prior assets or place large dependencies in project.
    target=args.work_home/'assets'/f'template-first-frame-{IMAGE_SHA}.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with target.open('xb') as stream:stream.write(image.read_bytes())
        target.chmod(0o644)
    validate_source(target,IMAGE_SHA,'persisted template PNG')
    report['first_frame']=dict(path=str(target),sha256=IMAGE_SHA)
    patcher=base.load_sibling('prepare_minimax_h3_vae_dtype_probe.py','generation_patcher')
    example=args.local_target.parent/'example/minimax_h3.py'
    report['example_patch']=patcher.patch_example(source,example)
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','generation_parent_common')
    installer=base.load_sibling('install_infinilm_private_runtime.py','generation_installer')
    hygon=base.load_sibling('prepare_minimax_h3_hygon_runtime.py','generation_hygon')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest differs')
    report['last_stage']='runtime-staging';checkpoint()
    with common.StageHeartbeat('generation-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),args.local_target)
        report['private_distributions']=installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target))
        os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        report['hygon_runtime']=hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    environment={'PYTHONPATH':os.pathsep.join([str(args.local_target),str(args.local_target.parent/'hygon-site-packages'),str(python_root),os.environ.get('PYTHONPATH','')]),'H3_BOUNDARY_TRACE_DIR':''}
    folder=args.run_dir/'native';folder.mkdir()
    report['last_stage']='Native-full-generation';checkpoint()
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--run-dir',str(folder),'--model-path',str(args.model_path),
        '--diag-extension',str(extension),'--extension-sha256',evidence['extension_sha256'],'--example',str(example),
        '--example-sha256',report['example_patch']['destination_sha256'],'--image',str(target)]
    base.run_worker(command,folder/'worker.log',environment)
    child=json.loads((folder/'native-generation.json').read_text(encoding='utf-8'))
    validate_counts(child['observed_calls'])
    if child.get('result')!='MEDIA_COMPLETE' or child.get('extension_sha256')!=capture.EXPECTED_EXTENSION:
        raise ValueError('Native media worker incomplete or wrong binary')
    validate_source(source,EXAMPLE_SHA,'original entry after generation')
    validate_source(original_extension,OFFICIAL_EXTENSION_SHA,'official extension after generation')
    report['generation']=child;report['result']='MEDIA_COMPLETE';report['last_stage']='complete';checkpoint()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    for name in ('extension-sha256','example-sha256'):p.add_argument('--'+name)
    for name in ('work-home','run-dir','local-target','model-path','diag-extension','example','image'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv)
    required=('diag_extension','extension_sha256','example','example_sha256','image') if args.worker else ('work_home','local_target')
    if any(getattr(args,key) is None for key in required):p.error('missing worker or parent paths')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.1-load-sync-template-Native-signature-preserved',result='FAILED',last_stage='preflight',
        quality_status='UNASSESSED',official_SSIM='NOT_TESTED',Qwen_parity='NOT_TESTED',performance_measurement_valid=False,
        scope='full-Native-FL2VA-media-and-call-structure-only',observed_calls={},
        workload=dict(task='fl2va',prompt='A dog falls asleep',seed=42,steps=50,num_frames=124,gpus=list(range(8)),canvas='pipeline-default'))
    start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds_including_setup_and_loading']=time.monotonic()-start
        base.write_json(args.run_dir/'native-generation.json',report)
    checkpoint()
    try:
        (worker if args.worker else orchestrate)(args,report,checkpoint);return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
