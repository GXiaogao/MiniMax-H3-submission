#!/usr/bin/env python3
"""Generate an unchanged legacy reference, then compare two existing FL2VA videos.

Reference tools are diagnostic only. No Native inference is replaced or repeated.
Neither provided-video comparison time nor different placement is a speedup result.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

spec=importlib.util.spec_from_file_location('reference_generation_helpers',Path(__file__).with_name('run_minimax_h3_load_sync_generation.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
RECEIPT_RELATIVE=Path('logs/infinifabric-h3-existing-media-20261107T090136Z-3046211/media-recovery.json')
RECEIPT_SHA='680e95cddece1f3b078866af7ca0d583dacf9915f3cbc4a45fecc1c9c16a6f90'
NATIVE_RELATIVE=Path('logs/infinifabric-h3-load-sync-generation-20261107T082020Z-3236632/native/fl2va-native-load-sync-50steps.mp4')
NATIVE_SHA='30cfd9ec32dd3e5cdc153b3237de91568bc2171e681681baa027f42b070537ab'
COMPARE_SHA='e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590'
QWEN_SHA='0ee6d35edd380adf4c5ef9cb73ef061a4e75de8aa4c20164eecafdc965fe56db'
H3_SHA='92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef'


def reference_argv(model,image,output):
    return g.official_argv(model,image,output)+['--legacy','--legacy-text-gpus=0,1,2,3',
        '--legacy-transformer-gpus=4,5,6,7','--legacy-memory-fraction=0.72']


def compare_command(script,model,native,reference,output):
    return [sys.executable,'-u',str(script),f'--model-path={model}','--tasks=fl2va',
        f'--infinilm-fl2va-video={native}',f'--reference-fl2va-video={reference}',f'--output-dir={output}']


def validate_native(work):
    receipt=work/RECEIPT_RELATIVE
    g.validate_source(receipt,RECEIPT_SHA,'returned media recovery report')
    data=json.loads(receipt.read_text(encoding='utf-8'))
    media=data.get('media',{})
    if (data.get('result')!='EXISTING_MEDIA_VALIDATED' or data.get('new_inference_executed') is not False
            or media.get('sha256')!=NATIVE_SHA or media.get('decoded_frame_count')!=124
            or (media.get('width'),media.get('height'),media.get('average_frame_rate'))!=(768,928,24.0)):
        raise ValueError('returned Native media provenance differs')
    native=work/NATIVE_RELATIVE
    g.validate_source(native,NATIVE_SHA,'existing Native video')
    return native


def parse_comparison(text):
    if (not re.search(r'Size:\s+PASS\s+\(PyTorch 768x928, InfiniLM 768x928\)',text)
            or not re.search(r'Length:\s+PASS\s+\(PyTorch 124 frames, InfiniLM 124 frames\)',text)):
        raise ValueError('official comparison lacks matching full size/length output')
    matches=re.findall(r'Average SSIM:\s+(-?[0-9]+(?:\.[0-9]+)?)%',text)
    if len(matches)!=1:raise ValueError('official comparison lacks unique SSIM output')
    score=float(matches[0])
    if not math.isfinite(score) or not -100<=score<=100:raise ValueError('invalid displayed SSIM')
    return dict(size_match=True,length_match=True,displayed_average_ssim_percent=score,
        metric_precision='official-stdout-rounded-to-3-decimals',quality_status='MEASURED_NO_ACCEPTANCE_THRESHOLD',
        performance_measurement_valid=False)


def preflight(args,report,checkpoint):
    import av
    import numpy as np
    import scipy
    import skimage
    from skimage.metrics import structural_similarity
    image=np.arange(256,dtype=np.uint8).reshape(16,16)
    score=float(structural_similarity(image,image,data_range=255))
    if score!=1.0:raise ValueError('SSIM dependency self-check failed')
    report.update(result='COMPARISON_DEPENDENCIES_READY',last_stage='complete',
        versions=dict(av=av.__version__,numpy=np.__version__,scipy=scipy.__version__,skimage=skimage.__version__))
    checkpoint()


def reference_worker(args,report,checkpoint):
    g.full.disable_trace()
    import torch
    from PIL import Image
    from infinilm.lib import _infinilm
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    g.validate_source(_infinilm.__file__,g.OFFICIAL_EXTENSION_SHA,'imported original extension')
    g.validate_source(args.example,g.EXAMPLE_SHA,'original reference entry')
    g.validate_source(args.image,g.IMAGE_SHA,'same template PNG')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=8:raise ValueError('reference requires eight visible DCUs')
    common=g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','reference_common')
    audit=g.base.load_sibling('diagnose_minimax_h3_vae_roundtrip.py','reference_audit')
    module=common.load_official_example(args.example)
    common.install_official_stage_tracing(module)
    original_load=module.load_pipeline
    with ExitStack() as observations:
        def load(*a,**kw):
            if (kw.get('legacy') is not True or kw.get('legacy_text_devices')!=list(range(4))
                    or kw.get('legacy_transformer_devices')!=list(range(4,8))):
                raise ValueError('must use the original legacy reference placement')
            report['last_stage']='original-reference-model-load';checkpoint()
            pipe,encoder,transformer=original_load(*a,**kw)
            if type(encoder) is not module.Qwen3VLForConditionalGeneration or type(transformer) is not module.MiniMaxH3Transformer3DModel:
                raise ValueError('reference core must be the exact original legacy classes')
            g.validate_source(inspect.getfile(type(encoder)),QWEN_SHA,'reference Qwen implementation')
            g.validate_source(inspect.getfile(type(transformer)),H3_SHA,'reference H3 implementation')
            if len(transformer.transformer_blocks)!=50 or transformer.config.num_layers!=50:raise ValueError('reference must load full50 main blocks')
            maps={name:{k:str(v) for k,v in model.hf_device_map.items()} for name,model in (('Qwen',encoder),('H3',transformer))}
            g.base.validate_device_map(maps['Qwen'],list(range(4)))
            g.base.validate_device_map(maps['H3'],list(range(4,8)))
            report['models']=dict(condition_encoder=type(encoder).__name__,transformer=type(transformer).__name__,main_blocks=50,device_maps=maps,
                qwen_source_sha256=QWEN_SHA,h3_source_sha256=H3_SHA)
            report['parameter_audits']={name:audit.audit_parameters(model) for name,model in
                (('Qwen',encoder),('H3',transformer),('vae',pipe.vae),('audio_vae',pipe.audio_vae))}
            # Preserve the protected baseline's exact .to(dtype=DTYPE) calls. Audit, do not repair it here.
            report['reference_precision_profile']='unchanged-original-entry; Native uses separately-declared VAE preservation patch'
            with Image.open(args.image) as image:
                height,width=resolve_canvas_size(*image.size,pipe.canvas_multiple,pipe.config.canvas_short_edge,pipe.config.canvas_max_pixels)
            if (width,height)!=(768,928):raise ValueError('reference default canvas differs from existing Native')
            report['canvas']=dict(width=width,height=height,source='unmodified-pipeline-default')
            for label,owner in (('video',pipe.scheduler),('audio',pipe.audio_scheduler)):
                g.validate_source(inspect.getfile(type(owner)),g.SCHEDULER_SHA,'reference scheduler')
                observations.enter_context(g.observe_schedule(owner,label,report['schedules'],checkpoint))
            for owner,method,label in ((encoder.model,'forward','condition_encoder_calls'),(transformer,'forward','transformer_calls'),
                    (pipe.scheduler,'step','video_scheduler_steps'),(pipe.audio_scheduler,'step','audio_scheduler_steps')):
                observations.enter_context(g.observe_calls(owner,method,label,report['observed_calls'],checkpoint))
            report['last_stage']='full-reference-generation';checkpoint()
            return pipe,encoder,transformer
        module.load_pipeline=load
        previous=sys.argv[:]
        try:
            output=args.run_dir/'fl2va-reference-original-50steps.mp4'
            report['official_argv']=reference_argv(args.model_path,args.image,output)
            sys.argv=[str(args.example),*report['official_argv']]
            with common.StageHeartbeat('original-reference-FL2VA'):module.main()
            for device in range(8):torch.cuda.synchronize(device)
        finally:sys.argv=previous;module.load_pipeline=original_load
    g.validate_counts(report['observed_calls'],g.validate_schedules(report['schedules']))
    report['last_stage']='reference-media-validation';checkpoint()
    report['media']=g.inspect_output(output,report['canvas'],common,args.run_dir)
    g.validate_source(args.example,g.EXAMPLE_SHA,'original entry after generation')
    report.update(result='MEDIA_COMPLETE',last_stage='complete');checkpoint()


def orchestrate(args,report,checkpoint):
    native=validate_native(args.work_home)
    image=args.work_home/'assets'/f'template-first-frame-{g.IMAGE_SHA}.png'
    g.validate_source(image,g.IMAGE_SHA,'template PNG')
    example=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    g.validate_source(example,g.EXAMPLE_SHA,'original entry')
    compare=Path(__file__).with_name('minimax_h3_compare.py')
    g.validate_source(compare,COMPARE_SHA,'unmodified compare script')
    marker=dict(line.split('=',1) for line in (args.work_home/'state/infinilm-extension.success').read_text().splitlines() if '=' in line)
    original_extension=Path(marker['extension']);g.validate_source(original_extension,g.OFFICIAL_EXTENSION_SHA,'original extension')
    report['native_input']=dict(path=str(native),sha256=NATIVE_SHA,receipt_sha256=RECEIPT_SHA)
    report['baseline_entry_sha256']=g.EXAMPLE_SHA;report['compare_source_sha256']=COMPARE_SHA
    common=g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','reference_parent_common')
    installer=g.base.load_sibling('install_infinilm_private_runtime.py','reference_installer')
    hygon=g.base.load_sibling('prepare_minimax_h3_hygon_runtime.py','reference_hygon')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest differs')
    report['last_stage']='runtime-staging';checkpoint()
    with common.StageHeartbeat('reference-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),args.local_target)
        installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target));os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        report['hygon_runtime']=hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    env={'PYTHONPATH':os.pathsep.join([str(args.local_target),str(args.local_target.parent/'hygon-site-packages'),os.environ.get('PYTHONPATH','')]),'H3_BOUNDARY_TRACE_DIR':''}
    for mode in ('preflight','reference'):
        folder=args.run_dir/mode;folder.mkdir()
        report['last_stage']=mode;checkpoint()
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',mode,'--run-dir',str(folder),
            '--model-path',str(args.model_path),'--example',str(example),'--image',str(image)]
        g.base.run_worker(command,folder/'worker.log',env)
        if mode=='preflight':
            ready=json.loads((folder/'comparison-preflight.json').read_text(encoding='utf-8'))
            if ready.get('result')!='COMPARISON_DEPENDENCIES_READY':raise ValueError('comparison dependency preflight incomplete')
            report['comparison_dependencies']=ready
    result=json.loads((args.run_dir/'reference/reference-media.json').read_text(encoding='utf-8'))
    if result.get('result')!='MEDIA_COMPLETE':raise ValueError('reference media worker incomplete')
    g.validate_counts(result['observed_calls'],g.validate_schedules(result['schedules']))
    reference=Path(result['media']['path'])
    if reference!=args.run_dir/'reference/fl2va-reference-original-50steps.mp4':raise ValueError('reference output path differs')
    g.validate_source(reference,result['media']['sha256'],'reference completed video')
    report['reference']=result;report['last_stage']='official-video-comparison';checkpoint()
    cmd=compare_command(compare,args.model_path,native,reference,args.run_dir/'official-comparison')
    report['compare_command']=cmd;checkpoint()
    g.base.run_worker(cmd,args.run_dir/'official-comparison.log',env)
    report['comparison']=parse_comparison((args.run_dir/'official-comparison.log').read_text(encoding='utf-8'))
    validate_native(args.work_home)
    g.validate_source(reference,result['media']['sha256'],'reference video after comparison')
    g.validate_source(example,g.EXAMPLE_SHA,'baseline after comparison')
    g.validate_source(original_extension,g.OFFICIAL_EXTENSION_SHA,'original extension after comparison')
    g.validate_source(compare,COMPARE_SHA,'compare script after comparison')
    report.update(result='QUALITY_COMPARISON_COMPLETE',last_stage='complete',quality_status='MEASURED_NO_ACCEPTANCE_THRESHOLD');checkpoint()
    print('QUALITY_COMPARISON_COMPLETE; Native_regenerated=false; performance_valid=false; official_acceptance=UNASSESSED',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('preflight','reference'))
    for name in ('work-home','run-dir','local-target','model-path','example','image'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv)
    required=('example','image') if args.worker=='reference' else () if args.worker else ('work_home','local_target')
    if any(getattr(args,key) is None for key in required):p.error('missing worker/parent paths')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    filename='comparison-preflight.json' if args.worker=='preflight' else 'reference-media.json' if args.worker else 'reference-comparison.json'
    report=dict(version='1.0-original-reference-and-official-quality-compare',result='FAILED',last_stage='preflight',
        quality_status='UNASSESSED',performance_measurement_valid=False,Native_regenerated=False,
        reference_only_legacy=True,observed_calls={},schedules={},
        workload=dict(task='fl2va',prompt='A dog falls asleep',steps=50,num_frames=124,seed=42,canvas='pipeline-default'),
        precision_caveat='Native VAE preservation patch versus unchanged original baseline; report audits, do not silently normalize reference')
    start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds_including_setup_and_loading']=time.monotonic()-start
        g.base.write_json(args.run_dir/filename,report)
    checkpoint()
    try:
        (preflight if args.worker=='preflight' else reference_worker if args.worker else orchestrate)(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
