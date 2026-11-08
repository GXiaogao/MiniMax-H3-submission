#!/usr/bin/env python3
"""Read-only recovery of an existing video after the v1.1 count-check defect.

This validates a prior artifact, never executes inference and never reports a
new generation or controlled performance measurement. Original evidence stays intact.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback

spec=importlib.util.spec_from_file_location('media_recovery_generation',Path(__file__).with_name('run_minimax_h3_load_sync_generation.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
PRIOR=Path('logs/infinifabric-h3-load-sync-generation-20261107T082020Z-3236632')
VIDEO='fl2va-native-load-sync-50steps.mp4'
COUNT_ERROR='ValueError: incomplete fresh Native encoding/50-step denoising schedules'
AV_WHEEL='av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl'


def validate_previous(data,text):
    expected=dict(task='fl2va',prompt='A dog falls asleep',steps=50,num_frames=124,seed=42,gpus=list(range(8)),canvas='pipeline-default')
    if (data.get('version')!='1.1-load-sync-template-Native-signature-preserved' or data.get('result')!='FAILED'
            or data.get('error')!=COUNT_ERROR or data.get('extension_sha256')!=g.capture.EXPECTED_EXTENSION
            or data.get('workload')!=expected or data.get('canvas',{}).get('width')!=768 or data.get('canvas',{}).get('height')!=928):
        raise ValueError('prior receipt differs from the specific count-check failure')
    model=data.get('models',{})
    if any(model.get(k)!=v for k,v in dict(main_blocks=50,tp_size=8,trace_enabled=False,graphs_enabled=False).items()):
        raise ValueError('prior Native model/trace provenance differs')
    # Recover only the explicit 49/49 loop in this returned run; do not infer generic schedules from counts.
    if (not re.search(r'100%\|[^\n\r]*49/49',text) or f'[done] Saved to:' not in text
            or f'/{PRIOR.name}/native/{VIDEO}' not in text
            or 'native_trace stage=output.media_encode status=DONE' not in text
            or 'native_trace stage=generation.pipeline_total status=DONE' not in text):
        raise ValueError('prior log does not confirm completed loop and media encoding')
    for label in ('transformer_calls','video_scheduler_steps','audio_scheduler_steps'):
        values=[int(x) for x in re.findall(r'call='+label+r' index=(\d+) status=DONE',text)]
        if values!=list(range(1,50)):raise ValueError('prior log missing/duplicate/nonsequential '+label)
    g.validate_counts(data.get('observed_calls',{}),49)
    if not re.search(r'call=condition_encoder_calls index=1 status=DONE',text):raise ValueError('no fresh encoder completion in prior log')
    return 49


def recover(args,report,checkpoint):
    native=args.work_home/PRIOR/'native'
    metadata=native/'native-generation.json';log=native/'worker.log';video=native/VIDEO
    data=json.loads(metadata.read_text(encoding='utf-8'))
    text=log.read_text(encoding='utf-8')
    report['validated_prior_evaluations']=validate_previous(data,text)
    report['source_evidence']=dict(run_dir=str(native.parent),receipt_sha256=g.base.sha256(metadata),worker_log_sha256=g.base.sha256(log))
    if data.get('official_argv')!=g.official_argv(Path('/public/home/aclsylqidf/shared/MiniMax-H3'),
            args.work_home/'assets'/f'template-first-frame-{g.IMAGE_SHA}.png',video):
        raise ValueError('prior exact model/image/workload/output arguments differ')
    image=args.work_home/'assets'/f'template-first-frame-{g.IMAGE_SHA}.png'
    g.validate_source(image,g.IMAGE_SHA,'original template input')
    before=g.base.sha256(video)
    report['last_stage']='media-runtime-staging';checkpoint()
    common=g.base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','recovery_common')
    installer=g.base.load_sibling('install_infinilm_private_runtime.py','recovery_installer')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:
        raise ValueError('runtime manifest differs')
    # Only AV is needed for media decoding; no Torch, model or GPU runtime staging.
    wheel=Path(manifest['wheel_dir'])/AV_WHEEL
    g.validate_source(wheel,installer.EXPECTED_WHEELS[AV_WHEEL],'offline AV wheel')
    installer.install_wheels_direct([wheel],args.local_target)
    sys.path.insert(0,str(args.local_target))
    report['last_stage']='existing-media-decode';checkpoint()
    report['media']=g.inspect_output(video,data['canvas'],common,args.run_dir)
    if g.base.sha256(video)!=before:raise ValueError('prior video changed during inspection')
    g.validate_source(metadata,report['source_evidence']['receipt_sha256'],'original receipt after inspection')
    g.validate_source(log,report['source_evidence']['worker_log_sha256'],'original log after inspection')
    report['result']='EXISTING_MEDIA_VALIDATED';report['last_stage']='complete';checkpoint()
    print(f'EXISTING_MEDIA_VALIDATED; new_inference=false; quality=UNASSESSED; original_video={video}',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work-home','run-dir','local-target'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.0-existing-media-recovery',result='FAILED',last_stage='prior-evidence-validation',
        new_inference_executed=False,quality_status='UNASSESSED',official_SSIM='NOT_TESTED',performance_measurement_valid=False)
    start=time.monotonic()
    def checkpoint():
        report['inspection_elapsed_seconds']=time.monotonic()-start
        g.base.write_json(args.run_dir/'media-recovery.json',report)
    checkpoint()
    try:recover(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
