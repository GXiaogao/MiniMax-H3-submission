#!/usr/bin/env python3
"""Isolated 0/1-block H3 diagnostic. Never a generation or competition workload."""
from __future__ import annotations
import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

spec=importlib.util.spec_from_file_location('h3_parity_base',Path(__file__).with_name('diagnose_minimax_h3_transformer_parity.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def validate_prefix_loading(info,count,total):
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('active reference weights failed loading')
    for name in info.get('unexpected_keys',[]):
        match=re.fullmatch(r'transformer_blocks\.(\d+)\..+',name)
        if not match or not count<=int(match[1])<total:
            raise ValueError('unexpected active/unknown weight: '+name)
    return {**info,'unexpected_keys':[]}

def prefix_engine_factory(factory,count):
    if count not in (0,1): raise ValueError('only diagnostic prefixes 0/1 supported')
    def create(*a,**kw):
        kw['hf_config_overrides']={**kw.get('hf_config_overrides',{}),'num_layers':count,'load_only_model_keys':True}
        return factory(*a,**kw)
    return create

def load_reference_prefix(factory,args,*a,**kw):
    if args.prefix not in (0,1): raise ValueError('single-GPU reference supports only diagnostic prefixes 0/1')
    kw['num_layers']=args.prefix
    # The small diagnostic model fits on one GPU. Automatic balancing separates
    # token_refiner output from the packed-layout indices used by index_copy.
    kw['device_map']={'':'cuda:0'}
    model,info=factory(*a,**kw)
    base.write_json(args.run_dir/'reference-prefix-loading.json',{
        'prefix':args.prefix,'requested_device_map':kw['device_map'],'unfiltered_loading_info':info})
    if len(model.transformer_blocks)!=args.prefix: raise RuntimeError('Reference prefix override did not take effect')
    return model,validate_prefix_loading(info,args.prefix,50)

def copy_inputs(source,target):
    base.read_bundle(source,'first-inputs')
    for suffix in ('json','npz'):
        shutil.copyfile(source/f'first-inputs.{suffix}',target/f'first-inputs.{suffix}')

def restore_inputs(directory,device):
    import torch
    arrays,meta=base.read_bundle(directory,'first-inputs')
    if set(arrays)!=set(base.INPUT_NAMES): raise ValueError('incomplete input bundle')
    return {k:torch.from_numpy(v).to(device=device,dtype=getattr(torch,meta['fields'][k]['torch_dtype'])) for k,v in arrays.items()}

def native(args):
    import torch
    import infinicore
    from infinicore.lib import _infinicore
    import infinilm.multimodal.minimax_h3 as bridge
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','prefix_native_common')
    devices=list(range(8))
    common.validate_device_counts(devices,torch_count=torch.cuda.device_count(),native_count=_infinicore.get_device_count(_infinicore.Device.Type.HYGON))
    staging=common.configure_weight_staging(infinicore,devices,_infinicore.get_device_count(_infinicore.Device.Type.CPU))
    factory=bridge.InferEngine
    bridge.InferEngine=prefix_engine_factory(factory,args.prefix)
    try:
        runner=bridge.MiniMaxH3TransformerRunner(args.model_path/'diffusers',device='hygon',tp_device_ids=devices)
    finally:
        bridge.InferEngine=factory
    keys=runner.engine.state_dict_keyname()
    blocks={int(m[1]) for k in keys if (m:=re.match(r'transformer_blocks\.(\d+)\.',k))}
    if blocks!=set(range(args.prefix)): raise RuntimeError('Native prefix override did not take effect')
    base.write_json(args.run_dir/'native-prefix-model.json',{'prefix':args.prefix,'expected_keys':keys,
        'config':runner.engine.hf_config,'weight_staging':staging,'adapter_sha256':base.sha256(inspect.getfile(bridge))})
    model=bridge.MiniMaxH3DiffusersAdapter(runner).eval()
    with torch.inference_mode():
        output=model(**restore_inputs(args.run_dir,'cuda:0'),return_dict=False)
    base.save_bundle(args.run_dir,'first-native-outputs',dict(zip(('video','audio'),output)))

def reference(args):
    import diffusers
    real=diffusers.MiniMaxH3Transformer3DModel
    class PrefixModel:
        @staticmethod
        def from_pretrained(*a,**kw):
            return load_reference_prefix(real.from_pretrained,args,*a,**kw)
    diffusers.MiniMaxH3Transformer3DModel=PrefixModel
    try: base.reference_forward(args)
    finally: diffusers.MiniMaxH3Transformer3DModel=real

def orchestrate(args,report,checkpoint):
    source=base.read_bundle(args.source_run,'first-inputs')[1]
    if source['sha256']!='e2b542a7102402d075f99af716e6c6480d0f4ef707b7cc6016cb8e5a38a8a2a7':
        raise ValueError('not the agreed 20261105T080500Z-408969 input snapshot')
    report['input_sha256']=source['sha256']
    report['source_run']=str(args.source_run)
    config=json.loads((args.model_path/'diffusers/transformer/config.json').read_text())
    if config.get('num_layers')!=50: raise ValueError('expected unchanged 50-block checkpoint')
    marker=dict(line.split('=',1) for line in (args.work_home/'state/infinilm-extension.success').read_text().splitlines() if '=' in line)
    if marker.get('build_version')!='1.5-hygon-fl2va-no-cpu-runtime': raise ValueError('incompatible extension')
    report['extension_sha256']=base.sha256(marker['extension'])
    if report['extension_sha256']!='e7d92053530cc1626dc112927788aa7f2116653a6803ac3cc7e23710610ed0dd':
        raise ValueError('extension changed since full-model comparison')
    if base.sha256(args.model_path/'diffusers/transformer/config.json')!='74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae':
        raise ValueError('checkpoint config changed')
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','prefix_parent_common')
    installer=base.load_sibling('install_infinilm_private_runtime.py','prefix_installer')
    hygon=base.load_sibling('prepare_minimax_h3_hygon_runtime.py','prefix_hygon')
    args.local_target.parent.mkdir(parents=True,exist_ok=True)
    # Only needed by the reference device-routing helpers; never edits the server example.
    example=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    if base.sha256(example)!='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4':
        raise ValueError('official example changed')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:
        raise ValueError('runtime manifest mismatch')
    report['last_stage']='staging';checkpoint()
    with common.StageHeartbeat('prefix-runtime-staging'):
        wheels=installer.verify_wheels(Path(manifest['wheel_dir']))
        installer.install_wheels_direct(wheels,args.local_target)
        installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target))
        os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    report['comparison']={}
    for count in (0,1):
        directory=args.run_dir/f'prefix-{count}';directory.mkdir()
        copy_inputs(args.source_run,directory)
        for worker in ('native','reference'):
            stage=f'prefix-{count}-{worker}';report['last_stage']=stage;checkpoint()
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',worker,'--prefix',str(count),
                '--run-dir',str(directory),'--model-path',str(args.model_path),'--example',str(example)]
            with common.StageHeartbeat(stage): base.run_worker(command,directory/f'{worker}.log',{})
            report['completed_workers'].append(stage);checkpoint()
        a,_=base.read_bundle(directory,'first-reference-outputs');b,_=base.read_bundle(directory,'first-native-outputs')
        if set(a)!=set(b) or set(a)!={'video','audio'}: raise ValueError('incomplete outputs')
        report['comparison'][str(count)]={k:base.compare_arrays(a[k],b[k]) for k in ('video','audio')};checkpoint()
    report['result']='COMPARISON_COMPLETE';report['last_stage']='complete'
    print(json.dumps(report['comparison'],indent=2),flush=True)
    print('PREFIX_DIAGNOSTIC COMPLETE; NOT_GENERATION; quality_status=UNASSESSED',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('native','reference'))
    p.add_argument('--prefix',type=int,choices=(0,1))
    for name in ('run-dir','model-path','first-frame','work-home','local-target','source-run','example'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.prefix is None or args.example is None: p.error('worker requires --prefix/--example')
        (native if args.worker=='native' else reference)(args);return 0
    if any(getattr(args,k) is None for k in ('work_home','local_target','source_run')): p.error('parent requires work-home/local-target/source-run')
    report={'version':'1.1-prefix-reference-single-device','result':'FAILED','last_stage':'preflight','quality_status':'UNASSESSED',
        'validation_scope':'0-and-1-main-block-isolation-with-common-input-and-output-modules',
        'not_full_model':True,'performance_measurement_valid':False,'completed_workers':[]}
    started=time.monotonic()
    def checkpoint():
        report['elapsed_seconds']=time.monotonic()-started;base.write_json(args.run_dir/'prefix-parity.json',report)
    try: orchestrate(args,report,checkpoint);return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally: checkpoint()

if __name__=='__main__': raise SystemExit(main())
