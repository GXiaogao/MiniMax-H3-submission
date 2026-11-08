#!/usr/bin/env python3
"""Fresh full 50-main-block H3 forward on the load-sync candidate; not generation."""
import argparse
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import importlib.util

spec=importlib.util.spec_from_file_location('full_load_sync_capture',Path(__file__).with_name('diagnose_minimax_h3_load_sync.py'))
capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(capture)
base=capture.base
prefix=capture.prefix
PREFIX_RELATIVE=Path('logs/infinifabric-h3-load-sync-capture-20261107T034425Z-2750046/load-sync-parity.json')
PREFIX_SHA='297113efb56811584038010eec12c389a7b7fb097d77cbca97614202af952264'
INPUT_SHA='e2b542a7102402d075f99af716e6c6480d0f4ef707b7cc6016cb8e5a38a8a2a7'
REFERENCE_SHA='92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef'


def disable_trace():
    os.environ.pop('H3_BOUNDARY_TRACE_DIR',None)


def full_engine_factory(factory):
    def create(*a,**kw):
        if 'num_layers' in kw.get('hf_config_overrides',{}):raise ValueError('full diagnostic forbids num_layers override')
        kw['enable_graph_compiling']=False
        return factory(*a,**kw)
    return create


def validate_native_full(keys,config):
    blocks={int(m[1]) for key in keys if (m:=re.match(r'transformer_blocks\.(\d+)\.',key))}
    if blocks!=set(range(50)) or config.get('num_layers')!=50:
        raise ValueError('Native model must declare all50 main blocks without override')


def validate_outputs(arrays):
    import numpy as np
    expected={'video':[1,8208,96],'audio':[1,414,32]}
    if set(arrays)!=set(expected):raise ValueError('incomplete full forward outputs')
    for name,shape in expected.items():
        if list(arrays[name].shape)!=shape or not np.isfinite(arrays[name]).all():
            raise ValueError('invalid full forward output: '+name)


def full_reference_factory(factory,get_sha,expected_sha=REFERENCE_SHA):
    def create(*a,**kw):
        if 'num_layers' in kw:raise ValueError('reference full diagnostic forbids layer override')
        model,loading=factory(*a,**kw)
        config=model.config
        layers=config.get('num_layers') if hasattr(config,'get') else getattr(config,'num_layers',None)
        if len(model.transformer_blocks)!=50 or layers!=50 or get_sha(model)!=expected_sha:
            raise ValueError('reference must use pinned full50-block implementation')
        return model,loading
    return create


def validate_prefix_result(work):
    path=work/PREFIX_RELATIVE
    if base.sha256(path)!=PREFIX_SHA:raise ValueError('returned prefix report changed')
    data=json.loads(path.read_text(encoding='utf-8'))
    if (data.get('result')!='COMPARISON_COMPLETE' or data.get('completed_workers')!=['native-tp1','native-tp8','reference']
            or data.get('input_sha256')!=INPUT_SHA
            or data.get('diagnostic_build',{}).get('extension_sha256')!='ee448772f8e03947f6f33fdaa670fd4079bdd88df31f4e2655ce405aa0b34047'
            or data.get('loading_comparison_summary',{}).get('weight_samples_exact_ranks_after')!=8):
        raise ValueError('prefix numerical evidence/provenance incomplete')
    return dict(report_sha256=PREFIX_SHA,path=str(path),scope='prior-one-block-diagnostic-not-full-model')


def validate_loaded_extension(actual,args):
    if Path(actual).resolve()!=args.diag_extension.resolve() or base.sha256(actual)!=args.extension_sha256:
        raise RuntimeError('loaded full-model candidate extension path/hash differs')
    if args.extension_sha256!=capture.EXPECTED_EXTENSION:raise RuntimeError('unexpected full-model candidate digest')


def native_worker(args):
    disable_trace()
    import torch
    import infinicore
    from infinicore.lib import _infinicore
    from infinilm.lib import _infinilm
    import infinilm.multimodal.minimax_h3 as bridge
    validate_loaded_extension(_infinilm.__file__,args)
    if base.sha256(inspect.getfile(bridge))!='5e5fc07bf0c02925c553603285fd163c44c973eb6c3de71349cf6c7677e3e4e5':
        raise ValueError('Native adapter changed')
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','full_native_common')
    devices=list(range(8))
    common.validate_device_counts(devices,torch_count=torch.cuda.device_count(),
        native_count=_infinicore.get_device_count(_infinicore.Device.Type.HYGON))
    staging=common.configure_weight_staging(infinicore,devices,_infinicore.get_device_count(_infinicore.Device.Type.CPU))
    original=bridge.InferEngine
    bridge.InferEngine=full_engine_factory(original)
    print('stage=Native-full50-load status=START; trace=DISABLED; num_layers_override=NONE',flush=True)
    try:runner=bridge.MiniMaxH3TransformerRunner(args.model_path/'diffusers',device='hygon',tp_device_ids=devices)
    finally:bridge.InferEngine=original
    keys=runner.engine.state_dict_keyname();config=runner.engine.hf_config
    validate_native_full(keys,config)
    if os.environ.get('H3_BOUNDARY_TRACE_DIR'):raise RuntimeError('trace unexpectedly enabled during full-model load')
    base.write_json(args.run_dir/'native-full-model.json',dict(main_blocks=50,tp_size=8,expected_keys=keys,
        config=config,weight_staging=staging,trace_enabled=False,graphs_enabled=False,
        extension_sha256=args.extension_sha256))
    model=bridge.MiniMaxH3DiffusersAdapter(runner).eval()
    print('stage=Native-full50-forward status=START; inputs=original-capture',flush=True)
    with torch.inference_mode():output=model(**prefix.restore_inputs(args.run_dir,'cuda:0'),return_dict=False)
    base.save_bundle(args.run_dir,'first-native-outputs',dict(zip(('video','audio'),output)))
    arrays,_=base.read_bundle(args.run_dir,'first-native-outputs');validate_outputs(arrays)
    print('stage=Native-full50-forward status=DONE',flush=True)


def reference_worker(args):
    disable_trace()
    import diffusers
    real=diffusers.MiniMaxH3Transformer3DModel
    create=full_reference_factory(real.from_pretrained,lambda model:base.sha256(inspect.getfile(type(model))))
    class Reference:
        from_pretrained=staticmethod(create)
    diffusers.MiniMaxH3Transformer3DModel=Reference
    try:base.reference_forward(args)
    finally:diffusers.MiniMaxH3Transformer3DModel=real
    arrays,_=base.read_bundle(args.run_dir,'first-reference-outputs');validate_outputs(arrays)


def complete_worker(run,worker,report,checkpoint):
    if worker not in ('native','reference') or worker in report['completed_workers']:
        raise ValueError('invalid or duplicate full-model worker completion')
    folder=run/worker
    path=folder/('native-full-model.json' if worker=='native' else 'reference-model.json')
    model=json.loads(path.read_text(encoding='utf-8'))
    if worker=='native':
        validate_native_full(model['expected_keys'],model['config'])
        if (model.get('main_blocks')!=50 or model.get('tp_size')!=8 or model.get('trace_enabled') is not False
                or model.get('graphs_enabled') is not False or model.get('extension_sha256')!=capture.EXPECTED_EXTENSION):
            raise ValueError('Native full model evidence differs')
    else:
        if (model.get('config',{}).get('num_layers')!=50 or model.get('implementation_sha256')!=REFERENCE_SHA
                or model.get('fp32_violations')!=[] or 'loading_info' not in model):
            raise ValueError('reference full model evidence differs')
        if any(model['loading_info'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('reference full weights incomplete')
        base.validate_device_map(model['device_map'],list(range(8)))
    arrays,bundle=base.read_bundle(folder,'first-native-outputs' if worker=='native' else 'first-reference-outputs')
    validate_outputs(arrays)
    report.setdefault('output_bundles',{})[worker]=bundle
    report.setdefault('worker_models',{})[worker]=dict(main_blocks=50,metadata_sha256=base.sha256(path))
    report['completed_workers'].append(worker);checkpoint()


def orchestrate(args,report,checkpoint):
    report['prefix_evidence']=validate_prefix_result(args.work_home)
    source=args.work_home/'logs/infinifabric-transformer-parity-20261105T080500Z-408969'
    _,meta=base.read_bundle(source,'first-inputs')
    if meta['sha256']!=INPUT_SHA:raise ValueError('original full-forward inputs changed')
    report['input_sha256']=INPUT_SHA
    extension,python_root,evidence=capture.validate_diagnostic(args.work_home)
    report['diagnostic_build']=evidence;checkpoint()
    config=args.model_path/'diffusers/transformer/config.json'
    if base.sha256(config)!='74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae':raise ValueError('checkpoint config changed')
    original_marker=dict(line.split('=',1) for line in (args.work_home/'state/infinilm-extension.success').read_text().splitlines() if '=' in line)
    report['official_extension_sha256']=base.sha256(original_marker['extension'])
    if report['official_extension_sha256']!='e7d92053530cc1626dc112927788aa7f2116653a6803ac3cc7e23710610ed0dd':raise ValueError('official extension changed')
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','full_parent_common')
    installer=base.load_sibling('install_infinilm_private_runtime.py','full_installer')
    hygon=base.load_sibling('prepare_minimax_h3_hygon_runtime.py','full_hygon')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest differs')
    example=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    if base.sha256(example)!='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4':raise ValueError('official entry changed')
    args.local_target.parent.mkdir(parents=True,exist_ok=True)
    report['last_stage']='runtime-staging';checkpoint()
    with common.StageHeartbeat('full-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),args.local_target)
        installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target))
        os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        report['hygon_runtime']=hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    environment={'PYTHONPATH':os.pathsep.join([str(args.local_target),str(args.local_target.parent/'hygon-site-packages'),str(python_root),os.environ.get('PYTHONPATH','')]),'H3_BOUNDARY_TRACE_DIR':''}
    for worker in ('native','reference'):
        folder=args.run_dir/worker;folder.mkdir();prefix.copy_inputs(source,folder)
        report['last_stage']=worker+'-full50';checkpoint()
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',worker,'--run-dir',str(folder),
            '--model-path',str(args.model_path),'--example',str(example),'--diag-extension',str(extension),
            '--extension-sha256',evidence['extension_sha256']]
        with common.StageHeartbeat(worker+'-full50'):base.run_worker(command,folder/'worker.log',environment)
        complete_worker(args.run_dir,worker,report,checkpoint)
    report['last_stage']='comparison';checkpoint()
    native,_=base.read_bundle(args.run_dir/'native','first-native-outputs')
    reference,_=base.read_bundle(args.run_dir/'reference','first-reference-outputs')
    report['comparison']={name:dict(base.compare_arrays(reference[name],native[name]),
        scope='all-elements-full50-block-single-forward',main_blocks=50) for name in ('video','audio')}
    report['result']='COMPARISON_COMPLETE';report['last_stage']='complete';checkpoint()
    print('FULL50_FORWARD COMPARISON_COMPLETE; generation=NOT_TESTED; Qwen_parity=NOT_TESTED; performance_valid=false',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=('native','reference'));p.add_argument('--extension-sha256')
    for name in ('run-dir','work-home','local-target','model-path','example','diag-extension'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.example is None or args.diag_extension is None or not args.extension_sha256:p.error('worker needs example/diag-extension/extension-sha256')
        (native_worker if args.worker=='native' else reference_worker)(args);return 0
    if args.work_home is None or args.local_target is None:p.error('parent needs work-home/local-target')
    report=dict(version='1.1-rebuilt-full50-first-forward-load-sync',result='FAILED',last_stage='preflight',completed_workers=[],
        requested_main_blocks=50,requested_transformer_forwards={'native':1,'reference':1},
        quality_status='UNASSESSED',Qwen_parity='NOT_TESTED',generation_status='NOT_TESTED',performance_measurement_valid=False,
        scope='full-Transformer-first-forward-not-generation',trace_enabled=False,graphs_enabled=False)
    start=time.monotonic()
    def checkpoint():report['elapsed_seconds']=time.monotonic()-start;base.write_json(args.run_dir/'full-forward-parity.json',report)
    try:orchestrate(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
