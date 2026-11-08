#!/usr/bin/env python3
"""Compare sampled actual-engine condition and first-block boundaries, TP=1/8/reference."""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import importlib.util

def sibling(name,alias):
    s=importlib.util.spec_from_file_location(alias,Path(__file__).with_name(name))
    m=importlib.util.module_from_spec(s);sys.modules[alias]=m;s.loader.exec_module(m);return m

prefix=sibling('diagnose_minimax_h3_prefix_parity.py','boundary_prefix')
base=prefix.base
NAMES=['condition_input','condition_projection'] + [f'condition_{i}_{stage}' for i in range(2) for stage in ('norm1','attention_output','after_attention','norm2','mlp_output','after_mlp')] + ['condition_final'] + ['packed_hidden','temb','adaln_projection','attention_input','attention_output','after_attention','mlp_input','mlp_output','after_mlp']
SAMPLING='linear-integer-indices-at-most-65536'
EXTENSION_RELATIVE=Path('diagnostics/h3-condition/20261106T071847Z-74362/InfiniLM/python/infinilm/lib/_infinilm.cpython-311-x86_64-linux-gnu.so')
BUILD_LOG_RELATIVE=Path('logs/infinifabric-build-h3-condition-20261106T071848Z-74380')
PATCHED_SOURCE_SHA='37d46589c7a0a6cc8994480eb6cf4d2abf56f268eab64c8cc9dec82853dbb479'

def sample_indices(total,count):
    import numpy as np
    if total<1 or count<1 or count>total or total>(1<<47):raise ValueError('invalid trace sampling dimensions')
    return np.zeros(1,dtype=np.int64) if count==1 else np.arange(count,dtype=np.int64)*(total-1)//(count-1)

def save_snapshot(folder,name,value):
    import numpy as np
    if (folder/f'{name}.json').exists() or (folder/f'{name}.f32').exists():raise FileExistsError(name)
    if hasattr(value,'detach'):
        tensor=value.detach();dtype=str(tensor.dtype).removeprefix('torch.')
        array=tensor.float().cpu().numpy()
    else:array=np.asarray(value);dtype=str(array.dtype)
    count=min(65536,array.size)
    samples=np.asarray(array.reshape(-1)[sample_indices(array.size,count)],dtype='<f4')
    if not np.isfinite(samples).all():raise ValueError('nonfinite saved samples')
    with (folder/f'{name}.f32').open('xb') as stream:stream.write(samples.tobytes())
    base.write_json(folder/f'{name}.json',{'shape':list(array.shape),'dtype':dtype,'elements':array.size,
        'sample_count':count,'storage_dtype':'float32','rank':0,'sampling':SAMPLING,'all_elements_finite_checked':False})

def read_snapshot(folder,name):
    import numpy as np
    meta=json.loads((folder/f'{name}.json').read_text())
    shape=meta['shape'];total=math.prod(shape)
    if not shape or any(type(n)!=int or n<1 for n in shape):raise ValueError('invalid boundary shape')
    if meta['elements']!=total or meta['sample_count']!=min(65536,total) or meta['sampling']!=SAMPLING:
        raise ValueError('invalid snapshot sampling')
    if meta['storage_dtype']!='float32' or meta['rank']!=0 or meta['dtype'] not in ('float32','bfloat16'):
        raise ValueError('unsupported snapshot dtype/rank')
    raw=(folder/f'{name}.f32').read_bytes()
    if len(raw)!=meta['sample_count']*4:raise ValueError('snapshot size mismatch')
    array=np.frombuffer(raw,dtype='<f4').copy()
    if not np.isfinite(array).all():raise ValueError('nonfinite saved samples')
    return array,{**meta,'data_sha256':base.sha256(folder/f'{name}.f32'),'metadata_sha256':base.sha256(folder/f'{name}.json')}

def compare_boundary(reference,native,name):
    a,am=read_snapshot(reference,name);b,bm=read_snapshot(native,name)
    if am['shape']!=bm['shape']:raise ValueError('boundary shape differs: '+name)
    result=base.compare_arrays(a,b)
    result.update(scope='saved-boundary-samples',full_shape=am['shape'],reference_dtype=am['dtype'],native_dtype=bm['dtype'],
                  reference_data_sha256=am['data_sha256'],native_data_sha256=bm['data_sha256'])
    return result

def validate_diagnostic(work):
    path=work/'state/infinilm-condition.success'
    marker=dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line)
    if marker.get('build_version')!='1.0-isolated-h3-condition-trace' or marker.get('cxx11_abi')!='1':
        raise ValueError('incompatible condition diagnostic marker')
    extension=Path(marker['extension']).resolve()
    run=(work/BUILD_LOG_RELATIVE).resolve()
    if extension!=(work/EXTENSION_RELATIVE).resolve() or Path(marker['run_dir']).resolve()!=run:
        raise ValueError('condition marker differs from returned build identity')
    manifest=Path(marker['artifacts_sha256']).resolve()
    if manifest!=run/'artifacts.sha256':raise ValueError('unexpected build digest manifest path')
    digests=[]
    for line in manifest.read_text().splitlines():
        fields=line.split(maxsplit=1)
        if len(fields)!=2:raise ValueError('malformed build digest manifest')
        digest,filename=fields
        if Path(filename.lstrip('*')).resolve()==extension:
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid extension hash')
            digests.append(digest)
    if len(digests)!=1:raise ValueError('expected exactly one extension hash in build manifest')
    expected=digests[0]
    if base.sha256(extension)!=expected:raise ValueError('diagnostic extension hash differs from build manifest')
    source=run/'minimax_h3_transformer.cpp'
    trace_meta=json.loads((run/'minimax_h3_transformer.cpp.trace.json').read_text())
    if base.sha256(source)!=PATCHED_SOURCE_SHA or trace_meta.get('patched_sha256')!=PATCHED_SOURCE_SHA:
        raise ValueError('condition trace source differs from pinned patch')
    if trace_meta.get('boundaries')!=NAMES or trace_meta.get('trace_version')!='1.0-isolated-h3-condition-trace':
        raise ValueError('condition trace source metadata differs')
    tree=extension.parents[2]
    official=work/'Projects/InfiniLM/python'
    checked={}
    for file in (official/'infinilm').rglob('*.py'):
        relative=file.relative_to(official);target=tree/relative
        checksum=base.sha256(file)
        if base.sha256(target)!=checksum:raise ValueError('diagnostic Python copy differs: '+str(relative))
        checked[str(relative)]=checksum
    if not checked:raise ValueError('no diagnostic Python sources verified')
    return extension,tree,{'extension_sha256':expected,'marker_sha256':base.sha256(path),
        'hash_validation':'extension-checked-against-build-manifest-not-independent-local-attestation',
        'manifest_sha256':base.sha256(manifest),'trace_source_sha256':PATCHED_SOURCE_SHA,
        'python_files_checked':len(checked),'python_hashes':checked,'build_marker':marker}


def native_worker(args):
    import torch
    import infinicore
    from infinilm.lib import _infinilm
    if Path(_infinilm.__file__).resolve()!=args.diag_extension.resolve():raise RuntimeError('wrong loaded diagnostic extension')
    if base.sha256(_infinilm.__file__)!=args.extension_sha256:raise RuntimeError('loaded diagnostic extension hash differs')
    import infinilm.multimodal.minimax_h3 as bridge
    trace=args.run_dir/'trace';trace.mkdir()
    os.environ['H3_BOUNDARY_TRACE_DIR']=str(trace)
    original=bridge.InferEngine
    def without_graph(*a,**kw):
        kw['enable_graph_compiling']=False
        return original(*a,**kw)
    bridge.InferEngine=without_graph
    try:prefix.native(args)
    finally:bridge.InferEngine=original;os.environ.pop('H3_BOUNDARY_TRACE_DIR',None)
    for name in NAMES:read_snapshot(trace,name)

def attach_reference(model,folder):
    handles=[]
    def before(label):
        def hook(module,args):save_snapshot(folder,label,args[0])
        return hook
    def after(label):
        def hook(module,args,output):save_snapshot(folder,label,output)
        return hook
    block=model.transformer_blocks[0]
    handles.append(block.register_forward_pre_hook(before('packed_hidden')))
    for module,label in [(model.time_embedder,'temb'),(block.adaln_proj.linear,'adaln_projection'),
                         (block.attn,'attention_output'),(block.ff,'mlp_output'),(block,'after_mlp')]:
        handles.append(module.register_forward_hook(after(label)))
    for module,label in [(block.attn,'attention_input'),(block.norm2,'after_attention'),(block.ff,'mlp_input')]:
        handles.append(module.register_forward_pre_hook(before(label)))
    if len(model.token_refiner.refiner_blocks)!=2:raise ValueError('expected two reference refiner blocks')
    handles.append(model.context_embedder.register_forward_pre_hook(before('condition_input')))
    handles.append(model.context_embedder.register_forward_hook(after('condition_projection')))
    handles.append(model.token_refiner.final_norm.register_forward_hook(after('condition_final')))
    for i,refiner in enumerate(model.token_refiner.refiner_blocks):
        for module,stage in [(refiner.norm1,'norm1'),(refiner.attn,'attention_output'),
                             (refiner.norm2,'norm2'),(refiner.ff,'mlp_output'),(refiner,'after_mlp')]:
            handles.append(module.register_forward_hook(after(f'condition_{i}_{stage}')))
        handles.append(refiner.norm2.register_forward_pre_hook(before(f'condition_{i}_after_attention')))
    return handles

def reference_worker(args):
    import diffusers
    real=diffusers.MiniMaxH3Transformer3DModel;handles=[]
    trace=args.run_dir/'trace';trace.mkdir()
    class Reference:
        @staticmethod
        def from_pretrained(*a,**kw):
            model,loading=prefix.load_reference_prefix(real.from_pretrained,args,*a,**kw)
            if base.sha256(inspect.getfile(type(model)))!='92d665e9fa10b1417088341dce5977663a180f9e6fa404fb9330062f353fe7ef':
                raise ValueError('reference implementation changed')
            handles.extend(attach_reference(model,trace));return model,loading
    diffusers.MiniMaxH3Transformer3DModel=Reference
    try:base.reference_forward(args)
    finally:
        diffusers.MiniMaxH3Transformer3DModel=real
        for handle in handles:handle.remove()
    for name in NAMES:read_snapshot(trace,name)

def orchestrate(args,report,checkpoint):
    source=args.work_home/'logs/infinifabric-transformer-parity-20261105T080500Z-408969'
    _,meta=base.read_bundle(source,'first-inputs')
    if meta['sha256']!='e2b542a7102402d075f99af716e6c6480d0f4ef707b7cc6016cb8e5a38a8a2a7':raise ValueError('original input differs')
    report['input_sha256']=meta['sha256']
    extension,python_root,evidence=validate_diagnostic(args.work_home)
    report['diagnostic_build']=evidence;checkpoint()
    config=args.model_path/'diffusers/transformer/config.json'
    if base.sha256(config)!='74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae':raise ValueError('checkpoint config changed')
    original_marker=dict(line.split('=',1) for line in (args.work_home/'state/infinilm-extension.success').read_text().splitlines() if '=' in line)
    report['official_extension_sha256']=base.sha256(original_marker['extension'])
    if report['official_extension_sha256']!='e7d92053530cc1626dc112927788aa7f2116653a6803ac3cc7e23710610ed0dd':raise ValueError('official extension changed')
    common=base.load_sibling('run_minimax_h3_fl2va_native_smoke.py','boundary_common')
    installer=base.load_sibling('install_infinilm_private_runtime.py','boundary_installer')
    hygon=base.load_sibling('prepare_minimax_h3_hygon_runtime.py','boundary_hygon')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest mismatch')
    example=args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    if base.sha256(example)!='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4':raise ValueError('official entry changed')
    args.local_target.parent.mkdir(parents=True,exist_ok=True)
    report['last_stage']='runtime-staging';checkpoint()
    with common.StageHeartbeat('boundary-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),args.local_target)
        installer.check_target_contents(args.local_target)
        sys.path.insert(0,str(args.local_target))
        os.environ['PYTHONPATH']=str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        runtime=hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',installer,
            args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    report['hygon_runtime']=runtime
    environment={'PYTHONPATH':os.pathsep.join([str(args.local_target),str(args.local_target.parent/'hygon-site-packages'),str(python_root),os.environ.get('PYTHONPATH','')]),'H3_BOUNDARY_TRACE_DIR':''}
    for worker,tp in [('native',1),('native',8),('reference',1)]:
        label=f'native-tp{tp}' if worker=='native' else 'reference'
        folder=args.run_dir/label;folder.mkdir();prefix.copy_inputs(source,folder)
        report['last_stage']=label;checkpoint()
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker',worker,'--tp-size',str(tp),
                 '--extension-sha256',evidence['extension_sha256'],'--prefix','1','--run-dir',str(folder),'--model-path',str(args.model_path),'--example',str(example),'--diag-extension',str(extension)]
        with common.StageHeartbeat(label):base.run_worker(command,folder/'worker.log',environment)
        for name in NAMES:read_snapshot(folder/'trace',name)
        report['completed_workers'].append(label);checkpoint()
    report['last_stage']='comparison';checkpoint()
    report['comparison']={label:{name:compare_boundary(args.run_dir/ref/'trace',args.run_dir/native/'trace',name) for name in NAMES}
        for label,ref,native in [('tp1_vs_reference','reference','native-tp1'),('tp8_vs_reference','reference','native-tp8'),('tp8_vs_tp1','native-tp1','native-tp8')]}
    report['result']='COMPARISON_COMPLETE';report['last_stage']='complete';checkpoint()
    print('CONDITION_DIAGNOSTIC COMPLETE; quality_status=UNASSESSED; sampled-boundaries-only',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-sha256');p.add_argument('--worker',choices=('native','reference'));p.add_argument('--tp-size',type=int,choices=(1,8),default=8)
    p.add_argument('--prefix',type=int,choices=(1,),default=1)
    for name in ('run-dir','work-home','local-target','model-path','example','diag-extension'):
        p.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path'))
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.diag_extension is None or args.example is None or not args.extension_sha256:p.error('worker needs diag-extension/example/extension-sha256')
        (native_worker if args.worker=='native' else reference_worker)(args);return 0
    if args.work_home is None or args.local_target is None:p.error('parent needs work-home/local-target')
    report={'version':'1.0-actual-engine-condition-boundaries','result':'FAILED','last_stage':'preflight','completed_workers':[],
            'quality_status':'UNASSESSED','performance_measurement_valid':False,'scope':'sampled-condition-and-first-block-boundaries-not-full-model'}
    start=time.monotonic()
    def checkpoint():report['elapsed_seconds']=time.monotonic()-start;base.write_json(args.run_dir/'condition-parity.json',report)
    try:orchestrate(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()

if __name__=='__main__':raise SystemExit(main())
