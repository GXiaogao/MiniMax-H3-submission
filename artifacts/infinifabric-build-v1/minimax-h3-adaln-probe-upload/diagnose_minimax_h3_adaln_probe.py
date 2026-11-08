#!/usr/bin/env python3
"""Real-weight AdaLN projection/allgather component diagnostic; no generation."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

def sibling(name,alias):
    spec=importlib.util.spec_from_file_location(alias,Path(__file__).with_name(name))
    module=importlib.util.module_from_spec(spec);sys.modules[alias]=module;spec.loader.exec_module(module);return module

base=sibling('diagnose_minimax_h3_transformer_parity.py','adaln_base')

def shard_ranges(width,world):
    if world not in (1,8) or width<=0 or width%world:raise ValueError('invalid shard width')
    return [(i*width//world,(i+1)*width//world) for i in range(world)]

def gather_expected(parts):
    import numpy as np
    if not parts or any(p.ndim!=2 or p.shape!=parts[0].shape for p in parts):raise ValueError('incompatible local matrices')
    return np.concatenate(parts,axis=1)

def marker_parts(rows,width,world):
    import numpy as np
    return [((rank*16)+np.arange(rows)[:,None]*8+np.arange(end-start)[None,:]%8).astype(np.float32)
            for rank,(start,end) in enumerate(shard_ranges(width,world))]

def compile_shim(directory,work,compiler=None):
    compiler=compiler or shutil.which('g++') or shutil.which('c++')
    log=directory/'compile.log'
    if not compiler:
        log.write_text('No g++/c++ compiler available\n');raise RuntimeError('C++ compiler unavailable; see compile.log')
    target=directory/'adaln_probe.so'
    command=[compiler,'-std=c++17','-O2','-shared','-fPIC','-pthread','-D_GLIBCXX_USE_CXX11_ABI=1',
        str(Path(__file__).with_name('adaln_probe_shim.cpp')),'-I'+str(work/'Projects/InfiniCore/include'),
        '-L'+str(work/'.infini/lib'),'-Wl,-rpath,'+str(work/'.infini/lib'),'-linfinicore_cpp_api','-linfiniop','-linfiniccl','-linfinirt','-o',str(target)]
    try:
        with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    except (OSError,subprocess.TimeoutExpired) as error:
        with log.open('a') as stream:stream.write(str(error)+'\n')
        raise RuntimeError('shim compilation failed; see compile.log') from error
    if result.returncode:raise RuntimeError('shim compilation failed; see compile.log')
    return target,command

def stage_runtime(args,report):
    installer=sibling('install_infinilm_private_runtime.py','adaln_installer')
    common=sibling('run_minimax_h3_fl2va_native_smoke.py','adaln_common')
    manifest=common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode')!='job-local-wheel-staging' or manifest.get('expected_wheels')!=installer.EXPECTED_WHEELS:raise ValueError('runtime manifest mismatch')
    local=args.local_target
    with common.StageHeartbeat('adaln-runtime-staging'):
        installer.install_wheels_direct(installer.verify_wheels(Path(manifest['wheel_dir'])),local)
        report['private_distributions']=installer.check_target_contents(local)
    sys.path.insert(0,str(local))

def load_weights(folder):
    from safetensors import safe_open
    names=['time_embedder.linear_1.weight','time_embedder.linear_1.bias','time_embedder.linear_2.weight',
           'time_embedder.linear_2.bias','transformer_blocks.0.adaln_proj.linear.weight','transformer_blocks.0.adaln_proj.linear.bias']
    found={};sources={}
    for file in sorted(folder.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as archive:
            for name in set(names).intersection(archive.keys()):
                if name in found:raise ValueError('duplicate checkpoint key '+name)
                found[name]=archive.get_tensor(name);sources[name]=file.name
    if set(found)!=set(names):raise ValueError('missing AdaLN/time embedding weights')
    return found,sources

def run_case(shim,activated,weight,bias,world,marker=False):
    import torch
    rows,in_width=activated.shape;width=weight.shape[0]
    xs=[];ws=[];bs=[];locals_=[];outputs=[]
    markers=marker_parts(rows,width,world) if marker else None
    for rank,(start,end) in enumerate(shard_ranges(width,world)):
        device=f'cuda:{rank}'
        xs.append(torch.tensor(markers[rank],device=device,dtype=torch.bfloat16) if marker else activated.to(device).contiguous())
        ws.append(weight[start:end].to(device).contiguous());bs.append(bias[start:end].to(device).contiguous())
        locals_.append(torch.empty((rows,end-start),device=device,dtype=torch.bfloat16))
        outputs.append(torch.empty((rows,width),device=device,dtype=torch.bfloat16))
    for rank in range(world):torch.cuda.synchronize(rank)
    def pointers(tensors):return (ctypes.c_void_p*world)(*[x.data_ptr() for x in tensors])
    error=ctypes.create_string_buffer(4096)
    rc=shim.adaln_probe(world,rows,in_width,width,int(marker),pointers(xs),pointers(ws),pointers(bs),pointers(locals_),pointers(outputs),error,len(error))
    if rc:raise RuntimeError('native shim: '+error.value.decode(errors='replace'))
    return [x.float().cpu().numpy() for x in locals_],[x.float().cpu().numpy() for x in outputs]

def probe(args,report,checkpoint):
    source=args.work_home/'logs/infinifabric-transformer-parity-20261105T080500Z-408969'
    arrays,meta=base.read_bundle(source,'first-inputs')
    if meta['sha256']!='e2b542a7102402d075f99af716e6c6480d0f4ef707b7cc6016cb8e5a38a8a2a7':raise ValueError('source input changed')
    report['input_sha256']=meta['sha256']
    config=args.model_path/'diffusers/transformer/config.json'
    if base.sha256(config)!='74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae':raise ValueError('checkpoint config changed')
    report['last_stage']='staging';checkpoint();stage_runtime(args,report)
    import torch
    import infinicore
    from diffusers.models.embeddings import Timesteps
    if torch.cuda.device_count()<8:raise ValueError('eight GPUs required')
    report['torch_version']=torch.__version__
    if not torch._C._GLIBCXX_USE_CXX11_ABI:raise ValueError('expected C++11 ABI=1')
    report['last_stage']='compile';checkpoint()
    target,command=compile_shim(args.run_dir,args.work_home)
    report['compile_command']=command;report['shim_sha256']=base.sha256(target)
    report['native_libraries']={n:base.sha256(args.work_home/'.infini/lib'/n) for n in ('libinfinicore_cpp_api.so','libinfiniop.so','libinfiniccl.so','libinfinirt.so')}
    shim=ctypes.CDLL(str(target));shim.adaln_probe.restype=ctypes.c_int
    shim.adaln_probe.argtypes=[ctypes.c_int]*5+[ctypes.POINTER(ctypes.c_void_p)]*5+[ctypes.c_char_p,ctypes.c_size_t]
    weights,sources=load_weights(config.parent);report['weight_sources']=sources
    prefix='time_embedder.'
    def w(name):return weights[prefix+name].to(device='cuda:0',dtype=torch.float32)
    with torch.inference_mode():
        timestep=torch.from_numpy(arrays['timestep']).to('cuda:0')
        embedding=Timesteps(256,flip_sin_to_cos=True,downscale_freq_shift=0)(timestep)
        hidden=torch.nn.functional.silu(torch.nn.functional.linear(embedding,w('linear_1.weight'),w('linear_1.bias')))
        temb=torch.nn.functional.linear(hidden,w('linear_2.weight'),w('linear_2.bias'))
        activated=torch.nn.functional.silu(temb).to(torch.bfloat16)
        weight=weights['transformer_blocks.0.adaln_proj.linear.weight'].to('cuda:0')
        bias=weights['transformer_blocks.0.adaln_proj.linear.bias'].to('cuda:0')
        if weight.dtype!=torch.bfloat16 or bias.dtype!=torch.bfloat16 or tuple(weight.shape)!=(96768,2688) or tuple(bias.shape)!=(96768,):raise ValueError('unexpected AdaLN weight dtype/shape')
        reference=torch.nn.functional.linear(activated,weight,bias).float().cpu().numpy()
        base.save_bundle(args.run_dir,'adaln-inputs',{'temb':temb,'activated':activated})
        base.save_bundle(args.run_dir,'adaln-reference',{'projection':reference})
        report['comparison']={}
        for world,marker in ((8,True),(1,False),(8,False)):
            stage='marker-tp8' if marker else f'projection-tp{world}'
            report['last_stage']=stage;checkpoint();print('stage='+stage,flush=True)
            parts,outputs=run_case(shim,activated,weight,bias,world,marker)
            expected=gather_expected(parts)
            result={'all_ranks_gather':[base.compare_arrays(expected,x) for x in outputs]}
            if marker:
                oracle=gather_expected(marker_parts(activated.shape[0],96768,8))
                result['marker_exact']=all((x==oracle).all() for x in outputs)
                if not result['marker_exact']:raise ValueError('rank/time marker gather mismatch')
            else:
                result['local_vs_reference']=[base.compare_arrays(reference[:,start:end],p) for p,(start,end) in zip(parts,shard_ranges(96768,world))]
                result['gathered_vs_reference']=[base.compare_arrays(reference,x) for x in outputs]
                base.save_bundle(args.run_dir,stage,{'gathered_rank0':outputs[0],'concatenated_locals':expected})
            report['comparison'][stage]=result;checkpoint()
    report['result']='COMPARISON_COMPLETE';report['last_stage']='complete';checkpoint()
    print('ADALN_COMPONENT COMPLETE; quality_status=UNASSESSED',flush=True)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','work-home','model-path','local-target'):p.add_argument('--'+name,type=Path,required=name!='local-target')
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.local_target is None:args.local_target=args.run_dir/'local-site'
    report={'version':'1.0-adaln-projection-gather','result':'FAILED','last_stage':'preflight','quality_status':'UNASSESSED',
            'performance_measurement_valid':False,'scope':'isolated-real-weight-projection-and-gather-not-actual-engine-intermediate'}
    start=time.monotonic()
    def checkpoint():report['elapsed_seconds']=time.monotonic()-start;base.write_json(args.run_dir/'adaln-probe.json',report)
    try:probe(args,report,checkpoint);return 0
    except Exception as error:report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()

if __name__=='__main__':raise SystemExit(main())
