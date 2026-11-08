#!/usr/bin/env python3
"""Independent InfiniCore cross-device noncontiguous shard-copy diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

MODES=('implicit','retained','retained_source_sync')
GROUPS=('mixed',)+MODES
CHILD_TIMEOUT_SECONDS=300
PREVIOUS_RUN='logs/infinifabric-h3-projection-20261107T015112Z-2822929'
PREVIOUS_SHA='ca84526c069ea5889dbc401aaf49b8ec6edaa6760750ed0b0e7c4b96588641a7'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def enqueue_copy(core,source,shape,dtype,rank,mode):
    if (mode not in MODES or type(rank)!=int or not 0<=rank<8 or len(shape)!=2
            or any(type(v)!=int or v<1 for v in shape) or shape[1]%8):
        raise ValueError('invalid copy shape/rank/mode')
    width=shape[1]//8
    view=source.narrow(1,rank*width,width)
    keep=None
    if mode!='implicit':
        keep=view.contiguous()
        if mode=='retained_source_sync':
            core.set_device(core.device('cuda',0))
            core.sync_stream()
    destination=core.empty([shape[0],width],dtype=dtype,device=core.device('cuda',rank))
    destination.copy_(view if keep is None else keep)
    # Parent holds packed sources until every queued destination transfer completes.
    return rank,destination,keep


def compare_full(expected,actual):
    import numpy as np
    a,b=np.asarray(expected),np.asarray(actual)
    if a.shape!=b.shape or not a.size:raise ValueError('comparison shape differs/empty')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite full copy result')
    different=a!=b
    return dict(scope='all-elements',shape=list(a.shape),elements=int(a.size),
        exact_equal=bool(not different.any()),mismatched_elements=int(np.count_nonzero(different)),
        max_abs=float(np.max(np.abs(a-b))))


def read_previous(work):
    path=work/PREVIOUS_RUN/'projection-parity.json'
    if sha256(path)!=PREVIOUS_SHA:raise ValueError('previous projection report changed')
    report=json.loads(path.read_text(encoding='utf-8'))
    if report.get('result')!='COMPARISON_COMPLETE':raise ValueError('previous projection incomplete')
    models={}
    for label in ('native-tp1','native-tp8'):
        model=work/PREVIOUS_RUN/label/'native-prefix-model.json'
        data=json.loads(model.read_text(encoding='utf-8'))
        if data.get('tp_size')!=(1 if label=='native-tp1' else 8):raise ValueError('previous model TP metadata differs')
        models[label]=dict(metadata_sha256=sha256(model),weight_staging=data['weight_staging'])
    return dict(report_sha256=PREVIOUS_SHA,models=models)


def probe_environment(torch,core,native):
    # HYGON uses Python's 'cuda' alias; zero-count CPU types have no Python device mapping.
    return dict(torch_version=torch.__version__,hip_version=torch.version.hip,
        core_python=str(Path(core.__file__).resolve()),core_extension_sha256=sha256(native.__file__),
        torch_device_count=torch.cuda.device_count(),
        native_device_count=native.get_device_count(native.Device.Type.HYGON),
        native_cpu_count=native.get_device_count(native.Device.Type.CPU),
        devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def load_readback_runtime(work):
    # ABI comes from the pinned repository include/infinirt.h and infinicore.h.
    path=(work/'.infini/lib/libinfinirt.so').resolve()
    runtime=ctypes.CDLL(str(path))
    runtime.infinirtSetDevice.argtypes=[ctypes.c_int,ctypes.c_int]
    runtime.infinirtSetDevice.restype=ctypes.c_int
    runtime.infinirtDeviceSynchronize.argtypes=[]
    runtime.infinirtDeviceSynchronize.restype=ctypes.c_int
    runtime.infinirtMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    runtime.infinirtMemcpy.restype=ctypes.c_int
    return runtime,dict(path=str(path),sha256=sha256(path),observer='synchronous-InfiniRT-D2H-without-Core-rearrange',hygon_enum=8,d2h_enum=2)


def readback(runtime,value,rank,dtype_name):
    import numpy as np
    if (type(rank)!=int or not 0<=rank<8 or dtype_name not in ('float32','bfloat16')
            or not value.is_contiguous() or not value.data_ptr()):
        raise ValueError('invalid direct readback tensor/rank/dtype')
    raw=np.empty(tuple(value.shape),dtype='<f4' if dtype_name=='float32' else '<u2')
    def check(code,operation):
        if code!=0:raise RuntimeError(f'{operation} failed status={code}')
    check(runtime.infinirtSetDevice(8,rank),'infinirtSetDevice')
    check(runtime.infinirtDeviceSynchronize(),'infinirtDeviceSynchronize')
    check(runtime.infinirtMemcpy(raw.ctypes.data,value.data_ptr(),raw.nbytes,2),'infinirtMemcpy(D2H)')
    return raw if dtype_name=='float32' else (raw.astype('<u4')<<16).view('<f4')


def complete_batch(submit,sync_all,observe,before_observe):
    # Worker-local native descriptor caches outlive all asynchronous use and observation.
    with ThreadPoolExecutor(max_workers=8) as pool:
        queued=list(pool.map(submit,range(8)))
        sync_all()
        observed=[]
        for rank,destination,keep in queued:
            before_observe(rank)
            observed.append(observe(destination,rank))
        return observed


def run_probe(args,report,checkpoint):
    # Provenance first; target-only imports follow so failed preflights remain readable.
    report['previous_projection']=read_previous(args.work_home)
    import numpy as np
    import torch
    import infinicore as core
    from infinicore.lib import _infinicore
    report['environment']=probe_environment(torch,core,_infinicore)
    if report['environment']['torch_device_count']<8 or report['environment']['native_device_count']<8:
        raise RuntimeError('eight Hygon devices required')
    if int(_infinicore.Device.Type.HYGON)!=8:raise RuntimeError('InfiniRT HYGON ABI differs')
    runtime,report['readback_runtime']=load_readback_runtime(args.work_home)
    report['library_hashes']={p.name:sha256(p) for p in (args.work_home/'.infini/lib').glob('libinfini*.so') if p.is_file()}
    report['last_stage']='copy-cases';checkpoint()

    def sync_all():
        for rank in range(8):
            core.set_device(core.device('cuda',rank));core.sync_device()
            torch.cuda.synchronize(rank)

    for rank in range(8):
        core.set_device(core.device('cuda',rank))
        warm=core.empty([1],dtype=core.float32,device=core.device('cuda',rank))
        core.sync_device()
        del warm
    shapes=([128,1024],[5376,7168])
    for shape in shapes:
        rows,cols=shape;width=cols//8
        for dtype_name in ('float32','bfloat16'):
            for epoch in range(3):
                # Small integers are exactly representable in both tested dtypes.
                matrix=((np.arange(rows,dtype=np.int32)[:,None]*11+
                         np.arange(cols,dtype=np.int32)[None,:]*7+epoch*13)%127-63).astype('float32')
                torch_source=torch.from_numpy(matrix).to(device='cuda:0',dtype=getattr(torch,dtype_name))
                torch.cuda.synchronize(0)
                source=core.from_torch(torch_source)
                if not source.is_contiguous():raise RuntimeError('source must be contiguous')
                for mode in (MODES if args.worker_group=='mixed' else (args.worker_group,)):
                    report['active_case']=dict(shape=shape,dtype=dtype_name,epoch=epoch,mode=mode)
                    report['last_stage']='pre-copy-sync';report.pop('active_rank',None);checkpoint()
                    sync_all()
                    gate=threading.Barrier(8,timeout=60)
                    def submit(rank):
                        torch.cuda.set_device(rank)
                        gate.wait()
                        return enqueue_copy(core,source,shape,getattr(core,dtype_name),rank,mode)
                    report['last_stage']='enqueue-and-complete-copy';checkpoint()
                    def before_observe(rank):
                        report['last_stage']='direct-D2H-readback';report['active_rank']=rank;checkpoint()
                        print(f'phase=direct-D2H-readback rank={rank} mode={mode} shape={shape} dtype={dtype_name} epoch={epoch}',flush=True)
                    arrays=complete_batch(submit,sync_all,
                        lambda value,rank:readback(runtime,value,rank,dtype_name),before_observe)
                    case=dict(shape=shape,dtype=dtype_name,epoch=epoch,mode=mode,ranks={})
                    for rank,actual in enumerate(arrays):
                        metric=compare_full(matrix[:,rank*width:(rank+1)*width],actual)
                        sample=actual.reshape(-1)[np.linspace(0,actual.size-1,min(4096,actual.size),dtype=np.int64)]
                        metric['sample_sha256']=hashlib.sha256(sample.astype('<f4').tobytes()).hexdigest()
                        case['ranks'][str(rank)]=metric
                    case['all_ranks_exact']=all(v['exact_equal'] for v in case['ranks'].values())
                    report['cases'].append(case);checkpoint()
                    print(f"copy_case shape={shape} dtype={dtype_name} epoch={epoch} mode={mode} all_ranks_exact={case['all_ranks_exact']}",flush=True)
                    del arrays,actual
                del source,torch_source,matrix
    report['mode_summary']={mode:dict(cases=sum(c['mode']==mode for c in report['cases']),
        exact_cases=sum(c['mode']==mode and c['all_ranks_exact'] for c in report['cases'])) for mode in MODES}
    report['result']='PROBE_COMPLETE';report['last_stage']='complete';checkpoint()
    print('SHARD_COPY_PROBE COMPLETE; full_model_quality=UNASSESSED; performance_measurement_valid=false',flush=True)


def run_child(args,group):
    folder=args.run_dir/group;folder.mkdir()
    log=folder/'worker.log'
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(folder),
        '--work-home',str(args.work_home),'--worker-group',group]
    start=time.monotonic();timed_out=False
    with log.open('w',encoding='utf-8') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while True:
                try:proc.wait(timeout=min(30,CHILD_TIMEOUT_SECONDS));break
                except subprocess.TimeoutExpired:
                    print(f'heartbeat group={group} elapsed_seconds={time.monotonic()-start:.1f}',flush=True)
                    if time.monotonic()-start>=CHILD_TIMEOUT_SECONDS:
                        timed_out=True;proc.terminate()
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait()
                        break
        finally:
            if proc.poll() is None:
                proc.kill();proc.wait()
    print(log.read_text(encoding='utf-8',errors='replace'),end='',flush=True)
    checkpoint_path=folder/'shard-copy.json'
    data={}
    if checkpoint_path.exists():
        try:data=json.loads(checkpoint_path.read_text(encoding='utf-8'))
        except (ValueError,OSError) as error:data=dict(checkpoint_read_error=str(error))
    return dict(returncode=proc.returncode,timed_out=timed_out,elapsed_seconds=time.monotonic()-start,
        log=str(log),checkpoint=data)


def group_complete(group,result):
    data=result.get('checkpoint',{})
    cases=data.get('cases',[])
    if (result.get('returncode')!=0 or result.get('timed_out') or data.get('result')!='PROBE_COMPLETE'
            or len(cases)!=(36 if group=='mixed' else 12)):
        return False
    expected={(tuple(shape),dtype,epoch,mode) for shape in ([128,1024],[5376,7168])
        for dtype in ('float32','bfloat16') for epoch in range(3)
        for mode in (MODES if group=='mixed' else (group,))}
    seen=set()
    for case in cases:
        identity=(tuple(case.get('shape',[])),case.get('dtype'),case.get('epoch'),case.get('mode'))
        if identity not in expected or identity in seen:return False
        seen.add(identity)
        output_shape=[case['shape'][0],case['shape'][1]//8]
        ranks=case.get('ranks',{})
        if set(ranks)!={str(r) for r in range(8)}:return False
        for metric in ranks.values():
            if (metric.get('scope')!='all-elements' or metric.get('shape')!=output_shape
                    or type(metric.get('elements'))!=int or metric['elements']!=math.prod(output_shape)
                    or type(metric.get('mismatched_elements'))!=int or not 0<=metric['mismatched_elements']<=metric['elements']
                    or type(metric.get('exact_equal'))!=bool or metric['exact_equal']!=(metric['mismatched_elements']==0)
                    or not isinstance(metric.get('max_abs'),(int,float)) or not math.isfinite(metric['max_abs'])):
                return False
        if case.get('all_ranks_exact')!=all(m['exact_equal'] for m in ranks.values()):return False
    return seen==expected


def orchestrate(args,report,checkpoint):
    report['groups']={}
    for group in GROUPS:
        report['last_stage']='group-'+group;checkpoint()
        print('group='+group+' status=START',flush=True)
        result=run_child(args,group)
        result['complete']=group_complete(group,result)
        report['groups'][group]=result;checkpoint()
        print(f"group={group} returncode={result['returncode']} complete={result['complete']}",flush=True)
    all_complete=all(r['complete'] for r in report['groups'].values())
    report['result']='PROBE_COMPLETE' if all_complete else 'INCOMPLETE_CHILD_FAILURE'
    report['last_stage']='complete' if all_complete else 'incomplete'
    checkpoint()
    print('SHARD_COPY_GROUPS '+report['result']+'; quality_status=UNASSESSED',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--work-home',type=Path,required=True)
    p.add_argument('--worker-group',choices=GROUPS)
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.1-isolated-native-shard-copy-controls',result='FAILED',last_stage='preflight',cases=[],worker_group=args.worker_group,
        performance_measurement_valid=False,quality_status='UNASSESSED',
        scope='synthetic-full-array-copy-only-not-model-generation',
        control_limits='retention and source sync are separate diagnostic controls; no production patch or root-cause verdict')
    start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds']=time.monotonic()-start
        write_json(args.run_dir/'shard-copy.json',report)
    try:
        if args.worker_group:run_probe(args,report,checkpoint)
        else:
            report['previous_projection']=read_previous(args.work_home);checkpoint()
            orchestrate(args,report,checkpoint)
        return 0 if report['result']=='PROBE_COMPLETE' else 1
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
