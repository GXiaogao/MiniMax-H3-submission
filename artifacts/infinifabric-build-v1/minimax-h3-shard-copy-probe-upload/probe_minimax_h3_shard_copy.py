#!/usr/bin/env python3
"""Independent InfiniCore cross-device noncontiguous shard-copy diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback

MODES=('implicit','retained','retained_source_sync')
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
    report['library_hashes']={p.name:sha256(p) for p in (args.work_home/'.infini/lib').glob('libinfini*.so') if p.is_file()}
    report['last_stage']='copy-cases';checkpoint()

    def sync_all():
        for rank in range(8):
            core.set_device(core.device('cuda',rank));core.sync_device()
            torch.cuda.synchronize(rank)

    def observe(value,rank):
        # Same bridge pattern as the existing Native adapter; copies are already complete.
        output=torch.empty(tuple(value.shape),dtype=core.utils.to_torch_dtype(value.dtype),device=f'cuda:{rank}')
        core.from_torch(output).copy_(value)
        core.set_device(core.device('cuda',rank));core.sync_device()
        torch.cuda.synchronize(rank)
        return output.float().cpu().numpy()

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
                for mode in MODES:
                    sync_all()
                    gate=threading.Barrier(8,timeout=60)
                    def submit(rank):
                        torch.cuda.set_device(rank)
                        gate.wait()
                        return enqueue_copy(core,source,shape,getattr(core,dtype_name),rank,mode)
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        queued=list(pool.map(submit,range(8)))
                    # Observe only after all destination transfers and source work finish.
                    # queued retains every explicit temporary across these synchronizations.
                    sync_all()
                    case=dict(shape=shape,dtype=dtype_name,epoch=epoch,mode=mode,ranks={})
                    for rank,destination,keep in queued:
                        actual=observe(destination,rank)
                        metric=compare_full(matrix[:,rank*width:(rank+1)*width],actual)
                        sample=actual.reshape(-1)[np.linspace(0,actual.size-1,min(4096,actual.size),dtype=np.int64)]
                        metric['sample_sha256']=hashlib.sha256(sample.astype('<f4').tobytes()).hexdigest()
                        case['ranks'][str(rank)]=metric
                    case['all_ranks_exact']=all(v['exact_equal'] for v in case['ranks'].values())
                    report['cases'].append(case);checkpoint()
                    print(f"copy_case shape={shape} dtype={dtype_name} epoch={epoch} mode={mode} all_ranks_exact={case['all_ranks_exact']}",flush=True)
                    del queued,destination,keep
                del source,torch_source,matrix
    report['mode_summary']={mode:dict(cases=sum(c['mode']==mode for c in report['cases']),
        exact_cases=sum(c['mode']==mode and c['all_ranks_exact'] for c in report['cases'])) for mode in MODES}
    report['result']='PROBE_COMPLETE';report['last_stage']='complete';checkpoint()
    print('SHARD_COPY_PROBE COMPLETE; full_model_quality=UNASSESSED; performance_measurement_valid=false',flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--work-home',type=Path,required=True)
    args=p.parse_args(argv);args.run_dir.mkdir(parents=True,exist_ok=True)
    report=dict(version='1.0-native-shard-copy-controls',result='FAILED',last_stage='preflight',cases=[],
        performance_measurement_valid=False,quality_status='UNASSESSED',
        scope='synthetic-full-array-copy-only-not-model-generation',
        control_limits='retention and source sync are separate diagnostic controls; no production patch or root-cause verdict')
    start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds']=time.monotonic()-start
        write_json(args.run_dir/'shard-copy.json',report)
    try:run_probe(args,report,checkpoint);return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';traceback.print_exc();return 1
    finally:checkpoint()


if __name__=='__main__':raise SystemExit(main())
