#!/usr/bin/env python3
"""Read-only v3 launch diagnosis: no installation, compilation or inference changes."""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import traceback
import uuid
import zipfile

from benchmark_support import Fixture,environment,SOURCE_SHA256
from protocol import DTYPES,sha256,write_json,validate_bundle,statistics_us
from timing import host_split,graph_interval,complete_samples
from hip_graph import Hip,Captured,Direct

HERE=Path(__file__).resolve().parent
CASES=((7,64,'6h'),(257,513,'6h'),(4096,2048,'6h'))
PATHS=('baseline','fused','direct')
GRAPH_SIZES=(1,)
PINNED_LIBRARIES={
    'libinfinicore_cpp_api.so':'7b02218a3d208d39a4387746e936808cb82bec73ed0a9cca5cf28675f9f0abcb',
    'libinfiniop.so':'3004db4633d57f7db73d46c96f083930b15aab524e95740a26d71f8714106a44',
    'libinfinirt.so':'1993bdeba63c137758fc2cb9d8ae772d6fdeb94999de260373ab52e1f4c7a467',
}


def counts(f,direct):return (f.native.gated_residual_hygon_launch_count(),direct.count())


def check_counts(before,after,path,calls,graph=False):
    delta=[a-b for a,b in zip(after,before)]
    expected=[0,0] if graph else [calls if path=='fused' else 0,calls if path=='direct' else 0]
    if delta!=expected:raise AssertionError(f'{path}: submissions {delta}, expected {expected}; graph={graph}')
    return delta


def correct(f,call,path,folder,label):
    """Poison output then require full-array bit identity; no cached-output success."""
    import numpy as np
    index=1 if path=='baseline' else 2
    # A valid final output must not be produced using a product left by warmup.
    if path=='baseline':f.buffers[0].fill_(float('nan'))
    f.buffers[index].fill_(float('nan'));f.sync()
    result=call()
    value=f.read(f.old if path=='baseline' else f.new) if result is None else f.read(result)
    bits=lambda t:t.contiguous().view(f.torch.uint8).numpy()
    finite=bool(f.torch.isfinite(value).all())
    equal=bool(np.array_equal(bits(value),bits(f.expected)))
    unchanged=f.input_hashes()==f.initial_input_hashes
    report={'output_poisoned_before_call':True,'intermediate_poisoned_before_call':path=='baseline',
            'finite':finite,'equal_reference_bits':equal,
            'inputs_unchanged':unchanged,'output_sha256':f.digest(value),'reference_sha256':f.digest(f.expected)}
    indices=np.unique(np.concatenate((np.arange(min(64,value.numel())),
                                     np.linspace(0,value.numel()-1,min(4096,value.numel()),dtype=np.int64))))
    np.savez_compressed(folder/(label+'_sample.npz'),indices=indices,
        actual=value.float().numpy().reshape(-1)[indices],expected=f.expected.float().numpy().reshape(-1)[indices])
    write_json(folder/(label+'_correctness.json'),report)
    if not (finite and equal and unchanged):raise AssertionError('Poison/replay correctness failed: '+label)
    return report


def phase_stats(rows):
    return {key[:-3]+'_us':statistics_us([r[key]/1000 for r in rows])
            for key in ('submit_ns','remaining_wait_ns','total_ns')}


def host_measure(f,hip,direct,calls,args,folder):
    report={'passed':False,'samples':[],'correctness':{}}
    write=lambda:write_json(folder/'host.json',report)
    write()
    try:
        for name,call in calls.items():
            before=counts(f,direct)
            report['correctness'][name+'_before']=correct(f,call,name,folder,'host_'+name+'_before')
            check_counts(before,counts(f,direct),name,1)
            before=counts(f,direct)
            for _ in range(args.warmup):call()
            hip.sync();check_counts(before,counts(f,direct),name,args.warmup)
        for i in range(args.samples):
            # Rotate and reverse to avoid assigning all paths a fixed temperature/order.
            order=list(PATHS[i%3:]+PATHS[:i%3])
            if i%2:order.reverse()
            for name in order:
                before=counts(f,direct)
                row,out=host_split(calls[name],hip.sync)
                row.update(index=i,path=name,submissions=check_counts(before,counts(f,direct),name,1))
                report['samples'].append(row)
                out=None
            write()
        for name,call in calls.items():
            before=counts(f,direct)
            report['correctness'][name+'_after']=correct(f,call,name,folder,'host_'+name+'_after')
            check_counts(before,counts(f,direct),name,1)
        report['statistics']={name:phase_stats([r for r in report['samples'] if r['path']==name]) for name in PATHS}
        report['passed']=complete_samples(report['samples'],args.samples,PATHS)
        if not report['passed']:raise RuntimeError('Incomplete host phase samples')
        return report
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write()


def graph_measure(f,hip,direct,calls,args,folder,k):
    report={'passed':False,'operations_per_graph':k,'scope':'single_operation_graph',
            'graphs':{},'samples':[],'correctness':{}}
    write=lambda:write_json(folder/f'graph_{k}.json',report)
    write()
    try:
        if k!=1:raise ValueError('Only single-operation graph measurement is supported')
        with ExitStack() as stack:
            graphs={}
            for name in ('baseline','fused'):
                before=counts(f,direct)
                graphs[name]=stack.enter_context(Captured(hip,calls[name],k,k*(2 if name=='baseline' else 1)))
                report['graphs'][name]={'topology':graphs[name].topology,
                    'capture_submissions':check_counts(before,counts(f,direct),name,k)}
                write()
                # Host counter increases at capture, not replay. Node counts and
                # poisoning establish replay work; never use host counter as GPU execution count.
                before=counts(f,direct)
                report['correctness'][name+'_before']=correct(f,graphs[name].launch,name,folder,f'graph_{k}_{name}_before')
                for _ in range(args.warmup):graphs[name].launch();hip.sync()
                check_counts(before,counts(f,direct),name,0,graph=True)
            for i in range(args.samples):
                for name in (('baseline','fused') if i%2==0 else ('fused','baseline')):
                    before=counts(f,direct)
                    row,_=host_split(graphs[name].launch,hip.sync)
                    row.update(graph_interval(graphs[name].elapsed_ms(),k))
                    row.update(index=i,path=name,submissions=check_counts(before,counts(f,direct),name,0,graph=True))
                    report['samples'].append(row)
                write()
            for name in ('baseline','fused'):
                before=counts(f,direct)
                report['correctness'][name+'_after']=correct(f,graphs[name].launch,name,folder,f'graph_{k}_{name}_after')
                check_counts(before,counts(f,direct),name,0,graph=True)
            report['statistics']={}
            for name in ('baseline','fused'):
                rows=[r for r in report['samples'] if r['path']==name]
                report['statistics'][name]=phase_stats(rows)|{
                    'device_graph_us_per_op':statistics_us([r['device_graph_us_per_op'] for r in rows])}
            report['passed']=complete_samples(report['samples'],args.samples,('baseline','fused'))
            if not report['passed']:raise RuntimeError('Incomplete graph phase samples')
        return report
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write()


def profiler_inventory():
    tools={name:shutil.which(name) for name in ('rocprof','rocprofv2','rocprofv3','roctracer','hyprof','dprof')}
    roots=[Path('/opt/dtk/bin'),Path('/opt/dtk/profiler/bin'),Path('/opt/rocm/bin')]
    candidates=[]
    for root in roots:
        if root.is_dir():
            candidates.extend(str(p) for p in root.iterdir() if p.is_file() and any(s in p.name.lower() for s in ('prof','trace')))
    return {'executables':tools,'directory_candidates':sorted(set(candidates)),
            'profiling_started':False,'note':'Inventory only; graph event intervals are not individual kernel traces.'}


def write_tables(folder,report):
    fields=['case','phase','operations_per_graph','path','metric','samples','p50_us','p95_us','mean_us','min_us','max_us']
    with (folder/'results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in report['cases']:
            groups=[('host',1,case.get('host',{})),*[(f'graph_{r["operations_per_graph"]}',r['operations_per_graph'],r) for r in case.get('graphs',[])]]
            for phase,k,data in groups:
                for name,stats in data.get('statistics',{}).items():
                    for metric,stat in stats.items():
                        writer.writerow({key:value for key,value in dict(case=case['name'],phase=phase,operations_per_graph=k,
                            path=name,metric=metric,**stat).items() if key in fields})


def worker(args):
    folder=args.worker.resolve()
    report={'schema':3,'passed':False,'separation_measured':False,'cases':[],
        'measurement_scope':'single_operation_graph','full_matrix_measured':False,
        'batch_graph_measured':False,
        'batch_capture_status':{'status':'unverified','excluded_operations':[16,64],
            'evidence':'20261028_102452_2a69c989 pasted log: K=16 baseline exposed only two type-200 nodes',
            'cause':'unknown: opaque node aggregation or incomplete capture has not been distinguished',
            'policy':'No multi-operation capture or division by an unverified work count'},
        'pure_kernel_time_measured':False,'official_model_tested':False,'eight_card_tested':False,
        'quick_smoke_only':args.quick,
        'protocol':{'samples':args.samples,'warmup':args.warmup,'graph_sizes':GRAPH_SIZES,
            'allocation':'preallocated only','seed':20261027,'input':'same 6H synthetic fixture families as benchmark v1',
            'host':'wall-clock submission, remaining stream wait and their total; independent synchronized single calls',
            'direct':'same preloaded host DSO and HSACO, prepared TensorArg; ctypes boundary differs from pybind/public API',
            'device':'single operation per HIP graph (K=1); internal event interval includes device graph scheduling and event overhead; no batch division',
            'counter':'host submissions at capture; zero during replay; graph nodes + poisoned-output bit check validate replay',
            'warning':'Do not subtract medians or total minus GPU to report CPU overhead. This is diagnostic raw capture, not production capture-safe support.'}}
    path=folder/'summary.json';write_json(path,report)
    try:
        torch,ic,native,device,env,bundle=environment(folder)
        for library,wanted in PINNED_LIBRARIES.items():
            actual={p['sha256'] for p in env['libraries'] if Path(p['path']).name==library}
            if actual!={wanted}:raise RuntimeError('Loaded library differs from 2026-10-28 baseline: '+library)
        report['environment']=env;report['bundle']=bundle;report['profilers']=profiler_inventory();write_json(path,report)
        hip=Hip(bundle['path'],ic.get_stream())
        report['hip_runtime']={'path':str(hip.path),'sha256':sha256(hip.path),'stream':hip.stream}
        # Preserve first-case host evidence even if this DTK rejects graph capture.
        specs=CASES[:1] if args.quick else CASES
        report['expected_cases']=len(DTYPES)*len(specs)
        for dtype in DTYPES:
            for n,h,layout in specs:
                name=f'{dtype}_{n}x{h}_{layout}'
                case={'name':name,'passed':False,'graphs':[]};report['cases'].append(case)
                target=folder/name;target.mkdir();report['current_case']=name;write_json(path,report)
                # Match the corresponding original fixture seed (1,3,7).
                seed=20261027+{7:1,257:3,4096:7}[n]
                f=Fixture(torch,ic,native,device,n,h,layout,dtype,seed);case['fixture']=f.metadata
                direct=Direct(bundle['path'],hip,f)
                try:
                    calls=f.calls('preallocated')|{'direct':direct}
                    print('HOST:',name,flush=True)
                    enabled=gc.isenabled();gc.disable()
                    try:case['host']=host_measure(f,hip,direct,calls,args,target)
                    finally:
                        if enabled:gc.enable()
                    write_json(path,report)
                    if hip.layout is None:
                        print('Calibrating runtime HIP graph node ABI',flush=True)
                        event_calibration=hip.calibrate()
                        before=counts(f,direct)
                        compute_calibration=hip.calibrate_computation(calls['baseline'],calls['fused'])
                        calibration_submissions=check_counts(before,counts(f,direct),'fused',1)
                        report['graph_calibration']={'event':event_calibration,
                            'computation':compute_calibration,
                            'capture_submissions':calibration_submissions}
                        write_json(path,report)
                    for k in GRAPH_SIZES:
                        print('GRAPH:',name,'operations=',k,flush=True)
                        enabled=gc.isenabled();gc.disable()
                        try:row=graph_measure(f,hip,direct,calls,args,target,k)
                        finally:
                            if enabled:gc.enable()
                        case['graphs'].append(row);write_json(path,report)
                    case['passed']=True
                finally:direct.close()
                print('PASS:',name,flush=True)
                for p,s in case['host']['statistics'].items():
                    print(f"  {p} submit/wait/total P50 us: {s['submit_us']['p50_us']:.3f} / {s['remaining_wait_us']['p50_us']:.3f} / {s['total_us']['p50_us']:.3f}",flush=True)
                for row in case['graphs']:
                    print('  graph K=',row['operations_per_graph'],'device P50 us/op:',
                        {p:round(s['device_graph_us_per_op']['p50_us'],3) for p,s in row['statistics'].items()},flush=True)
                del f,direct,calls;gc.collect();write_json(path,report)
        if validate_bundle(bundle['path'],SOURCE_SHA256)!=bundle:raise RuntimeError('Kernel bundle changed during diagnosis')
        if any(name in sys.modules for name in ('triton','ninetoothed')):raise RuntimeError('Compiler imported in runtime diagnosis')
        report['passed']=len(report['cases'])==report['expected_cases'] and all(c['passed'] for c in report['cases'])
        report['separation_measured']=report['passed']
        report['full_matrix_measured']=report['passed'] and not args.quick
        if not report['passed']:raise RuntimeError('Incomplete diagnostic case matrix')
        return 0
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';traceback.print_exc();return 1
    finally:
        write_json(path,report);write_tables(folder,report)
        print('SUMMARY:',path,'passed=',report['passed'],flush=True)


def run_worker(command,env,log_path,timeout):
    with log_path.open('w',encoding='utf-8',newline='\n') as log:
        with subprocess.Popen(command,env=env,start_new_session=True,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1) as child:
            errors=[]
            def pump():
                try:
                    for line in child.stdout:
                        log.write(line);log.flush();print(line,end='',flush=True)
                except Exception as exc:errors.append(exc)
            thread=threading.Thread(target=pump,daemon=True);thread.start()
            try:code=child.wait(timeout=timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                if sys.platform=='linux':os.killpg(child.pid,signal.SIGKILL)
                else:child.kill()
                child.wait();code=124
            thread.join(timeout=10)
            if thread.is_alive():raise RuntimeError('Worker log stream did not close')
            if errors:raise errors[0]
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=40)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.quick:args.samples,args.warmup=6,2
    if args.samples<2 or args.samples%2 or args.warmup<1 or args.timeout<1:
        parser.error('Use an even samples count >=2 and positive warmup/timeout')
    if args.worker:return worker(args)
    home=Path(os.environ['WORK_HOME']).resolve(strict=True);logs=home/'logs';logs.mkdir(exist_ok=True)
    folder=logs/('h3_gated_launch_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
    print('REPORT DIRECTORY:',folder,flush=True)
    parent={'passed':False,'started':datetime.now().isoformat(),'arguments':vars(args)|{'worker':None}}
    code=1
    try:
        active=logs/'h3_gated_active_bundle.txt';bundle=Path(active.read_text().strip())
        if not bundle.is_absolute():raise RuntimeError('Active bundle must be absolute')
        parent['selected_bundle']=validate_bundle(bundle,SOURCE_SHA256)
        parent['active_path_file_sha256']=sha256(active)
        parent['script_hashes']={p.name:sha256(p) for p in HERE.iterdir() if p.is_file()}
        env=dict(os.environ,INFINICORE_H3_GATED_BUNDLE=str(bundle),LOG_DIR=str(folder))
        command=[sys.executable,'-X','faulthandler','-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker',str(folder)]
        parent['command']=command;write_json(folder/'run.json',parent)
        code=run_worker(command,env,folder/'worker.log',args.timeout)
        parent['exit_code']=code
        summary=json.loads((folder/'summary.json').read_text())
        parent['passed']=code==0 and summary.get('passed') is True
        if not parent['passed']:code=code or 1
        if sha256(active)!=parent['active_path_file_sha256']:raise RuntimeError('Active bundle selection changed during run')
    except Exception as exc:
        parent['passed']=False;parent['error']=f'{type(exc).__name__}: {exc}';traceback.print_exc();code=1
    finally:
        parent['finished']=datetime.now().isoformat();write_json(folder/'run.json',parent)
        files=sorted(p for p in folder.rglob('*') if p.is_file())
        write_json(folder/'files.json',[{'path':p.relative_to(folder).as_posix(),'sha256':sha256(p),'size':p.stat().st_size} for p in files])
        archive=folder.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files+[folder/'files.json']:z.write(p,folder.name+'/'+p.relative_to(folder).as_posix())
        print('EVIDENCE ZIP:',archive,flush=True)
        success='SINGLE-OP SMOKE PASSED' if args.quick else 'SINGLE-OP DIAGNOSIS COMPLETE'
        print(success if parent['passed'] else 'DIAGNOSIS FAILED / PARTIAL; retain evidence',flush=True)
        print('K=1 only; K=16/64 capture remains unverified. Graph timing includes device scheduling and event overhead; pure kernel/model/8-card performance NOT measured.',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
