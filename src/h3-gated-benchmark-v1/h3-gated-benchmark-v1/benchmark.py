#!/usr/bin/env python3
"""Offline single-DCU benchmark of existing InfiniCore paths; never installs/builds."""
import argparse
import csv
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

from protocol import (CASES, DTYPES, check_launches, comparison, complete, device_sample,
                      host_sample, path_order, sha256, validate_bundle, write_json)

SOURCE_SHA256 = 'd040cfa621319b10b48942d4a17f9d7a641dd08b79f831501a0607c813d7f4d9'
HERE = Path(__file__).resolve().parent
MODES = ('preallocated', 'allocating')


def command_info(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {'command':command, 'exit_code':proc.returncode, 'stdout':proc.stdout, 'stderr':proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command':command, 'error':str(exc)}


def environment(folder):
    if sys.platform!='linux' or sys.version_info[:2]!=(3,11) or sys.prefix!=sys.base_prefix:
        raise RuntimeError('Use normal initialized Linux Python 3.11, not the compiler venv')
    if any(os.environ.get(k) for k in ('HSA_OVERRIDE_GFX_VERSION','TRITON_OVERRIDE_ARCH')):
        raise RuntimeError('Architecture overrides must be unset')
    import torch
    import infinicore as ic
    from infinicore.lib import _infinicore as native
    if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Expected exactly one visible Hygon device')
    props = torch.cuda.get_device_properties(0)
    if getattr(props,'gcnArchName','').split(':')[0]!='gfx906':
        raise RuntimeError('Expected real gfx906')
    device=ic.device('cuda',0)
    if device._underlying.type!=type(device._underlying).Type.HYGON or ic.use_ntops:
        raise RuntimeError('Expected native InfiniCore HYGON dispatcher, not Python ntops')
    torch.cuda.set_device(0); ic.set_device(device)
    if ic.is_graph_recording(): raise RuntimeError('Graph recording must be disabled')
    bundle=validate_bundle(os.environ['INFINICORE_H3_GATED_BUNDLE'], SOURCE_SHA256)
    import check_sources
    check_sources.main()
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if any(s in line for s in ('libinfinicore_cpp_api.so','libinfiniop.so','libinfinirt.so','libamdhip64.so','libgalaxyhip.so')):
            path=line.split(maxsplit=5)[-1]
            if path.startswith('/') and Path(path).is_file(): mapped.add(path)
    if not any('libinfinicore_cpp_api.so' in p for p in mapped):
        raise RuntimeError('Cannot identify loaded InfiniCore native library')
    sources={}
    for name in ('InfiniCore','InfiniLM'):
        repo=Path(os.environ['WORK_HOME'])/'Projects'/name
        sources[name]={key:command_info(['git','-C',str(repo),*args]) for key,args in (
            ('commit',['rev-parse','HEAD']),('status',['status','--short','--branch']),
            ('submodules',['submodule','status','--recursive']))}
        diff=command_info(['git','-C',str(repo),'diff','--binary'])
        write_json(folder/(name+'_diff.json'),diff)
        config=repo/'.xmake/linux/x86_64/xmake.conf'
        if config.is_file(): (folder/(name+'_xmake.conf')).write_bytes(config.read_bytes())
    packages={}
    for name in ('torch','numpy','InfiniCore','InfiniLM'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    env={'python':sys.executable,'python_version':platform.python_version(),'host':platform.node(),
         'os':platform.platform(),'packages':packages,'hip':torch.version.hip,'device':str(props),
         'visible_devices':torch.cuda.device_count(),'infinicore':ic.__file__,
         'libraries':[{'path':p,'sha256':sha256(p)} for p in sorted(mapped)],
         'extension':{'path':native.__file__,'sha256':sha256(native.__file__)},'sources':sources,
         'device_stream':str(ic.get_stream()),'loadavg':list(os.getloadavg()),
         'affinity':sorted(os.sched_getaffinity(0)),
         'environment':{k:os.environ.get(k) for k in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES',
             'ROCR_VISIBLE_DEVICES','OMP_NUM_THREADS','INFINICORE_H3_GATED_BUNDLE')},
         'clock_control':'No clocks changed or guaranteed fixed; run on an otherwise idle card',
         'gpu_memory_free_total':list(torch.cuda.mem_get_info())}
    return torch,ic,native,device,env,bundle


class Fixture:
    def __init__(self, torch, ic, native, device, n, h, layout, dtype_name, seed):
        self.torch,self.ic,self.native,self.device=torch,ic,native,device
        self.dtype=getattr(torch,dtype_name)
        self.dtype_name=dtype_name
        generator=torch.Generator(device='cpu').manual_seed(seed)
        self.host=[(torch.rand((n,h),generator=generator)*2-1).to(self.dtype) for _ in range(3)]
        # Include cancellation and signed-zero witnesses in every timed input.
        self.host[0].view(-1)[:4]=torch.tensor([1.,-1.,0.,-0.],dtype=self.dtype)
        self.host[1].view(-1)[:4]=1
        self.host[2].view(-1)[:4]=torch.tensor([-1.,1.,-0.,0.],dtype=self.dtype)
        from rounding_reference import torch_add
        self.expected=torch_add(self.host[0],(self.host[1].float()*self.host[2].float()).to(self.dtype),self.dtype)
        self.owners=[];self.views=[]
        for index,host in enumerate(self.host):
            if layout=='strided':
                owner=torch.full((n,2*h+3),-7.,dtype=self.dtype,device='cuda')
                view=owner[:,1:2*h+1:2]
                view.copy_(host)
            elif index==1 and layout in ('6h','batch1_6h'):
                owner=torch.full((n,6*h),-7.,dtype=self.dtype,device='cuda')
                view=owner[:,5*h:6*h];view.copy_(host)
            else:
                owner=host.cuda();view=owner
            self.owners.append(owner)
            self.views.append(view.unsqueeze(0) if layout=='batch1_6h' else view)
        if layout=='batch1_6h': self.expected=self.expected.unsqueeze(0)
        self.sync()
        self.inputs=[self.wrap(t) for t in self.views]
        shape=tuple(self.views[0].shape)
        self.buffers=[torch.empty(shape,dtype=self.dtype,device='cuda') for _ in range(4)]
        self.temp,self.old,self.new,self.read_buffer=[self.wrap(t) for t in self.buffers]
        self.initial_input_hashes=self.input_hashes()
        self.metadata={'seed':seed,'dtype':dtype_name,'shape':list(shape),'layout':layout,
                       'input_strides':[list(t.stride()) for t in self.views],
                       'input_storage_offsets':[t.storage_offset() for t in self.views],
                       'output_stride':list(self.buffers[1].stride()),
                       'input_storage_sha256':self.initial_input_hashes,
                       'cpu_expected_sha256':self.digest(self.expected),
                       'shape_source':'synthetic operator case; not official model configuration'}

    def wrap(self,t):
        return self.ic.strided_from_blob(t.data_ptr(),list(t.shape),list(t.stride()),
            dtype=getattr(self.ic,self.dtype_name),device=self.device)

    def sync(self):
        self.ic.sync_device();self.torch.cuda.synchronize()

    def digest(self,t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes()).hexdigest()

    def input_hashes(self):
        self.sync()
        return [self.digest(t) for t in self.owners]

    def calls(self,mode):
        r,g,b=self.inputs
        if mode=='preallocated':
            def baseline():
                self.ic.mul(g,b,out=self.temp)
                return self.ic.add(r,self.temp,out=self.old)
            def fused(): return self.ic.gated_residual(r,g,b,out=self.new)
        else:
            def baseline(): return self.ic.add(r,self.ic.mul(g,b))
            def fused(): return self.ic.gated_residual(r,g,b)
        return {'baseline':baseline,'fused':fused}

    def read(self,value):
        self.read_buffer.copy_(value);self.sync()
        return self.buffers[3].cpu()

    def validate(self,calls,folder,label):
        import numpy as np
        outputs={};counts={}
        for path,call in calls.items():
            before=self.native.gated_residual_hygon_launch_count()
            outputs[path]=self.read(call())
            delta=self.native.gated_residual_hygon_launch_count()-before
            check_launches(path,delta,1);counts[path]=delta
        def bits(t): return t.contiguous().view(self.torch.uint8).numpy().reshape(-1)
        refbits=bits(self.expected)
        result={}
        for path,value in outputs.items():
            finite=bool(self.torch.isfinite(value).all())
            same=bool(np.array_equal(bits(value),refbits))
            result[path]={'finite':finite,'equals_reference_bits':same,'sha256':self.digest(value)}
        exact=bool(np.array_equal(bits(outputs['baseline']),bits(outputs['fused'])))
        unchanged=self.input_hashes()==self.initial_input_hashes
        result.update(equal_baseline_fused_bits=exact,inputs_unchanged=unchanged,native_launches=counts)
        # Full arrays are validated and hashed. Only deterministic excerpts are
        # archived to avoid multi-GB uploads for synthetic stress shapes.
        length=self.expected.numel()
        indices=np.unique(np.concatenate((np.arange(min(64,length)),np.linspace(0,length-1,min(4096,length),dtype=np.int64))))
        samples={k:v.float().numpy().reshape(-1)[indices] for k,v in outputs.items()}
        samples['expected']=self.expected.float().numpy().reshape(-1)[indices]
        samples['indices']=indices
        for k,v in zip(('residual','gate','branch'),self.host): samples[k]=v.float().numpy().reshape(-1)[indices]
        np.savez_compressed(folder/(label+'_sample.npz'),**samples)
        write_json(folder/(label+'_correctness.json'),result)
        if not exact or not unchanged or not all(result[p]['finite'] and result[p]['equals_reference_bits'] for p in outputs):
            raise AssertionError('Full-array correctness or input immutability failed: '+label)
        return result


def run_case(fixture,mode,args,folder,case_index):
    folder.mkdir()
    report={'passed':False,'correct_before':False,'correct_after':False,'launch_counts_ok':False,
            'mode':mode,'fixture':fixture.metadata,'samples':[]}
    path=folder/'result.json'
    calls=fixture.calls(mode)
    def save(): write_json(path,report)
    save()
    try:
        report['before']=fixture.validate(calls,folder,'before');report['correct_before']=True;save()
        # These calls initialize both dispatcher paths. Their latency is not part
        # of steady-state distributions; no auto-tuning or kernel compile occurs.
        for name in path_order(0,case_index):
            before=fixture.native.gated_residual_hygon_launch_count()
            for _ in range(args.warmup): calls[name]()
            fixture.sync()
            check_launches(name,fixture.native.gated_residual_hygon_launch_count()-before,args.warmup)
        gc_was_enabled=gc.isenabled();gc.disable()
        measured_counts={'baseline':0,'fused':0}
        try:
            for index in range(args.samples):
                for name in path_order(index,case_index):
                    before=fixture.native.gated_residual_hygon_launch_count()
                    call=calls[name]
                    # Keep returned tensors alive until work completes. Repeated
                    # calls never modify the common residual/gate/branch inputs.
                    device_us,output=device_sample(call,fixture.sync,
                        lambda:fixture.ic.DeviceEvent(enable_timing=True),args.iterations)
                    output=None
                    host_us,output=host_sample(call,fixture.sync)
                    output=None
                    delta=fixture.native.gated_residual_hygon_launch_count()-before
                    check_launches(name,delta,args.iterations+1)
                    measured_counts[name]+=delta
                    report['samples'].append({'index':index,'path':name,'device_us':device_us,'host_us':host_us,
                        'batch_iterations':args.iterations,'native_launches':delta})
                # Logging is outside all measurements; retain progress on a crash.
                save()
                if (index+1)%10==0: print(f'  samples {index+1}/{args.samples}',flush=True)
        finally:
            if gc_was_enabled:gc.enable()
        report['launch_counts']=measured_counts;report['launch_counts_ok']=True
        report['after']=fixture.validate(calls,folder,'after');report['correct_after']=True
        for key in ('host','device'):
            report[key]=comparison(*[[x[key+'_us'] for x in report['samples'] if x['path']==p] for p in ('baseline','fused')])
        report['passed']=True
        print('PASS',folder.name,flush=True)
        for metric in ('device','host'):
            stat=report[metric]
            print(f"  {metric} P50/P95 us: baseline {stat['baseline']['p50_us']:.3f}/{stat['baseline']['p95_us']:.3f}; "
                  f"fused {stat['fused']['p50_us']:.3f}/{stat['fused']['p95_us']:.3f}; "
                  f"P50 speedup {stat['p50_speedup']:.4f}x",flush=True)
        return report
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:save()


def tables(folder,report):
    fields=['case','dtype','shape','layout','mode','metric','baseline_p50_us','baseline_p95_us',
            'fused_p50_us','fused_p95_us','p50_speedup','p50_latency_reduction_percent']
    with (folder/'results.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        lines=['# 门控残差算子性能实测','',
               '完整性状态：'+('通过' if report['passed'] else '未完成/失败；下列为已有记录'),'',
               '合成输入、单卡、公开 InfiniCore Python 接口；非正式模型或端到端视频性能。',
               'device 为当前设备流内每批重复调用的平均耗时分布，含主机提交空隙；host 为独立单次接口调用到设备同步完成。',
               'P50/P95 均为微秒，不删除离群值。分配口径分开比较；小于1x表示变慢。','',
               '|用例|dtype|分配口径|计时|基线P50/P95 µs|融合P50/P95 µs|P50加速比|',
               '|---|---|---|---|---|---|---|']
        for row in report['cases']:
            if not row.get('passed'):continue
            f=row['fixture'];name=f"{f['shape']}/{f['layout']}"
            for metric in ('host','device'):
                c=row[metric];a,b=c['baseline'],c['fused']
                writer.writerow(dict(zip(fields,[name,f['dtype'],str(f['shape']),f['layout'],row['mode'],metric,
                    a['p50_us'],a['p95_us'],b['p50_us'],b['p95_us'],c['p50_speedup'],c['p50_latency_reduction_percent']])))
                lines.append(f"|{name}|{f['dtype']}|{row['mode']}|{metric}|{a['p50_us']:.3f}/{a['p95_us']:.3f}|"
                             f"{b['p50_us']:.3f}/{b['p95_us']:.3f}|{c['p50_speedup']:.4f}x|")
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


def worker(args):
    folder=args.worker.resolve()
    report={'schema':1,'run_id':folder.name,'passed':False,'performance_tested':False,
            'official_model_tested':False,'eight_card_tested':False,'cases':[],
            'scope':'synthetic single-device InfiniCore public API microbenchmark',
            'quick_smoke_only':args.quick,
            'protocol':{'samples':args.samples,'batch_iterations':args.iterations,'warmup':args.warmup,
                'order':'alternating AB/BA, shifted by case','percentiles':'linear interpolation; no outlier trimming',
                'device_metric':'current InfiniCore stream event interval / batch iterations; includes host submission gaps',
                'host_metric':'one public Python API call plus final device synchronization; initial sync excluded',
                'allocation_modes':MODES,'cache_policy':'warm repeated fixed inputs; no cache flush',
                'seed':args.seed,'input_generation':'CPU torch.rand, no GPU reference computation'}}
    path=folder/'summary.json';write_json(path,report)
    try:
        torch,ic,native,device,env,bundle=environment(folder)
        report['environment']=env;report['bundle']=bundle;write_json(path,report)
        print('ENVIRONMENT:',json.dumps(env,ensure_ascii=False),flush=True)
        print('PROTOCOL:',json.dumps(report['protocol'],ensure_ascii=False),flush=True)
        print('BUNDLE:',bundle['path'],bundle['manifest_sha256'],flush=True)
        cases=CASES[:2] if args.quick else CASES
        report['expected_case_modes']=len(DTYPES)*len(cases)*len(MODES)
        for dtype in DTYPES:
            for spec_index,(n,h,layout) in enumerate(cases):
                fixture=Fixture(torch,ic,native,device,n,h,layout,dtype,args.seed+spec_index)
                for mode in MODES:
                    name=f'{dtype}_{n}x{h}_{layout}_{mode}'
                    report['current_case']=name;write_json(path,report)
                    print('RUN:',name,flush=True)
                    row=run_case(fixture,mode,args,folder/name,len(report['cases']))
                    report['cases'].append(row);write_json(path,report)
                del fixture
                gc.collect()
        report['backend']=native.gated_residual_hygon_backend_info()
        if any(k in sys.modules for k in ('triton','ninetoothed')):
            raise RuntimeError('Benchmark process unexpectedly imported a GPU compiler')
        # Detect asset replacement during the run instead of mixing revisions.
        if validate_bundle(bundle['path'],SOURCE_SHA256)!=bundle:
            raise RuntimeError('Kernel assets changed during benchmarking')
        report['passed']=complete(report['cases'],report['expected_case_modes'],args.samples)
        if not report['passed']:raise RuntimeError('Incomplete timing/correctness coverage')
        report['performance_tested']=True
        return 0
    except BaseException as exc:
        report['passed']=False;report['error']=f'{type(exc).__name__}: {exc}';traceback.print_exc();return 1
    finally:
        write_json(path,report);tables(folder,report)
        print('SUMMARY:',path,'passed=',report['passed'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true',help='small smoke run only; not the report experiment')
    p.add_argument('--samples',type=int,default=60)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--warmup',type=int,default=25)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.quick:args.samples,args.iterations,args.warmup=6,5,3
    if args.samples<2 or args.samples%2 or args.iterations<1 or args.warmup<1 or args.timeout<1:
        p.error('Use positive counts and an even samples count >=2')
    if args.worker:return worker(args)
    home=Path(os.environ['WORK_HOME']).resolve(strict=True)
    logs=home/'logs';logs.mkdir(exist_ok=True)
    folder=logs/('h3_gated_benchmark_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
    print('REPORT DIRECTORY:',folder,flush=True)
    parent={'passed':False,'started':datetime.now().isoformat(),'arguments':vars(args)|{'worker':None}}
    code=1
    try:
        active=home/'logs/h3_gated_active_bundle.txt'
        bundle=Path(active.read_text().strip())
        if not bundle.is_absolute():raise RuntimeError('Active bundle path must be absolute')
        parent['selected_bundle']=validate_bundle(bundle,SOURCE_SHA256)
        parent['active_path_file_sha256']=sha256(active)
        env=dict(os.environ,INFINICORE_H3_GATED_BUNDLE=str(bundle),LOG_DIR=str(folder))
        parent['script_hashes']={f.name:sha256(f) for f in HERE.iterdir() if f.is_file()}
        command=[sys.executable,'-X','faulthandler','-u',str(Path(__file__).resolve()),*sys.argv[1:],'--worker',str(folder)]
        parent['command']=command;write_json(folder/'run.json',parent)
        # Child inherits stdout so run_logged shows progress; outer timeout can
        # still archive partial evidence after native failures or hangs.
        with subprocess.Popen(command,env=env,start_new_session=True) as child:
            try:code=child.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
        parent['exit_code']=code
        summary=json.loads((folder/'summary.json').read_text())
        parent['passed']=code==0 and summary.get('passed') is True
        if not parent['passed']:code=code or 1
    except Exception as exc:
        parent['error']=f'{type(exc).__name__}: {exc}';traceback.print_exc();code=1
    finally:
        parent['finished']=datetime.now().isoformat();write_json(folder/'run.json',parent)
        files=sorted(f for f in folder.rglob('*') if f.is_file())
        write_json(folder/'files.json',[{'path':f.relative_to(folder).as_posix(),'sha256':sha256(f),'size':f.stat().st_size} for f in files])
        archive=folder.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in files+[folder/'files.json']:z.write(f,folder.name+'/'+f.relative_to(folder).as_posix())
        print('EVIDENCE ZIP:',archive,flush=True)
        print('BENCHMARK PASSED' if parent['passed'] else 'BENCHMARK FAILED; inspect run.json and summary.json',flush=True)
        print('Performance results are synthetic operator measurements, not model or 8-card speedups.',flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
