"""Archive and independently recalculate saved server benchmark evidence (no GPU)."""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(data): return hashlib.sha256(data).hexdigest()


def save_original(source,target):
    if target.exists():
        assert digest(source.read_bytes())==digest(target.read_bytes()),'Existing evidence differs'
    else:shutil.copyfile(source,target)


def recompute(values):
    x=np.asarray(values,dtype=np.float64)
    assert x.size==60 and np.isfinite(x).all() and (x>0).all()
    return dict(samples=len(x),p50_us=float(np.quantile(x,.5)),p95_us=float(np.quantile(x,.95)),
                min_us=float(x.min()),max_us=float(x.max()),mean_us=float(x.mean()),
                stdev_us=float(x.std(ddof=1)))


def close_dict(actual,expected):
    assert set(actual)==set(expected)
    for key in actual:
        assert math.isclose(actual[key],expected[key],rel_tol=2e-12,abs_tol=1e-10),(key,actual[key],expected[key])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path);p.add_argument('log',type=Path)
    args=p.parse_args()
    out=ROOT/'logs/2026-10-28-gated-benchmark-server';out.mkdir(exist_ok=True)
    save_original(args.archive,out/args.archive.name);save_original(args.log,out/'server-terminal.log')
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        members=z.infolist();names=[x.filename for x in members]
        assert len(names)==len(set(names))
        prefix=args.archive.stem+'/'
        blobs={}
        for item in members:
            name=item.filename
            assert name.startswith(prefix) and '\\' not in name and not item.is_dir()
            relative=PurePosixPath(name[len(prefix):])
            assert not relative.is_absolute() and '..' not in relative.parts
            assert (item.external_attr>>16)&0o170000 !=0o120000
            destination=evidence.joinpath(*relative.parts)
            assert destination.resolve().is_relative_to(evidence.resolve())
            data=z.read(item);blobs[relative.as_posix()]=data
            destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():assert destination.read_bytes()==data
            else:destination.write_bytes(data)
    read=lambda name:json.loads(blobs[name])
    inventory=read('files.json')
    assert {x['path'] for x in inventory}==set(blobs)-{'files.json'}
    assert len(inventory)==len({x['path'] for x in inventory})
    for item in inventory:
        assert len(blobs[item['path']])==item['size']
        assert digest(blobs[item['path']])==item['sha256']
    summary,run=read('summary.json'),read('run.json')
    assert run['passed'] is True and run['exit_code']==0
    assert summary['passed'] is True and summary['performance_tested'] is True
    assert summary['quick_smoke_only'] is False and summary['expected_case_modes']==48
    assert run['selected_bundle']==summary['bundle']
    assert summary['protocol']['samples']==60 and summary['protocol']['batch_iterations']==50 and summary['protocol']['warmup']==25
    for name,expected in run['script_hashes'].items():
        assert digest((ROOT/'scripts/h3-gated-benchmark-v1'/name).read_bytes())==expected,name
    assert len(run['script_hashes'])==8
    csv_rows=list(csv.DictReader(io.StringIO(blobs['results.csv'].decode('utf-8'))))
    assert len(csv_rows)==96
    cases=summary['cases'];assert len(cases)==48
    identities=set();records=0;sample_arrays=0;stats=0;launches=0
    for case_index,row in enumerate(cases):
        fixture=row['fixture'];dtype=fixture['dtype'];shape=fixture['shape'];layout=fixture['layout'];mode=row['mode']
        identity=(dtype,tuple(shape),layout,mode);assert identity not in identities;identities.add(identity)
        name=f"{dtype}_{shape[-2]}x{shape[-1]}_{layout}_{mode}"
        assert read(name+'/result.json')==row
        assert all(row[k] is True for k in ('passed','correct_before','correct_after','launch_counts_ok'))
        raw=row['samples'];assert len(raw)==120;records+=len(raw)
        for index in range(60):
            pair=raw[2*index:2*index+2]
            wanted=['baseline','fused'] if (case_index+index)%2==0 else ['fused','baseline']
            assert [x['path'] for x in pair]==wanted
            assert all(x['index']==index and x['batch_iterations']==50 for x in pair)
            assert all(x['native_launches']==(51 if x['path']=='fused' else 0) for x in pair)
        assert row['launch_counts']=={'baseline':0,'fused':3060};launches+=row['launch_counts']['fused']
        for metric in ('device','host'):
            sums={path:recompute([x[metric+'_us'] for x in raw if x['path']==path]) for path in ('baseline','fused')}
            for path in sums:close_dict(row[metric][path],sums[path]);stats+=1
            ratio=sums['baseline']['p50_us']/sums['fused']['p50_us']
            assert math.isclose(row[metric]['p50_speedup'],ratio,rel_tol=1e-12)
            reduction=100*(1-sums['fused']['p50_us']/sums['baseline']['p50_us'])
            assert math.isclose(row[metric]['p50_latency_reduction_percent'],reduction,rel_tol=1e-12,abs_tol=1e-10)
            matches=[x for x in csv_rows if x['dtype']==dtype and x['shape']==str(shape) and x['layout']==layout and x['mode']==mode and x['metric']==metric]
            assert len(matches)==1
            c=matches[0]
            for path in sums:
                for pct in ('p50','p95'):
                    assert math.isclose(float(c[path+'_'+pct+'_us']),sums[path][pct+'_us'],rel_tol=1e-12)
            assert math.isclose(float(c['p50_speedup']),ratio,rel_tol=1e-12)
        for phase in ('before','after'):
            check=read(name+'/'+phase+'_correctness.json');assert check==row[phase]
            assert check['inputs_unchanged'] is True and check['equal_baseline_fused_bits'] is True
            assert check['native_launches']=={'baseline':0,'fused':1}
            assert all(check[k]['finite'] is True and check[k]['equals_reference_bits'] is True for k in ('baseline','fused'))
            assert check['baseline']['sha256']==check['fused']['sha256']==fixture['cpu_expected_sha256']
            with np.load(io.BytesIO(blobs[name+'/'+phase+'_sample.npz']),allow_pickle=False) as arrays:
                assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
                for path in ('baseline','fused'):
                    assert arrays[path].dtype==np.float32 and np.array_equal(arrays[path].view(np.uint32),arrays['expected'].view(np.uint32))
                assert len(arrays['indices'])==len(set(arrays['indices']))
                # An independent CPU recheck is exact in FP32. Half/BF16 samples
                # are promoted to float32 for storage; their original full bits
                # cannot be independently recovered from hashes alone.
                if dtype=='float32':
                    import sys
                    sys.path.insert(0,str(ROOT/'scripts/h3-gated-rounding-v3'))
                    from rounding_reference import add_down
                    expected=add_down(arrays['residual'],(arrays['gate']*arrays['branch']).astype(np.float32))
                    assert np.array_equal(expected.view(np.uint32),arrays['expected'].view(np.uint32))
            sample_arrays+=1
    # All specified shapes, layouts and allocation variants must appear.
    specs=((7,64,'contiguous'),(7,64,'6h'),(257,513,'contiguous'),(257,513,'6h'),
           (257,513,'strided'),(257,513,'batch1_6h'),(4096,2048,'contiguous'),(4096,2048,'6h'))
    want={(d,((1,n,h) if layout=='batch1_6h' else (n,h)),layout,m)
          for n,h,layout in specs for d in ('float32','float16','bfloat16') for m in ('preallocated','allocating')}
    assert identities==want
    groups={}
    for mode in ('preallocated','allocating'):
        for metric in ('device','host'):
            values=[x[metric]['p50_speedup'] for x in cases if x['mode']==mode]
            groups[mode+'/'+metric]={'cases':len(values),'faster_cases':sum(x>1 for x in values),
                'speedup_min':min(values),'speedup_max':max(values),
                'note':'range across heterogeneous synthetic cases; not an aggregate model speedup'}
    log=args.log.read_text(encoding='utf-8')
    assert 'BENCHMARK PASSED' in log and 'exit code: 0' in log
    assert sum(line.startswith('PASS ') for line in log.splitlines())==48
    report={'passed':True,'run_id':summary['run_id'],'zip_sha256':digest(args.archive.read_bytes()),
        'manifest_files_checked':len(inventory),'script_hashes_checked':len(run['script_hashes']),
        'case_modes':len(cases),'raw_records':records,'timing_values':records*2,'stats_recomputed':stats,
        'csv_rows_checked':len(csv_rows),'sample_npz_checked':sample_arrays,'measured_fused_launches':launches,
        'correctness_limit':'Server reports full-array bitwise checks; local audit recalculated saved excerpts and verified full-array hash agreement, not unavailable full tensors.',
        'timing_limit':'Normalized recorded samples verified. Raw event durations and new device execution are unavailable; stream intervals contain host submission gaps.',
        'groups':groups,'official_model_tested':False,'eight_card_tested':False}
    (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
