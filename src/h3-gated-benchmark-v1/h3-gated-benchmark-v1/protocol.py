"""CPU-only timing protocol and integrity gates; no inference or compiler imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

V3_REVISION = 'v3: preserve directed signed zero through integer selection and bitcast'
V3_SEMANTICS = 'FP32 mul nearest-even then add toward negative infinity; F16/BF16 unchanged'
DTYPES = ('float32', 'float16', 'bfloat16')
# Synthetic operator shapes, not official MiniMax-H3 evaluation dimensions.
CASES = ((7,64,'contiguous'), (7,64,'6h'),
         (257,513,'contiguous'), (257,513,'6h'), (257,513,'strided'), (257,513,'batch1_6h'),
         (4096,2048,'contiguous'), (4096,2048,'6h'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def statistics_us(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Timing samples must be finite and positive; none may be silently dropped')
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered)-1)*p
        low = math.floor(index)
        return ordered[low]+(ordered[math.ceil(index)]-ordered[low])*(index-low)
    return {'samples':len(values), 'p50_us':percentile(.5), 'p95_us':percentile(.95),
            'min_us':min(values), 'max_us':max(values), 'mean_us':statistics.mean(values),
            'stdev_us':statistics.stdev(values) if len(values)>1 else 0.0}


def per_call_us(elapsed_ms, iterations):
    if iterations <= 0 or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise ValueError('Invalid device timer or iteration count')
    return elapsed_ms*1000/iterations


def path_order(index, case_index):
    return ('baseline','fused') if (index+case_index)%2==0 else ('fused','baseline')


def host_sample(call, synchronize, clock=time.perf_counter_ns):
    synchronize()
    start = clock()
    output = call()
    synchronize()
    duration = (clock()-start)/1000
    statistics_us([duration])
    return duration, output


def device_sample(call, synchronize, event_factory, iterations):
    start, end = event_factory(), event_factory()
    synchronize()
    start.record()
    output = None
    for _ in range(iterations):
        output = call()
    end.record()
    end.synchronize()
    return per_call_us(start.elapsed_time(end), iterations), output


def check_launches(path, delta, calls):
    wanted = calls if path == 'fused' else 0
    if path not in ('baseline','fused') or delta != wanted:
        raise AssertionError(f'{path}: expected {wanted} NineToothed launches, got {delta}')


def comparison(baseline, fused):
    old, new = statistics_us(baseline), statistics_us(fused)
    return {'baseline':old, 'fused':new, 'p50_speedup':old['p50_us']/new['p50_us'],
            'p50_latency_reduction_percent':100*(1-new['p50_us']/old['p50_us'])}


def complete(rows, wanted, samples):
    if len(rows) != wanted:
        return False
    for row in rows:
        if not all(row.get(k) is True for k in ('passed','correct_before','correct_after','launch_counts_ok')):
            return False
        raw = row.get('samples', [])
        if len(raw)!=2*samples or {(x['index'],x['path']) for x in raw}!={(i,p) for i in range(samples) for p in ('baseline','fused')}:
            return False
        try:
            statistics_us([x['host_us'] for x in raw]); statistics_us([x['device_us'] for x in raw])
        except (ValueError, KeyError):
            return False
    return True


def validate_bundle(root, source_sha256, required_files=None):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root/'bundle.json').read_text())
    if manifest.get('revision') != V3_REVISION or manifest.get('semantics') != V3_SEMANTICS:
        raise ValueError('Benchmark requires the validated rounding-v3 bundle')
    if manifest.get('source_hashes',{}).get('gated_residual.py') != source_sha256:
        raise ValueError('NineToothed kernel source identity differs from validated v3')
    required = required_files if required_files is not None else {
        'hip_library.txt','libh3_gated_host.so',*[f'export_{d}/kernel.hsaco' for d in DTYPES]}
    if not required.issubset(manifest.get('files',{})):
        raise ValueError('Incomplete kernel bundle manifest')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path)!=expected:
            raise ValueError('Bundle file missing, escaping root or changed: '+name)
    return {'path':str(root),'manifest_sha256':sha256(root/'bundle.json'),'manifest':manifest}
