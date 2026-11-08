"""Independently audit saved single-operation launch evidence; no GPU execution.

Run with the original ZIP and terminal log. Originals are retained byte-for-byte;
derived audit files go to logs/2026-10-28-gated-launch-single-op-server.
"""
from pathlib import Path, PurePosixPath
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/2026-10-28-gated-launch-single-op-server'
PACKAGE = ROOT / 'artifacts/h3-gated-launch-diagnosis-v3-single-op.tar.gz'
PACKAGE_SHA = '85ba7f96c8f29686a87e1a9d17840671882c2e1b5cdc20ac69f83e3cec46eea8'
PATHS = ('baseline', 'fused', 'direct')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keep_original(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, f'Existing evidence differs: {path}'
    else:
        path.write_bytes(data)


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-10), (a, b)


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size == 40 and np.isfinite(values).all() and (values > 0).all()
    return dict(samples=40, p50_us=float(np.quantile(values, .5)),
                p95_us=float(np.quantile(values, .95)), min_us=float(values.min()),
                max_us=float(values.max()), mean_us=float(values.mean()),
                stdev_us=float(values.std(ddof=1)))


def chain(topology, expected_types):
    """Check a connected, directed single chain without using server validators."""
    types, edges = topology['node_types'], topology['edges']
    size = len(types)
    assert size == len(expected_types) and len(edges) == size - 1
    assert len({tuple(edge) for edge in edges}) == len(edges)
    if 'node_handles' in topology:
        assert len(topology['node_handles']) == size
        assert len(set(topology['node_handles'])) == size
    links = {}
    incoming = [0] * size
    for start, end in edges:
        assert 0 <= start < size and 0 <= end < size and start != end
        assert start not in links
        links[start] = end
        incoming[end] += 1
    roots = [i for i, count in enumerate(incoming) if count == 0]
    assert len(roots) == 1 and max(incoming) <= 1
    ordered = []
    node = roots[0]
    while node not in ordered:
        ordered.append(node)
        if node not in links:
            break
        node = links[node]
    assert len(ordered) == size and ordered[-1] not in links
    assert [types[i] for i in ordered] == expected_types


def audit_phase(phase, graph=False):
    assert phase['passed'] is True
    paths = PATHS[:2] if graph else PATHS
    records = phase['samples']
    assert len(records) == 40 * len(paths)
    for index in range(40):
        if graph:
            order = list(paths if index % 2 == 0 else paths[::-1])
        else:
            order = list(paths[index % 3:] + paths[:index % 3])
            if index % 2:
                order.reverse()
        group = records[index * len(paths):(index + 1) * len(paths)]
        assert [r['path'] for r in group] == order
        for row in group:
            assert row['index'] == index
            values = [row[k] for k in ('submit_ns', 'remaining_wait_ns', 'total_ns')]
            assert all(type(v) is int and v > 0 for v in values)
            assert values[0] + values[1] == values[2]
            expected_counts = [int(row['path'] == 'fused'), int(row['path'] == 'direct')]
            assert row['submissions'] == ([0, 0] if graph else expected_counts)
            if graph:
                assert row['operations'] == 1 and row['event_ms'] > 0
                close(row['device_graph_us_per_op'], row['event_ms'] * 1000)
                # Same-stream events must fit inside the synchronized wall envelope.
                assert row['device_graph_us_per_op'] <= row['total_ns'] / 1000 + 1
    if graph:
        assert phase['operations_per_graph'] == 1 and phase['scope'] == 'single_operation_graph'
        assert set(phase['graphs']) == set(paths)
        for name, types in (('baseline', [7, 200, 200, 7]), ('fused', [7, 200, 7])):
            entry = phase['graphs'][name]
            chain(entry['topology'], types)
            assert entry['capture_submissions'] == [int(name == 'fused'), 0]
    recalculated = {}
    for name in paths:
        selected = [r for r in records if r['path'] == name]
        metrics = {key + '_us': [r[key + '_ns'] / 1000 for r in selected]
                   for key in ('submit', 'remaining_wait', 'total')}
        if graph:
            metrics['device_graph_us_per_op'] = [r['event_ms'] * 1000 for r in selected]
        recalculated[name] = {key: statistics(values) for key, values in metrics.items()}
        assert set(phase['statistics'][name]) == set(metrics)
        for key, expected in recalculated[name].items():
            saved = phase['statistics'][name][key]
            assert set(saved) == set(expected)
            for field in expected:
                close(saved[field], expected[field])
    assert set(phase['statistics']) == set(paths)
    return recalculated


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: this evidence audit uses assertions.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    blobs = {}
    prefix = args.archive.stem + '/'
    with zipfile.ZipFile(args.archive) as archive:
        entries = archive.infolist()
        assert len(entries) == len({entry.filename for entry in entries})
        for entry in entries:
            assert entry.filename.startswith(prefix) and '\\' not in entry.filename
            assert not entry.is_dir() and (entry.external_attr >> 16) & 0o170000 != 0o120000
            name = PurePosixPath(entry.filename[len(prefix):])
            assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name)
            target = OUT / 'evidence' / name
            assert target.resolve().is_relative_to((OUT / 'evidence').resolve())
            data = archive.read(entry)
            blobs[str(name)] = data
    read = lambda name: json.loads(blobs[name])
    inventory = read('files.json')
    assert len(inventory) == 206 and len({x['path'] for x in inventory}) == 206
    assert {x['path'] for x in inventory} == set(blobs) - {'files.json'}
    for entry in inventory:
        assert len(blobs[entry['path']]) == entry['size']
        assert digest(blobs[entry['path']]) == entry['sha256']
    summary, run = read('summary.json'), read('run.json')
    assert run['passed'] is True and run['exit_code'] == 0
    assert run['arguments']['samples'] == 40 and run['arguments']['warmup'] == 10
    assert run['arguments']['quick'] is False
    for key in ('passed', 'separation_measured', 'full_matrix_measured'):
        assert summary[key] is True
    for key in ('batch_graph_measured', 'pure_kernel_time_measured', 'official_model_tested',
                'eight_card_tested', 'quick_smoke_only'):
        assert summary[key] is False
    assert summary['measurement_scope'] == 'single_operation_graph'
    assert summary['expected_cases'] == 9 and summary['protocol']['graph_sizes'] == [1]
    assert summary['protocol']['samples'] == 40 and summary['protocol']['warmup'] == 10
    assert summary['protocol']['allocation'] == 'preallocated only'
    assert digest(PACKAGE.read_bytes()) == PACKAGE_SHA
    with tarfile.open(PACKAGE) as package:
        assert len(run['script_hashes']) == 11
        for name, expected in run['script_hashes'].items():
            assert digest(package.extractfile('h3-gated-launch-diagnosis-v1/' + name).read()) == expected, name
    previous = json.loads((ROOT / 'logs/2026-10-28-gated-benchmark-server/evidence/summary.json').read_text())
    assert run['selected_bundle'] == summary['bundle'] == previous['bundle']
    for lib in ('libinfinicore_cpp_api.so', 'libinfiniop.so', 'libinfinirt.so'):
        get = lambda doc: {x['sha256'] for x in doc['environment']['libraries'] if PurePosixPath(x['path']).name == lib}
        assert get(summary) == get(previous) and len(get(summary)) == 1
    calibration = summary['graph_calibration']
    assert calibration['event']['node_layout'] == {'event': 7}
    chain(calibration['event']['event_only_graph'], [7, 7])
    assert calibration['computation']['node_layout'] == {'event': 7, 'compute': 200}
    chain(calibration['computation']['baseline_computation_graph'], [200, 200])
    chain(calibration['computation']['fused_computation_graph'], [200])
    assert calibration['capture_submissions'] == [1, 0]
    csv_rows = list(csv.DictReader(io.StringIO(blobs['results.csv'].decode('utf-8'))))
    assert len(csv_rows) == 153
    expected_cases = {f'{dtype}_{n}x{h}_6h' for dtype in ('float32', 'float16', 'bfloat16')
                      for n, h in ((7, 64), (257, 513), (4096, 2048))}
    assert len(summary['cases']) == 9 and {c['name'] for c in summary['cases']} == expected_cases
    results, outliers = [], []
    raw_count = stat_count = correctness_count = 0
    for case in summary['cases']:
        name, fixture = case['name'], case['fixture']
        assert case['passed'] is True and len(case['graphs']) == 1
        assert len([c for c in previous['cases'] if c['fixture'] == fixture and c['mode'] == 'preallocated']) == 1
        computed = {}
        for phase_name, phase in [('host', case['host']), ('graph_1', case['graphs'][0])]:
            assert read(name + '/' + phase_name + '.json') == phase
            stats = audit_phase(phase, graph=phase_name == 'graph_1')
            computed[phase_name] = stats
            raw_count += len(phase['samples'])
            wanted_checks = {p + '_' + time for p in stats for time in ('before', 'after')}
            assert set(phase['correctness']) == wanted_checks
            for key, check in phase['correctness'].items():
                filename = name + '/' + phase_name + '_' + key
                assert read(filename + '_correctness.json') == check
                for field in ('output_poisoned_before_call', 'finite', 'equal_reference_bits', 'inputs_unchanged'):
                    assert check[field] is True
                assert check['intermediate_poisoned_before_call'] is key.startswith('baseline_')
                assert check['output_sha256'] == check['reference_sha256'] == fixture['cpu_expected_sha256']
                with np.load(io.BytesIO(blobs[filename + '_sample.npz']), allow_pickle=False) as arr:
                    assert set(arr.files) == {'indices', 'actual', 'expected'}
                    size = math.prod(fixture['shape'])
                    indices = np.unique(np.r_[np.arange(min(64, size)), np.linspace(0, size - 1, min(4096, size), dtype=np.int64)])
                    assert np.array_equal(arr['indices'], indices)
                    assert arr['actual'].dtype == arr['expected'].dtype == np.float32
                    assert arr['actual'].shape == arr['expected'].shape == indices.shape
                    assert np.isfinite(arr['actual']).all() and np.isfinite(arr['expected']).all()
                    assert np.array_equal(arr['actual'].view(np.uint32), arr['expected'].view(np.uint32))
                correctness_count += 1
            for path, metrics in stats.items():
                for metric, values in metrics.items():
                    matches = [r for r in csv_rows if (r['case'], r['phase'], r['path'], r['metric']) == (name, phase_name, path, metric)]
                    assert len(matches) == 1 and int(matches[0]['operations_per_graph']) == 1
                    for field in ('samples', 'p50_us', 'p95_us', 'mean_us', 'min_us', 'max_us'):
                        close(float(matches[0][field]), values[field])
                    stat_count += 1
                maximum = max((r for r in phase['samples'] if r['path'] == path), key=lambda r: r['total_ns'])
                outliers.append(dict(case=name, phase=phase_name, path=path, maximum_wall_sample=maximum))
        host, graph = computed['host'], computed['graph_1']
        row = dict(case=name, dtype=fixture['dtype'], shape=fixture['shape'])
        for path in PATHS:
            row[path + '_submit_p50_us'] = host[path]['submit_us']['p50_us']
            row[path + '_total_p50_us'] = host[path]['total_us']['p50_us']
        for path in PATHS[:2]:
            row[path + '_graph_p50_us'] = graph[path]['device_graph_us_per_op']['p50_us']
            row[path + '_graph_p95_us'] = graph[path]['device_graph_us_per_op']['p95_us']
        for phase in ('submit', 'total', 'graph'):
            row[phase + '_p50_ratio'] = row['baseline_' + phase + '_p50_us'] / row['fused_' + phase + '_p50_us']
        results.append(row)
    assert raw_count == 1800 and stat_count == 153 and correctness_count == 90
    log = args.log.read_text(encoding='utf-8')
    assert 'SINGLE-OP DIAGNOSIS COMPLETE' in log and 'exit code: 0' in log
    assert blobs['worker.log'].decode('utf-8').replace('\r\n', '\n').strip() in log.replace('\r\n', '\n')
    assert sum(line.startswith('PASS: ') for line in log.splitlines()) == 9
    keep_original(OUT / args.archive.name, args.archive.read_bytes())
    keep_original(OUT / 'server-terminal.log', args.log.read_bytes())
    for name, data in blobs.items():
        keep_original(OUT / 'evidence' / name, data)
    result = dict(passed=True, run_id=args.archive.stem, zip_sha256=digest(args.archive.read_bytes()),
                  manifest_files_checked=206, published_package_sha256=PACKAGE_SHA, script_hashes_checked=11,
                  cases_checked=9, same_fixture_as_prior_benchmark=9, raw_records_checked=raw_count,
                  statistics_recomputed=stat_count, csv_rows_checked=153, correctness_records_checked=90,
                  sample_arrays_checked=90, graph_topologies_checked=18, calibration_topologies_checked=3,
                  host_submit_faster_cases=sum(r['submit_p50_ratio'] > 1 for r in results),
                  host_total_faster_cases=sum(r['total_p50_ratio'] > 1 for r in results),
                  graph_interval_faster_cases=sum(r['graph_p50_ratio'] > 1 for r in results),
                  all_slow_samples_retained=True, batch_graph_measured=False, pure_kernel_time_measured=False,
                  official_model_tested=False, eight_card_tested=False,
                  correctness_limit='Server full-array checks/hash agreement plus local bitwise excerpt comparison. No full tensors or input arrays in this ZIP; CPU arithmetic was not recomputed locally.',
                  timing_limit='Saved event durations and wall intervals audited without GPU rerun. K=1 graph intervals include graph scheduling/event overhead; host submit is not pure CPU time. No median subtraction.',
                  results=results, maximum_wall_samples=outliers)
    (OUT / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (OUT / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'maximum_wall_samples')}, indent=2))
    for row in results:
        print(row['case'], 'submit/total/graph ratios:', *(round(row[k + '_p50_ratio'], 4) for k in ('submit', 'total', 'graph')))


if __name__ == '__main__':
    main()
