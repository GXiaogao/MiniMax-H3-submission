"""Recheck saved server evidence; does not execute any GPU or reference model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'logs/2026-10-27-gated-rounding-v3-server'
E = BASE / 'evidence'
PACKAGE = ROOT / 'scripts/h3-gated-rounding-v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_metric(actual, expected, saved):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    error = np.abs(actual - expected)
    current = {'max_abs': float(error.max()), 'mean_abs': float(error.mean()),
               'outside_tolerance': int(np.count_nonzero(error > saved['atol'] + saved['rtol'] * np.abs(expected)))}
    for key, value in current.items():
        assert value == saved[key], (key, value, saved[key])
    assert current['outside_tolerance'] == 0
    return current


def main():
    summary = read(E / 'summary.json')
    assert summary['run_id'] == '20261027_201509_22845adc' and summary['passed']
    records = read(E / 'files.json')
    assert len(records) == 211 and len({r['path'] for r in records}) == 211
    for r in records:
        p = (E / r['path']).resolve()
        assert p.is_relative_to(E.resolve()) and p.stat().st_size == r['size'] and sha(p) == r['sha256'], r
    for name, digest in summary['script_hashes'].items():
        assert sha(PACKAGE / name) == digest, name
    counts = {'ops':44, 'h3_off':18, 'h3_on':18, 'bridge_off':12, 'bridge_on':12, 'mha':85}
    stages = {}
    for row in summary['stages']:
        stage = row['stage']; data = read(E / stage / 'result.json')
        assert row['status'] == 'PASS' and row['exit_code'] == 0
        assert data['run_id'] == summary['run_id'] and data['stage'] == stage and data['passed']
        assert len(data['cases']) == counts[stage] and all(c['passed'] for c in data['cases'])
        stages[stage] = data
    assert set(stages) == set(counts)
    for prefix, calls in [('h3',36), ('bridge',24)]:
        assert stages[prefix+'_off']['native_launches'] == 0
        assert stages[prefix+'_on']['native_launches'] == calls
    assert stages['ops']['native_mul_add_comparisons'] == 36
    assert stages['ops']['native_launch_count'] == stages['ops']['numeric_cases'] == 36
    assert stages['ops']['rejection_cases'] == 8
    for p in (E / 'ops').glob('*.npz'):
        with np.load(p, allow_pickle=False) as a:
            assert np.isfinite(a['actual']).all() and np.array_equal(a['actual'], a['expected']), p
    assert len(list((E / 'ops').glob('*.npz'))) == 36
    metrics = []
    for stage in ('h3_off','h3_on'):
        report = read(E / stage / 'model_report.json')
        assert report['status'] == 'passed' and len(report['cases']) == 6
        for model in report['cases']:
            assert model['parameters_exact'] == 42 and model['status'] == 'passed'
            for run in model['runs']:
                assert run['status'] == 'passed' and run['inputs_unchanged']
                path = E / stage / model['name'].replace('/','-') / f"outputs_{run['repeat']}.npz"
                with np.load(path, allow_pickle=False) as a:
                    for field in ('video','audio'):
                        metrics.append(check_metric(a[field],a['reference_'+field],run[field]))
    for stage in ('bridge_off','bridge_on'):
        report = read(E / stage / 'model_report.json')
        assert report['status'] == 'passed' and len(report['models']) == 3
        for model in report['models']:
            assert model['status'] == 'passed'
            for i, run in enumerate(model['runs']):
                assert run['status'] == 'passed' and run['inputs_unchanged'] and run['handoff']['values_exact']
                path = E / stage / model['dtype'] / f'case_{i}.npz'
                assert sha(path) == run['arrays']['sha256']
                with np.load(path, allow_pickle=False) as a:
                    metrics.append(check_metric(a['encoder_actual'],a['encoder_reference'],run['encoder_error']))
                    for field in ('video','audio'):
                        metrics.append(check_metric(a[field+'_actual'],a[field+'_conditional_reference'],run['conditional_h3_error'][field]))
                        metrics.append(check_metric(a[field+'_actual'],a[field+'_chain_reference'],run['chain_error'][field]))
    assert len(metrics) == 192
    pairs = []
    for row in summary['off_on_comparisons']:
        stage, rel = row['path'].split('/',1)
        assert row['exact']
        with np.load(E/(stage+'_off')/rel,allow_pickle=False) as off, np.load(E/(stage+'_on')/rel,allow_pickle=False) as on:
            assert set(off.files) == set(on.files) == set(row['keys'])
            for key in off.files:
                assert off[key].dtype == on[key].dtype and np.array_equal(off[key],on[key]), (row['path'],key)
                if off[key].dtype.kind in 'buifc':assert np.isfinite(off[key]).all() and np.isfinite(on[key]).all()
            pairs.append({'path':row['path'],'fields':len(off.files),'exact':True})
    assert len(pairs) == 30 and len({r['path'] for r in pairs}) == 30
    probes = []
    for label in ('before','after'):
        folder = E/'rounding_probes'/('rounding_probe_'+label)
        report = read(folder/'report.json');assert report['passed'] and len(report['cases']) == 3
        for row in report['cases']:
            with np.load(folder/(row['layout']+'.npz'),allow_pickle=False) as a:
                expected = a['reference_down' if label=='after' else 'reference_nearest']
                assert np.isfinite(a['candidate']).all()
                assert np.array_equal(a['candidate'].view(np.uint32),expected.view(np.uint32))
                assert np.array_equal(a['original'].view(np.uint32),a['reference_down'].view(np.uint32))
                if label=='after':assert np.array_equal(a['candidate'].view(np.uint32),a['original'].view(np.uint32))
                probes.append({'label':label,'layout':row['layout'],'elements':int(a['candidate'].size),'bitwise_match':True})
    spec = importlib.util.spec_from_file_location('h3_audit_abi', PACKAGE/'abi.py')
    abi = importlib.util.module_from_spec(spec);spec.loader.exec_module(abi)
    exports = []
    bundle = read(E/'bundle_evidence/bundle.json')
    assert sha(E/'bundle_evidence/bundle.json') == summary['bundle']['manifest_sha256']
    for dtype in ('float32','float16','bfloat16'):
        folder = E/'bundle_evidence'/('export_'+dtype)
        result = abi.validate_abi((folder/'kernel.amdgcn').read_text(),read(folder/'metadata.json'))
        assert sha(folder/'kernel.hsaco') == bundle['files'][f'export_{dtype}/kernel.hsaco']
        exports.append({'dtype':dtype,**result})
    result = {'run_id':summary['run_id'],'acceptance':'saved correctness evidence verified',
        'zip_sha256':sha(BASE/f"h3_gated_rounding_integration_{summary['run_id']}.zip"),
        'file_manifest_verified':len(records),'script_hashes_matched':len(summary['script_hashes']),
        'numeric_op_arrays':36,'model_reference_comparisons_recomputed':len(metrics),
        'off_on_output_pairs':pairs,'rounding_probes':probes,'exports':exports,'stages':counts,
        'native_calls':{'ops':36,'h3_off':0,'h3_on':36,'bridge_off':0,'bridge_on':24},
        'server_active_bundle':summary['bundle']['path'],
        'limits':['Local audit rechecks saved arrays; no local GPU execution',
                  'Original mul+add outputs in ops are checked by server but not separately saved in ops NPZ',
                  'No measured performance comparison','No official full model or eight-card validation']}
    (BASE/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('off_on_output_pairs','rounding_probes','exports')},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
