"""Prepare an isolated rebuild with source hash gates and a run-local receipt."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('clean_build_load_sync',Path(__file__).with_name('prepare_minimax_h3_load_sync_trace.py'))
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def builder_text(data):
    text,_=base.builder_text(data)
    text=base.replace_once(text,'/diagnostics/h3-load-sync/','/diagnostics/h3-load-sync-rebuild/')
    text=base.replace_once(text,'infinifabric-build-h3-load-sync-','infinifabric-rebuild-h3-load-sync-')
    text=base.replace_once(text,'readonly SUCCESS_MARKER="${STATE_DIR}/infinilm-load-sync.success"',
                           'readonly SUCCESS_MARKER="${RUN_DIR}/infinilm-load-sync-rebuild.success"')
    anchor='run_stage "trace-source-evidence"'
    text=base.replace_once(text,anchor,
        'run_stage "verify-reconstructed-source" "${PYTHON}" "${H3_REBUILD_VERIFY_HELPER:?}" '
        '--verify-root "${LOCAL_PROJECT_DIR}" --expected "${H3_REBUILD_EXPECTED_HASHES:?}" '
        '--output "${RUN_DIR}/source-hash-verification.json"\n'+anchor)
    return text

def verify(root,expected_path,output):
    expected=json.loads(expected_path.read_text(encoding='utf-8'))
    mapping={'patched_sha256':'csrc/models/minimax_h3/minimax_h3_transformer.cpp',
             'quant_patched_sha256':'csrc/layers/quantization/none_quantization.cpp',
             'rank_patched_sha256':'csrc/engine/rank_worker.cpp'}
    for name,h in expected['helper_hashes'].items():
        mapping[name]=('csrc/engine/' if name=='h3_weight_load_sync.hpp' else 'csrc/models/minimax_h3/')+name
    checks={}
    for label,rel in mapping.items():
        got=hashlib.sha256((root/rel).read_bytes()).hexdigest()
        want=expected.get(label,expected['helper_hashes'].get(label))
        checks[rel]={'sha256':got,'expected':want,'matches':got==want}
    result={'result':'SOURCE_HASHES_MATCH' if all(v['matches'] for v in checks.values()) else 'SOURCE_HASH_MISMATCH','files':checks}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['result']!='SOURCE_HASHES_MATCH':raise ValueError('Rebuilt source differs from recorded server trace')
    print('SOURCE_HASH_GATE PASSED: 7 output file hashes match previous server trace',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--builder',type=Path);p.add_argument('--destination',type=Path)
    p.add_argument('--verify-root',type=Path);p.add_argument('--expected',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.builder:
        if a.destination is None:p.error('destination required')
        if a.destination.exists():raise FileExistsError(a.destination)
        a.destination.write_text(builder_text(a.builder.read_bytes()),encoding='utf-8',newline='\n')
    else:
        if any(v is None for v in [a.verify_root,a.expected,a.output]):p.error('verify-root, expected and output required')
        verify(a.verify_root,a.expected,a.output)
