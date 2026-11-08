"""Reconstruct selected modified sources from fixed inputs; never compile or run a model."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

def sha(data):return hashlib.sha256(data).hexdigest()

def restore(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError(out)
    folder=root/'artifacts/infinifabric-build-v1'
    expected=json.loads((root/'configs/load-sync-source-hashes.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('restore_loading_patch',folder/'prepare_minimax_h3_load_sync_trace.py')
    patcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(patcher)
    builder=(folder/'build_infinilm_extension.sh').read_bytes()
    # Existing generator validates the exact fixed base builder before we reuse its Python payload.
    isolated_builder,build_meta=patcher.builder_text(builder)
    text=builder.decode('utf-8')
    anchor='    "${PYTHON}" -c \'\n'
    start=text.index(anchor,text.index('patch_hygon_fl2va_no_cpu_runtime()'))+len(anchor)
    payload=text[start:text.index('\n\' "${LOCAL_PROJECT_DIR}"',start)]
    paths=['models/qwen3_vl/qwen3_vl_for_conditional_generation.cpp','models/qwen3_5/qwen3_5_vision.cpp',
           'engine/infer_engine.cpp','engine/rank_worker.cpp','layers/attention/backends/static_attn.cpp',
           'models/minimax_h3/minimax_h3_transformer.cpp','layers/quantization/none_quantization.cpp']
    originals={}
    archive=folder/'InfiniLM-infinifabric-v3.tar.gz'
    with tarfile.open(archive) as t:
        for rel in paths:originals['csrc/'+rel]=t.extractfile('InfiniLM/csrc/'+rel).read()
    checks={}
    def check(label,data,key=None):
        actual=sha(data);want=key or expected[label]
        if actual!=want:raise ValueError('Server trace hash mismatch: '+label)
        checks[label]=actual
    check('rank_archive_sha256',originals['csrc/engine/rank_worker.cpp'])
    check('source_sha256',originals['csrc/models/minimax_h3/minimax_h3_transformer.cpp'])
    check('quant_source_sha256',originals['csrc/layers/quantization/none_quantization.cpp'])
    for name,h in expected['helper_hashes'].items():check(name,(folder/name).read_bytes(),h)
    out.mkdir(parents=True)
    for rel,data in originals.items():
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    argv=sys.argv
    try:
        sys.argv=['fixed-base-patch',str(out)]
        exec(compile(payload,'fixed-base-builder-source-transform','exec'),{})
    finally:sys.argv=argv
    h3=out/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'
    quant=out/'csrc/layers/quantization/none_quantization.cpp'
    rank=out/'csrc/engine/rank_worker.cpp'
    check('rank_source_sha256',rank.read_bytes())
    patcher.patch_files(h3,quant,rank)
    check('patched_sha256',h3.read_bytes())
    check('quant_patched_sha256',quant.read_bytes())
    check('rank_patched_sha256',rank.read_bytes())
    changes=[]
    for rel,data in originals.items():
        now=(out/rel).read_bytes()
        if now!=data:
            changes.extend(difflib.unified_diff(data.decode('utf-8').splitlines(True),now.decode('utf-8').splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    for name in expected['helper_hashes']:
        rel=('csrc/engine/' if name=='h3_weight_load_sync.hpp' else 'csrc/models/minimax_h3/')+name
        changes.extend(difflib.unified_diff([], (out/rel).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+rel))
    (out/'candidate.patch').write_text(''.join(changes),encoding='utf-8',newline='\n')
    (out/'build-isolated.sh').write_text(isolated_builder,encoding='utf-8',newline='\n')
    result={'result':'SOURCE_HASHES_MATCH_SERVER_TRACE','verified_hashes':checks,
            'archive_sha256':sha(archive.read_bytes()),'builder':build_meta,
            'scope':'selected changed sources plus helpers, not a complete source checkout',
            'binary_rebuild_verified':False,'model_correctness_verified':False}
    (out/'restoration-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(restore(Path(__file__).resolve().parents[1],args.output),indent=2))
