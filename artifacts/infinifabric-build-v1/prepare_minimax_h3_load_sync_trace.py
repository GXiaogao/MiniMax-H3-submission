#!/usr/bin/env python3
"""Build an isolated real-weight loading candidate retaining projection diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

_spec=importlib.util.spec_from_file_location('load_sync_projection_patch',Path(__file__).with_name('prepare_minimax_h3_projection_trace.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
sha=base.sha
replace_once=base.replace_once
ARCHIVE_RANK_SHA='d88ee44890a79e1b5a20b2fe36286f09e9c6cd34e3324fa67de34f1db3a709f1'
# The original builder applies its existing no-CPU allocation patch first.
RANK_SHA='5c2f0d4679c372b2e1fac1212f8fe69529ac33d623aca8f0cd0879cc6b52f36b'
VERSION='1.0-isolated-h3-load-sync-projection-trace'
HELPER='h3_weight_load_sync.hpp'


def instrument_rank(data):
    if sha(data)!=RANK_SHA:raise ValueError('unrecognized or already patched rank worker source')
    text=data.decode()
    text=replace_once(text,'#include "rank_worker.hpp"','#include "rank_worker.hpp"\n#include "h3_weight_load_sync.hpp"')
    text=replace_once(text,'                    model_->load_parameter(local_param_name, local_param);',
        '                    auto staged_param = weight_load_sync::stage(local_param, rank_info_.device);\n'
        '                    model_->load_parameter(local_param_name, staged_param);')
    text=replace_once(text,'                    model_->load_parameters_no_sync(local_params, local_params_strict);',
        '                    auto staged_params = local_params;\n'
        '                    for (auto &[name, tensor] : staged_params) {\n'
        '                        tensor = weight_load_sync::stage(tensor, rank_info_.device);\n'
        '                    }\n'
        '                    model_->load_parameters_no_sync(staged_params, local_params_strict);')
    return text


def patch_files(path,quant,rank):
    data=rank.read_bytes();text=instrument_rank(data)
    backup=rank.with_suffix(rank.suffix+'.before-load-sync')
    helper_target=rank.with_name(HELPER)
    if backup.exists() or helper_target.exists():raise FileExistsError('loading backup/helper exists')
    meta=base.patch_files(path,quant)
    backup.write_bytes(data)
    rank.write_text(text,encoding='utf-8',newline='\n')
    helper=Path(__file__).with_name(HELPER);shutil.copyfile(helper,helper_target)
    meta.update(trace_version=VERSION,rank_archive_sha256=ARCHIVE_RANK_SHA,rank_source_sha256=RANK_SHA,rank_patched_sha256=sha(text.encode()),
        rank_backup=str(backup),loading_control='hygon-cross-device-full-parameter-staging-before-existing-local-slice')
    meta['helper_hashes'][HELPER]=sha(helper.read_bytes())
    path.with_suffix(path.suffix+'.trace.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta


def builder_text(data):
    text,meta=base.builder_text(data)
    for old,new in [('1.0-isolated-h3-projection-trace',VERSION),
        ('/diagnostics/h3-projection/','/diagnostics/h3-load-sync/'),
        ('infinifabric-build-h3-projection-','infinifabric-build-h3-load-sync-'),
        ('infinilm-projection.success','infinilm-load-sync.success'),
        ('"h3-projection-trace-patch"','"h3-load-sync-trace-patch"')]:text=replace_once(text,old,new)
    anchor='"${LOCAL_PROJECT_DIR}/csrc/layers/quantization/none_quantization.cpp" '
    text=replace_once(text,anchor,anchor+'"${LOCAL_PROJECT_DIR}/csrc/engine/rank_worker.cpp" '
        '"${LOCAL_PROJECT_DIR}/csrc/engine/h3_weight_load_sync.hpp" ')
    anchor='--quant-source "${LOCAL_PROJECT_DIR}/csrc/layers/quantization/none_quantization.cpp"'
    text=replace_once(text,anchor,anchor+' --rank-source "${LOCAL_PROJECT_DIR}/csrc/engine/rank_worker.cpp"')
    meta.update(builder_sha256=sha(text.encode()),destination_root='${WORK_HOME}/diagnostics/h3-load-sync/${H3_BOUNDARY_BUILD_ID}/InfiniLM',
        success_marker='${STATE_DIR}/infinilm-load-sync.success')
    return text,meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--quant-source',type=Path);p.add_argument('--rank-source',type=Path)
    p.add_argument('--destination',type=Path);p.add_argument('--builder',action='store_true');args=p.parse_args()
    if args.builder:
        if args.destination is None:p.error('--builder requires --destination')
        if args.destination.exists():raise FileExistsError(args.destination)
        text,meta=builder_text(args.source.read_bytes());args.destination.parent.mkdir(parents=True,exist_ok=True)
        with args.destination.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
        args.destination.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    else:
        if args.quant_source is None or args.rank_source is None:p.error('patch requires quant-source/rank-source')
        meta=patch_files(args.source,args.quant_source,args.rank_source)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
