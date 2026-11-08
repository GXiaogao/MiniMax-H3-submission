#!/usr/bin/env python3
"""Prepare a separate optional-trace InfiniLM build; never edit official files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

CPP_SHA='c881444814a634ea3b90446f08622c7e2343861df13accaafa395ee9b942aa1a'
BUILD_SHA='78b75da3a8d7a4a51981e67d816280fdfca8d7a815aaf8efe78f58235bc1486e'
BOUNDARIES=['packed_hidden','temb','adaln_projection','attention_input','attention_output','after_attention','mlp_input','mlp_output','after_mlp']

def sha(data):return hashlib.sha256(data).hexdigest()
def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('expected exact single source anchor: '+old[:70])
    return text.replace(old,new)

def instrument_text(data):
    if sha(data)!=CPP_SHA:raise ValueError('unrecognized or already patched H3 source')
    text=data.decode('utf-8')
    text=replace_once(text,'#include <vector>','#include <vector>\n#include "h3_boundary_trace.hpp"')
    text=replace_once(text,'    require_tensor(input.video_hidden_states, "video_hidden_states");',
        '    if (boundary_trace::enabled() && transformer_blocks_.size()!=1)\n'
        '        throw std::runtime_error("boundary trace supports only the isolated one-block diagnostic");\n'
        '    require_tensor(input.video_hidden_states, "video_hidden_states");')
    text=replace_once(text,'    auto temb = time_embedder_->forward(input.timestep.value());',
        '    boundary_trace::save("packed_hidden", hidden_states);\n'
        '    auto temb = time_embedder_->forward(input.timestep.value());\n'
        '    boundary_trace::save("temb", temb);')
    text=replace_once(text,'    auto modulation = adaln_proj_->forward(temb)->view(',
        '    auto modulation_raw = adaln_proj_->forward(temb);\n'
        '    boundary_trace::save("adaln_projection", modulation_raw);\n'
        '    auto modulation = modulation_raw->view(')
    text=replace_once(text,'    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);',
        '    boundary_trace::save("attention_input", normalized);\n'
        '    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);\n'
        '    boundary_trace::save("attention_output", attention_output);')
    text=replace_once(text,'    auto output = gated_update(hidden_states, gate_msa, attention_output);',
        '    auto output = gated_update(hidden_states, gate_msa, attention_output);\n'
        '    boundary_trace::save("after_attention", output);')
    text=replace_once(text,'    auto mlp_output = ff_->forward(normalized);',
        '    boundary_trace::save("mlp_input", normalized);\n'
        '    auto mlp_output = ff_->forward(normalized);\n'
        '    boundary_trace::save("mlp_output", mlp_output);')
    text=replace_once(text,'    output = gated_update(output, gate_mlp, mlp_output);',
        '    output = gated_update(output, gate_mlp, mlp_output);\n'
        '    boundary_trace::save("after_mlp", output);')
    return text,{'source_sha256':sha(data),'patched_sha256':sha(text.encode()),'boundaries':BOUNDARIES,
                 'sampling':'linear-integer-indices-at-most-65536','rank':0,'requires_graph_disabled':True}

def patch_file(path):
    data=path.read_bytes();text,meta=instrument_text(data)
    backup=path.with_suffix(path.suffix+'.before-boundary-trace')
    if backup.exists():raise FileExistsError(backup)
    helper=Path(__file__).with_name('h3_boundary_trace.hpp')
    target=path.with_name(helper.name)
    if target.exists():raise FileExistsError(target)
    backup.write_bytes(data);shutil.copyfile(helper,target)
    path.write_text(text,encoding='utf-8',newline='\n')
    meta.update(backup=str(backup),helper_sha256=sha(helper.read_bytes()))
    path.with_suffix(path.suffix+'.trace.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta

def builder_text(data):
    if sha(data)!=BUILD_SHA:raise ValueError('unrecognized base build script')
    text=data.decode('utf-8')
    changes=[('readonly BUILD_VERSION="1.5-hygon-fl2va-no-cpu-runtime"','readonly BUILD_VERSION="1.0-isolated-h3-boundary-trace"'),
        ('readonly PERSISTENT_PROJECT_DIR="${WORK_HOME}/Projects/InfiniLM"','readonly PERSISTENT_PROJECT_DIR="${WORK_HOME}/diagnostics/h3-boundary/${H3_BOUNDARY_BUILD_ID:?}/InfiniLM"'),
        ('readonly RUN_DIR="${LOG_DIR}/infinifabric-build-infinilm-extension-${RUN_ID}"','readonly RUN_DIR="${LOG_DIR}/infinifabric-build-h3-boundary-${RUN_ID}"'),
        ('readonly SUCCESS_MARKER="${STATE_DIR}/infinilm-extension.success"','readonly SUCCESS_MARKER="${STATE_DIR}/infinilm-boundary.success"'),
        ('[[ -f "${PERSISTENT_PROJECT_DIR}/python/infinilm/lib/__init__.py" ]]',
         '[[ -f "${WORK_HOME}/Projects/InfiniLM/python/infinilm/lib/__init__.py" ]]'),
        ('run_stage "hygon-fl2va-no-cpu-runtime-patch" patch_hygon_fl2va_no_cpu_runtime',
         'run_stage "hygon-fl2va-no-cpu-runtime-patch" patch_hygon_fl2va_no_cpu_runtime\n'
         'run_stage "h3-boundary-trace-patch" "${PYTHON}" "${H3_BOUNDARY_PATCHER:?}" --source "${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/minimax_h3_transformer.cpp"\n'
         'run_stage "trace-source-evidence" cp -- "${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/minimax_h3_transformer.cpp" '
         '"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_boundary_trace.hpp" '
         '"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/minimax_h3_transformer.cpp.trace.json" "${RUN_DIR}/"'),
        ('mkdir -p "${DESTINATION_DIR}"',
         'mkdir -p "${PERSISTENT_PROJECT_DIR}"\n'
         'cp -R -- "${WORK_HOME}/Projects/InfiniLM/python" "${PERSISTENT_PROJECT_DIR}/"\n'
         'mkdir -p "${DESTINATION_DIR}"')]
    for old,new in changes:text=replace_once(text,old,new)
    meta={'source_sha256':sha(data),'builder_sha256':sha(text.encode()),
          'destination_root':'${WORK_HOME}/diagnostics/h3-boundary/${H3_BOUNDARY_BUILD_ID}/InfiniLM',
          'success_marker':'${STATE_DIR}/infinilm-boundary.success'}
    return text,meta

def write_builder(source,destination):
    if destination.exists():raise FileExistsError(destination)
    text,meta=builder_text(source.read_bytes());destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    destination.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path);p.add_argument('--builder',action='store_true');args=p.parse_args()
    if args.builder and args.destination is None:p.error('--builder requires --destination')
    print(json.dumps(write_builder(args.source,args.destination) if args.builder else patch_file(args.source),indent=2))
if __name__=='__main__':main()
