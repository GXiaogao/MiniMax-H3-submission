#!/usr/bin/env python3
"""Prepare an isolated actual-weight projection/control diagnostic build."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

_spec=importlib.util.spec_from_file_location('projection_attention_patch',Path(__file__).with_name('prepare_minimax_h3_attention_trace.py'))
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
sha=base.sha
replace_once=base.replace_once
QUANT_SHA='4692bb718c511baa5c560e03364663641e8adbb7eb12d594d397b8e23f4f8fa1'
HELPERS=base.HELPERS+['h3_projection_trace.hpp']
STAGES=['out_weight','out_original','out_local','out_control_sum','out_original_after_control']


def instrument_text(data):
    text,meta=base.instrument_text(data)
    text=replace_once(text,'#include "h3_attention_trace.hpp"','#include "h3_attention_trace.hpp"\n#include "h3_projection_trace.hpp"')
    text=replace_once(text,'    auto projected = linear_forward(out_proj_, output);',
        '    auto projected = [&]() {\n'
        '        projection_trace::Scope scope(attention_trace::active);\n'
        '        return linear_forward(out_proj_, output);\n'
        '    }();')
    meta.update(patched_sha256=sha(text.encode()),trace_version='1.0-isolated-h3-projection-trace',
                projection_boundaries=STAGES,projection_scope='original-return-with-separate-recomputation-controls')
    return text,meta


def instrument_quant(data):
    if sha(data)!=QUANT_SHA:raise ValueError('unrecognized or already patched quantization source')
    text=data.decode()
    text=replace_once(text,'#include <optional>','#include <optional>\n#include "../../models/minimax_h3/h3_projection_trace.hpp"\n'
        'namespace h3_projection = infinilm::models::minimax_h3::projection_trace;')
    serial,rest=text.split('infinicore::Tensor NoneQuantization::forward_allreduce(',1)
    serial=replace_once(serial,'    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();',
        '    if (h3_projection::active && (has_bias || alpha != 1.0f))\n'
        '        throw std::runtime_error("projection trace requires no bias and alpha=1");\n'
        '    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();')
    for expression in ['infinicore::op::linear_packed(input_contiguous, weight, bias_opt, alpha)',
                       'infinicore::op::linear(input_contiguous->contiguous(), weight->contiguous(), bias_opt, alpha)']:
        serial=replace_once(serial,'return '+expression+';',
            'auto original = '+expression+';\n        h3_projection::inspect_serial(weight, original);\n        return original;')
    rest=replace_once(rest,'    if (alpha != 1.0f) {',
        '    if (h3_projection::active && (has_bias || alpha != 1.0f))\n'
        '        throw std::runtime_error("projection trace requires no bias and alpha=1");\n'
        '    if (alpha != 1.0f) {')
    old='''    if (weight_prepacked_) {
        return infinicore::op::linear_allreduce_packed(
            input_contiguous, weight, bias_opt, communicator);
    }
    return infinicore::op::linear_allreduce(
        input_contiguous, weight->contiguous(), bias_opt, communicator);'''
    new='''    auto original = weight_prepacked_
        ? infinicore::op::linear_allreduce_packed(input_contiguous, weight, bias_opt, communicator)
        : infinicore::op::linear_allreduce(input_contiguous, weight->contiguous(), bias_opt, communicator);
    h3_projection::inspect_control(input_contiguous, weight, bias_opt, communicator, weight_prepacked_, original);
    return original;'''
    rest=replace_once(rest,old,new)
    return serial+'infinicore::Tensor NoneQuantization::forward_allreduce('+rest


def patch_files(path,quant):
    data=path.read_bytes();qdata=quant.read_bytes()
    text,meta=instrument_text(data);qtext=instrument_quant(qdata)
    backup=path.with_suffix(path.suffix+'.before-projection-trace')
    qbackup=quant.with_suffix(quant.suffix+'.before-projection-trace')
    if any(p.exists() for p in [backup,qbackup]+[path.with_name(n) for n in HELPERS]):raise FileExistsError('diagnostic backup/helper exists')
    backup.write_bytes(data);qbackup.write_bytes(qdata)
    hashes={}
    for name in HELPERS:
        helper=Path(__file__).with_name(name);shutil.copyfile(helper,path.with_name(name));hashes[name]=sha(helper.read_bytes())
    path.write_text(text,encoding='utf-8',newline='\n');quant.write_text(qtext,encoding='utf-8',newline='\n')
    meta.update(backup=str(backup),quant_backup=str(qbackup),quant_source_sha256=QUANT_SHA,
                quant_patched_sha256=sha(qtext.encode()),helper_hashes=hashes)
    path.with_suffix(path.suffix+'.trace.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return meta


def builder_text(data):
    text,meta=base.builder_text(data)
    for old,new in [('1.0-isolated-h3-attention-trace','1.0-isolated-h3-projection-trace'),
        ('/diagnostics/h3-attention/','/diagnostics/h3-projection/'),
        ('infinifabric-build-h3-attention-','infinifabric-build-h3-projection-'),
        ('infinilm-attention.success','infinilm-projection.success'),
        ('"h3-attention-trace-patch"','"h3-projection-trace-patch"')]:text=replace_once(text,old,new)
    anchor='--source "${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/minimax_h3_transformer.cpp"'
    text=replace_once(text,anchor,anchor+' --quant-source "${LOCAL_PROJECT_DIR}/csrc/layers/quantization/none_quantization.cpp"')
    anchor='"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_attention_trace.hpp" '
    text=replace_once(text,anchor,anchor+'"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_projection_trace.hpp" '
                     '"${LOCAL_PROJECT_DIR}/csrc/layers/quantization/none_quantization.cpp" ')
    meta.update(builder_sha256=sha(text.encode()),destination_root='${WORK_HOME}/diagnostics/h3-projection/${H3_BOUNDARY_BUILD_ID}/InfiniLM',
                success_marker='${STATE_DIR}/infinilm-projection.success')
    return text,meta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--quant-source',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--builder',action='store_true');args=p.parse_args()
    if args.builder:
        if args.destination is None:p.error('--builder requires --destination')
        if args.destination.exists():raise FileExistsError(args.destination)
        text,meta=builder_text(args.source.read_bytes());args.destination.parent.mkdir(parents=True,exist_ok=True)
        with args.destination.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
        args.destination.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    else:
        if args.quant_source is None:p.error('patch requires --quant-source')
        meta=patch_files(args.source,args.quant_source)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
