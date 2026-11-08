#!/usr/bin/env python3
"""Isolated, optional all-rank trace of the first H3 refiner Attention."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

_spec = importlib.util.spec_from_file_location(
    'attention_condition_patch', Path(__file__).with_name('prepare_minimax_h3_condition_trace.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
sha = base.sha
replace_once = base.replace_once
ATTENTION_BOUNDARIES = ['input', 'q_raw', 'k_raw', 'v_raw', 'q_norm', 'k_norm', 'mha', 'projected']
HELPERS = ['h3_boundary_trace.hpp', 'h3_attention_trace.hpp']


def instrument_text(data):
    text, meta = base.instrument_text(data)
    text = replace_once(text, '} // namespace infinilm::models::minimax_h3::boundary_trace',
        '} // namespace infinilm::models::minimax_h3::boundary_trace\n#include "h3_attention_trace.hpp"')
    text = replace_once(text, '    auto attention_output = attn_->forward(normalized1);',
        '    auto attention_output = [&]() {\n'
        '        attention_trace::Scope scope(boundary_trace::condition_block == 0);\n'
        '        return attn_->forward(normalized1);\n'
        '    }();')
    text = replace_once(text,
        '    auto hidden_states_mutable = hidden_states;\n'
        '    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);',
        '    attention_trace::save("input", hidden_states);\n'
        '    auto hidden_states_mutable = hidden_states;\n'
        '    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);\n'
        '    attention_trace::save("q_raw", query);\n'
        '    attention_trace::save("k_raw", key);\n'
        '    attention_trace::save("v_raw", value);')
    text = replace_once(text,
        '    key = key->view({batch_size, sequence_length, num_heads_, head_dim_});\n\n'
        '    if (rotary_cos_sin_cache.has_value()) {',
        '    key = key->view({batch_size, sequence_length, num_heads_, head_dim_});\n'
        '    attention_trace::save("q_norm", query);\n'
        '    attention_trace::save("k_norm", key);\n\n'
        '    if (rotary_cos_sin_cache.has_value()) {')
    text = replace_once(text,
        '    output = output->view({batch_size, sequence_length, inner_dim_});\n'
        '    return linear_forward(out_proj_, output);',
        '    output = output->view({batch_size, sequence_length, inner_dim_});\n'
        '    attention_trace::save("mha", output);\n'
        '    auto projected = linear_forward(out_proj_, output);\n'
        '    attention_trace::save("projected", projected);\n'
        '    return projected;')
    meta.update(patched_sha256=sha(text.encode()),
                trace_version='1.0-isolated-h3-attention-trace',
                attention_boundaries=ATTENTION_BOUNDARIES, all_tp_ranks=True,
                attention_scope='first-token-refiner-attention-only',
                projection_scope='original-combined-output-projection-and-reduction')
    return text, meta


def patch_file(path):
    data = path.read_bytes()
    text, meta = instrument_text(data)
    backup = path.with_suffix(path.suffix + '.before-attention-trace')
    targets = [path.with_name(name) for name in HELPERS]
    if backup.exists() or any(target.exists() for target in targets):
        raise FileExistsError('diagnostic backup/helper already exists')
    backup.write_bytes(data)
    hashes = {}
    for name, target in zip(HELPERS, targets):
        helper = Path(__file__).with_name(name)
        shutil.copyfile(helper, target)
        hashes[name] = sha(helper.read_bytes())
    path.write_text(text, encoding='utf-8', newline='\n')
    meta.update(backup=str(backup), helper_hashes=hashes)
    path.with_suffix(path.suffix + '.trace.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


def builder_text(data):
    text, meta = base.builder_text(data)
    for old, new in [
        ('1.0-isolated-h3-condition-trace', '1.0-isolated-h3-attention-trace'),
        ('/diagnostics/h3-condition/', '/diagnostics/h3-attention/'),
        ('infinifabric-build-h3-condition-', 'infinifabric-build-h3-attention-'),
        ('infinilm-condition.success', 'infinilm-attention.success'),
        ('"h3-condition-trace-patch"', '"h3-attention-trace-patch"'),
    ]:
        text = replace_once(text, old, new)
    text = replace_once(text,
        '"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_boundary_trace.hpp" ',
        '"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_boundary_trace.hpp" '
        '"${LOCAL_PROJECT_DIR}/csrc/models/minimax_h3/h3_attention_trace.hpp" ')
    meta.update(builder_sha256=sha(text.encode()),
                destination_root='${WORK_HOME}/diagnostics/h3-attention/${H3_BOUNDARY_BUILD_ID}/InfiniLM',
                success_marker='${STATE_DIR}/infinilm-attention.success')
    return text, meta


def write_builder(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    text, meta = builder_text(source.read_bytes())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    destination.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--builder', action='store_true')
    args = parser.parse_args()
    if args.builder and args.destination is None:
        parser.error('--builder requires --destination')
    print(json.dumps(write_builder(args.source, args.destination) if args.builder else patch_file(args.source), indent=2))


if __name__ == '__main__':
    main()
