#!/usr/bin/env python3
"""Add optional H3 condition-branch samples in a separate diagnostic build."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

_spec = importlib.util.spec_from_file_location(
    'condition_boundary_patch', Path(__file__).with_name('prepare_minimax_h3_boundary_trace.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
sha = base.sha
replace_once = base.replace_once
STAGES = ['norm1', 'attention_output', 'after_attention', 'norm2', 'mlp_output', 'after_mlp']
BOUNDARIES = ['condition_input', 'condition_projection'] + [
    f'condition_{i}_{stage}' for i in range(2) for stage in STAGES
] + ['condition_final'] + base.BOUNDARIES


def instrument_text(data):
    text, meta = base.instrument_text(data)
    text = replace_once(text, '#include "h3_boundary_trace.hpp"', '''#include "h3_boundary_trace.hpp"
namespace infinilm::models::minimax_h3::boundary_trace {
inline thread_local size_t condition_block = 0;
inline void save_condition(const char *stage, const infinicore::Tensor &value) {
    if (!enabled()) return;
    const std::string name = "condition_" + std::to_string(condition_block) + "_" + stage;
    save(name.c_str(), value);
}
} // namespace infinilm::models::minimax_h3::boundary_trace''')
    text = replace_once(text, '''    auto attention_output = attn_->forward(norm1_->forward(hidden_states));
    auto output = infinicore::op::add(hidden_states, attention_output);
    auto mlp_output = ff_->forward(norm2_->forward(output));
    return infinicore::op::add(output, mlp_output);''', '''    auto normalized1 = norm1_->forward(hidden_states);
    boundary_trace::save_condition("norm1", normalized1);
    auto attention_output = attn_->forward(normalized1);
    boundary_trace::save_condition("attention_output", attention_output);
    auto output = infinicore::op::add(hidden_states, attention_output);
    boundary_trace::save_condition("after_attention", output);
    auto normalized2 = norm2_->forward(output);
    boundary_trace::save_condition("norm2", normalized2);
    auto mlp_output = ff_->forward(normalized2);
    boundary_trace::save_condition("mlp_output", mlp_output);
    auto result = infinicore::op::add(output, mlp_output);
    boundary_trace::save_condition("after_mlp", result);
    return result;''')
    text = replace_once(text, '''    auto output = hidden_states;
    for (const auto &block : refiner_blocks_) {
        output = block->forward(output);
    }
    return final_norm_->forward(output);''', '''    auto output = hidden_states;
    if (boundary_trace::enabled() && refiner_blocks_.size() != 2)
        throw std::runtime_error("condition trace requires exactly two token refiner blocks");
    size_t trace_block = 0;
    for (const auto &block : refiner_blocks_) {
        boundary_trace::condition_block = trace_block++;
        output = block->forward(output);
    }
    auto result = final_norm_->forward(output);
    boundary_trace::save("condition_final", result);
    return result;''')
    text = replace_once(text, '    auto context_embeds = linear_forward(context_embedder_, text_input);',
        '    boundary_trace::save("condition_input", text_input);\n'
        '    auto context_embeds = linear_forward(context_embedder_, text_input);\n'
        '    boundary_trace::save("condition_projection", context_embeds);')
    meta.update(patched_sha256=sha(text.encode()), boundaries=BOUNDARIES,
                trace_version='1.0-isolated-h3-condition-trace', refiner_blocks=2)
    return text, meta


def patch_file(path):
    data = path.read_bytes()
    text, meta = instrument_text(data)
    backup = path.with_suffix(path.suffix + '.before-condition-trace')
    helper = Path(__file__).with_name('h3_boundary_trace.hpp')
    target = path.with_name(helper.name)
    if backup.exists() or target.exists():
        raise FileExistsError('diagnostic backup/helper already exists')
    backup.write_bytes(data)
    shutil.copyfile(helper, target)
    path.write_text(text, encoding='utf-8', newline='\n')
    meta.update(backup=str(backup), helper_sha256=sha(helper.read_bytes()))
    path.with_suffix(path.suffix + '.trace.json').write_text(
        json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


def builder_text(data):
    text, meta = base.builder_text(data)
    for old, new in [
        ('1.0-isolated-h3-boundary-trace', '1.0-isolated-h3-condition-trace'),
        ('/diagnostics/h3-boundary/', '/diagnostics/h3-condition/'),
        ('infinifabric-build-h3-boundary-', 'infinifabric-build-h3-condition-'),
        ('infinilm-boundary.success', 'infinilm-condition.success'),
        ('"h3-boundary-trace-patch"', '"h3-condition-trace-patch"'),
    ]:
        text = replace_once(text, old, new)
    meta.update(builder_sha256=sha(text.encode()),
                destination_root='${WORK_HOME}/diagnostics/h3-condition/${H3_BOUNDARY_BUILD_ID}/InfiniLM',
                success_marker='${STATE_DIR}/infinilm-condition.success')
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
