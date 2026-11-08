#!/usr/bin/env python3
"""Observe the existing Native FL2VA path; never substitute model computation."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

VERSION = '1.0-native-chain-observer'
SAMPLE_LIMIT = 65536


def is_tensor(value):
    import numpy as np
    torch = sys.modules.get('torch')
    return isinstance(value, np.ndarray) or (torch is not None and torch.is_tensor(value))


def sample_array(value):
    """No RNG, no mutation; bounded host copy, identical sampling before/after steps."""
    import numpy as np
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        stride = max(1, math.ceil(flat.size / SAMPLE_LIMIT))
        return flat[::stride].astype(np.float64, copy=True)
    flat = value.detach().reshape(-1)
    stride = max(1, math.ceil(flat.numel() / SAMPLE_LIMIT))
    return flat[::stride].float().cpu().numpy().astype(np.float64, copy=True)


def tensor_summary(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        size, finite, device = value.size, bool(np.isfinite(value).all()), 'cpu'
    else:
        import torch
        size, finite, device = value.numel(), bool(torch.isfinite(value).all().item()), str(value.device)
    sampled = sample_array(value)
    valid = bool(sampled.size and np.isfinite(sampled).all())
    report = {
        'shape': list(value.shape), 'dtype': str(value.dtype), 'device': device,
        'numel': size, 'all_finite': finite, 'finite_check_scope': 'all-elements',
        'statistics_scope': 'deterministic-strided-sample', 'sample_count': sampled.size,
        'sample_min': float(sampled.min()) if valid else None,
        'sample_max': float(sampled.max()) if valid else None,
        'sample_mean': float(sampled.mean()) if valid else None,
        'sample_std': float(sampled.std()) if valid else None,
        'sample_rms': float(np.sqrt(np.mean(sampled * sampled))) if valid else None,
    }
    if size <= 64 and valid:
        report['values'] = sampled.tolist()
    return report


def summarize(value, depth=0):
    if is_tensor(value):
        return tensor_summary(value)
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth < 3 and isinstance(value, dict):
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if depth < 3 and isinstance(value, (tuple, list)):
        return [summarize(v, depth + 1) for v in value[:64]]
    return {'type': type(value).__name__}


def summarize_output(value):
    # Qwen returns many hidden-state slots, some aliases; record the consumed layer only.
    states = getattr(value, 'hidden_states', None)
    if states is not None:
        return {'hidden_state_count': len(states),
                'hidden_state_50': summarize(states[50]) if len(states) > 50 else None,
                'last_hidden_state': summarize(getattr(value, 'last_hidden_state', None))}
    posterior = value[0] if isinstance(value, (list, tuple)) and value else value
    if hasattr(posterior, 'logvar') and hasattr(posterior, 'mean'):
        return {'posterior_mean': summarize(posterior.mean), 'posterior_logvar': summarize(posterior.logvar)}
    return summarize(value)


def nonfinite_paths(value, prefix=''):
    if isinstance(value, dict):
        if value.get('all_finite') is False:
            yield prefix
        for key, item in value.items():
            yield from nonfinite_paths(item, f'{prefix}.{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from nonfinite_paths(item, f'{prefix}[{i}]')


class ChainTrace:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / 'native-chain.jsonl'
        self.originals = []
        self.counts = Counter()
        self.nonfinite = []
        self.sources = {}

    def emit(self, event):
        event = {'version': VERSION, **event}
        self.nonfinite.extend(f"{event.get('stage')}:{p}" for p in nonfinite_paths(event))
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        print(f"chain_trace stage={event.get('stage')} status={event.get('status')} call={event.get('call', '-')}", flush=True)

    def save_tensor(self, name, value):
        import numpy as np
        if not is_tensor(value):
            return
        size = value.size if isinstance(value, np.ndarray) else value.numel()
        if size > 4_000_000:
            return
        array = value if isinstance(value, np.ndarray) else value.detach().float().cpu().numpy()
        path = self.run_dir / f'{name}.npz'
        np.savez_compressed(path, tensor=array)
        self.emit({'stage': name, 'status': 'SNAPSHOT', 'path': path.name,
                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(array.shape)})

    def wrap(self, owner, name, stage, *, track_update=False):
        original = getattr(owner, name)
        signature = inspect.signature(original)
        source = inspect.getsourcefile(inspect.unwrap(original))
        if source and Path(source).is_file():
            self.sources[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()

        @functools.wraps(original)
        def observed(*args, **kwargs):
            self.counts[stage] += 1
            call = self.counts[stage]
            inputs = dict(signature.bind_partial(*args, **kwargs).arguments)
            self.emit({'stage': stage, 'status': 'START', 'call': call, 'inputs': summarize(inputs)})
            before = sample_array(inputs['sample']) if track_update else None
            if stage == 'vae.decode':
                value = next((v for v in inputs.values() if is_tensor(v)), None)
                self.save_tensor(f'decoder-input-{call:03d}', value)
            started = time.monotonic()
            try:
                output = original(*args, **kwargs)
                event = {'stage': stage, 'status': 'DONE', 'call': call,
                         'observer_inclusive_seconds': time.monotonic() - started,
                         'output': summarize_output(output)}
                if track_update:
                    import numpy as np
                    after_tensor = output[0] if isinstance(output, (tuple, list)) else output.prev_sample
                    after = sample_array(after_tensor)
                    delta = after - before
                    finite = bool(np.isfinite(delta).all())
                    event['update'] = {'scope': 'deterministic-strided-sample', 'sample_count': delta.size,
                                       'max_abs_delta': float(np.abs(delta).max()) if finite and delta.size else None,
                                       'rms_delta': float(np.sqrt(np.mean(delta * delta))) if finite and delta.size else None}
                if name == 'set_timesteps':
                    event['schedule'] = {k: summarize(getattr(owner, k, None)) for k in ('timesteps', 'sigmas')}
                self.emit(event)
                if stage == 'condition.encode':
                    self.save_tensor(f'condition-normalized-{call:03d}', output)
                return output
            except BaseException as error:
                self.emit({'stage': stage, 'status': 'FAILED', 'call': call,
                           'error': f'{type(error).__name__}: {error}'})
                raise

        self.originals.append((owner, name, original))
        setattr(owner, name, observed)

    def attach_pipeline(self, pipe, text_encoder, transformer):
        from diffusers.modular_pipelines.minimax_h3 import encoders
        audit = load_sibling('diagnose_minimax_h3_vae_roundtrip.py', 'chain_vae_audit')
        self.emit({'stage': 'pipeline.loaded', 'status': 'DONE',
                   'components': {k: f'{type(v).__module__}.{type(v).__name__}' for k, v in
                                  [('text_encoder', text_encoder), ('transformer', transformer), ('vae', pipe.vae)]},
                   'vae_audit': audit.audit_parameters(pipe.vae),
                   'vae_config': summarize(dict(pipe.vae.config)),
                   'pipeline_config': {k: summarize(getattr(pipe, k, None)) for k in
                                       ('patch_size', 'pixel_mean', 'pixel_std', 'keyframe_noise_aug',
                                        'keyframe_encode_seed', 'text_encoder_layer', 'vae_latent_channels')},
                   'scheduler_config': summarize(dict(pipe.scheduler.config))})
        self.wrap(encoders, 'encode_vae_condition', 'condition.encode')
        self.wrap(text_encoder.model, 'forward', 'qwen.forward')
        self.wrap(transformer, 'forward', 'transformer')
        self.wrap(pipe.vae, 'encode', 'vae.encode')
        self.wrap(pipe.vae, 'decode', 'vae.decode')
        for label, scheduler in [('video', pipe.scheduler), ('audio', pipe.audio_scheduler)]:
            self.wrap(scheduler, 'set_timesteps', f'schedule.{label}')
            self.wrap(scheduler, 'step', f'scheduler.{label}', track_update=True)
            self.wrap(scheduler, 'scale_noise', f'scale_noise.{label}')

    def close(self):
        while self.originals:
            owner, name, original = self.originals.pop()
            setattr(owner, name, original)

    def finish(self, exit_code):
        required = ['condition.encode', 'qwen.forward', 'transformer', 'scheduler.video', 'vae.decode']
        missing = [stage for stage in required if not self.counts[stage]]
        effective_rc = exit_code or (2 if missing else 0)
        summary = {'version': VERSION, 'exit_code': effective_rc,
                   'result': 'DIAGNOSTIC_COMPLETE' if effective_rc == 0 else 'INCOMPLETE',
                   'validation_scope': 'native-chain-observations-not-model-correctness',
                   'quality_status': 'UNASSESSED', 'performance_measurement_valid': False,
                   'counts': dict(self.counts), 'missing_stages': missing,
                   'nonfinite_observations': self.nonfinite, 'source_sha256': self.sources}
        (self.run_dir / 'native-chain-summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        return effective_rc


def load_sibling(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def invoke_observed_example(common, trace, example, argv):
    module = common.load_official_example(example)
    common.install_official_stage_tracing(module)
    original_load = module.load_pipeline

    @functools.wraps(original_load)
    def load_pipeline(*args, **kwargs):
        if kwargs.get('legacy'):
            raise RuntimeError('Native chain diagnostic refuses the legacy backend')
        pipe, text_encoder, transformer = original_load(*args, **kwargs)
        trace.attach_pipeline(pipe, text_encoder, transformer)
        return pipe, text_encoder, transformer

    module.load_pipeline = load_pipeline
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(example), *argv]
        module.main()
    finally:
        trace.close()
        module.load_pipeline = original_load
        sys.argv = previous_argv


def main():
    common = load_sibling('run_minimax_h3_fl2va_native_smoke.py', 'native_chain_common')
    args = common.parse_args()
    trace = ChainTrace(args.run_dir)
    trace.emit({'stage': 'diagnostic', 'status': 'START', 'quality_status': 'UNASSESSED',
                'performance_measurement_valid': False,
                'example_sha256': hashlib.sha256(args.example.read_bytes()).hexdigest(),
                'first_frame_sha256': hashlib.sha256(args.first_frame.read_bytes()).hexdigest()})
    common.invoke_official_example = lambda example, argv: invoke_observed_example(common, trace, example, argv)
    rc = 1
    try:
        rc = common.main()
    finally:
        trace.close()
        effective_rc = trace.finish(rc)
    return effective_rc


if __name__ == '__main__':
    raise SystemExit(main())
