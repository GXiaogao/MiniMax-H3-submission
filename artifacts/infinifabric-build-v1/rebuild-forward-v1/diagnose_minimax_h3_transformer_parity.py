#!/usr/bin/env python3
"""Compare one real Native H3 forward with an isolated Diffusers reference.

This is a component diagnostic, never a completed generation or benchmark.
Native Qwen output is held fixed in both forwards; Qwen parity is not tested.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

VERSION = '1.0-first-h3-forward-parity'
INPUT_NAMES = ('hidden_states', 'audio_hidden_states', 'encoder_hidden_states',
               'timestep', 'timestep_indices', 'token_tags', 'position_ids',
               'video_indices', 'audio_indices', 'text_indices')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(directory, name, values):
    import numpy as np
    arrays, fields = {}, {}
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            array, dtype = value.copy(), str(value.dtype)
        elif hasattr(value, 'detach'):
            tensor = value.detach().cpu()
            dtype = str(tensor.dtype).removeprefix('torch.')
            # NumPy has no standard bfloat16. Float32 represents every BF16 value exactly.
            array = (tensor.float() if dtype == 'bfloat16' else tensor).numpy().copy()
        else:
            raise TypeError(f'expected tensor for {key}: {type(value).__name__}')
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError(f'unsupported or nonfinite tensor: {key}')
        arrays[key] = array
        fields[key] = {'torch_dtype': dtype, 'storage_dtype': str(array.dtype), 'shape': list(array.shape)}
    path = Path(directory)/f'{name}.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    metadata = {'sha256': sha256(path), 'fields': fields}
    write_json(path.with_suffix('.json'), metadata)
    return metadata


def read_bundle(directory, name):
    import numpy as np
    path = Path(directory)/f'{name}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256(path) != metadata['sha256']:
        raise ValueError(f'SHA256 mismatch: {path.name}')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(metadata['fields']):
            raise ValueError('bundle fields differ from manifest')
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, array in arrays.items():
        info = metadata['fields'][key]
        if list(array.shape) != info['shape'] or str(array.dtype) != info['storage_dtype']:
            raise ValueError(f'bundle shape/dtype mismatch: {key}')
        if not np.isfinite(array).all():
            raise ValueError(f'nonfinite bundle tensor: {key}')
    return arrays, metadata


def compare_arrays(reference, native):
    import numpy as np
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(native, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError('comparison shape mismatch or empty tensor')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite comparison tensor')
    delta = b-a
    norm_a, norm_b = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    return {'shape': list(a.shape), 'elements': a.size, 'scope': 'all-elements',
            'mae': float(np.abs(delta).mean()), 'max_abs': float(np.abs(delta).max()),
            'rmse': float(np.sqrt(np.mean(delta*delta))),
            'relative_l2': float(np.linalg.norm(delta.ravel())/norm_a) if norm_a else None,
            'cosine': float(np.dot(a.ravel(), b.ravel())/(norm_a*norm_b)) if norm_a and norm_b else None,
            'reference_std': float(a.std()), 'native_std': float(b.std()),
            'exact_equal': bool(np.array_equal(a,b))}


class CaptureComplete(BaseException):
    """Intentional component-only stop; cannot be swallowed by ordinary error handlers."""


class FirstForwardCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.original, self.owner = None, None
        self.complete = False

    def attach(self, owner):
        self.owner, self.original = owner, owner.forward
        original = self.original
        signature = inspect.signature(original)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            tensors = {k:v for k,v in bound.items() if k in INPUT_NAMES}
            extras = {k:v for k,v in bound.items() if k not in INPUT_NAMES}
            if any(k not in ('return_dict','attention_kwargs') or v not in (False, None, {}) for k,v in extras.items()):
                raise ValueError('unsupported forward options: cannot replay faithfully')
            print('stage=native-first-forward status=START', flush=True)
            save_bundle(self.directory, 'first-inputs', tensors)
            output = original(*args, **kwargs)
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise ValueError('expected Native video/audio output tuple')
            save_bundle(self.directory, 'first-native-outputs', dict(zip(('video','audio'),output)))
            self.complete = True
            print('stage=native-first-forward status=CAPTURED; full_generation=false', flush=True)
            raise CaptureComplete()
        owner.forward = observed

    def close(self):
        if self.owner is not None:
            self.owner.forward = self.original


def validate_device_map(mapping, devices):
    if not mapping:
        raise ValueError('reference model has no device map')
    for name, value in mapping.items():
        text = str(value)
        ordinal = int(text) if text.isdigit() else int(text[5:]) if text.startswith('cuda:') and text[5:].isdigit() else -1
        if ordinal not in devices:
            raise ValueError(f'reference offload/unknown device prohibited: {name} -> {value}')


def run_worker(command, log_path, environment):
    # Separate processes ensure native engines/communicators are gone before reference loading.
    env = {**os.environ, **environment}
    with Path(log_path).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace', env=env)
        try:
            for line in proc.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
            rc = proc.wait()
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
        if rc:
            raise RuntimeError(f'worker failed with exit code {rc}; log={log_path}')


def load_sibling(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_capture(args):
    import torch
    import infinicore
    from infinicore.lib import _infinicore
    common = load_sibling('run_minimax_h3_fl2va_native_smoke.py','parity_native_common')
    devices = list(range(8))
    common.validate_device_counts(devices, torch_count=torch.cuda.device_count(),
        native_count=_infinicore.get_device_count(_infinicore.Device.Type.HYGON))
    staging = common.configure_weight_staging(infinicore, devices,
        _infinicore.get_device_count(_infinicore.Device.Type.CPU))
    module = common.load_official_example(args.example)
    capture = FirstForwardCapture(args.run_dir)
    original_load = module.load_pipeline

    @functools.wraps(original_load)
    def load(*a, **kw):
        if kw.get('legacy'): raise ValueError('capture requires Native backend')
        pipe, text, transformer = original_load(*a, **kw)
        capture.attach(transformer)
        write_json(args.run_dir/'native-capture-model.json', {
            'weight_staging': staging, 'transformer_class':type(transformer).__name__,
            'adapter_sha256':sha256(inspect.getfile(type(transformer))),
            'qwen_class':type(text).__name__, 'torch_version':torch.__version__})
        return pipe, text, transformer
    module.load_pipeline = load
    config = SimpleNamespace(model_path=args.model_path, first_frame=args.first_frame,
        output=args.run_dir/'NOT_A_GENERATION.mp4',text='A dog falls asleep',
        gpus='0,1,2,3,4,5,6,7',num_frames=124,steps=10,seed=42,height=384,width=576)
    previous = sys.argv[:]
    try:
        sys.argv = [str(args.example), *common.build_official_argv(config)]
        module.main()
    except CaptureComplete:
        if not capture.complete: raise RuntimeError('capture stopped without outputs')
    else:
        raise RuntimeError('capture hook did not stop after first forward')
    finally:
        capture.close(); module.load_pipeline = original_load; sys.argv = previous
    arrays, _ = read_bundle(args.run_dir,'first-inputs')
    if set(arrays) != set(INPUT_NAMES):
        raise ValueError(f'incomplete captured inputs: {sorted(set(INPUT_NAMES)-set(arrays))}')


def reference_forward(args):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    common = load_sibling('run_minimax_h3_fl2va_native_smoke.py','parity_reference_common')
    module = common.load_official_example(args.example)
    devices = list(range(8))
    if torch.cuda.device_count() < 8: raise RuntimeError('eight GPUs required')
    arrays, metadata = read_bundle(args.run_dir,'first-inputs')
    if set(arrays) != set(INPUT_NAMES): raise ValueError('incomplete H3 input bundle')
    print('stage=reference-model-load status=START; diagnostic_only=true',flush=True)
    model, loading = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.model_path/'diffusers'), subfolder='transformer', dtype=torch.bfloat16,
        device_map='balanced', max_memory=module.make_max_memory(devices,0.72),
        low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True)
    evidence = {'loading_info':loading,'implementation_sha256':sha256(inspect.getfile(type(model))),
        'device_map':{k:str(v) for k,v in model.hf_device_map.items()},'config':dict(model.config)}
    write_json(args.run_dir/'reference-model.json',evidence)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('reference weight loading mismatches; inspect reference-model.json')
    validate_device_map(model.hf_device_map,devices)
    pins = tuple(model._keep_in_fp32_modules)
    violations = [name for name,p in model.named_parameters()
                  if any(part in pins for part in name.split('.')) and p.dtype != torch.float32]
    evidence['fp32_violations'] = violations
    write_json(args.run_dir/'reference-model.json',evidence)
    if violations: raise RuntimeError('reference mixed precision modules were downcast')
    model.eval().requires_grad_(False)
    module.install_legacy_output_device_hooks(model)
    device = module.module_execution_device(model.context_embedder)
    inputs = {key:torch.from_numpy(value).to(device=device,dtype=getattr(torch,metadata['fields'][key]['torch_dtype']))
              for key,value in arrays.items()}
    print('stage=reference-first-forward status=START; inputs=exact-native-capture',flush=True)
    with torch.inference_mode():
        output = model(**inputs,return_dict=False)
    save_bundle(args.run_dir,'first-reference-outputs',dict(zip(('video','audio'),output)))
    print('stage=reference-first-forward status=DONE',flush=True)


def orchestrate(args, report, checkpoint):
    common = load_sibling('run_minimax_h3_fl2va_native_smoke.py','parity_parent_common')
    installer = load_sibling('install_infinilm_private_runtime.py','parity_installer')
    hygon = load_sibling('prepare_minimax_h3_hygon_runtime.py','parity_hygon')
    patcher = load_sibling('prepare_minimax_h3_vae_dtype_probe.py','parity_patcher')
    source = args.work_home/'Projects/InfiniLM/examples/minimax_h3.py'
    required = [source,args.first_frame,args.model_path/'diffusers/transformer/config.json',
                args.work_home/'state/infinilm-extension.success']
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    marker = dict(line.split('=',1) for line in required[-1].read_text().splitlines() if '=' in line)
    if marker.get('build_version') != '1.5-hygon-fl2va-no-cpu-runtime':
        raise RuntimeError('incompatible native extension marker')
    report['extension_sha256'] = sha256(marker['extension'])
    report['first_frame_sha256'] = sha256(args.first_frame)
    report['transformer_config_sha256'] = sha256(required[2])
    args.local_target.parent.mkdir(parents=True,exist_ok=True)
    patched = args.local_target.parent/'example/minimax_h3.py'
    report['example_patch'] = patcher.patch_example(source,patched)
    report['last_stage'] = 'runtime-staging'; checkpoint()
    manifest = common.read_json(args.work_home/'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode') != 'job-local-wheel-staging' or manifest.get('expected_wheels') != installer.EXPECTED_WHEELS:
        raise RuntimeError('runtime manifest/helper mismatch')
    with common.StageHeartbeat('runtime-staging'):
        wheels = installer.verify_wheels(Path(manifest['wheel_dir']))
        installer.install_wheels_direct(wheels,args.local_target)
        report['private_distributions'] = installer.check_target_contents(args.local_target)
        os.environ['PYTHONPATH'] = str(args.local_target)+os.pathsep+os.environ.get('PYTHONPATH','')
        sys.path.insert(0,str(args.local_target))
        report['hygon_runtime'] = hygon.stage_runtime(args.work_home/'state/minimax-h3-hygon-runtime.json',
            installer,args.local_target.parent/'hygon-site-packages',args.local_target.parent/'hygon-reconstructed')
    for worker in ('native','reference'):
        report['last_stage'] = worker; checkpoint()
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',worker,
                   '--run-dir',str(args.run_dir),'--example',str(patched),
                   '--model-path',str(args.model_path),'--first-frame',str(args.first_frame)]
        with common.StageHeartbeat(worker):
            run_worker(command,args.run_dir/f'{worker}.log',{})
        report['completed_workers'].append(worker); checkpoint()
    report['last_stage']='comparison'; checkpoint()
    native, _ = read_bundle(args.run_dir,'first-native-outputs')
    reference, _ = read_bundle(args.run_dir,'first-reference-outputs')
    if set(native)!=set(reference) or set(native)!={'video','audio'}:
        raise ValueError('comparison outputs are incomplete')
    report['comparison'] = {key:compare_arrays(reference[key],native[key]) for key in ('video','audio')}
    report['result']='COMPARISON_COMPLETE'; report['last_stage']='complete'; checkpoint()
    print(json.dumps(report['comparison'],indent=2),flush=True)
    print('PARITY_DIAGNOSTIC COMPLETE; quality_status=UNASSESSED; Qwen_parity=NOT_TESTED',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=('native','reference'))
    for name in ('run-dir','local-target','work-home','model-path','first-frame','example'):
        parser.add_argument('--'+name,type=Path,required=name in ('run-dir','model-path','first-frame'))
    args=parser.parse_args(argv)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:
        if args.example is None: parser.error('--example is required for workers')
        (native_capture if args.worker=='native' else reference_forward)(args)
        return 0
    if args.work_home is None or args.local_target is None: parser.error('--work-home and --local-target are required')
    report={'version':VERSION,'result':'FAILED','quality_status':'UNASSESSED',
            'validation_scope':'single-H3-forward-not-generation-not-Qwen-parity',
            'performance_measurement_valid':False,'last_stage':'preflight',
            'workload_source':{'gpus':8,'num_frames':124,'steps':10,'seed':42,'width':576,'height':384},
            'requested_transformer_forwards':{'native':1,'reference':1}, 'completed_workers':[]}
    start=time.monotonic()
    def checkpoint():
        report['elapsed_seconds']=time.monotonic()-start
        write_json(args.run_dir/'transformer-parity.json',report)
    try:
        orchestrate(args,report,checkpoint)
        return 0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        traceback.print_exc()
        return 1
    finally:
        checkpoint()
        print(f'RESULT_JSON: {args.run_dir / "transformer-parity.json"}',flush=True)


if __name__=='__main__':
    raise SystemExit(main())
