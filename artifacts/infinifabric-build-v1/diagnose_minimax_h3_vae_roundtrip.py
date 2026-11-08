#!/usr/bin/env python3
"""Isolate the existing video VAE. This is not a FL2VA generation or score run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

VERSION = '1.0-isolated-vae-roundtrip'


def prepare_image(path, *, width=576, height=384):
    from PIL import Image
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    # FL2VA stretches its first keyframe onto the requested canvas; only followers are cover-cropped.
    return image.resize((width, height), Image.Resampling.LANCZOS)


def reconstruction_metrics(expected, actual):
    import numpy as np
    expected, actual = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError('reconstruction shapes differ or are empty')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('nonfinite pixels in reconstruction comparison')
    delta = actual - expected
    mse = float(np.mean(delta * delta))
    return {'mae': float(np.mean(np.abs(delta))), 'mse': mse,
            'psnr_db': -10 * math.log10(mse) if mse > 0 else None, 'exact_match': mse == 0}


def audit_parameters(model):
    pins = list(getattr(model, '_keep_in_fp32_modules', []) or [])
    dtype_elements, devices, violations = Counter(), Counter(), []
    for name, parameter in model.named_parameters():
        dtype_elements[str(parameter.dtype)] += parameter.numel()
        devices[str(parameter.device)] += parameter.numel()
        if any(pin in name.split('.') for pin in pins) and str(parameter.dtype) != 'torch.float32':
            violations.append(name)
    return {'required_fp32_modules': pins, 'dtype_elements': dict(dtype_elements),
            'device_elements': dict(devices), 'fp32_violations': violations}


def tensor_stats(tensor):
    import torch
    data = tensor.detach().float()
    all_finite = bool(torch.isfinite(data).all().item())
    return {'shape': list(data.shape), 'dtype': str(tensor.dtype), 'device': str(tensor.device),
            'all_finite': all_finite,
            'min': data.min().item() if all_finite else None,
            'max': data.max().item() if all_finite else None,
            'mean': data.mean().item() if all_finite else None,
            'std': data.std(unbiased=False).item() if all_finite else None}


def load_common():
    path = Path(__file__).with_name('run_minimax_h3_fl2va_native_smoke.py')
    spec = importlib.util.spec_from_file_location('vae_probe_common', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_diagnostic(args, report, checkpoint):
    common = load_common()
    script_dir = Path(__file__).resolve().parent
    def stage(name):
        report['last_stage'] = name
        checkpoint()
        print(f'stage={name} status=START', flush=True)

    stage('input-validation')
    vae_dir = args.model_path / 'diffusers/vae'
    for path in (args.first_frame, vae_dir / 'config.json', args.model_path / 'diffusers/model_index.json'):
        if not path.is_file():
            raise FileNotFoundError(f'required input missing: {path}')
    report['first_frame_sha256'] = hashlib.sha256(args.first_frame.read_bytes()).hexdigest()
    report['vae_config'] = json.loads((vae_dir / 'config.json').read_text(encoding='utf-8'))
    report['vae_config_sha256'] = hashlib.sha256((vae_dir / 'config.json').read_bytes()).hexdigest()
    if report['vae_config'].get('_class_name') != 'AutoencoderKLMiniMaxH3':
        raise RuntimeError('unexpected VAE class in checkpoint config')

    stage('job-local-runtime-staging')
    installer = common.load_helper(script_dir / 'install_infinilm_private_runtime.py',
                                   'install_infinilm_private_runtime.py', 'vae_probe_installer')
    hygon = common.load_helper(script_dir / 'prepare_minimax_h3_hygon_runtime.py',
                              'prepare_minimax_h3_hygon_runtime.py', 'vae_probe_hygon')
    manifest = common.read_json(args.work_home / 'state/infinilm-runtime-private.json')
    if manifest.get('runtime_mode') != 'job-local-wheel-staging' or manifest.get('expected_wheels') != installer.EXPECTED_WHEELS:
        raise RuntimeError('runtime state and installer wheel sets differ')
    with common.StageHeartbeat('job-local-runtime-staging'):
        wheels = installer.verify_wheels(Path(manifest['wheel_dir']))
        installer.install_wheels_direct(wheels, args.local_target)
        report['private_distributions'] = installer.check_target_contents(args.local_target)
        sys.path.insert(0, str(args.local_target))
        os.environ['PYTHONPATH'] = str(args.local_target) + os.pathsep + os.environ.get('PYTHONPATH', '')
        report['hygon_runtime'] = hygon.stage_runtime(
            args.work_home / 'state/minimax-h3-hygon-runtime.json', installer,
            args.local_target.parent / 'hygon-site-packages', args.local_target.parent / 'hygon-reconstructed')

    import numpy as np
    import torch
    import diffusers
    from diffusers import AutoencoderKLMiniMaxH3
    from PIL import Image, ImageDraw
    if not torch.cuda.is_available():
        raise RuntimeError('torch.cuda is unavailable')
    device = torch.device('cuda:0')
    report.update(torch_version=torch.__version__, diffusers_version=diffusers.__version__,
                  device_name=torch.cuda.get_device_name(0), visible_devices=torch.cuda.device_count())
    implementation = Path(inspect.getfile(AutoencoderKLMiniMaxH3))
    report['vae_implementation_sha256'] = hashlib.sha256(implementation.read_bytes()).hexdigest()
    stage('vae-model-load')
    with common.StageHeartbeat('vae-model-load'):
        vae, info = AutoencoderKLMiniMaxH3.from_pretrained(
            str(vae_dir), torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
    report['loading_info'] = info
    checkpoint()
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError('VAE checkpoint loading reported key mismatches; inspect loading_info')
    vae.to(device=device)
    vae.eval().requires_grad_(False)
    report['parameter_audit'] = audit_parameters(vae)
    report['use_tiling'] = bool(vae.use_tiling)
    report['loaded_vae_config'] = dict(vae.config)
    checkpoint()
    if report['parameter_audit']['fp32_violations']:
        raise RuntimeError('VAE modules required in float32 have another dtype')

    stage('prepare-reconstruction-input')
    source = prepare_image(args.first_frame)
    source.save(args.run_dir / 'input-preprocessed.png')
    model_index = json.loads((args.model_path / 'diffusers/model_index.json').read_text(encoding='utf-8'))
    pixel_mean = model_index.get('pixel_mean', [0.485, 0.456, 0.406])
    pixel_std = model_index.get('pixel_std', [0.229, 0.224, 0.225])
    if len(pixel_mean) != 3 or len(pixel_std) != 3 or any(x <= 0 for x in pixel_std):
        raise ValueError('invalid RGB normalization in model index')
    # 22 repeated image frames exercise one complete temporal reconstruction.
    # This diagnostic is separate from, and never substituted for, the 124-frame generation workload.
    frame_count = 22
    report['diagnostic_input'] = {'frames': frame_count, 'width': 576, 'height': 384,
                                'pixel_mean': pixel_mean, 'pixel_std': pixel_std,
                                'posterior_selection': 'mode', 'source': 'repeated supplied image'}
    reference = np.asarray(source, dtype=np.float32) / 255
    pixels = torch.from_numpy(reference.copy()).permute(2, 0, 1)[None, :, None].to(device)
    mean = torch.tensor(pixel_mean, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(pixel_std, device=device).view(1, 3, 1, 1, 1)
    pixels = ((pixels - mean) / std).repeat(1, 1, frame_count, 1, 1)
    report['encoder_input'] = tensor_stats(pixels)
    with torch.inference_mode():
        stage('vae-encode')
        with common.StageHeartbeat('vae-encode'):
            posterior = vae.encode(pixels, return_dict=False)[0]
            latent = posterior.mode()
        report['posterior_mean'] = tensor_stats(posterior.mean)
        report['posterior_logvar'] = tensor_stats(posterior.logvar)
        report['decoder_input'] = tensor_stats(latent)
        checkpoint()
        if not all(report[key]['all_finite'] for key in ('posterior_mean', 'posterior_logvar', 'decoder_input')):
            raise RuntimeError('nonfinite VAE posterior')
        stage('vae-decode')
        with common.StageHeartbeat('vae-decode'):
            # Raw posterior mode goes directly to decode: no diffusion latent scaling here.
            decoded = vae.decode(latent, return_dict=False)[0]
        report['decoder_output'] = tensor_stats(decoded)
        checkpoint()
        if not report['decoder_output']['all_finite']:
            raise RuntimeError('nonfinite VAE decoder output')
        if list(decoded.shape) != [1, 3, frame_count, 384, 576]:
            raise RuntimeError(f'unexpected reconstructed shape: {list(decoded.shape)}')
        unclamped = decoded.float() * std + mean
        report['outside_rgb_range_fraction'] = ((unclamped < 0) | (unclamped > 1)).float().mean().item()
        reconstructed = unclamped.clamp(0, 1)[0].permute(1, 2, 3, 0).cpu().numpy()

    stage('save-evidence')
    selected = [0, 1, 4, 10, 21]
    report['reconstruction_frames'] = len(reconstructed)
    report['per_frame_metrics'] = {str(i): reconstruction_metrics(reference, reconstructed[i]) for i in range(frame_count)}
    canvas = Image.new('RGB', (3 * 288, 2 * 216), 'white')
    draw = ImageDraw.Draw(canvas)
    images = [('input', source)]
    for i in selected:
        image = Image.fromarray(np.rint(reconstructed[i] * 255).astype(np.uint8))
        image.save(args.run_dir / f'reconstruction-{i:03}.png')
        images.append((f'reconstruction frame {i}', image))
    for i, (label, image) in enumerate(images):
        left, top = i % 3 * 288, i // 3 * 216
        draw.text((left + 6, top + 5), label, fill='black')
        canvas.paste(image.resize((288, 192)), (left, top + 24))
    canvas.save(args.run_dir / 'vae-roundtrip-contact-sheet.jpg', quality=94)
    print('EVIDENCE: vae-roundtrip-contact-sheet.jpg and vae-roundtrip.json', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'local-target', 'model-path', 'first-frame', 'work-home'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    report = {'diagnostic_version': VERSION, 'result': 'FAILED', 'quality_status': 'UNASSESSED',
              'validation_scope': 'isolated-vae-roundtrip-not-generation', 'last_stage': 'initialization'}
    started = time.monotonic()
    def checkpoint():
        report['elapsed_seconds'] = time.monotonic() - started
        (args.run_dir / 'vae-roundtrip.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    try:
        print(f'diagnostic_version={VERSION}', flush=True)
        run_diagnostic(args, report, checkpoint)
        report.update(result='DIAGNOSTIC_COMPLETE', last_stage='complete')
        print('VAE_ROUNDTRIP DIAGNOSTIC_COMPLETE; quality_status=UNASSESSED', flush=True)
        return 0
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        return 1
    finally:
        checkpoint()
        print(f'RESULT_JSON: {args.run_dir / "vae-roundtrip.json"}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
