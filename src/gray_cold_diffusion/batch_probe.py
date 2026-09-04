"""Isolated memory-fit probes. No learned weights or real checkpoints are saved."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from .color import denormalize_rgb, rgb_to_normalized_lab
from .factory import OFFICIAL_MODE, build_model_and_bridge
from .io import update_ema
from .metrics import delta_e76, psnr, ssim, trajectory_monotonic_fraction

OOM_EXIT = 42
RESULT_MARKER = 'BATCH_PROBE_RESULT='


def candidate_batches(effective_batch):
    if effective_batch < 1:
        raise ValueError('effective batch must be positive')
    return [b for b in range(effective_batch, 0, -1) if effective_batch % b == 0]


def classify_result(returncode, result, reserve_gb=2.0, reserve_fraction=0.1):
    if returncode == OOM_EXIT and result.get('status') == 'cuda_oom':
        return 'cuda_oom'
    if returncode != 0 or result.get('status') != 'ok':
        raise RuntimeError(f'probe failed with non-OOM error (exit {returncode}): {result}')
    if not (reserve_gb >= 0 and 0 <= reserve_fraction < 1):
        raise ValueError('invalid headroom requirement')
    headroom = max(reserve_gb * 1024**3, result['total_bytes'] * reserve_fraction)
    # initial_free includes other GPU users; reserved is the peak of this fresh
    # process's caching allocator, not just currently-live tensors.
    fits_peak = result['initial_free_bytes'] - result['peak_reserved_bytes'] >= headroom
    fits_now = result['final_free_bytes'] >= headroom
    return 'pass' if fits_peak and fits_now else 'insufficient_headroom'


def run_child(config_path, batch_size, effective_batch, attempt_dir, timeout=300):
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = attempt_dir / 'result.json'
    command = [sys.executable, '-u', '-m', 'gray_cold_diffusion.batch_probe',
               '--config', str(config_path), '--batch-size', str(batch_size),
               '--effective-batch', str(effective_batch), '--result', str(result_path)]
    with (attempt_dir / 'worker.log').open('w') as log:
        # subprocess.run kills and waits for this worker on timeout. An OOM
        # process also exits completely before another CUDA context is started.
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    return completed.returncode, result


def select_batch(config_path, effective_batch, report_dir, runner=run_child):
    report_dir = Path(report_dir)
    report = {'effective_batch': effective_batch, 'attempts': [],
              'reserve_gib': 2, 'reserve_fraction': 0.1,
              'scope': 'largest passing divisor; not absolute max batch or fastest batch',
              'selected': None, 'status': 'probing'}

    def persist():
        (report_dir / 'batch_probe.json').write_text(json.dumps(report, indent=2))

    persist()
    for batch in candidate_batches(effective_batch):
        attempt = {'batch_size': batch, 'grad_accum': effective_batch // batch}
        report['attempts'].append(attempt)
        print(f'PROBE batch={batch}, accumulation={effective_batch // batch} (effective={effective_batch})', flush=True)
        try:
            code, result = runner(config_path, batch, effective_batch, report_dir / f'batch_{batch}')
            attempt.update(result=result, returncode=code)
            status = classify_result(code, result)
            attempt['status'] = status
        except (Exception, KeyboardInterrupt) as exc:
            attempt.update(status='fatal', error=f'{type(exc).__name__}: {exc}')
            report['status'] = 'stopped_on_error'
            persist()
            raise
        print(f'  {status}', flush=True)
        if status == 'pass':
            report['selected'] = {'batch_size': batch, 'grad_accum': effective_batch // batch}
            report['status'] = 'selected'
            persist()
            return report
        persist()
    report['status'] = 'no_safe_batch'
    persist()
    raise RuntimeError('No batch passed with required memory headroom; training was NOT started')


def probe_workload(config, batch_size, effective_batch, device, updates=3):
    """Match the tensor workload, with random data; CPU support is test-only."""
    device = torch.device(device)
    if config['mode'] != OFFICIAL_MODE or config['training'].get('amp', False):
        raise ValueError('probe requires the FP32 official RGB baseline')
    if batch_size < 1 or effective_batch % batch_size or updates < 2:
        raise ValueError('invalid batch/accumulation or insufficient warmup steps')
    torch.manual_seed(int(config['seed']))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.init()
        initial_free, total = torch.cuda.mem_get_info(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        initial_free = total = 0
    model, bridge = build_model_and_bridge(config)
    model, bridge = model.to(device), bridge.to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    size = int(config['data']['image_size'])
    accum = effective_batch // batch_size
    times = []
    for _ in range(updates):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            target = torch.rand(batch_size, 3, size, size, device=device) * 2 - 1
            t = torch.randint(1, bridge.steps + 1, (batch_size,), device=device)
            prediction = model(bridge.degrade(target, None, t), t)
            loss = (prediction - target).abs().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('non-finite probe loss (not an OOM)')
            (loss / accum).backward()
        optimizer.step()  # allocates the real Adam moment buffers
        optimizer.zero_grad(set_to_none=True)
        update_ema(ema, model, float(config['training']['ema_decay']))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    # Keep optimizer/EMA resident and use the real crop-validation metric path,
    # including all trajectory states and converted Lab states.
    with torch.no_grad():
        anchor = target.mean(1, keepdim=True).expand_as(target)
        t = torch.full((batch_size,), bridge.steps, device=device, dtype=torch.long)
        direct = denormalize_rgb(ema(anchor, t))
        state, trajectory = bridge.sample(ema, anchor, True)
        labs = [rgb_to_normalized_lab(denormalize_rgb(x)) for x in trajectory]
        reference = denormalize_rgb(target)
        target_lab = rgb_to_normalized_lab(reference)
        values = [psnr(denormalize_rgb(state), reference),
                  ssim(denormalize_rgb(state), reference),
                  delta_e76(rgb_to_normalized_lab(direct), target_lab),
                  trajectory_monotonic_fraction(labs, target_lab)]
        if not all(bool(torch.isfinite(x).all()) for x in values):
            raise RuntimeError('non-finite validation probe (not an OOM)')
    result = {'status': 'ok', 'batch_size': batch_size, 'grad_accum': accum,
              'effective_batch': effective_batch, 'updates': updates,
              'seconds_per_update_after_warmup': sum(times[1:]) / len(times[1:]),
              'parameters': sum(p.numel() for p in model.parameters()),
              'initial_free_bytes': initial_free, 'total_bytes': total,
              'device': str(device), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'checks': ['training_forward_backward', 'adam_states', 'ema', 'full_gray_crop_validation'],
              'limits': 'synthetic inputs; no DataLoader IO or full-scene preview; not a stability guarantee'}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        result.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      final_free_bytes=torch.cuda.mem_get_info(device)[0],
                      gpu=torch.cuda.get_device_name(device))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--effective-batch', required=True, type=int)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; batch probing requires the actual training GPU')
        result = probe_workload(config, args.batch_size, args.effective_batch, 'cuda')
    except torch.cuda.OutOfMemoryError as exc:
        result = {'status': 'cuda_oom', 'error': str(exc), 'batch_size': args.batch_size}
        Path(args.result).write_text(json.dumps(result, indent=2))
        print(RESULT_MARKER + json.dumps(result), flush=True)
        return OOM_EXIT
    Path(args.result).write_text(json.dumps(result, indent=2))
    print(RESULT_MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
