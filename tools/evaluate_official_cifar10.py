"""Evaluate the official full-gray CIFAR recipe on the complete held-out split."""
import argparse
from collections import defaultdict
import contextlib
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from official_cifar_resume_support import log_preview_color

PIN = 'f8b1379151ff0cccba49112cf61d439bd4dd4ad9'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize(events):
    totals = defaultdict(lambda: defaultdict(float))
    count = 0
    for event in events:
        count += event['batch_size']
        for phase, result in event['phases'].items():
            for metric, value in result['mean'].items():
                totals[phase][metric] += value * event['batch_size']
    if not count:
        raise ValueError('No images evaluated')
    means = {p: {k: v/count for k, v in values.items()} for p, values in totals.items()}
    for values in means.values():
        values['delta_e76_gain_vs_gray'] = means['xt']['delta_e76'] - values['delta_e76']
        values['rgb_mae_gain_vs_gray'] = means['xt']['rgb_mae'] - values['rgb_mae']
    return {'count': count, 'phases': means}


def main():
    project = Path(__file__).resolve().parents[1]
    official = project/'data/official_cold_diffusion/decolor-diffusion'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-dir', type=Path, default=official)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=10000, help='10000 for full test; smaller is explicitly a subset')
    args = parser.parse_args()
    args.official_dir = args.official_dir.resolve()
    if args.batch_size < 1 or not 1 <= args.limit <= 10000:
        parser.error('batch-size must be positive; limit must be 1..10000')
    rev = subprocess.check_output(['git', '-C', str(args.official_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if rev != PIN:
        raise ValueError(f'Expected official commit {PIN}, got {rev}')
    if not torch.cuda.is_available():
        raise RuntimeError('Official sampler requires CUDA')
    checkpoint = (args.checkpoint or args.official_dir/'results/cifar10_rgb_fullgray_100k/model_100000.pt').resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    expected = {'image_size': [32, 32], 'timesteps': 20, 'loss_type': 'l1',
                'train_routine': 'Final', 'sampling_routine': 'x0_step_down',
                'to_lab': False, 'decolor_routine': 'Linear', 'decolor_total_remove': True}
    if 'signature' in state:
        for key, value in expected.items():
            if state['signature'].get(key) != value:
                raise ValueError(f'Checkpoint recipe mismatch: {key}')
    step = int(state['step'])
    output = (args.output_dir or args.official_dir/f'evaluation/cifar10_test_step{step}').resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is nonempty; choose a new --output-dir: {output}')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.official_dir))
    from diffusion import GaussianDiffusion
    from diffusion.model.get_model import get_model
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    recipe = SimpleNamespace(model='UnetConvNext', dataset='cifar10')
    model = get_model(recipe, with_time_emb=True).cuda()
    one_shot = get_model(recipe, with_time_emb=False).cuda()
    diffusion = GaussianDiffusion(
        model, image_size=(32, 32), device_of_kernel='cuda', channels=3,
        one_shot_denoise_fn=one_shot, timesteps=20, loss_type='l1',
        train_routine='Final', sampling_routine='x0_step_down',
        forward_process_type='Decolorization', decolor_routine='Linear',
        decolor_total_remove=True, batch_size=args.batch_size, to_lab=False,
    ).cuda()
    diffusion.load_state_dict(state['ema'], strict=True)
    del state
    diffusion.eval()
    dataset = datasets.CIFAR10(root=str(args.official_dir/'data'), train=False,
                              download=False, transform=transforms.Compose([
                                  transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)]))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(args.limit)),
                                         batch_size=args.batch_size, shuffle=False, num_workers=0)
    metadata = {'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint),
                'step': step, 'official_commit': rev, 'split': 'CIFAR10 test', 'count': args.limit,
                'full_test': args.limit == 10000, 'seed': 42, 'order': 'dataset index ascending',
                'weights': 'ema', 'batch_size': args.batch_size, 'recipe': expected,
                'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(),
                'sources_sha256': {str(p): file_hash(p) for p in [
                    Path(__file__), Path(__file__).with_name('official_cifar_resume_support.py'),
                    args.official_dir/'diffusion/diffusion.py']}}
    (output/'evaluation_config.json').write_text(json.dumps(metadata, indent=2))
    metric_trainer = SimpleNamespace(step=step, results_folder=output, save_and_sample_every=1000,
                                     metric_split='test' if args.limit == 10000 else 'test_subset')
    events, offset = [], 0
    with torch.no_grad(), (output/'sampler.log').open('w') as sampler_log:
        for batch_index, (images, _) in enumerate(loader):
            images = images.cuda()
            with contextlib.redirect_stdout(sampler_log):
                samples = diffusion.sample(batch_size=len(images), img=images, t=20)
            samples['og'] = images
            metric_trainer.metric_indices = list(range(offset, offset + len(images)))
            event = log_preview_color(metric_trainer, samples)
            events.append(event)
            if batch_index == 0:
                for phase, tensor in samples.items():
                    save_image(((tensor + 1)*.5).clamp(0, 1), output/f'{phase}.png', nrow=6)
            offset += len(images)
            print(f'Evaluated {offset}/{args.limit}', flush=True)
    summary = summarize(events)
    summary.update(step=step, checkpoint_sha256=metadata['checkpoint_sha256'], full_test=metadata['full_test'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    print(f'Evaluation complete: {output}')


if __name__ == '__main__':
    main()
