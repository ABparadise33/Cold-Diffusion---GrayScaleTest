"""Step-grouped, reproducibly random full-scene validation previews."""
import json
from pathlib import Path
import random

from PIL import Image
import torch

from .color import denormalize_rgb, normalize_rgb
from .data import _to_tensor
from .io import save_stage_strip, save_tensor_image
from .official_colorization import channel_gray
from .tiling import TiledModel


def select_preview_images(paths, seed, step, count=5):
    """Local RNG: drawing previews must not change training shuffle/crop RNG."""
    paths = sorted((Path(path) for path in paths), key=lambda path: str(path))
    if count < 1 or not paths:
        raise ValueError('preview count and validation set must be nonempty')
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError('validation preview filenames must have unique stems')
    sampling_seed = f'official-validation-preview:{seed}:{step}'
    return random.Random(sampling_seed).sample(paths, min(count, len(paths))), sampling_seed


@torch.no_grad()
def save_full_scene_previews(trainer):
    cfg = trainer.config['training']
    count = int(cfg.get('preview_count', 5))
    selected, sampling_seed = select_preview_images(
        trainer.val_loader.dataset.items, trainer.config['seed'], trainer.step, count,
    )
    step_output = trainer.output / 'previews' / f'step_{trainer.step:06d}'
    step_output.mkdir(parents=True, exist_ok=True)
    output = step_output
    if any(step_output.iterdir()):
        # Preview precedes checkpoint saving. An interrupted run can revisit a
        # step; retain its earlier evidence without blocking checkpoint recovery.
        attempt = 1
        while True:
            output = step_output / f'retry_{attempt:03d}'
            try:
                output.mkdir()
                break
            except FileExistsError:
                attempt += 1
        print(f'existing preview retained; writing new attempt to {output}')
    manifest = {
        'step': trainer.step, 'source': 'validation split; not training or UIEB test',
        'sampling': 'without replacement within step; independent draws across steps',
        'sampling_seed': sampling_seed, 'requested_count': count, 'actual_count': len(selected),
        'selected_images': [str(path) for path in selected], 'completed_images': [],
        'sampler': trainer.bridge.sampler, 'tile_size': cfg['preview_tile_size'],
        'tile_overlap': cfg['preview_tile_overlap'], 'display_max_side': cfg['preview_max_side'],
        'standalone_prediction': 'original geometry; no resize', 'status': 'in_progress',
    }
    metadata_path = output / 'preview.json'
    metadata_path.write_text(json.dumps(manifest, indent=2))
    trainer.ema.eval()
    model = TiledModel(trainer.ema, int(cfg['preview_tile_size']), int(cfg['preview_tile_overlap']))
    # Process full scenes one at a time; do not batch five large DIV2K images.
    for selected_path in selected:
        row = _save_one_preview(trainer, model, selected_path, output, cfg)
        manifest['completed_images'].append(row)
        metadata_path.write_text(json.dumps(manifest, indent=2))
    manifest['status'] = 'complete'
    metadata_path.write_text(json.dumps(manifest, indent=2))
    print(f'validation_previews={output} images={len(selected)}')


def _save_one_preview(trainer, model, selected, output, cfg):
    with Image.open(selected) as image:
        rgb = _to_tensor(image.convert('RGB')).unsqueeze(0).to(trainer.device)
    anchor = channel_gray(normalize_rgb(rgb))
    t = torch.tensor([trainer.bridge.steps], device=trainer.device)
    direct = denormalize_rgb(model(anchor, t))
    x = anchor.clone()
    # Keep trajectories on CPU, as before, releasing all scene tensors per image.
    stages = [('s=T full gray', denormalize_rgb(x).cpu())]
    for s in range(trainer.bridge.steps, 0, -1):
        x = trainer.bridge.reverse_step(model, x, s)
        stages.append((f'update {trainer.bridge.steps-s+1}/{trainer.bridge.steps}', denormalize_rgb(x).cpu()))
    predicted = denormalize_rgb(x)
    filename = f'{selected.stem}.png'
    save_tensor_image(predicted[0], output / 'predictions' / filename)
    save_tensor_image(direct[0], output / 'direct_predictions' / filename)
    save_stage_strip([('reference (original)', rgb), ('full gray', denormalize_rgb(anchor)),
                      ('direct', direct), (trainer.bridge.sampler, predicted)],
                     output / 'samples' / filename, max_side=int(cfg['preview_max_side']))
    save_stage_strip(stages, output / 'trajectories' / filename, max_side=int(cfg['preview_max_side']))
    return {'image': str(selected), 'filename': filename, 'original_hw': list(rgb.shape[-2:])}
