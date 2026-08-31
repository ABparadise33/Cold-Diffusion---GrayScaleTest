from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .color import denormalize_rgb, normalized_lab_to_rgb


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float):
    with torch.no_grad():
        model_params = dict(model.named_parameters())
        for name, param in ema.named_parameters():
            param.mul_(decay).add_(model_params[name], alpha=1.0 - decay)
        model_buffers = dict(model.named_buffers())
        for name, buffer in ema.named_buffers():
            buffer.copy_(model_buffers[name])


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def save_tensor_image(image: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(image).save(path)


def save_stage_strip(
    stages: list[tuple[str, torch.Tensor]],
    path: Path,
    image_index: int = 0,
    display_scale: int = 1,
):
    """Save one sample's stages from left to right without reducing resolution."""
    if display_scale < 1:
        raise ValueError("display_scale must be >= 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for label, images in stages:
        image = images[image_index] if images.ndim == 4 else images
        tile = _tensor_to_pil(image)
        if display_scale > 1:
            tile = tile.resize(
                (tile.width * display_scale, tile.height * display_scale),
                Image.Resampling.BICUBIC,
            )
        tiles.append((label, tile))

    font_size = max(16, 5 * display_scale)
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:
        font = ImageFont.load_default()
    label_h = font_size + 12
    canvas = Image.new(
        "RGB",
        (sum(tile.width for _, tile in tiles), max(tile.height for _, tile in tiles) + label_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, tile in tiles:
        draw.text((x + 6, 5), label, fill="black", font=font)
        canvas.paste(tile, (x, label_h))
        x += tile.width
    canvas.save(path)


def save_labeled_grid(rows: list[tuple[str, torch.Tensor]], path: Path, max_images: int = 4):
    """Save rows of BCHW RGB tensors with compact labels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(label, images[:max_images]) for label, images in rows]
    tile_h, tile_w = rows[0][1].shape[-2:]
    label_h = 22
    canvas = Image.new("RGB", (tile_w * max_images, (tile_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, images) in enumerate(rows):
        y = row_index * (tile_h + label_h)
        draw.text((4, y + 3), label, fill="black")
        for col, image in enumerate(images):
            canvas.paste(_tensor_to_pil(image), (col * tile_w, y + label_h))
    canvas.save(path)


def save_trajectory_grid(
    trajectory: list[torch.Tensor],
    path: Path,
    image_index: int = 0,
    display_scale: int = 1,
    color_space: str = "lab",
):
    if color_space not in {"lab", "rgb"}:
        raise ValueError(f"unsupported trajectory color space: {color_space}")
    to_rgb = normalized_lab_to_rgb if color_space == "lab" else denormalize_rgb
    stages = []
    for index, state in enumerate(trajectory):
        stages.append((f"reverse {index}/{len(trajectory)-1}", to_rgb(state)))
    save_stage_strip(stages, path, image_index=image_index, display_scale=display_scale)


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
