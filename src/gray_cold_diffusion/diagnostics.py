"""Intermediate-start diagnostics; the normal training/sampler path is unchanged."""
from __future__ import annotations

import torch
from torch import nn

from .bridge import GrayBridge


@torch.no_grad()
def sample_from_step(
    model: nn.Module,
    bridge: GrayBridge,
    initial_state: torch.Tensor,
    anchor: torch.Tensor,
    start_step: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Algorithm 2 from a known D_t image, preserving the original time labels.

    The anchor is still gray(source), not the partial-color initial image.
    This adds a diagnostic entry point without modifying GrayBridge.sample.
    """
    if not 1 <= start_step <= bridge.steps:
        raise ValueError(f"start_step must be in [1, {bridge.steps}]")
    if initial_state.shape != anchor.shape:
        raise ValueError("initial_state and anchor must have identical geometry")
    x = initial_state.clone()
    trajectory = [x.clone()]
    for step in range(start_step, 0, -1):
        t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
        prediction = model(x, t)
        x = (
            x - bridge.degrade(prediction, anchor, t)
            + bridge.degrade(prediction, anchor, t - 1)
        ).clamp(-1.0, 1.0)
        trajectory.append(x.clone())
    return x, trajectory


def check_diagnostic_checkpoint(
    config: dict, baseline: dict, step: int, expected_step: int, *, color_space: str = "lab",
    expected_saturation_factor: float = 1.0,
):
    """Refuse a different experiment, representation, recipe, or checkpoint age."""
    mismatches = []
    for key in ("experiment", "mode", "seed", "model", "diffusion"):
        if config.get(key) != baseline.get(key):
            mismatches.append(key)
    data = config.get("data", {})
    expected_data = baseline.get("data", {})
    if data.get("image_size") != expected_data.get("image_size"):
        mismatches.append("data.image_size")
    if expected_saturation_factor not in (1.0, 1.25, 1.5, 2.0):
        raise ValueError("unsupported diagnostic saturation factor")
    if color_space != "rgb" and expected_saturation_factor != 1.0:
        raise ValueError("higher-saturation diagnostics currently require RGB")
    for key in ("saturation_factor", "reference_saturation_factor"):
        expected_factor = expected_saturation_factor if key == "saturation_factor" else 1.0
        if (float(data.get(key, 1.0)) != expected_factor
                or float(expected_data.get(key, 1.0)) != expected_factor):
            mismatches.append(f"data.{key}")
    training = config.get("training", {})
    expected_training = baseline.get("training", {})
    for key in ("learning_rate", "ema_decay", "ema_update_every", "amp", "max_steps"):
        if training.get(key) != expected_training.get(key):
            mismatches.append(f"training.{key}")
    batch = training.get("batch_size", 0) * training.get("grad_accum", 0)
    expected_batch = expected_training.get("batch_size", 0) * expected_training.get("grad_accum", 0)
    if batch != expected_batch:
        mismatches.append("training.effective_batch_size")
    if step != expected_step:
        mismatches.append(f"checkpoint_step={step}, expected={expected_step}")
    allowed_modes = {
        "lab": {"cold_gray", "natural_lab_colorization"},
        "rgb": {"natural_rgb_colorization"},
    }
    if color_space not in allowed_modes:
        raise ValueError("color_space must be lab or rgb")
    if config.get("mode") not in allowed_modes[color_space]:
        mismatches.append(f"requires a {color_space.upper()} model")
    if mismatches:
        raise ValueError(f"checkpoint does not match the selected DIV2K {color_space.upper()} control: " + "; ".join(mismatches))
