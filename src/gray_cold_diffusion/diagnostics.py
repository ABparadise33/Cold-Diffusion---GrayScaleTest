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


def check_diagnostic_checkpoint(config: dict, baseline: dict, step: int, expected_step: int):
    """Refuse a different experiment, representation, recipe, or checkpoint age."""
    mismatches = []
    for key in ("experiment", "mode", "seed", "model", "diffusion"):
        if config.get(key) != baseline.get(key):
            mismatches.append(key)
    data = config.get("data", {})
    expected_data = baseline.get("data", {})
    if data.get("image_size") != expected_data.get("image_size"):
        mismatches.append("data.image_size")
    for key in ("saturation_factor", "reference_saturation_factor"):
        if float(data.get(key, 1.0)) != 1.0:
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
    if config.get("mode") != "cold_gray":
        mismatches.append("requires paired cold_gray Lab mode")
    if mismatches:
        raise ValueError("checkpoint does not match the DIV2K UIEB-style control: " + "; ".join(mismatches))
