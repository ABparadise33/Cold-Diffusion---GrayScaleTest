"""Inference-only intermediate starts; leave the full-gray training operator untouched."""
import math

import torch

from .official_colorization import channel_gray


def retained_color_start_step(steps, percent):
    if not math.isfinite(percent) or not 0 <= percent < 100:
        raise ValueError('retained raw color must be >=0 and <100 percent')
    position = int(steps) * (1 - percent / 100)
    start = round(position)
    if not 1 <= start <= steps or not math.isclose(position, start, abs_tol=1e-7):
        raise ValueError(f'{percent:g}% is not an exact state for T={steps}; do not silently round timesteps')
    return start


def partial_raw_input(bridge, raw_state, start_step):
    if not 1 <= start_step <= bridge.steps:
        raise ValueError('start step must be in 1..T')
    if start_step == bridge.steps:
        return channel_gray(raw_state)  # exactly the existing full-gray endpoint
    t = torch.full((len(raw_state),), start_step, device=raw_state.device, dtype=torch.long)
    return bridge.degrade(raw_state, None, t)  # never accepts a reference/GT image


@torch.no_grad()
def sample_from_step(bridge, model, state, start_step):
    if not 1 <= start_step <= bridge.steps:
        raise ValueError('start step must be in 1..T')
    if start_step == bridge.steps:
        return bridge.sample(model, state, return_trajectory=True)
    x = state.clone()
    trajectory = [x.clone()]
    for s in range(start_step, 0, -1):
        x = bridge.reverse_step(model, x, s)
        trajectory.append(x.clone())
    return x, trajectory
