"""Novel Cold Diffusion operator that removes chroma from whole pixels."""
import torch
from torch import nn

from .official_colorization import SAMPLERS, channel_gray


class SpatialChromaMask(nn.Module):
    """Replace an increasing fraction of RGB pixels with channel-mean gray.

    A continuous random score is assigned to every pixel.  Thresholding the
    same score map at every timestep makes the masks nested: t=0 keeps every
    pixel colored and t=T keeps none.  Training draws fresh score maps; sampling
    uses an isolated, deterministic generator so validation does not perturb
    the training RNG.
    """

    def __init__(self, steps=20, sampler="paper_algorithm2", sampling_seed=42):
        super().__init__()
        if steps < 1 or sampler not in SAMPLERS:
            raise ValueError("invalid spatial-chroma diffusion steps/sampler")
        self.steps = int(steps)
        self.sampler = sampler
        self.sampling_seed = int(sampling_seed)
        self.register_buffer("retention", torch.linspace(1, 0, self.steps + 1))

    @staticmethod
    def random_scores(candidate):
        return torch.rand_like(candidate[:, :1])

    def sampling_scores(self, candidate):
        generator = torch.Generator(device="cpu").manual_seed(self.sampling_seed)
        shape = (len(candidate), 1, *candidate.shape[-2:])
        return torch.rand(shape, generator=generator).to(
            device=candidate.device, dtype=candidate.dtype
        )

    def degrade(self, candidate, anchor, t, mask_scores=None):
        """Apply D(x,t); ``anchor`` remains unused for bridge API compatibility."""
        if candidate.ndim != 4 or candidate.shape[1] != 3:
            raise ValueError("spatial-chroma states must be BCHW RGB")
        if t.ndim != 1 or len(t) != len(candidate):
            raise ValueError("one timestep per image is required")
        if bool(((t < 0) | (t > self.steps)).any()):
            raise ValueError("degradation timestep must be in 0..T")
        scores = self.random_scores(candidate) if mask_scores is None else mask_scores
        expected = (len(candidate), 1, *candidate.shape[-2:])
        if tuple(scores.shape) != expected:
            raise ValueError(f"mask score shape must be {expected}, got {tuple(scores.shape)}")
        keep = scores < self.retention[t.long()].view(-1, 1, 1, 1)
        return torch.where(keep, candidate, channel_gray(candidate))

    @torch.no_grad()
    def reverse_step(self, model, x, s, mask_scores):
        if not 1 <= s <= self.steps:
            raise ValueError("reverse step must be in 1..T")
        t = torch.full((len(x),), s, device=x.device, dtype=torch.long)
        pred = model(x, t)
        k = s if self.sampler == "paper_algorithm2" else s - 1
        if k > 0:
            kt = torch.full_like(t, k)
            x = (
                x
                - self.degrade(pred, None, kt, mask_scores)
                + self.degrade(pred, None, kt - 1, mask_scores)
            )
        return x

    @torch.no_grad()
    def sample(self, model, anchor, return_trajectory=False, mask_scores=None):
        if anchor.ndim != 4 or anchor.shape[1] != 3:
            raise ValueError("full-gray BCHW RGB input required")
        if not torch.allclose(anchor, channel_gray(anchor), atol=1e-6, rtol=0):
            raise ValueError("spatial-chroma sampling starts at full gray")
        scores = self.sampling_scores(anchor) if mask_scores is None else mask_scores
        x = anchor.clone()
        trajectory = [x.clone()] if return_trajectory else None
        for s in range(self.steps, 0, -1):
            x = self.reverse_step(model, x, s, scores)
            if trajectory is not None:
                trajectory.append(x.clone())
        return (x, trajectory) if return_trajectory else x
