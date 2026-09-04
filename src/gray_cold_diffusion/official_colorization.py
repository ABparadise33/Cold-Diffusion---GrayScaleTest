"""Paper RGB operator and explicitly separate literal-upstream sampler control."""
import torch
from torch import nn
import torch.nn.functional as F

from .official_convnext import UnetConvNextBlock

UPSTREAM_COMMIT = "f8b1379151ff0cccba49112cf61d439bd4dd4ad9"
SAMPLERS = ("paper_algorithm2", "official_code")


def channel_gray(x):
    return x.mean(dim=1, keepdim=True).expand_as(x)


class OfficialColorizer(nn.Module):
    """Upstream network, with one-based state labels and pad/crop geometry adapter.

    Divisible training inputs pass unmodified. Boundary padding is only for
    arbitrary original-size inference, not a resize or extra learned layer.
    """

    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8), steps=20):
        super().__init__()
        if dim < 4 or dim % 2 or len(dim_mults) < 2 or steps < 1:
            raise ValueError("invalid official model dimensions/timesteps")
        self.steps = int(steps)
        self.multiple = 2 ** (len(dim_mults) - 1)
        self.network = UnetConvNextBlock(
            dim=dim, dim_mults=dim_mults, channels=3,
            with_time_emb=True, residual=False,
        )

    def forward(self, x, t):
        if t.ndim != 1 or len(t) != len(x):
            raise ValueError("one timestep per image is required")
        if bool(((t < 1) | (t > self.steps)).any()):
            raise ValueError("public state timestep must be in 1..T")
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % self.multiple, (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        out = self.network(x, t - 1)
        return out[..., :height, :width]


class RGBDecolorization(nn.Module):
    def __init__(self, steps=20, sampler="paper_algorithm2"):
        super().__init__()
        if steps < 1 or sampler not in SAMPLERS:
            raise ValueError("invalid RGB diffusion steps/sampler")
        self.steps = int(steps)
        self.sampler = sampler
        self.register_buffer("schedule", torch.linspace(0, 1, self.steps + 1))

    def degrade(self, candidate, anchor, t):
        """The legacy call signature is retained, but anchor is deliberately unused.

        In the paper D(candidate,t) depends ONLY on candidate, never gray(raw).
        """
        if bool(((t < 0) | (t > self.steps)).any()):
            raise ValueError("degradation timestep must be in 0..T")
        a = self.schedule[t.long()].view(-1, 1, 1, 1)
        return (1 - a) * candidate + a * channel_gray(candidate)

    @torch.no_grad()
    def reverse_step(self, model, x, s):
        if not 1 <= s <= self.steps:
            raise ValueError("reverse step must be in 1..T")
        t = torch.full((len(x),), s, device=x.device, dtype=torch.long)
        pred = model(x, t)
        k = s if self.sampler == "paper_algorithm2" else s - 1
        if k > 0:
            kt = torch.full_like(t, k)
            x = x - self.degrade(pred, None, kt) + self.degrade(pred, None, kt - 1)
        return x

    @torch.no_grad()
    def sample(self, model, anchor, return_trajectory=False):
        if anchor.ndim != 4 or anchor.shape[1] != 3:
            raise ValueError("full-gray BCHW RGB input required")
        if not torch.allclose(anchor, channel_gray(anchor), atol=1e-6, rtol=0):
            raise ValueError("official baseline starts at FULL gray, not partial color")
        x = anchor.clone()
        trajectory = [x.clone()] if return_trajectory else None
        for s in range(self.steps, 0, -1):
            x = self.reverse_step(model, x, s)
            # No clamping: it changes the algorithm and its channel-mean invariant.
            if trajectory is not None:
                trajectory.append(x.clone())
        return (x, trajectory) if return_trajectory else x
