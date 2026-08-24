import torch

from gray_cold_diffusion.bridge import GrayBridge
from gray_cold_diffusion.color import gray_anchor


def test_bridge_endpoints():
    torch.manual_seed(2)
    clean = torch.rand(2, 3, 8, 8) * 2 - 1
    raw = torch.rand(2, 3, 8, 8) * 2 - 1
    anchor = gray_anchor(raw)
    bridge = GrayBridge(steps=8)
    t0 = torch.zeros(2, dtype=torch.long)
    t8 = torch.full((2,), 8, dtype=torch.long)
    assert torch.allclose(bridge.degrade(clean, anchor, t0), clean)
    assert torch.allclose(bridge.degrade(clean, anchor, t8), anchor)


def test_algorithm2_oracle_reconstructs_exactly():
    torch.manual_seed(3)
    clean = torch.rand(2, 3, 8, 8) * 2 - 1
    anchor = gray_anchor(torch.rand_like(clean) * 2 - 1)
    bridge = GrayBridge(steps=8)
    x = anchor.clone()
    for step in range(8, 0, -1):
        t = torch.full((2,), step, dtype=torch.long)
        x = x - bridge.degrade(clean, anchor, t) + bridge.degrade(clean, anchor, t - 1)
    assert torch.allclose(x, clean, atol=1e-6)
