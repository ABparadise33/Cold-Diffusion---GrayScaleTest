"""One real default-network forward/backward step, no persistent training run."""
import argparse
from pathlib import Path

import torch
import yaml

from gray_cold_diffusion.factory import build_model_and_bridge
from gray_cold_diffusion.official_colorization import channel_gray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'configs/div2k_official_rgb_sat1_50k.yaml').read_text())
    model, bridge = build_model_and_bridge(config)
    model, bridge = model.to(device), bridge.to(device)
    count = sum(p.numel() for p in model.parameters())
    assert count == 56615708
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    target = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    t = torch.tensor([20], device=device)
    gray = bridge.degrade(target, None, t)
    assert torch.equal(gray[:, 0], gray[:, 1]) and torch.equal(gray[:, 1], gray[:, 2])
    assert torch.allclose(gray, channel_gray(target))
    prediction = model(gray, t)
    loss = (prediction - target).abs().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    print(f'OFFICIAL NETWORK OK: parameters={count:,}, FP32, batch1, 128x128, device={device}')
    print(f'one-step synthetic L1={loss.item():.6f}; not a trained result or duration benchmark')
    if device.type == 'cuda':
        print(f'peak_allocated_GiB={torch.cuda.max_memory_allocated(device) / 1024**3:.3f}')


if __name__ == '__main__':
    main()
