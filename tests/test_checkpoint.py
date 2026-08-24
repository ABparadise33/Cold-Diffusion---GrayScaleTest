from pathlib import Path

import torch

from gray_cold_diffusion.io import atomic_torch_save


def test_atomic_checkpoint_can_continue(tmp_path: Path):
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "latest.pt"
    atomic_torch_save({"step": 7, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["step"] == 7
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
