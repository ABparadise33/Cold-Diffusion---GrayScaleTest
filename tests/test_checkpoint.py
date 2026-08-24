from pathlib import Path

import torch

from gray_cold_diffusion.engine import _restore_cuda_rng_states
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


def test_cuda_rng_restore_converts_states_to_cpu(monkeypatch):
    captured = []

    def capture(states):
        captured.extend(states)

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", capture)
    _restore_cuda_rng_states([torch.tensor([1, 2, 3], dtype=torch.uint8)])

    assert len(captured) == 1
    assert captured[0].device.type == "cpu"
    assert captured[0].dtype == torch.uint8
