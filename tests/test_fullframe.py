import pytest
import torch
from torch import nn

from gray_cold_diffusion.fullframe import run_fullframe
from gray_cold_diffusion.tiling import TiledModel


def test_whole_image_first_and_no_retry_on_other_errors():
    model = nn.Identity()
    calls = []
    def operation(active):
        calls.append(active)
        return 123
    result, route = run_fullframe(operation, model, device='cpu')
    assert result == 123 and route['method'] == 'full_image' and calls == [model]
    def bad(active):
        raise RuntimeError('shape bug')
    with pytest.raises(RuntimeError, match='shape bug'):
        run_fullframe(bad, model, device='cpu')


def test_oom_restarts_complete_operation_and_restores_rng(monkeypatch):
    model = nn.Identity()
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: torch.tensor([1], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    draws = []
    def operation(active):
        draws.append(torch.rand(1).item())
        if not isinstance(active, TiledModel):
            raise torch.cuda.OutOfMemoryError('simulated')
        return active.tile_size
    result, route = run_fullframe(operation, model, device='cuda', fallback_tile=512)
    assert result == 512 and route['method'] == 'oom_tiled_fallback'
    assert draws[0] == draws[1]
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_fullframe(operation, model, device='cuda', fallback_tile=None)


def test_cpu_memory_errors_are_not_tiled():
    def operation(active):
        raise torch.cuda.OutOfMemoryError('not a CUDA execution')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_fullframe(operation, nn.Identity(), device='cpu')
