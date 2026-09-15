"""Whole-image inference first; retry an entire prediction only on CUDA OOM."""
import gc

import torch

from .tiling import TiledModel


def run_fullframe(operation, model, *, device, fallback_tile=512, overlap=64):
    """operation(model) must restart from the original input on every call.

    Return (result, route). Do not retry individual reverse steps: a result must
    use one spatial policy for its entire trajectory and Direct prediction.
    """
    if fallback_tile is not None and not 0 <= overlap < fallback_tile:
        raise ValueError('invalid OOM fallback tile/overlap')
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
    try:
        return operation(model), {'method': 'full_image', 'tile_size': None, 'overlap': None}
    except torch.cuda.OutOfMemoryError:
        if torch.device(device).type != 'cuda' or fallback_tile is None:
            raise
    # Outside except: release the traceback and intermediate activations first.
    gc.collect()
    torch.cuda.empty_cache()
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)
    print(f'CUDA OOM: restarting the complete prediction with tile={fallback_tile}, overlap={overlap}')
    result = operation(TiledModel(model, fallback_tile, overlap))
    return result, {'method': 'oom_tiled_fallback', 'tile_size': fallback_tile,
                    'overlap': overlap, 'reason': 'CUDA out of memory during full-image prediction'}
