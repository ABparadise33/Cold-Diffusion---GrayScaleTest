# Failure log

| Date | Attempt | Expected | Observed | Cause and evidence | Action |
|---|---|---|---|---|---|
| 2026-08-24 | Resume the CUDA smoke checkpoint on an RTX 4090 | Continue from step 2 to step 4 | `TypeError: RNG state must be a torch.ByteTensor` | `torch.load(map_location=cuda)` moved the saved CUDA RNG byte tensor onto CUDA, but `torch.cuda.set_rng_state_all` requires CPU byte tensors | Convert every saved CUDA RNG state to CPU before restoration and add a regression test |
