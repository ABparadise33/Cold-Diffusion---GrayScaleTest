# Failure log

| Date | Attempt | Expected | Observed | Cause and evidence | Action |
|---|---|---|---|---|---|
| 2026-08-24 | Resume the CUDA smoke checkpoint on an RTX 4090 | Continue from step 2 to step 4 | `TypeError: RNG state must be a torch.ByteTensor` | `torch.load(map_location=cuda)` moved the saved CUDA RNG byte tensor onto CUDA, but `torch.cuda.set_rng_state_all` requires CPU byte tensors | Convert every saved CUDA RNG state to CPU before restoration and add a regression test |
| 2026-08-25 | Initialize all FlowIE-compatible metrics | Load CLIP-IQA after MUSIQ | `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools 82+ removed `pkg_resources`, but the current `openai-clip` dependency still imports it | Pin setuptools below 82 and rerun evaluation |
| 2026-09-02 | Increase DIV2K saturation by scaling RGB away from channel-mean gray | Controlled chroma increase across factors 1.25, 1.5, and 2.0 | Bright/colorful previews contain substantial clipped pixels: up to 26.5% at 1.5 and 49.0% at 2.0 among four inspected examples | Fixed RGB extrapolation pushes one or more channels outside the sRGB [0,1] gamut and the current transform clamps them | Keep 1.0/1.25/1.5/2.0 as the first controlled RGB ablation and record clipping as a confound; later compare a separate Lab/HSV or gamut-aware transform rather than changing this baseline post hoc |
