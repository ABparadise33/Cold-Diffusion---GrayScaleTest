# UIEB + DIV2K full-gray pilot

This run uses **UIEB reference images** and **DIV2K HR images** as color targets.
Both are converted by the same sRGB channel-mean grayscale operator. It does not
train UIEB raw→reference restoration, and adds no saturation scaling or spatial masks.
Domain sampling is 50:50 in expectation (replacement), not exactly in every batch.
Defaults: fresh weights, T20, upstream ConvNeXt, FP32, effective batch32, 10k steps.
UIEB's existing train/val/test membership is preserved; test images are excluded.
The preparer rejects cross-split identical file bytes and paths, records SHA256s,
and makes namespaced symlinks without copying image data.

On the existing CUDA machine, after pulling this revision and activating/setup of
the project's environment:

```bash
UIEB_REFERENCE_DIR=/absolute/path/UIEB/reference-890 \
DIV2K_DATA_ROOT=/absolute/path/DIV2K \
bash scripts/train_mixed_uieb_div2k_4090.sh
```

Default Python is `.venv/bin/python`; override with `MIXED_PYTHON` if needed.
The DIV2K root must contain `DIV2K_train_HR` and `DIV2K_valid_HR`.
Resume is explicit: add `--resume` to the same command. Do not reuse an old
single-domain output/checkpoint. Model source fingerprints are strict on resume.

## Records

All model outputs are under `outputs/uieb_div2k_rgb_fullgray_pilot/`:

- `debug_diagnostics.jsonl`: append-only exposure counts every50 steps; fixed
  center-crop diagnostics at startup and every1000 steps plus final validation.
  Each domain contributes two fixed train and two fixed val images. These are
  diagnostic subsets, **not** full-domain validation aggregates.
- Each diagnostic records target/gray/direct online/direct EMA/sampled EMA RGB
  state range, nonfinite check, clipping fraction, unclipped chroma RMS,
  Lab a/b means, chroma ratio, Delta-E76 and PSNR. Trajectory contains every step.
  RGB/Lab roundtrip and full-gray endpoint errors check representation and operator.
- Training timestep histograms include actual t=T exposure; domain and timestep
  counts restart on resume and are explicitly labelled as such.
- `debug_previews/step_*/train|val/`: fixed target/gray/direct/sample strips.
- `full_gray_metrics.csv`: full combined validation center-crop metrics.
- `previews/step_*/`: existing random validation full-scene comparisons/trajectories.
- `run_manifest.json`: config, source fingerprints, dataset hashes, sampling rule.
- `checkpoints/`: latest/best/final weights; `logs/mixed_*.log` at repository root
  captures console output including failures.

## How to interpret failures

1. Target chroma absent or roundtrip error large: inspect source/normalization/export.
2. No t=T exposures after substantial training or nonzero endpoint color: inspect
   timestep selection and forward operator.
3. Direct output colored, sample gray: inspect first reverse step and subsequent
   chroma decay, clipping, update formula and timestep labels.
4. Both gray: inspect loss/fit/targets. Online vs EMA separates EMA lag from online fit.
5. Fixed train samples improve but fixed val samples do not: investigate
   generalization; subset observations alone do not prove overfitting.
6. More colorful but worse Delta-E: appearance change is not accurate restoration.

Diagnostics do not increase target saturation or alter endpoint sampling/loss.
Review at10k before a longer run. GPU training was not executed on the Mac.
