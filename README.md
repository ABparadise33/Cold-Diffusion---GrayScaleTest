# Cold Diffusion - GrayScaleTest

50k-step pilot for this question:

> Can a grayscale anchor become an interpretable restoration path from a raw
> underwater image to its paired natural-looking reference?

The Cold model follows the paper's core pattern: predict the clean target at
every time step, then use Algorithm 2 to update the sampler state. This repo
adds a paired Lab-space bridge whose endpoint is the raw image's luminance with
zero chroma.

## Fast path

```bash
python -m pip install -e .

python tools/make_uieb_split.py \
  --raw-dir /path/to/UIEB/raw-890 \
  --reference-dir /path/to/UIEB/reference-890 \
  --output splits/uieb_seed42.json

python train.py \
  --config configs/cold_gray_50k.yaml \
  --raw-dir /path/to/UIEB/raw-890 \
  --reference-dir /path/to/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json
```

Continue the same run to 100k:

```bash
python train.py \
  --config configs/cold_gray_50k.yaml \
  --raw-dir /path/to/UIEB/raw-890 \
  --reference-dir /path/to/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json \
  --resume auto \
  --max-steps 100000
```

`latest.pt` stores model, EMA, optimizer, AMP scaler, step, best score, and RNG
state. Increasing `--max-steps` is enough to continue beyond 50k.

## Required comparison

Run the three configs on the same split:

- `configs/rgb_oneshot_50k.yaml`
- `configs/gray_oneshot_50k.yaml`
- `configs/cold_gray_50k.yaml`

The first useful decision is at 10k. The hard budget is 50k; do not assume
50k is converged. Metrics and samples are written every 5k.

## Outputs

```text
outputs/<experiment>/
  checkpoints/latest.pt
  checkpoints/best.pt
  checkpoints/step_005000.pt
  metrics.csv
  samples/step_005000.png
  trajectories/step_005000.png
```

Metrics include PSNR, SSIM, Lab Delta-E76, and the fraction of reverse steps
whose color error decreases.

## Device defaults

- CUDA: AMP enabled; batch 16 by default in the NVIDIA config override.
- MPS: FP32; default config uses physical batch 4 and accumulation 4.
- CPU: supported for tests only.

Override device and batch settings from the command line:

```bash
python train.py ... --device cuda --batch-size 16 --grad-accum 1
```

## Evaluate a checkpoint

```bash
python evaluate.py \
  --checkpoint outputs/cold_gray_50k/checkpoints/best.pt \
  --raw-dir /path/to/UIEB/raw-890 \
  --reference-dir /path/to/UIEB/reference-890 \
  --split-file splits/uieb_seed42.json \
  --split test
```

## Tests

```bash
pytest -q
```

Conceptual reference: [Cold Diffusion: Inverting Arbitrary Image Transforms
Without Noise](https://arxiv.org/abs/2208.09392) and its
[official implementation](https://github.com/arpitbansal297/Cold-Diffusion-Models).
