# Research brief

## Question and motivation

Can a raw underwater image be projected to a grayscale anchor and then restored
through Cold Diffusion states that progressively recover natural luminance and
chroma? The useful result is an interpretable, controllable restoration path,
not merely a pleasing final image.

## Hypothesis

Algorithm 2 will improve over direct reconstruction from the same trained Cold
model, and most reverse steps will reduce paired-reference color error.

## Current limitation

Standard Reflow/latent diffusion moves from random noise to a clean latent, so
its middle states are not calibrated restoration levels. Standard Cold
Diffusion colorization also uses `gray(reference)`, which does not match the
remaining haze and contrast loss in `gray(raw underwater)`.

## Leverage

Paired UIEB images let us define a conditional Lab bridge ending at
`(L_raw, 0, 0)` and measure every generated state against its designed paired
trajectory.

## Smallest falsifiable test

Overfit 50 pairs for 2k steps, then train the Cold gray model on the fixed
720/80/90 split. On the same checkpoint and Test 90, compare direct
`R(gray, T)` against Algorithm 2. Use 128px crops, 8 Cold steps, and a 50k cap
with validation every 5k.

## Failure meaning

- Algorithm 2 does not beat direct reconstruction: iteration adds no value.
- Both gray methods lose clearly to RGB: the hard gray bottleneck discards
  necessary evidence.
- Final quality improves but state errors are not monotonic: the path is not an
  interpretable restoration trajectory.

## Success continuation

Repeat 30k/50k with three seeds, then test a gray-anchor latent Reflow model.

## Evaluation

PSNR, SSIM, Delta-E76, direct-vs-Algorithm-2 output, reverse-step monotonicity,
and qualitative inspection. Both outputs must come from the same checkpoint,
split, original-resolution image, and metric preprocessing.

## Budget and stop rule

Stop early if validation PSNR/Delta-E76 and trajectory monotonicity do not
improve across three consecutive 5k evaluations. Continue past 50k only if at
least one primary metric is still improving and Cold beats gray one-shot.
The first rented-RTX-4090 run must pass the CUDA/resume smoke test before paid
training; use the measured step rate rather than a theoretical duration.

## Natural-image color-prior ablation (DIV2K)

### Question and motivation

Does a colorization model trained only on high-quality natural images transfer
a useful, ordered color prior to out-of-domain underwater photographs? Does a
stronger training-target saturation improve that transfer, or only produce
over-colorization?

### Hypothesis

Increasing the training-target saturation from 1.0 through 1.25, 1.5, and 2.0
will recover progressively more chroma on underwater inputs, but the benefit
may stop or reverse when sRGB gamut clipping dominates. A factor counts as an
improvement only if paired UIEB Delta-E76/LPIPS also improve and the reverse
path remains mostly monotonic.

### Current limitation and leverage

The first pilot mixes paired UIEB supervision with a custom Lab bridge. The new
ablation removes both choices: DIV2K supplies unpaired high-resolution natural
images, and the forward path uses the paper-style RGB channel-mean grayscale
anchor. This isolates the learned natural-image color prior.

### Smallest falsifiable test

Train four otherwise identical 50k-step models on DIV2K train HR with 128px
random crops, 20 Cold steps, seed 42, and saturation factors 1.0, 1.25, 1.5,
and 2.0. The
saturation transform expands each RGB pixel away from its channel mean:
`clip(gray + factor * (rgb - gray), 0, 1)`. Validate on DIV2K validation HR and
evaluate both checkpoints on the unchanged UIEB seed-42 Test 90. Both runs use
the same grayscale endpoint computed from the unmodified image; only the color
target changes.

### Failure meaning, evaluation, and stop rule

- Higher saturation alone, with worse Delta-E76/LPIPS or obvious wrong colors,
  is hallucination rather than restoration.
- The current RGB transform clips out-of-gamut values to [0,1]. Preview images
  show that this can affect roughly one quarter of pixels at 1.5 and nearly half
  at 2.0 in already colorful/bright scenes. Treat clipping as a measured
  confound, not as an incidental implementation detail.
- Non-monotonic UIEB trajectories reject the interpretation of middle states as
  ordered restoration strength, even if the last image looks vivid.
- Stop a run after three consecutive 5k validations without improvement. Select
  the factor using paired color/perceptual metrics and qualitative color errors,
  not saturation alone. A later ablation may move chroma scaling to Lab, HSV, or
  another gamut-aware color space, but that change must remain separate from
  this first RGB baseline.

### CIE Lab factor-1 control

The RGB sweep produced a strong warm/brown UIEB bias. Before attributing that
result to natural-image training itself, run one controlled factor-1 model in
the earlier UIEB Lab representation. Keep DIV2K, saturation 1.0, seed 42,
128px crops, T=20, architecture, optimizer, effective batch, validation cadence,
and 50k cap unchanged; change only the state representation and gray endpoint
from RGB channel mean to `(L, 0, 0)`. Evaluate the RGB and Lab checkpoints on
the same DIV2K validation images and UIEB Test 90. If Lab is also brown on
UIEB while remaining credible on DIV2K, the natural-to-underwater domain shift
or grayscale information bottleneck—not RGB saturation scaling—is the likely
failure mechanism.

The decisive in-domain check uses all 100 original-resolution DIV2K validation
images rather than the trainer's single fixed 128px center crop. A tiled,
overlap-feathered evaluator preserves output geometry while fitting the 4090.
If the brown bias is already systematic on these in-domain images, reject the
domain-shift explanation and inspect the target/state conversion, objective,
and checkpoint before any additional training.
