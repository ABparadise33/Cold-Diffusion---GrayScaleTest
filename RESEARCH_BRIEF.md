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
