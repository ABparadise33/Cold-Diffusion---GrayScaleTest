# Research brief

## Question and motivation

Can a raw underwater image be projected to a grayscale anchor and then restored
through Cold Diffusion states that progressively recover natural luminance and
chroma? The useful result is an interpretable, controllable restoration path,
not merely a pleasing final image.

## Hypothesis

Algorithm 2 will improve over a matched grayscale one-shot U-Net, and most
reverse steps will reduce paired-reference color error.

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

Overfit 50 pairs for 2k steps, then train RGB one-shot, gray one-shot, and Cold
gray models on the same 720/80/90 split. Use 128px crops, 8 Cold steps, and a
50k cap with validation every 5k.

## Failure meaning

- Cold does not beat gray one-shot: iteration adds no value.
- Both gray methods lose clearly to RGB: the hard gray bottleneck discards
  necessary evidence.
- Final quality improves but state errors are not monotonic: the path is not an
  interpretable restoration trajectory.

## Success continuation

Repeat 30k/50k with three seeds, then test a gray-anchor latent Reflow model.

## Evaluation

PSNR, SSIM, Delta-E76, direct-vs-Algorithm-2 output, reverse-step monotonicity,
and qualitative inspection. Use identical architecture and split for the three
pixel-space models.

## Budget and stop rule

Stop early if validation PSNR/Delta-E76 and trajectory monotonicity do not
improve across three consecutive 5k evaluations. Continue past 50k only if at
least one primary metric is still improving and Cold beats gray one-shot.
