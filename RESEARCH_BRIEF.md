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

### Implementation-regression gate

Before another DIV2K run, retrain the original UIEB Lab configuration from
step zero under a new output name. Save one fixed validation example every
1,000 steps. If the fresh UIEB run develops the new brown bias, treat that as a
shared implementation regression and stop. If it recovers the earlier cool,
low-chroma behavior, run DIV2K through the same paired loader, Lab state,
T=8, architecture, and optimizer. Test factor 1 first; do not start the higher
Lab-chroma factors until the fixed preview passes visual inspection.

## Partial-chroma endpoint diagnostic (2026-09-04)

1. **Question/motivation:** Is DIV2K's brown output specifically a failure to
   infer colors from a completely gray input? This distinguishes loss of color
   evidence from a general Lab conversion or sampler failure.
2. **Hypothesis:** The same model restores substantially better hue/color from
   a partially desaturated input than from full gray.
3. **Limitation:** The existing all-gray test confounds recoloring ambiguous
   content with ordinary recovery of attenuated but still-observed chroma.
4. **Leverage:** Every intermediate Lab state was already represented during
   training. No loss, endpoint, time embedding, or training schedule change is
   needed to start inference from an existing intermediate state.
5. **Smallest test:** Use the existing verified 50k DIV2K Lab checkpoint
   (`natural_lab_colorization`, seed 42, T=20, 128px crops, factor 1). On four
   fixed DIV2K validation images, start at t=10,15,18 (50%,25%,10% remaining a/b).
   Default inference never starts at full gray; t=20 is an explicit optional
   control. A separate T=8 UIEB-recipe replication is not a prerequisite.
6. **Failure meaning:** If even partial-color inputs become brown, endpoint
   ambiguity alone is insufficient; examine training fit and sampler behavior.
   If only full gray fails, test endpoint supervision next, without claiming
   that missing semantic color inference has been solved.
7. **Success continuation:** Only after the small diagnostic, expand the same
   checkpoint test to Val100 or a separately controlled endpoint-training test.
8. **Evaluation:** Compare the attenuated input, Direct, Algorithm 2 and reference
   per image/step using Delta-E76, RGB PSNR and Lab chroma/a/b diagnostics. The
   input already contains target color information: this is a synthetic
   diagnostic, not a grayscale-colorization or underwater-restoration score.
   Include analytical chroma rescaling as a sanity control: partial Lab
   desaturation is exactly invertible before quantization when retention > 0.
9. **Budget/stop:** Inference only on four images, no new training launched here.
   Do not retrain a T=8 model for this diagnostic. Preserve the existing
   checkpoint's T=20 time labels, verify its embedded training step and Lab
   config, and never substitute UIEB or RGB weights. Preserve original output
   sizes and separate preview folders.

## RGB factor-1 intermediate-start control (2026-09-04)

- **Question/motivation:** Does the existing RGB factor-1 model also avoid sepia
  when color evidence remains? This tests whether the Lab observation extends
  to the already-trained RGB saturation experiments without new GPU training.
- **Hypothesis/limitation:** Partial RGB inputs will recover diverse colors;
  the Lab result alone cannot establish that, because both representation and
  grayscale operator differ. Percentages are path coefficients, not identical
  perceptual color strengths across RGB and Lab.
- **Leverage/minimal test:** Existing `div2k_rgb_sat1_50k` EMA at step 50000,
  original T=20. Use the same seed42 four DIV2K validation images, RGB
  channel-mean anchor, t=10/15/18 (50%/25%/10% color difference retained), plus
  t=20 full-gray control on the exact same checkpoint. No Lab normalization of
  RGB weights, no saturation edit, no new training.
- **Evaluation:** Preserve original geometry and 512/64 tiling. Compare input,
  Direct, Algorithm 2, analytic inverse, reference and actual-t trajectories;
  record RGB PSNR, Delta-E76, Lab chroma and a/b means. Inputs come from the
  unmodified factor-1 source. For r>0, `g + (input-g)/r` is an almost exact
  nonlearned inverse; success is diagnostic, not a restoration benchmark.
- **Failure/success:** If RGB remains brown from partial inputs, investigate
  its weights/fit/path before a sweep. If it restores color, next compare the
  existing RGB saturation models with identical source-derived inputs, not
  factor-specific target-derived inputs; this follow-on is not launched here.
- **Budget/stop:** Four images, inference only. Reject wrong mode, saturation,
  training step or config; stop on a failed sanity control. Do not scale to
  Val100, UIEB, or additional saturation models automatically.

## UIEB raw-only RGB saturation transfer (2026-09-04)

1. **Question/motivation:** Do the four existing DIV2K RGB saturation priors
   correct underwater color rather than simply undo artificial desaturation?
   This is the first paired underwater test of the partial-color approach.
2. **Hypothesis:** At a fixed raw-derived input and timestep, a higher training
   saturation factor improves reference color/perceptual error over factor 1
   AND over leaving raw unchanged, without simply amplifying its color cast.
3. **Limitation:** Natural self-desaturation is analytically invertible and
   retains correct target hues; underwater raw contains wrong hues and haze.
4. **Leverage:** Reuse RGB factors 1/1.25/1.5/2, each at training step 50000 and
   T=20. The fixed Underwater_FlowIE-compatible UIEB seed42 Test90 is available.
5. **Smallest real test:** Default to t=15 (25% retained RGB-gray) for all four
   models and all 90 fixed test images. No new training. Other starts may be
   explicitly selected, but the default changes only the checkpoint factor.
6. **Failure meaning:** If outputs merely approach raw or worsen paired color
   error, the test has not demonstrated enhancement. Higher chroma alone is not
   success. Training-target RGB gamut clipping remains a confound, especially
   at factor 2, and is not removed by this evaluation.
7. **Success continuation:** Inspect differences by image, compare the frozen
   baselines, and validate on another fixed split/dataset before retraining.
   Results on Test90 are exploratory if later used to choose a factor.
8. **Evaluation:** Construct both x_t and gray anchor exclusively from raw,
   before any factor-specific changes. GT is unchanged and used only for
   scoring/side-by-side display. Compare raw, analytic inversion (recovers raw,
   NOT GT), Direct and Algorithm 2; record RGB PSNR/SSIM, Delta-E76, chroma,
   output-minus-raw changes and color-error trajectory monotonicity. Reuse the
   existing 14-metric FlowIE evaluator at legacy 256px metric size while saved
   predictions remain original-size. Verify identical raw-derived input and
   reference hashes across models and export a combined factor table.
9. **Budget/stop:** Four sequential inference runs, one default start each,
   no model training or dataset downloads. Save all 90 predictions per model,
   but only four fixed comparison/trajectory examples unless requested. Stop
   on checkpoint/config/split mismatches or leakage checks; never silently
   substitute best.pt or an earlier checkpoint. Reassess after this sweep.
