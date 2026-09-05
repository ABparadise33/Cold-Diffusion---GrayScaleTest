# Research brief

## Active experiment: nested spatial chroma-mask pilot (2026-09-05)

1. **Question/motivation:** Can a DIV2K natural-image prior fill locally absent
   chroma instead of amplifying weak but invertible underwater chroma?
2. **Hypothesis:** Keeping full RGB at a shrinking subset of pixels and making
   every other pixel exactly channel-mean gray creates a meaningful missing-data
   task. It may still fail through ambiguity/domain shift; it is not a claimed fix.
3. **Current limitation:** The completed 5% amplitude run is near raw and remains
   analytically invertible. Full-gray transfer is weak and does not isolate whether
   sparse genuine hints can support completion.
4. **Leverage:** Hold DIV2K, saturation1, T20, seed42, upstream56.6M model, optimizer
   and effective batch32 fixed. Change only the operator to nested pixel masks.
5. **Smallest falsifiable test:** Authorized 10k pilot, validation every1k on all
   Val100 center crops plus five seeded random full-scene previews. Training samples
   fresh masks; each Algorithm2 trajectory reuses one fixed mask map.
6. **Failure meaning:** Gray regions, speckled incoherent color, or no improving
   validation evidence by10k means stop before50k. It does not prove that all
   alternative hint geometry or larger colorization datasets fail.
7. **Success continuation:** Resume the same run to50k, then add a matched UIEB
   Test90 0%/5%-spatial inference comparison. Do not use old amplitude checkpoints.
8. **Evaluation:** Train/validation L1, PSNR, SSIM, Delta-E76, chroma ratio,
   monotonicity, original-size predictions and trajectories. Direct is disabled.
9. **Budget/stop:** Default10k; only explicit resume extends to50k. No sat2 or
   T100/T200 change. No real paid-GPU training has been launched from this machine.

## Current decision: partial color reverts to raw; rethink information removal (2026-09-05)

1. **Question/motivation:** Can a natural-image color prior fill genuinely
   missing colors rather than amplify attenuated underwater color cast?
2. **Hypothesis:** Reducing chroma amplitude is the wrong information bottleneck:
   it stays invertible for nonzero retention. Spatially removing chroma at most
   pixels would require inference of missing colors. This is a proposed new
   operator experiment, not an authorized training run or a demonstrated fix.
3. **Current limitation:** Official5/25% Test90 DE21.931/21.889 versus raw21.808;
   output-vs-raw MAE1.115/.537 on0–255 floats. Repeated Test90 is exploratory.
   Actual0.5% neural inference is untested; analytical float32 inversion already
   recovers raw to0.000705/255 on a spatial-stride8 control of all90 images.
4. **Leverage:** Same natural data/model/sat1/T20 as a controlled starting point;
   color masking can remove local chroma without pretending that .5% amplitude
   means .5% pixels or .5% information. Keep these meanings explicitly separate.
5. **Smallest test:** Before training, define nested deterministic masks and an
   endpoint-consistent forward/reverse operator, test preservation of visible
   hints and non-uniqueness of hidden chroma. Then, only after approval, propose
   20 DIV2K train scenes and10 disjoint validation scenes, bounded2k optimizer
   updates or2 GPU-hours (whichever first); hold saturation/T fixed.
6. **Failure meaning:** Failure to beat gray/simple hint-propagation on held-out
   missing-color pixels means no evidence of useful learned color completion;
   do not scale. Passing this does not establish water dehazing or correct hues
   for all ambiguous objects. A new mask input to old weights is an OOD probe,
   not a matched-training comparison.
7. **Success continuation:** Scale natural validation only after the toy gate;
   revisit UIEB with unchanged names and raw baselines, then independent data.
8. **Evaluation:** Masked-region Delta-E/chroma plus full-image quality and hint
   consistency; predicted original-size PNGs/trajectories. No extra Direct
   evaluation by default. Pure chroma operators still preserve per-pixel RGB
   mean in Algorithm2, so full underwater brightness/haze repair is out of scope.
9. **Budget/stop:** No sat2/T100/T200 training or modified schedule implemented.
   Increasing uniform T cannot remove invertibility and reduces endpoint
   sampling probability at fixed update budget; nonlinear T20 could include
   1%/.5% but requires matched retraining and is lower priority than task design.

## Active diagnostic: official checkpoint, 5% / 25% raw color retained (2026-09-05)

1. **Question/motivation:** Can weak raw color cues prevent gray/brown local
   outputs in underwater transfer without retraining? Region-specific color
   fidelity matters more than simply making water blue.
2. **Hypothesis:** Starting the same official checkpoint at x19 (5%) or x15
   (25%) improves fish/reef colors versus its x20 full-gray result. This is an
   inference ablation, not proof that the model recognizes semantic categories.
3. **Limitation:** User export official_rgb_retrain reports UIEB90 full-gray
   paper Algorithm2 PSNR15.499, SSIM.73743, Delta-E76 25.934; viewed examples
   show blue water but gray/brown reefs. Direct is similar. Input ambiguity,
   training/data insufficiency and domain shift remain unresolved alternatives.
4. **Leverage:** Reuse the DIV2K saturation1 step050000 EMA weights and unchanged
   official own-prediction operator, T20, UIEB seed42 Test90, original-size256/32
   tiling. Retained color is from raw ONLY, never UIEB reference.
5. **Smallest test:** Analytic partial-step and 0%-equivalence CPU tests first;
   then two UIEB90 inference passes at5/25%, paper_algorithm2, no training.
   Model time starts at19/15, not20 with an incorrectly labeled partial input.
6. **Failure meaning:** If colors remain wrong, partial raw cues did not solve
   the transfer failure at these levels. If outputs merely revert to raw, the
   test demonstrates inversion of desaturation, not underwater enhancement.
7. **Success continuation:** Compare the same fish/reef regions and the full
   paired test metrics against unchanged raw before changing data or training.
8. **Evaluation:** Retain Direct, final predictions and named trajectories;
   PSNR/SSIM/Delta-E against GT, raw and partial-input baselines, output-vs-raw
   difference and per-image records. Optional legacy14 IQA metrics remain
   available. Full-gray baseline remains separate; do not overwrite it.
9. **Budget/stop:** No new training, dataset download or local real-checkpoint
   inference. Stop after the two requested conditions and review. Do not
   attribute a regional blue pattern to verified semantic understanding.

## Active gate: paper-aligned full-gray RGB baseline (2026-09-04)

1. **Question/motivation:** Can a source-aligned Cold Diffusion colorizer learn
   diverse DIV2K colors from full gray? A credible baseline is necessary before
   interpreting underwater transfer, saturation scaling, or negative results.
2. **Hypothesis:** The official ConvNeXt/attention restoration network and its
   own-prediction RGB degradation yield a stronger full-gray baseline than our
   custom fixed-anchor/small-UNet runs. This changes multiple implementation
   choices and is a baseline repair, not an isolated causal ablation.
3. **Limitation:** Prior RGB factor1 forward states match the paper, but the
   sampler, model size, EMA warmup, training budget and data do not all match.
   Partial-color inversion is not a full-gray colorization test.
4. **Leverage:** Pin upstream commit f8b1379151ff0cccba49112cf61d439bd4dd4ad9;
   preserve its default ConvNeXt/linear-attention network and time embedding.
   Use D(x,s)=(1-s/T)x+(s/T)mean_RGB(x), with no reverse-state clamp.
5. **Smallest test:** CPU analytic/operator/gradient/resume smoke first, then a
   fresh DIV2K800 run, factor1, seed42, 128px crops, T20, Adam2e-5, FP32,
   effective batch32, EMA .995 after upstream-style 2k warmup. Stop at 50k for
   review; this is not a claimed replication of the paper's 700k data budget.
6. **Failure meaning:** A continuing sepia bias does not establish that the
   paper is wrong or that insufficient diversity/capacity/training is excluded.
   Direct-versus-sampler tests isolate inference only, not learned weights.
7. **Success continuation:** Review full-gray in-domain validation before
   authorizing more steps, saturation >1, larger data, or underwater transfer.
8. **Evaluation:** All100 fixed DIV2K center crops; Direct/Algorithm2, validation
   L1, RGB PSNR/SSIM, Delta-E76, chroma relative to target and gray control.
   Each1k save a fixed full-scene validation image using explicitly recorded
   tiled inference, preserving the standalone PNG's original geometry. Keep
   horizontal batch/trajectory previews in separate directories.
9. **Budget/stop:** No paid training launched locally. 50k cap, resumable only
   with compatible new-baseline weights; no reuse of old UIEB/DIV2K checkpoints.
   Reassess at 5k/10k using color evidence, not PSNR alone. Retain latest/best
   and final numbered checkpoint; do not accumulate every large 5k checkpoint.

Indexing contract: public state s=1..T maps to upstream model label t=s-1.
Default `paper_algorithm2` implements all T paper updates. The explicitly
named `official_code` evaluation control reproduces the inspected source's
T model calls / T-1 effective updates (last call is a no-op). Never silently
mix the two or label this DIV2K50k adaptation a complete CIFAR/CelebA700k
reproduction. Earlier research sections below are historical plans, not proof
of upstream fidelity or current authorization to run saturation sweeps.

### Pre-training batch-fit gate (2026-09-04)

User requests trying a larger batch and falling back after OOM. Test descending
divisors of effective batch32 (32/16/8/4/2/1), each in an isolated subprocess,
using the actual FP32 model, Adam state, EMA, gradient accumulation and full-gray
crop validation. Three synthetic optimizer steps are a memory-fit smoke, not
quality evidence or a throughput optimum. Require max(2GiB,10% device capacity)
memory headroom. Only a typed CUDA OOM or insufficient headroom permits retry;
unrelated errors, worker termination and timeout stop the launcher. Select the
largest passing divisor and set accumulation=32/batch before starting the real
fresh/resume command. No dataset/weights/steps/LR changes and no automatic
restart of an already-running training process. Store every attempt and chosen
configuration outside the training output to preserve fresh-run safeguards.

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
