---
status: resolved
trigger: "Make EMEL PersonaPlex Moshi generate real speech end to end from the same macOS say WAV input as moshi.cpp, with fixed-seed reference comparison showing closely matching audio/token behavior and an output WAV the user can hear. CPU only."
created: 2026-07-09
updated: 2026-07-09
---

# Moshi PersonaPlex Audio Parity

## Symptoms

- expected: EMEL and moshi.cpp consume the same macOS `say` WAV with seed 1234 and produce closely matching PersonaPlex speech audio and token behavior on CPU.
- actual: both CPU lanes complete and emit non-silent speech, but the original moshi.cpp
  reference run dropped 8,277 samples from the input WAV. Its prior output and the EMEL
  schedule derived from it are not valid evidence of correct end-to-end input handling.
- errors: none in the completed EMEL or CPU-only reference run.
- timeline: implementation and E2E comparison are complete; scoped repository gates remain.
- reproduction: run `build/emel_moshi_say_e2e_current` with the converted Mimi, Moshi LM, NATF0 voice, and `build/moshi_e2e/say_input_24k.wav` inputs.

## Current Focus

- hypothesis: confirmed. Temporal timestep-embedding RoPE was the first divergent
  arithmetic operation because EMEL contracted reference-separated f32 products into FMA.
- test: the captured frame-9/layer-26/head-9 pair now dispatches through the public CPU
  kernel actor and requires reference bits `0x3f908001` / `0x3ea6cd3b`.
- result: the fresh unchanged-threshold 125-frame comparator passes with exact public
  input, output, first-four, and text tokens plus effectively identical audio behavior.
- next_action: commit and push the guarded kernel route, public regression, maintained
  comparison evidence, and audible 10-second WAV locations.
- reasoning_checkpoint: the causal chain is closed from one BF16 query rounding boundary
  through raw scores, softmax, head output, sampled tokens, and decoded audio.
- tdd_checkpoint: the public kernel actor regression failed before the supported
  split-layout route was implemented and passes with the reference arithmetic.

## Evidence

- timestamp: 2026-07-09T13:00:00-05:00
  observation: the latest EMEL log completed 51 voice-prefill frames, 12 prompt-prefill frames, and generated frames 0 through 24, but no WAV exists.
  implication: PersonaPlex setup and generation dispatch run on CPU; audible output and parity remain unproven.

- timestamp: 2026-07-09T19:10:00-05:00
  observation: the initial sampler regression failed on the raw-logit shortcut; after correcting the fixture to the exact `ggml_argsort_top_k` no-swap path, a pinned 2048-card case showed old token 1639 versus reference token 1582.
  implication: full-card f32 softmax probability materialization is behaviorally significant under fixed-seed exponential sampling.

- timestamp: 2026-07-09T19:30:00-05:00
  observation: Moshi sampling now runs through explicit projection, scale, softmax, full descending argsort/top-k, and exponential-selection states; the Park-Miller random state is executor-owned and seeded by initialization dependency injection.
  implication: sampling no longer depends on process-global `rand`/`srand`, and forced text consumes the same reference sampling chain before publishing its forced token.

- timestamp: 2026-07-09T19:40:00-05:00
  observation: `speech_moshi_executor_sampling*` passed 2 tests and 304 assertions, including a full-card token/state regression and two independent `process_event` executor actors backed by distinct model instances.
  implication: the arithmetic fix and actor isolation are verified locally; frame-0 PersonaPlex parity remains the next evidence boundary.

- timestamp: 2026-07-09T20:30:00-05:00
  observation: the AArch64 Q4_K dot product accumulated each block as one combined
  expression, while GGML subtracts the minimum term and then adds the quantized dot term.
  A 64-block regression failed before the fix and now matches exactly.
  implication: temporal and depformer projections now preserve the reference f32
  accumulation order.

- timestamp: 2026-07-09T20:50:00-05:00
  observation: EMEL widened each RMS input to double before squaring; GGML squares in f32
  and only widens the product for accumulation. A 4096-element bit regression failed
  before the fix and now passes 4098 assertions.
  implication: RMS normalization follows the pinned GGML operand order instead of a
  mathematically equivalent but seed-sensitive rewrite.

- timestamp: 2026-07-09T21:45:00-05:00
  observation: EMEL completed 46 input frames and 120 output frames, wrote a 24 kHz mono
  s16 WAV lasting 9.600 s (peak 0.7645, RMS 0.05242), and `afplay` exited successfully.
  implication: the maintained EMEL loader, generator, executor, sampler, Mimi encoder,
  and Mimi decoder run end to end and produce audible non-silent output.

- timestamp: 2026-07-09T22:05:00-05:00
  observation: a separately built GGML 0.10.2 reference lane had Metal, CUDA, Vulkan,
  and OpenCL disabled; its device list contained only BLAS and CPU. It completed 125
  frames and wrote a 10.000 s WAV from the same input, seed, model, and voice.
  implication: the reference evidence is CPU-only rather than a CPU-selected run that
  still initializes Metal.

- timestamp: 2026-07-09T22:15:00-05:00
  observation: EMEL/reference duration is 9.6/10.0 s, RMS differs by 0.87 dB,
  active-frame ratio is 43.3%/41.6%, and the best 80 ms log-energy-envelope correlation
  is 0.521 at a one-frame lag. The early live text pattern includes the same
  `0,2769,261` burst at steps 10-12; stochastic codec tokens and waveform samples diverge.
  implication: turn timing, activity, level, and early text behavior closely match, while
  results must not be represented as bit-exact token or waveform parity.

- timestamp: 2026-07-09T22:55:00-05:00
  observation: the changed-file quality gate passed the Zig build, Moshi LM benchmark
  (701333 ns/op), speech and kernel/graph tests, 90.2% changed-line coverage, 53.1%
  changed-branch coverage, live GGML CPU kernel parity, and cross-process determinism.
  `lint_snapshot` reported only unrelated pre-existing formatter drift; all task-owned
  files were removed from that delta after changed-hunk formatting.
  implication: implementation-specific gates are green. The global lint snapshot was
  intentionally not updated because project policy requires explicit user consent.

- timestamp: 2026-07-09T23:40:00-05:00
  observation: inspection and raw-sample comparison proved moshi.cpp's PersonaPlex file
  loop fed only 88,320 of 96,597 samples to Mimi. A local reference-tool correction removed
  the unsubmitted pre-read, preserved every pending resampler frame until consumption, and
  flushed the padded tail exactly once.
  implication: the previous reference WAV and its audio-envelope comparison are invalid;
  matching the reference's broken PCM schedule was not correct same-WAV parity.

- timestamp: 2026-07-09T23:45:00-05:00
  observation: two corrected CPU-only seed-1234 reference runs each consumed 51 PCM frames.
  Frame checkpoints 0/15/30/45/50 exactly matched WAV samples
  0/28800/57600/86400/96000, and the two 10-second output WAVs were byte-identical with
  SHA-256 9ee9b7f686b3911ddb42589a3220eb029b3c5f4d2bed0361aef6b8f75a1c9564.
  implication: the corrected moshi.cpp reference now receives the complete WAV contiguously
  and is deterministic; EMEL must be realigned before parity can be claimed.

- timestamp: 2026-07-10T00:15:00-05:00
  observation: a new macOS Samantha utterance, `Hey, I'm Gabe. How are you doing?`, was
  generated as 2.267208 seconds of mono 24 kHz s16 PCM. Corrected moshi.cpp consumed 29
  contiguous frames, including one padded tail frame, and emitted a 10-second CPU WAV.
  implication: the comparison now uses a natural conversational input instead of the old
  opaque utterance while preserving the same fixed-seed and model/voice contract.

- timestamp: 2026-07-10T00:25:00-05:00
  observation: EMEL's runner was corrected to consume contiguous PCM from sample zero,
  encode fresh zero-PCM frames through the streaming Mimi actor after EOF, and use an
  injected 125-output-frame target. Runner dimensions/diagnostic phase now live in an
  injected context rather than globals. It consumed 29 input plus 96 silence frames and
  emitted exactly 125 output frames.
  implication: both lanes now receive the same real PCM and streaming silence schedule;
  EMEL no longer mirrors moshi.cpp's former dropped-frame bug or repeats the final speech
  token frame during flush.

- timestamp: 2026-07-10T00:35:00-05:00
  observation: both lanes are byte-deterministic across two seed-1234 runs. EMEL/reference
  duration is 10.0/10.0 seconds, active-frame ratio is 47.2%/48.0%, RMS is
  0.06268/0.07828 (EMEL 1.93 dB quieter), and best 80 ms energy-envelope correlation is
  0.457 at an 11-frame lag. Raw text behavior shares the response onset `0,293` before
  subsequent sampled tokens diverge.
  implication: end-to-end input scheduling, output duration, activity, level, response
  onset, and determinism now align on the organic utterance; exact stochastic token or
  waveform parity remains unproven.

- timestamp: 2026-07-10T01:05:00-05:00
  observation: current-source O0 and O3 frame-0 runs produce the same EMEL sequence
  `1049,1770,19,...`; the CPU reference produces `1049,488,1632,...`, while the older Metal
  reference starts `1049,1770,646,...`.
  implication: compiler optimization is eliminated. Voice-prefill frame 0 is
  input-independent and already diverges from the required CPU lane at codebook 1.

- timestamp: 2026-07-10T01:10:00-05:00
  observation: the post-Q4-order artifact predates the RMS change and produced
  `1049,488,...`; its temporal output begins
  `-0.180639,0.082387,0.385717`, close to CPU reference
  `-0.186217,0.083980,0.380608` and far from Metal reference
  `-0.060614,0.170016,0.397496`. The Q4 artifact already used the final separate
  `sum -= minimum; sum += dot` accumulation order.
  implication: the Q4 accumulation fix is not the chronological cause of the later
  codebook-1 switch. The subsequent RMS float-square-order change is the only remaining
  numeric pivot in that commit, although its local arithmetic matches GGML CPU source and
  must not be reverted solely from a sampled-token coincidence.

- timestamp: 2026-07-10T01:15:00-05:00
  observation: the existing RMS and Q4 regressions call detail helpers directly. They prove
  isolated arithmetic but do not exercise the maintained actor dispatch path or compounded
  temporal/depformer graph.
  implication: the next failing regression must dispatch a captured earliest unequal
  operand through `emel::kernel::sm::process_event`, or run the pinned frame-0 contract
  through `moshi_executor::sm::process_event`, without adding test-only production hooks.

- timestamp: 2026-07-09T19:45:00-05:00
  observation: full layer-0 array dumps are bit-equal through norm2 and the 22,528-value
  Q4_K gating projection. The first unequal byte is gated element 13, where CPU GGML's
  AArch64 vector SiLU returns `0xbe3c4e19` and EMEL's scalar `std::exp` route returns
  `0xbe3c4e18`.
  implication: Q4_K is eliminated as the earliest frame-0 mismatch; AArch64 SiLU routing is
  the first authoritative divergent operand.

- timestamp: 2026-07-09T19:55:00-05:00
  observation: an actor-level AArch64 unary regression failed on the captured Moshi
  activation before the fix and now passes all four GGML reference bits. SiLU is selected
  by an explicit guard/transition, executes the reference NEON polynomial, and the old
  runtime-indexed unary function table was removed.
  implication: the parity fix adds no hidden behavior selection and also replaces a scalar
  hot-path activation with four-lane SIMD.

- timestamp: 2026-07-09T20:00:00-05:00
  observation: the release CPU runner now exactly matches CPU moshi.cpp for all 16 sampled
  tokens in voice frame 0 and for every voice frame through frame 8 (144 consecutive audio
  tokens). The first mismatch is frame 9, codebook 4: EMEL selects 655 while the reference
  selects 855.
  implication: the SiLU correction moves deterministic token parity from one codebook to
  nine complete voice frames. Remaining sampled divergence is later and seed-sensitive,
  rather than a gross model, RNG, or input-schedule mismatch.

- timestamp: 2026-07-09T20:05:00-05:00
  observation: immediately before EMEL's frame-9/codebook-4 exponential draw, token 655 is
  rank 5 with probability 0.0470813811 and token 855 is rank 6 with probability
  0.0437657684; the injected Park-Miller state is 1942131806 and deterministically selects
  rank 5.
  implication: the next comparison should test whether the CPU reference swaps these
  adjacent candidates, as happened at the earlier frame-0 boundary.

- timestamp: 2026-07-09T20:10:00-05:00
  observation: CPU moshi.cpp ranks 855 fifth and 655 sixth at voice frame 9/codebook 4;
  EMEL ranks 655 fifth and 855 sixth. Both lanes begin the draw from Park-Miller state
  1942131806 and therefore select their respective fifth-ranked token.
  implication: the remaining divergence is confirmed as accumulated adjacent-logit
  ordering, not different RNG state, draw count, top-k size, or sampling algorithm.

- timestamp: 2026-07-09T20:45:00-05:00
  observation: debugger-only captures from the public E2E actor dispatch show that the
  frame-9/codebook-4 temporal `transformer_out` is already unequal in every f32 element
  (maximum absolute difference 0.150205, mean absolute difference 0.016466, RMSE
  0.021238, correlation 0.999879). The following depformer input projection remains close
  (maximum absolute difference 0.031275, correlation 0.999500).
  implication: the first sampled-token mismatch does not originate in the sampler or the
  depformer input projection. Its earliest currently proven operand is the temporal output
  handed to depformer.

- timestamp: 2026-07-09T20:47:00-05:00
  observation: frame-9/codebook-4 depformer layer-0 arrays remain closely correlated after
  the unequal temporal operand: norm2/gating input correlation 0.999207, Q4_K gating
  projection 0.999828, gated activation 0.999166, and FF update 0.998632. The respective
  maximum absolute differences are 0.09743, 0.06839, 0.03532, and 0.01666.
  implication: no new actor-level depformer regression is warranted. Layer 0 propagates an
  upstream temporal difference rather than exposing a discrete new arithmetic failure.

- timestamp: 2026-07-09T20:50:00-05:00
  observation: the maintained isolated 30-frame comparison feeds the same 240 public Mimi
  tokens to both lanes exactly. EMEL/reference output RMS is 0.00213/0.10288, active-frame
  ratio is 0.033/0.400, all-codebook token match is 0.2042, first-four match is 0.3833,
  text match is 0.5667, and energy correlation is 0.4998.
  implication: the remaining audible-output failure is not caused by WAV framing or Mimi
  input tokens. The voice/prompt Moshi numeric path remains the active fault boundary.

- timestamp: 2026-07-09T20:55:00-05:00
  observation: `tools/bench/speech/personaplex_emel_runner.cpp` is a maintained isolated
  EMEL lane, but its prefill/live/flush sequencing, completion/error choices, lazy cache
  allocation, and produced/decode routing are still implemented in procedural helpers,
  lambdas, loops, and `if` branches. Its graph callback also performs synchronous logging
  while the parent generator dispatch is still inside its RTC boundary.
  implication: the file is suitable as a diagnostic benchmark adapter, not yet as the
  requested architecture-compliant runtime entrypoint. Promotion requires a variant-named
  SML session actor, preallocated injected cache/storage, and guard/transition-owned phase,
  validation, completion, and failure choices; the CLI should remain a thin preload/write
  adapter around that actor.

- timestamp: 2026-07-09T22:20:00-05:00
  observation: commit `0e8737de` promotes PersonaPlex orchestration into a variant-named
  SML session actor with injected temporal/depformer caches, storage, sampling policy, and
  frame target. The maintained 125-frame CPU comparison produces both 10-second WAVs. The
  29 actual WAV frames match all 232 public Mimi input tokens exactly; output match is
  0.161 overall, 0.192 for the first four codebooks, and 0.608 for text. Audio RMS is
  0.03488/0.07828, active ratio is 0.168/0.480, and best log-energy correlation is 0.3603.
  implication: the maintained architecture and same-WAV input contract are satisfied. The
  only maintained report threshold still failing is first-four output-token match
  0.192 < 0.300, so numeric temporal-transformer isolation remains required.

- timestamp: 2026-07-09T22:35:00-05:00
  observation: scheduled CPU-reference copies and debugger-only EMEL copies from the public
  actor lane show frame-9 temporal layer 0 is bit-for-bit identical at norm1 (4,096 f32),
  QKV projection (12,288), attention pre-output (4,096), attention residual (4,096),
  norm2/gating input (4,096), gating projection (22,528), gated activation (11,264), FF
  update (4,096), and final layer output (4,096).
  implication: temporal layer 0, including its history-dependent attention result and both
  Q4_K projections, is eliminated. A regression against any layer-0 operation would be
  already passing and would not reproduce the remaining fault; isolation must continue at
  layer 1 or later before production changes.

- timestamp: 2026-07-09T22:42:00-05:00
  observation: frame-9 temporal outputs remain bit-exact through layer 25. Layer 26 is the
  first unequal layer: its QKV projection is exact, but attention pre-output differs in
  exactly 128 of 4,096 values, indices 1,152-1,279 (head 9). The head-local maximum is
  0.000115395. Subsequent layer-26 feed-forward work amplifies the difference to a
  0.00133740 maximum at the layer output and layer 31 reaches 0.0398528.
  implication: the first maintained temporal divergence is isolated to layer-26 attention
  head 9 rather than a layer-specific Q4_K projection, gating operation, or residual path.

- timestamp: 2026-07-09T22:44:00-05:00
  observation: layer 26 uses the same causal RoPE/RMSNorm/SiLU block, tensor shapes, and
  Q4_K projection types as neighboring layers; there is no cross-attention, layer scale,
  weights-per-step schedule, or model-specific route. EMEL and reference BF16 score-dot
  disassembly also use the same 16-element f32 multiply and sequential f64 accumulation.
  implication: a hidden layer route, dtype dispatch, and attention-score reduction order
  are eliminated; query rotation, cached K/V, or softmax remained the candidate boundary.

- timestamp: 2026-07-09T22:46:00-05:00
  observation: all ten frame-9 layer-26/head-9 K and V cache rows are bit-exact BF16. The
  post-RoPE query differs by one ULP in 23 of 128 f32 lanes, but only lane 44 crosses a
  BF16 tie: EMEL `0x3f908000` rounds to `0x3f90`, while reference `0x3f908001` rounds to
  `0x3f91`. Replacing that one query BF16 value makes every one of the ten raw score dots
  bit-exact. Before substitution, raw scores differ by at most 0.0102539 and normalized
  weights by at most 0.0000842214.
  implication: RoPE is the earliest causal operand divergence. K/V state, dot accumulation,
  masking, and softmax only propagate the changed BF16 query.

- timestamp: 2026-07-09T22:48:00-05:00
  observation: AArch64 disassembly shows EMEL lowers
  `real * cos - imag * sin` to `fnmul` + `fmadd` and the imaginary result to `fmul` +
  `fmadd`. The reference builds four independent GGML multiply tensors before add/subtract,
  forcing each product to round to f32 first.
  implication: the arithmetic defect is contraction across a reference-mandated f32 round
  point, not a different RoPE frequency, position, pairing, or trigonometric function.

- timestamp: 2026-07-09T22:56:00-05:00
  observation: after routing split-layout timestep RoPE through explicit CPU kernel actor
  transitions with separate f32 product round points, the maintained 30-frame comparison
  matches 232/232 actual-WAV input tokens, 240/240 public output tokens, 120/120 first-four
  tokens, and 30/30 text tokens. EMEL/reference RMS is 0.10287923/0.10287953, both active
  ratios are 0.400, and log-energy correlation is 0.999999957 at zero lag.
  implication: the isolated RoPE defect fully explains the previously failing public-token
  report at this focused horizon. The unchanged 125-frame comparison is now the required
  maintained end-to-end verification rather than another arithmetic search.

- timestamp: 2026-07-09T23:02:00-05:00
  observation: the fresh maintained 125-frame comparison at
  `build/personaplex_compare_organic_rope_fix_10s/personaplex_compare.json` passes every
  unchanged threshold. Actual-WAV input is 232/232, public output is 1,000/1,000,
  first-four output is 500/500, and text is 125/125. EMEL/reference RMS is
  0.078284165/0.078284263, active ratio is 0.480/0.480, and log-energy correlation is
  0.999999967 at zero lag. EMEL took 57.90 seconds versus 259.12 seconds for the pinned
  one-thread CPU reference, a 4.475x reference/EMEL ratio.
  implication: the same-WAV, fixed-seed, CPU-only PersonaPlex comparison is fully passing;
  the earlier 0.192 first-four failure is eliminated without weakening any threshold.

- timestamp: 2026-07-09T23:18:00-05:00
  observation: after rebuilding the maintained runner from the final visible guarded and
  unchecked actor source, the unchanged 125-frame comparison at
  `build/personaplex_compare_organic_rope_guarded_10s/personaplex_compare.json` reproduced
  the exact prior WAV hashes and passed again: 232/232 actual-WAV input tokens, 1,000/1,000
  public output tokens, 500/500 first-four output tokens, and 125/125 text tokens. Audio
  energy correlation is 0.999999967 at zero lag with equal 0.480 activity. EMEL took
  59.20 seconds and the one-thread CPU moshi.cpp reference took 298.05 seconds, so the
  measured reference/EMEL ratio is 5.035x.
  implication: the result is reproducible from the committed architecture rather than a
  stale diagnostic binary, and both audible WAVs are retained with the final report.

- timestamp: 2026-07-09T23:26:00-05:00
  observation: the changed-file gate passed both affected test shards, 96.5% changed-line
  coverage, 58.2% changed-branch coverage, live GGML kernel parity, and cross-process
  determinism. The aggregate command remained nonzero because a stale repository benchmark
  dependency manifest forced a full-suite comparison that lacks the pre-existing
  `sm_scheduler/idle_async` and `sm_scheduler/busy_worker_async` entries, and the global
  lint snapshot has unrelated drift. After local formatting, neither task-owned file is in
  the lint delta; no snapshot was updated.
  implication: all implementation-specific lanes are green, while the remaining gate
  failures are structural repository baselines outside this change.

- timestamp: 2026-07-09T20:15:00-05:00
  observation: the corrected release EMEL lane generated a 10.000-second, mono 24 kHz s16
  WAV from the organic input in 58.90 seconds on one CPU thread. The matching CPU
  moshi.cpp run took 314.356 seconds; EMEL is 5.34x faster for this end-to-end run.
  implication: the native unpacked EMEL path now has both audible output and measured
  single-thread throughput evidence against the same effective Q4_K model class.

- timestamp: 2026-07-09T20:25:00-05:00
  observation: the final scoped gates passed 75 AArch64 tests (8,139 assertions), 90.2%
  changed-line coverage, 58.4% changed-branch coverage, the
  AArch64 benchmark lane, live GGML kernel parity, and cross-process determinism. The only
  final command failure is the repository-wide pre-existing lint snapshot delta; task-owned
  AArch64 guard/SM entries were removed from that delta and the snapshot was not updated.
  implication: implementation-specific verification is green without weakening gates or
  changing the user-controlled lint baseline.

## Eliminated

- hypothesis: the Moshi reference swaps the first two top-k indices before drawing exponentials.
  evidence: `moshi_sample_top_k` calls `ggml_argsort_top_k`, which is a full descending argsort plus a view; the swap exists only in the separate `GGML_OP_TOP_K` CPU kernel.
  conclusion: preserve descending argsort order without the `GGML_OP_TOP_K` swap.

- hypothesis: PersonaPlex depformer requires RoPE.
  evidence: the converted model and source config carry `moshi.lm.depformer.pos_emb =
  "none"`; the reference tensor diagnostic labeled `REF_DEP_Q_ROPE` is the unrotated Q
  tensor for this model.
  conclusion: RoPE remains a kernel primitive selected by model metadata; no PersonaPlex
  depformer RoPE actor or hardcoded route was added.

## Resolution

- root_cause: Moshi's split-layout timestep RoPE used the right frequency, position,
  pairing, trigonometric values, and K/V state, but its action-level expression contracted
  `real*cos - imag*sin` and `real*sin + imag*cos` into AArch64 FMA instructions. The pinned
  GGML reference materializes four independent f32 multiply nodes before add/subtract.
  At frame 9, temporal layer 26, head 9, query lane 44, that missing f32 round point changed
  `0x3f908001` to `0x3f908000`, crossing BF16 `0x3f91` to `0x3f90`. One changed query lane
  altered all ten scores and became the first sampled-token divergence.
- fix: add an explicit guarded split-timestep RoPE route to both CPU kernel actors, keep
  each product's f32 round point, and route temporal query/key rotation through the
  executor's injected kernel actor. Query rotate/copy and key rotate/copy are explicit,
  statically bounded completion phases; success/error choices remain in guards and
  transitions. Model period, tensor dimensions, position, caches, seed, temperatures,
  top-k values, session storage, and frame target remain dependency-injected or event/model
  supplied rather than process-global.
- verification: the captured public kernel actor regression matches both reference output
  bits. The focused 30-frame comparison reaches exact public token/text parity and
  0.999999957 audio-energy correlation. The final rebuilt 125-frame CPU-only report passes with
  exact 232/232 actual-WAV input tokens, 1,000/1,000 public output tokens, 500/500
  first-four tokens, 125/125 text tokens, 0.999999967 energy correlation, equal 0.480
  activity, and nearly identical RMS. It reproduces the prior output hashes from the final
  actor source. Both lanes emit audible 10-second 24 kHz mono WAVs.
- files_changed: CPU kernel timestep-RoPE detail, AArch64/x86 guards/actions/transitions,
  Moshi executor query/key RoPE phases, public kernel actor regression/helpers, and this
  persistent evidence record.
