---
status: resolved
trigger: "Make EMEL PersonaPlex Moshi generate real speech end to end from the same macOS say WAV input as moshi.cpp, with fixed-seed reference comparison showing closely matching audio/token behavior and an output WAV the user can hear. CPU only."
created: 2026-07-09
updated: 2026-07-09
---

# Moshi PersonaPlex Audio Parity

## Symptoms

- expected: EMEL and moshi.cpp consume the same macOS `say` WAV with seed 1234 and produce closely matching PersonaPlex speech audio and token behavior on CPU.
- actual: both CPU lanes now complete, emit non-silent speech WAVs, and follow the same
  51-frame voice plus 12-frame prompt setup. Seeded codec tokens are not sample-identical
  after small arithmetic differences reorder stochastic winners.
- errors: none in the completed EMEL or CPU-only reference run.
- timeline: implementation and E2E comparison are complete; scoped repository gates remain.
- reproduction: run `build/emel_moshi_say_e2e_current` with the converted Mimi, Moshi LM, NATF0 voice, and `build/moshi_e2e/say_input_24k.wav` inputs.

## Current Focus

- hypothesis: resolved. The raw-logit shortcut, process-global RNG, Q4_K accumulation
  grouping, and widened RMS square changed the seeded distribution path.
- test: focused arithmetic/actor regressions plus complete EMEL and Metal-free reference
  runs from the same WAV and seed.
- expecting: matched phase counts and speech-turn behavior with close audio activity/level;
  exact token identity is not expected once a stochastic winner differs.
- next_action: none for the runtime objective. The repository lint snapshot has unrelated
  pre-existing drift and was not updated without user consent.
- reasoning_checkpoint:
- tdd_checkpoint:

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

- root_cause: reference-sensitive sampling and numeric operand order differed at four
  boundaries: raw-logit top-k, process-global RNG state, Q4_K block accumulation, and RMS
  square widening.
- fix: materialize full-card f32 probabilities, use actor-owned Park-Miller state, model
  sampling phases and seed/top-k choices as explicit guarded transitions, and preserve
  GGML Q4_K/RMS arithmetic order. Runtime sampling parameters remain injected; model
  positional/prompt/voice/delay policy remains GGUF-derived.
- verification: complete EMEL and CPU-only reference WAVs, successful `afplay` of both,
  focused sampler/RNG/statechart/Q4_K/RMS/generation regressions, quantitative A/B
  metrics, scoped benchmark, coverage, kernel parity, and determinism. The only global
  gate exception is unrelated lint snapshot drift documented above.
- files_changed: kernel Q4_K/RMS numeric paths, Moshi executor sampling states/actions/
  guards/events/context/detail, focused tests, and this evidence log.
