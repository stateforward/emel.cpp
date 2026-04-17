# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around explicit Boost.SML orchestration, with
runtime behavior modeled as explicit actors instead of ad hoc control flow. The repo now ships
maintained GGUF generation slices plus one explicit maintained trimodal embedding slice for
`augmem/TE-75M-GGUF`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Latest shipped milestone: `v1.11`

Status: `v1.11` shipped on 2026-04-14 and adds one truthful maintained
`TE-75M-q8_0.gguf` trimodal embedding slice without widening EMEL into a generic multimodal
platform.

Release summary: EMEL now has explicit `omniembed` model acceptance, repo-owned text/image/audio
embedding lanes, one shared normalized embedding contract with supported Matryoshka truncation, and
stored upstream-golden proof plus tiny cross-modal smoke checks for the user-trained TE model.

## Latest Shipped Milestone: v1.11 TE-75M GGUF Trimodal Embedding Runtime

**Shipped:** 2026-04-14

**Delivered:**
- One maintained TE fixture pinned at `tests/models/TE-75M-q8_0.gguf` with explicit provenance
  and checksum tracking.
- Truthful `omniembed` model-family support instead of aliasing TE onto an existing LLM runtime.
- Explicit text, vision, and audio embedding lanes with one shared normalized embedding contract
  and supported Matryoshka truncation.
- Stored upstream-golden proof plus tiny cross-modal smoke checks integrated into the normal repo
  gate flow.

## Previous Milestone: v1.10 Planner Family AGENTS Hard Cutover

<details>
<summary>Implementation complete on 2026-04-05; closeout still pending</summary>

**Goal:** Hard-cut `src/emel/batch/planner` and its child planner-mode submachines over to the
`AGENTS.md` naming, layout, and SML orchestration contract without broadening the change into
unrelated machine families.

**Delivered before closeout:**
- Renamed and reorganized the planner-family files, aliases, events, and states into the canonical
  planner-owned surface.
- Brought planner and planner-mode machines into rule compliance for destination-first transitions,
  event naming, and persistent-state ownership.
- Preserved maintained batching behavior with focused planner-family proof.
- Kept the work bounded to `src/emel/batch/planner` and its child modes.

</details>

## Previous Shipped Milestone: v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice

<details>
<summary>Shipped on 2026-04-02</summary>

**Goal:** Prove one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice through the
existing EMEL generator, paritychecker, and benchmark workflow, with `Q4_K_M` as the maintained
truth anchor and without broadening into generic Liquid-family support.

**Delivered:**
- One official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with reproducible provenance under
  `tests/models/`
- One explicit Liquid request-conditioning contract derived from the official primary chat template
- Explicit `lfm2` runtime bring-up on the maintained generator path
- Maintained parity, regression protection, and benchmark publication for the same slice

</details>

## Requirements

### Validated

- ✓ v1.0 proved one canonical `Llama-68M-Chat-v1-Q2_K.gguf` slice end to end through
  `tools/paritychecker` with real GGUF/model loading, bounded generation, and subprocess coverage.
- ✓ v1.1 added one truthful canonical generation benchmark in `tools/bench`, published through the
  existing compare, snapshot, and docsgen workflow.
- ✓ v1.2 shipped an EMEL-owned flash-attention path plus hard cutover to `emel::tensor::sm` on the
  canonical Llama generation slice.
- ✓ v1.3 shipped optimized AArch64 flash execution and maintained optimized-vs-shared attribution
  on the canonical ARM workload.
- ✓ v1.4 shipped EMEL-owned vectorized AArch64 `q2_K/q3_K/q6_K x q8_K` hot-path kernels and
  maintained `1/10/100/1000` parity on the canonical ARM slice.
- ✓ v1.5 closed the canonical ARM quantized-path contract at
  `native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`
  and restored checked-in flash publication so stored evidence matches live proof.
- ✓ v1.6 documented one official canonical `Qwen3-0.6B-Q8_0.gguf` fixture and bound the maintained
  Qwen path to one explicit GGUF-derived conditioning contract with structured chat-message input
  and no implicit raw fallback.
- ✓ v1.6 added native `src/emel` `q8_0` runtime support plus explicit canonical `qwen3` topology
  handling on the shipped generator path.
- ✓ v1.6 proved maintained canonical Qwen parity against `llama.cpp` and protected the prior
  Llama anchor with stored compare coverage on `1/10/100/1000`, `--dump`, and `--attribution`.
- ✓ v1.6 published one truthful canonical Qwen benchmark compare/docs path aligned with the same
  parity-backed formatter/runtime contract.
- ✓ v1.7 extracted an explicit `src/emel/generator/prefill` orchestration machine for prefill
  slots, snapshot, compute dispatch, and handoff.
- ✓ v1.7 kept prefill request-scoped orchestration data on typed runtime/internal events instead of
  generator context phase fields.
- ✓ v1.7 collapsed the old top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.
- ✓ v1.8 published one truthful maintained Qwen3 executable-size comparison on final linked
  executables for the canonical `hello` -> first-token path.
- ✓ v1.9 documented one official maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture and bound
  the maintained Liquid path to explicit executable metadata truth plus one explicit structured
  chat contract.
- ✓ v1.9 added explicit `lfm2` model/runtime support for one maintained Liquid ARM slice through
  the shipped generator path.
- ✓ v1.9 proved maintained Liquid parity against `llama.cpp`, preserved additive maintained Qwen
  coverage, and published one truthful Liquid benchmark/docs path aligned with the parity-backed
  slice.
- ✓ v1.9 repaired its missing closeout artifacts, validation layer, and milestone bookkeeping so
  the milestone could pass audit and archive cleanly.
- ✓ v1.11 Phase 49 added a repo-owned `embeddings/generator` actor that initializes through the
  maintained conditioner/tokenizer seam and returns normalized TE text embeddings on
  `TE-75M-q8_0.gguf`.
- ✓ v1.11 Phase 49 proved supported Matryoshka truncation at `768/512/256/128`, explicit invalid
  dimension rejection, and callback/error coverage while keeping global quality gates green at
  `90.2%` line coverage.
- ✓ v1.11 Phase 50 derives the maintained TE image contract from the declared
  `mobilenetv4_conv_medium.e180_r384_in12k` encoder family and runs the real `image_encoder.*`
  convolutional tower plus shared `image_projection.*` head natively in `src/`.
- ✓ v1.11 Phase 50 proves normalized `1280`-dimensional image embeddings, supported truncation,
  and explicit malformed-image rejection on the maintained in-memory RGBA path while keeping
  repo-wide quality gates green.
- ✓ v1.11 Phase 51 derives the maintained TE audio frontend from the declared
  `efficientat_mn20_as` encoder family and runs the real `audio_encoder.features.*` tower plus
  shared `audio_projection.*` head natively in `src/`.
- ✓ v1.11 Phase 51 proves normalized `1280`-dimensional audio embeddings, supported truncation,
  and explicit malformed-audio rejection on the maintained mono PCM contract while keeping
  repo-wide quality gates green.
- ✓ v1.11 Phase 52 proves text, image, and audio all land on one deterministic
  `embeddings/generator` result contract with shared normalization, shared truncation behavior,
  and uniform invalid-dimension rejection while keeping repo-wide quality gates green.
- ✓ v1.11 Phase 53 proves the maintained TE anchors against stored upstream text/image/audio
  golden vectors, keeps tiny cross-modal smoke checks in repo-owned doctests, and preserves
  WordPiece parity across the TE and BERT GGUF vocab surfaces while repo-wide quality gates stay
  green.

### Active

- [ ] Decide whether the next maintained embedding milestone should add `TE-75M-q5_0.gguf`.
- [ ] Decide whether EMEL should expose a stable public embedding API for the maintained TE slice.
- [ ] Decide whether to add bounded same-modality embedding batches on the TE path.

### Out of Scope

- `TE-75M-q5_0.gguf` as an additional maintained slice before `TE-75M-q8_0.gguf` is proven
- Generic image/audio file decoding, resampling, or transcoding in the first maintained slice
- Vector-index, retrieval service, reranking, or ANN-search features inside EMEL
- Multimodal generation, captioning, transcription, or chat claims for TE-75M
- Broad `omniembed` or arbitrary embedding-model support beyond one maintained TE-75M slice
- New public C ABI or broad CLI API for multimedia embedding requests before the runtime surface is
  proven

## Next Milestone Goals

- Decide whether the next milestone should add `TE-75M-q5_0.gguf`, a public embedding API, or
  small-batch embedding throughput for the maintained TE path.
- Close out and archive `v1.10` formally so its phase history and audit trail move into
  `MILESTONES.md`.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. The
repo remains governed by `AGENTS.md` and `docs/rules/sml.rules.md`, and the user has now requested
support for their trained `augmem/TE-75M-GGUF` embedding model. Upstream model sources show a
trimodal shared-space design with text, image, and audio encoder lanes projecting into one
1280-dimensional embedding space. The current repo already has mature GGUF ingestion, text request
shaping, and kernel/operator infrastructure, but it does not yet have a maintained `omniembed`
family or `vision` / `audio` encoder actor families.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and `AGENTS.md`; modality routing must stay in
  guards, states, and transitions, not hidden in helper branching.
- **Maintained fixture**: `v1.11` is pinned to `tests/models/TE-75M-q8_0.gguf` for the first
  truthful slice.
- **Input boundary**: Keep the first maintained slice on narrow in-memory image/audio contracts;
  generic media file decoding is out of scope.
- **Proof boundary**: TE behavior must be proven with stored upstream golden embeddings and tiny
  cross-modal smoke checks before any benchmark or support claim is made.
- **Scope boundary**: Stop at feature extraction and shared-space embeddings; do not widen into
  vector search, multimodal generation, or public API commitments.
- **Repository state**: `v1.10` is ready for closeout but not yet archived, so `v1.11` must
  preserve prior phase history while continuing numbering from the next free phase slot.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start decomposition with `generator/prefill` instead of splitting the whole generator at once | Prefill was the clearest request-phase boundary and the largest current duplication source | ✓ Good |
| Collapse the prefill compute-routing matrix before broader extraction | File movement alone would have just relocated the cartesian product instead of reducing it | ✓ Good |
| Keep prefill request-scoped data on typed runtime/internal events | This preserved explicit behavior modeling and avoided context phase flags | ✓ Good |
| Defer decode extraction until the prefill pattern is proven | Decode also owns sampling, rendering, and loop control, so it was the riskier first cut | ✓ Good |
| Defer `attention::any` / `sm_any` extraction until after prefill collapse | Attention mode is only one axis of the duplication and should not hide unresolved top-level routing | ⚠ Revisit |
| Keep v1.9 fixed to one official Liquid Thinking GGUF slice | The repo needs one truthful maintained Liquid anchor before any broader family claims | ✓ Good |
| Use GGUF/config metadata as the maintained truth source for Liquid | Official prose and executable metadata disagree on context length, so docs must follow executable truth | ✓ Good |
| Scope v1.10 to the planner family only | The user asked for planner and planner-submachine compliance, not a broader machine-family rewrite | ✓ Good |
| Pin the first maintained TE slice to `TE-75M-q8_0.gguf` | It keeps the first `omniembed` milestone on the narrowest truthful quant/runtime surface with minimal quality loss | ✓ Locked |
| Treat TE support as explicit `omniembed` model-family work | The Hugging Face GGUF API reports `gguf.architecture=omniembed`; aliasing would overstate support | ✓ Locked |
| Prove TE behavior with upstream golden embeddings and tiny cross-modal smoke checks | The obvious existing generation parity lane does not apply cleanly to `omniembed`, so v1.11 needs its own deterministic proof seam | ✓ Locked |
| Keep generic media decoding, vector search, and public API expansion out of v1.11 | The milestone should stop at one truthful maintained embedding slice, not a broad multimodal platform | ✓ Locked |
| Define the TE proof corpus as deterministic in-memory payload contracts | Phase 47 must anchor one reproducible text/image/audio triplet set before runtime work broadens into encoder execution | ✓ Locked |
| Free `text/encoders` for embedding producers and move tokenizer families under `text/tokenizers` | It keeps tokenizer implementation concerns separate from embedding-capable text producers and preserves room for future hidden-state embedding dispatch | ✓ Locked |
| Keep `embeddings/generator` as the milestone embedding orchestrator | It gives v1.11 an explicit embedding contract without forcing a full generator-domain move in the same milestone | ✓ Locked |
| Validate `omniembed` as a modality-family contract instead of freezing TE-only internals into model acceptance | The user wants general architecture support, and later modality lanes need stable text/image/audio family bindings plus Matryoshka metadata | ✓ Locked |
| Treat `*/forward` as a future reuse seam, not a required modality domain | Every modality has internal forward computation, but a public `forward` domain should only exist when multiple top-level contracts share that hidden-state path | ✓ Locked |
| Vendor the `mdbr-leaf-ir` WordPiece vocab as the maintained text-token truth for TE tests | `TE-75M-q8_0.gguf` omits the tokenizer metadata needed for the text lane, so Phase 49 must pin the upstream vocab asset explicitly instead of inventing one | ✓ Locked |
| Keep Phase 52 on shared proof over the existing embedding actor instead of another orchestration split | The runtime already had one shared publication/truncation surface, so the truthful missing work was contract proof, not more machine churn | ✓ Locked |
| Keep WPM stored-vocab lookup compatible with both raw/`##` and `▁` word-start conventions | The maintained TE vocab and the existing BERT GGUF parity fixture store WordPiece pieces differently, so Phase 53 proof could not regress either truth surface | ✓ Locked |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-14 after v1.11 milestone closeout*
