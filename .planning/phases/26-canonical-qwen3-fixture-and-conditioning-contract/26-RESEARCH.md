# Phase 26: Canonical Qwen3 Fixture And Conditioning Contract - Research

**Researched:** 2026-03-27
**Domain:** Maintained Qwen3 fixture identity, GGUF-driven request conditioning, parity/bench contract publication
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Fixture Truth Anchor
- Treat `tests/models/Qwen3-0.6B-Q8_0.gguf` as the only maintained v1.6 generation fixture, with
  explicit checksum, source, and download provenance in `tests/models/README.md`.
- Update maintained tool-facing fixture identity and help text so drift fails on the old Llama
  anchor instead of silently reusing it.
- Keep the maintained acceptance boundary on the existing paritychecker and benchmark surfaces
  rather than inventing a Qwen-only harness.

### Formatter Input Contract
- Widen the maintained formatter-facing request shape now from one flat prompt string to explicit
  structured chat messages so the primary GGUF chat template can be represented honestly.
- Keep the existing injected formatter seam as the integration boundary for generator and
  conditioner, but update the request payload model as needed to carry structured messages.
- Scope the widening narrowly to the maintained generator, paritychecker, and benchmark surfaces
  required for the canonical Qwen3 slice; do not turn this phase into a broad new product surface.

### Template Source And Selection
- Treat GGUF `tokenizer.chat_template` metadata as the formatter source of truth when it is
  present, rather than hard-coding model-specific prompt text in shared formatter code.
- On maintained paths, honor only the primary GGUF `tokenizer.chat_template`; named template
  variants stay deferred.
- Build formatter behavior behind the existing injected `formatter_ctx + format_fn` seam already
  used by generator and conditioner, rather than adding formatter specialization to the shared
  formatter layer.

### Unsupported Template Behavior
- Classify maintained support explicitly: `no_template`, `supported_contract`, or
  `unsupported_template`. Do not silently fall back to raw formatting when a template is declared.
- Hard-fail maintained paritychecker and benchmark setup when the model declares a primary template
  that the current maintained formatter contract does not support.
- Do not provide a maintained-path raw/manual override for unsupported templates.

### Contract Visibility
- Keep the resolved formatter contract versioned and explicit on maintained proof artifacts so old
  baseline or prompt assumptions fail clearly.
- Publish the resolved formatter contract in maintained operator-facing parity and benchmark output,
  not just in internal binding state.

### the agent's Discretion
- The exact split between tool-local metadata extraction, template validation, and injected
  formatter-adapter ownership can stay local as long as the maintained contract is explicit and
  identical on both EMEL and reference paths.
- The exact structured message payload fields can stay additive as long as they are sufficient to
  represent the primary template honestly and do not introduce implicit behavior.

### Deferred Ideas (OUT OF SCOPE)
- Named-template selection from `chat_template_names` / `chat_template_values`.
- Any maintained raw/manual override path for unsupported templates.
- Full arbitrary `tokenizer.chat_template` rendering via the Jinja parser/formatter stack beyond
  the canonical maintained slice.
- Broader Qwen chat/tool-use product surfaces beyond what the canonical maintained parity/bench
  slice needs.
- Broader Qwen-family support beyond the canonical dense `Qwen3-0.6B-Q8_0.gguf` slice.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FIX-01 | The repo documents one official `Qwen3-0.6B-Q8_0.gguf` fixture with checksum, source, and stable maintained path identity under `tests/models/`. | Local disk verification, maintained tool constant audit, parity baseline fixture contract pattern |
| COND-01 | The maintained Qwen3 slice uses one explicit canonical request-conditioning contract for both EMEL and `llama.cpp`, derived from the primary GGUF `tokenizer.chat_template`, with structured chat-message input and no implicit raw fallback on the maintained path. | Existing `formatter_ctx + format_fn` seam, local GGUF metadata, parity/bench raw-prompt drift audit, explicit contract publication pattern |
</phase_requirements>

## Summary

Phase 26 can stay narrow. The repo already has the right orchestration seam: `generator::sm` injects
`formatter_ctx + format_fn` into `text::conditioner::sm`, and both maintained tools already own the
fixture constants, baseline contract strings, and reference tokenization setup. The current drift is
that all maintained generation paths still pass plain raw strings and hard-code the Llama fixture
identity.

The local canonical fixture provides the truth anchor directly. `tests/models/Qwen3-0.6B-Q8_0.gguf`
exists locally, hashes to `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`,
is `639446688` bytes (`610M`), reports `general.architecture=qwen3`, `tokenizer.ggml.pre=qwen2`,
and exposes exactly one primary `tokenizer.chat_template`. There are no named template variants in
this file, so Phase 26 does not need template-name selection.

**Primary recommendation:** Keep Phase 26 tool-surface scoped: move the maintained fixture anchor to
`Qwen3-0.6B-Q8_0.gguf`, widen request structs to span-based chat messages plus explicit template
options, bind a metadata-derived formatter adapter through the existing injection seam, and publish a
versioned formatter-contract string anywhere parity/bench currently publish fixture/runtime contract.

## Project Constraints (from CLAUDE.md)

- Preserve the RTC/no-queue Boost.SML actor model; no self-dispatch, no queue policies, no hidden async fallback.
- Keep guards pure and actions bounded, allocation-free, and branch-free; model runtime control flow in guards/states, not `if`/`switch` inside actions.
- Do not mirror dispatch-local request data into machine context; propagate per-dispatch data through typed events only.
- Keep public event payloads small and immutable; avoid owning containers in events unless allocation-free dispatch is proven.
- Use the existing `src/` SML machines as source of truth and ask before changing machine structure.
- Treat performance as first-class; no fallback that silently widens hot-path behavior claims.
- Test with doctest/`ctest`; run `scripts/quality_gates.sh` after implementation changes.

## Standard Stack

### Core
| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| `src/emel/text/formatter/format.hpp` | repo HEAD | Stable injected formatter ABI (`void * formatter_ctx`, `format_fn`) | Already used by `generator` and `conditioner`; phase can stay additive |
| `src/emel/text/conditioner/sm.hpp` | repo HEAD | Format + tokenize orchestration | Existing maintained formatting boundary |
| `src/emel/generator/sm.hpp` | repo HEAD | Maintained generation entrypoint | Already threads formatter injection into conditioner |
| GGUF metadata in local fixture | `Qwen3-0.6B-Q8_0.gguf` | Source of truth for architecture and primary chat template | Avoids hard-coded Qwen prompt logic |

### Supporting
| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `tools/paritychecker/parity_runner.cpp` local Jinja helpers | repo HEAD | Existing EMEL/reference Jinja parse/render seam | Reuse for maintained contract adapter, not generic product templating |
| `tools/bench/generation_bench.cpp` GGUF KV extraction | repo HEAD | Existing maintained fixture metadata extraction | Narrowest way to derive contract on bench path without loader refactor |
| doctest + `ctest` | repo HEAD | Unit/subprocess regression coverage | Existing gate for request-shape and tool-surface drift |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Metadata-derived adapter via injected seam | Hard-code Qwen prompt text in shared formatter | Violates locked decision and creates hidden model specialization |
| Span-based chat-message payloads | `std::vector`/`std::string` event payloads | Conflicts with repo event/context rules and raises dispatch-allocation risk |
| Explicit unsupported-template failure | Silent `format_raw` fallback | Recreates the exact drift this phase is meant to stop |

## Architecture Patterns

### Recommended Project Structure
```text
tests/models/README.md                     # fixture provenance ledger
src/emel/text/formatter/format.hpp         # widened request ABI only
src/emel/text/conditioner/events.hpp       # widened prepare request
src/emel/generator/events.hpp              # widened generate request
tools/paritychecker/parity_runner.*        # fixture anchor, metadata extraction, contract publication
tools/paritychecker/parity_main.cpp        # maintained generation help text / CLI parsing
tools/bench/generation_bench.cpp           # maintained fixture anchor + identical reference conditioning
tools/bench/bench_main.cpp                 # operator-facing formatter-contract publication
tests/text/conditioner/text_conditioner_tests.cpp
tests/text/formatter/formatter_tests.cpp
tests/generator/lifecycle_tests.cpp
tools/paritychecker/paritychecker_tests.cpp
```

### Pattern 1: Widen request payloads with non-owning chat-message spans
**What:** Replace flat maintained prompt fields with explicit chat-message spans and explicit template options.
**When to use:** For `generator::event::generate`, `conditioner::event::prepare`, and `formatter::format_request`.
**Why:** Satisfies the approved structured-message contract without violating repo rules on event ownership and context mirroring.
**Recommended shape:**
```cpp
struct chat_message {
  std::string_view role;
  std::string_view content;
};

struct format_request {
  std::span<const chat_message> messages = {};
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
};
```

### Pattern 2: Bind formatter behavior from GGUF metadata at tool setup, not in shared formatter code
**What:** Read `tokenizer.chat_template` during parity/bench fixture setup, classify support, and inject a formatter adapter via `formatter_ctx + format_fn`.
**When to use:** Maintained paritychecker and bench generation setup for the canonical Qwen3 fixture.
**Why:** Current repo search shows `model::data` has chat-template storage fields but the maintained load/setup paths do not yet populate or use them. Tool-local extraction is the narrowest truthful Phase 26 change.

### Pattern 3: Publish formatter contract where fixture/runtime contracts are already published
**What:** Add formatter-contract identity alongside existing `fixture`, `contract`, and runtime-contract output.
**When to use:** Parity baseline files, `generation parity ok` output, and `tools/bench` compare headers.
**Why:** Existing proof surfaces already fail clearly on contract drift; formatter drift should use the same mechanism.

### Anti-Patterns to Avoid
- **Shared formatter hard-coded for Qwen:** Locked out by context; keep shared formatter ABI generic.
- **Event payloads with owning containers:** Repo rules push toward `std::span`/`std::string_view`.
- **Updating only EMEL conditioning:** Bench/parity reference paths currently tokenize raw strings too; both sides must use the same resolved contract.
- **Implicit template defaults:** The Qwen template branches on `tools`, `messages`, `add_generation_prompt`, and `enable_thinking`; maintained paths must set explicit values, not rely on omission.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Template source discovery | Hard-coded prompt strings or string-matched architecture branches | GGUF `tokenizer.chat_template` + explicit support classifier | Keeps truth tied to the fixture, not code guesses |
| Formatter integration | New formatter actor/product layer | Existing `formatter_ctx + format_fn` seam | Phase stays additive and brownfield-safe |
| Arbitrary Jinja implementation | New custom renderer/parser | Existing repo Jinja parser/formatter helpers where needed | Repo already carries this machinery; Phase 26 only needs one maintained slice |
| Drift publication | Ad hoc logs | Existing parity baseline / bench contract output pattern | Makes old fixture/prompt assumptions fail clearly |

**Key insight:** The hard part here is not rendering one template string. It is keeping fixture identity, request shape, EMEL conditioning, reference conditioning, and operator-facing proof artifacts on the same declared contract.

## Common Pitfalls

### Pitfall 1: Widening to chat messages with owning containers
**What goes wrong:** `std::vector` or `std::string` lands in public request events.
**Why it happens:** Structured chat APIs often default to owning containers.
**How to avoid:** Use non-owning `std::span<const chat_message>` and `std::string_view` fields only.
**Warning signs:** New event/context fields start storing message arrays or copied prompt text.

### Pitfall 2: Assuming local `model::data` already carries chat-template metadata
**What goes wrong:** Planner schedules a loader-wide metadata refactor inside Phase 26.
**Why it happens:** `model::data.hpp` already has `chat_template*` fields, but current repo search shows no maintained population/use path.
**How to avoid:** Keep Phase 26 on tool-local GGUF KV extraction plus injected adapter binding.
**Warning signs:** Work expands into shared loader/state-machine changes before fixture/contract drift is fixed.

### Pitfall 3: Leaving maintained paths on raw prompt tokenization
**What goes wrong:** EMEL or reference side still tokenizes `hello` directly while claiming Qwen template support.
**Why it happens:** Both `parity_runner.cpp` and `generation_bench.cpp` currently call `format_raw` and raw `llama_tokenize`.
**How to avoid:** Route both sides through the same resolved formatter contract before tokenization/generation.
**Warning signs:** `opts.text` / `spec.prompt` still flow straight into `event::generate` or `llama_tokenize`.

### Pitfall 4: Relying on implicit Qwen template defaults
**What goes wrong:** Output drifts when `enable_thinking` or `add_generation_prompt` default behavior changes.
**Why it happens:** The local primary template contains explicit branches for those flags.
**How to avoid:** Freeze one maintained formatter-contract version with explicit values for all optional knobs.
**Warning signs:** Contract publication omits option values or setup code leaves them unset.

## Code Examples

Verified repo patterns to reuse:

### Existing formatter injection seam
```cpp
// Source: src/emel/generator/actions.hpp
emel::text::conditioner::event::bind bind_ev{ctx.model->vocab_data};
bind_ev.formatter_ctx = ctx.formatter_ctx;
bind_ev.format_prompt = ctx.format_prompt;
ev.ctx.phase_accepted = ctx.conditioner->process_event(bind_ev);
```

### Existing formatter call site in conditioner
```cpp
// Source: src/emel/text/conditioner/actions.hpp
emel::text::formatter::format_request request = {};
request.output = ev.ctx.formatted;
request.output_capacity = ev.ctx.formatted_capacity;
request.output_length_out = &ev.ctx.formatted_length;
ev.ctx.format_accepted = ctx.format_prompt(ctx.formatter_ctx, request, &err);
```

### Existing explicit contract publication pattern
```cpp
// Source: tools/paritychecker/parity_runner.cpp
std::fprintf(file, "format=%.*s\n", ...);
std::fprintf(file, "contract=%.*s\n", ...);
std::fprintf(file, "fixture=%s\n", k_generation_fixture_name);
```

## State of the Art

| Old Approach | Current Recommended Approach | When Changed | Impact |
|--------------|------------------------------|--------------|--------|
| Maintained generation uses flat raw prompt strings and Llama fixture constants | Maintained generation uses one Qwen fixture anchor plus explicit metadata-derived formatter contract | Phase 26 | Prevents silent Llama/raw drift |
| Shared formatter boundary only carries `input` text | Shared formatter boundary carries non-owning structured chat messages plus explicit options | Phase 26 | Makes primary GGUF template representable without product-surface explosion |
| Proof artifacts publish runtime contract only | Proof artifacts publish runtime contract and formatter contract | Phase 26 | Makes prompt-conditioning drift visible and reviewable |

**Deprecated/outdated:**
- Raw maintained generation conditioning on `format_raw` for the canonical Qwen slice.
- Llama-specific maintained fixture help text in paritychecker/bench generation surfaces.

## Open Questions

1. **Which optional Qwen template knobs become part of the maintained contract string?**
   - What we know: The local primary template branches on `tools`, `messages`, `add_generation_prompt`, and `enable_thinking`.
   - What's unclear: Whether v1.6 wants `enable_thinking=true` or `false` for its single maintained slice.
   - Recommendation: Freeze it explicitly in the Phase 26 contract string and require parity/bench setup to pass that exact value on both EMEL and reference paths.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest via CMake/CTest |
| Config file | `CMakeLists.txt` + `tools/paritychecker/CMakeLists.txt` |
| Quick run command | `./build/coverage/emel_tests_bin --test-case=*conditioner*,*formatter*,*generator*` |
| Full suite command | `scripts/quality_gates.sh` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FIX-01 | Maintained fixture provenance and tool anchor fail on old Llama identity | subprocess / unit | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ |
| COND-01 | Structured-message contract flows through formatter -> conditioner -> generator and maintained tools publish the same resolved contract | unit + subprocess | `./build/coverage/emel_tests_bin --test-case=*conditioner*,*formatter*,*generator*` and `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ |

### Sampling Rate
- **Per task commit:** `./build/coverage/emel_tests_bin --test-case=*conditioner*,*formatter*,*generator*`
- **Per wave merge:** `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- **Phase gate:** `scripts/quality_gates.sh`

### Wave 0 Gaps
- None in framework/infrastructure. Existing doctest + paritychecker test targets are already present.
- Required new coverage is behavioral, not infrastructural: add structured-message formatter cases, generator/conditioner contract flow cases, and paritychecker fixture/contract drift cases in existing test files.

## Sources

### Primary (HIGH confidence)
- Local repo: `src/emel/generator/events.hpp`, `src/emel/generator/actions.hpp`, `src/emel/text/conditioner/events.hpp`, `src/emel/text/conditioner/actions.hpp`, `src/emel/text/formatter/format.hpp`, `tools/paritychecker/parity_runner.cpp`, `tools/bench/generation_bench.cpp`, `tools/bench/bench_main.cpp`, `tests/text/conditioner/text_conditioner_tests.cpp`, `CMakeLists.txt`
- Local fixture inspection: `tests/models/Qwen3-0.6B-Q8_0.gguf` via Python `gguf` reader and local SHA256
- Official Qwen GGUF repo: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF

### Secondary (MEDIUM confidence)
- Official llama.cpp templates guidance: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - almost entirely repo-local and fixture-local
- Architecture: HIGH - existing seams and current drift were verified directly in code
- Pitfalls: HIGH - derived from repo rules plus current maintained-path raw-prompt behavior

**Research date:** 2026-03-27
**Valid until:** 2026-04-03
