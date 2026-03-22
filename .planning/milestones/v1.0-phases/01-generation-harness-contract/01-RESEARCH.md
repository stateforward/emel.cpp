# Phase 1: Generation Harness Contract - Research

**Researched:** 2026-03-07
**Domain:** paritychecker CLI and harness scaffolding for a brownfield Boost.SML inference slice
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

No user constraints - all decisions at Claude's discretion.

Locked constraints already established in `.planning/PROJECT.md`, `.planning/ROADMAP.md`, and
`docs/rules/sml.rules.md`:
- The target model for the first slice is `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
- Acceptance is in `tools/paritychecker/`, not a new public C API example.
- The slice must preserve the RTC actor model and no-queue invariant.
</user_constraints>

<research_summary>
## Summary

Phase 1 should not try to prove generation. It should establish a stable harness contract so later
phases can wire real EMEL loading and generation into an already-valid paritychecker mode. The
existing tool already has a clean mode enum in `tools/paritychecker/parity_runner.hpp`, CLI parsing
in `tools/paritychecker/parity_main.cpp`, and mode dispatch in `tools/paritychecker/parity_runner.cpp`.

The safest brownfield move is to extend those existing seams rather than introduce a second tool or
parallel CLI. Phase 1 should therefore add a `generation` mode, explicit generation-oriented option
fields, deterministic argument validation, and a pinned Llama-68M slice contract. It should stop
short of fake generation logic, because the real load and generate path belongs to Phases 2-4.

**Primary recommendation:** Add a generation CLI and request envelope now, but keep the actual
EMEL load and decode work out of Phase 1 so the harness contract stays clean and auditable.
</research_summary>

<standard_stack>
## Standard Stack

This phase reuses the repo's existing stack rather than introducing anything new.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++20 | repo standard | Tool and runtime implementation | Already enforced by `CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` |
| Boost.SML | pinned in repo | Downstream actor orchestration contract | The harness must prepare for the real actors instead of bypassing them |
| doctest | vendored in repo | Existing paritychecker subprocess regression surface | Later phases rely on the same test harness |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scripts/paritychecker.sh` toolchain | repo standard | Build and run the paritychecker target and tests | Use for every plan verification in this phase |
| `tests/models/README.md` fixture catalog | repo standard | Stable model identity and metadata | Use to pin the first-slice target fixture |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending `tools/paritychecker/` | New demo binary | Violates the agreed acceptance boundary |
| Stable harness contract first | Immediate generation stub | Encourages fake parity without the real EMEL path |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```text
tools/paritychecker/
├── parity_runner.hpp      # mode enum and request options
├── parity_main.cpp        # CLI parsing and usage text
├── parity_runner.cpp      # mode dispatch and harness helpers
└── paritychecker_tests.cpp
```

### Pattern 1: Extend Existing Mode Dispatch
**What:** Add `generation` beside the current tokenizer, GBNF, kernel, and Jinja modes.
**When to use:** When the tool already has a single accepted entry surface.
**Example:** Update `parity_mode` and `parity_options` in `tools/paritychecker/parity_runner.hpp`,
then align parsing in `tools/paritychecker/parity_main.cpp`.

### Pattern 2: Separate Contract From Implementation
**What:** Make the CLI and request envelope explicit before wiring the real EMEL generation path.
**When to use:** When downstream phases need stable inputs and verification, but the real behavior
depends on later actor integration.
**Example:** Validate generation inputs and fixture selection in Phase 1, then add EMEL loader and
generator dispatch in later phases.

### Anti-Patterns to Avoid
- **Reference-first generation stub:** do not route the new mode straight into `llama.cpp` and
  call that "generation parity."
- **Ad hoc temporary flags:** avoid one-off CLI branches that do not map cleanly into later EMEL
  load/generate phases.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Model fixture discovery | New external manifest | `tests/models/README.md` plus explicit pinned path rules | The repo already tracks fixture identity locally |
| Build verification flow | New custom shell flow | `scripts/paritychecker.sh` | Existing standard paritychecker gate already builds and tests the tool |
| Mode routing surface | Secondary CLI wrapper | `parity_mode` + `run_parity(...)` | The current tool already centralizes dispatch there |

**Key insight:** Phase 1 wins by reusing the current tool's seams, not by inventing a second
harness layer.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Making generation mode mean "already generates"
**What goes wrong:** The phase overreaches and mixes CLI contract work with half-wired generation.
**Why it happens:** It is tempting to chase visible output immediately.
**How to avoid:** Keep Phase 1 scoped to CLI contract, fixture selection, and request shaping only.
**Warning signs:** The plan starts modifying EMEL loader or generator internals in Phase 1.

### Pitfall 2: Breaking existing parity modes
**What goes wrong:** Tokenizer, GBNF, kernel, or Jinja parity regress while adding generation mode.
**Why it happens:** Shared CLI or mode-dispatch code is edited without compatibility checks.
**How to avoid:** Preserve current defaults and verify with `scripts/paritychecker.sh`.
**Warning signs:** Existing paritychecker tests fail after enum or parser changes.

### Pitfall 3: Ambiguous fixture selection
**What goes wrong:** The harness accepts "some llama model" instead of the pinned Llama-68M target.
**Why it happens:** The repo contains multiple `.gguf` files under `tests/models/`.
**How to avoid:** Encode one clear first-slice target and deterministic error messaging.
**Warning signs:** The new mode behavior changes depending on directory order or ad hoc file choice.
</common_pitfalls>

## Validation Architecture

Phase 1 can be validated with the existing paritychecker build-and-test path. The target is not a
working generation implementation yet; the target is a stable tool contract that preserves existing
paritychecker behavior.

- Quick verification: `scripts/paritychecker.sh`
- Full verification: `scripts/paritychecker.sh`
- Additional spot checks:
  - `build/paritychecker_zig/paritychecker --help`
  - `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello"`

## Open Questions

1. **What exact Phase 1 behavior should generation mode return before Phase 2 load wiring exists?**
   - What we know: the mode and inputs must be real by the end of this phase.
   - What's unclear: whether the preferred placeholder is a deterministic "not yet wired" error or
     a deeper harness preparation return path.
   - Recommendation: decide during planning and make the behavior explicit in tests or usage text.

2. **Should the first-slice fixture be implicit or explicit on the CLI?**
   - What we know: the roadmap wants an unambiguous Llama-68M contract.
   - What's unclear: whether that should be a dedicated flag, default resolution rule, or stricter
     validation around `--model`.
   - Recommendation: keep the contract explicit in Phase 1 and optimize convenience later.

## Metadata

**Research scope:**
- Core technology: paritychecker CLI and harness structure
- Patterns: brownfield mode extension, fixture pinning, request-envelope staging
- Pitfalls: false generation proof, mode regressions, ambiguous fixture choice

**Confidence breakdown:**
- Harness structure: HIGH - the extension points are already visible in the repo
- Validation path: HIGH - `scripts/paritychecker.sh` already defines the standard check
- Open questions: MEDIUM - the placeholder behavior choice is still a planning decision

**Research date:** 2026-03-07
**Valid until:** 2026-04-06

---
*Phase: 01-generation-harness-contract*
*Research completed: 2026-03-07*
*Ready for planning: yes*
