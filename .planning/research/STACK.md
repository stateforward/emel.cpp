# Stack Research

**Domain:** Canonical Qwen3-0.6B parity and benchmark slice for EMEL's brownfield generation stack
**Researched:** 2026-03-27
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Official `Qwen/Qwen3-0.6B-GGUF` fixture `Qwen3-0.6B-Q8_0.gguf` | Official Qwen GGUF release | Canonical model artifact for v1.6 | The official GGUF repo gives one clear provenance chain, license, and file identity for the milestone. That is a stronger truth anchor than a community conversion or an ad hoc quant choice. |
| Existing EMEL runtime surfaces in `src/emel/model`, `src/emel/generator`, `tools/paritychecker`, and `tools/bench` | Current repo | Shipped runtime and maintained acceptance boundary | The user asked for parity and benchmark work against Qwen3-0.6B, not a new harness. The existing generator, paritychecker, and compare/docs flow remain the honest boundary. |
| Existing conditioner, formatter, and Jinja stack in `src/emel/text/**` | Current repo | Deterministic prompt conditioning for one canonical request contract | Official Qwen3 guidance routes local use through chat templates and notes thinking-mode caveats. EMEL already has formatter injection and Jinja components, so this seam is the right place to solve conditioning without changing actor structure. |
| `llama.cpp` CPU reference path already vendored through bench/parity builds | Current repo; official Qwen docs say Qwen3 support starts at `b5092` | Reference parity and benchmark comparison | The repo already uses `llama.cpp` as the reference implementation. Official Qwen docs explicitly document Qwen3 usage through `llama.cpp` with `--jinja`, so this remains the correct comparison target. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `doctest` | Current repo | Loader, generator, and parity regression coverage | Use for failing tests that prove one Qwen3 slice works and to keep the milestone honest when runtime changes land. |
| `huggingface-cli` from `huggingface_hub` | Current upstream | Local acquisition of the canonical fixture | Use only to fetch the official artifact into `tests/models/` and verify checksum/provenance. It is not a runtime dependency. |
| Existing benchmark/docs scripts (`scripts/bench.sh`, docsgen flow) | Current repo | Maintained compare publication | Use after parity is real so benchmark claims stay on the existing operator workflow. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Zig C/C++ toolchain | Default build path | No new compiler or runtime stack is needed for v1.6. |
| `ctest` targets `emel_tests` and `lint_snapshot` | Required verification surfaces | Keep the normal repo gates instead of inventing a Qwen-only check path. |
| `scripts/quality_gates.sh` | Required repo gate | Run after implementation changes; keep benchmark drift as warning-only unless the milestone explicitly changes policy. |

## Installation

```bash
# Fetch the official canonical fixture
huggingface-cli download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir tests/models

# Build the maintained surfaces
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER="zig cc" -DCMAKE_CXX_COMPILER="zig c++"
cmake --build build --target emel_tests paritychecker bench_runner
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| One official Qwen3 `Q8_0` fixture from `Qwen/Qwen3-0.6B-GGUF` | Community quantized GGUFs such as `Q4_K_M` conversions | Use community conversions only for local experiments after the official slice is working. They are the wrong truth anchor for milestone acceptance. |
| Keep v1.6 to one Qwen3-0.6B fixture | Broaden immediately to Qwen3.5, Qwen3Next, or MoE variants | Only after one canonical Qwen3 slice is proven on maintained parity and benchmark surfaces. |
| A documented prompt-conditioning contract through EMEL's formatter seam | Continue shipping raw `hello` text via `format_raw` and call it official Qwen behavior | Use raw formatting only as an explicit interim probe. It is not truthful as the milestone's final operator contract for an instruct model. |
| Existing paritychecker and bench surfaces | A benchmark-only or parity-only Qwen harness | Only if the project intentionally widens the acceptance boundary in a later milestone. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Community-converted GGUF as the canonical milestone fixture | Provenance, chat template content, and tokenizer metadata may drift from the official Qwen release | Use `Qwen/Qwen3-0.6B-GGUF` as the v1.6 truth anchor |
| Llama-only architecture guards such as `architecture == "llama"` on the maintained Qwen path | They make the repo look Qwen-ready while still rejecting the actual model family | Replace hard Llama gates only where the canonical Qwen slice is truly supported |
| Raw prompt formatting plus argmax generation presented as the official Qwen contract | Official Qwen docs route local use through chat templates and warn that thinking-mode behavior is a bad fit for greedy decoding | Define one explicit conditioning contract for parity and benchmark and apply it in both EMEL and `llama.cpp` |
| Tool-only Qwen compute or benchmark scaffolding | It would violate the repo rule that `src/` runtime paths are the source of truth | Land runtime support in `src/emel`, then prove it through the existing tools |
| Whole-family Qwen scope in the first milestone | Qwen3 has distinct architecture and conditioning concerns; broad rollout would hide whether the first slice is real | Keep v1.6 to one official Qwen3-0.6B slice |

## Stack Patterns by Variant

**If the GGUF chat template metadata is available in EMEL runtime:**
- Render one deterministic canonical request through the formatter/conditioner seam before tokenization.
- Because official Qwen3 local-use guidance expects chat-template conditioning rather than raw prompt bytes.

**If chat template metadata is not yet populated in EMEL runtime:**
- Treat that as a real milestone gap and either add metadata support or document an explicit interim conditioning contract.
- Because silently falling back to `format_raw` would overstate parity with the official Qwen workflow.

**If the maintained reference path uses `llama.cpp`:**
- Keep the same fixture, prompt-conditioning contract, and CPU-only runtime shape as EMEL.
- Because benchmark and parity claims are only truthful when both sides execute the same operator-facing request contract.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Official Qwen3 model card | Official Qwen3 GGUF model card | Both document `Qwen3-0.6B` / `Qwen3-0.6B-GGUF` as `qwen3` with 28 layers, 16 Q heads, 8 KV heads, and 32,768 context. |
| `llama.cpp` `b5092` or later | Official Qwen3 `llama.cpp` guide | Official Qwen docs state Qwen3 and Qwen3MoE are supported from `b5092`. |
| Current EMEL tokenizer-pre variants (`QWEN2`, `QWEN35`) | Current `llama.cpp` vocab pre-types | Pre-tokenizer coverage is partially present already, but that does not imply architecture/runtime readiness for `qwen3`. |
| Current EMEL formatter injection | Current conditioner/generator initialization | The seam exists today; the work is choosing and implementing the right canonical formatter contract. |

## Sources

- https://huggingface.co/Qwen/Qwen3-0.6B - official model card; verified architecture `qwen3`, 28 layers, 16/8 GQA heads, 32,768 context, and the existence of thinking/non-thinking behavior.
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF - official GGUF model card; verified the official `Qwen3-0.6B-Q8_0.gguf` artifact and `llama.cpp` usage guidance.
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/main - official file listing; verified the single official GGUF file name for the 0.6B repo.
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html - official Qwen docs; verified `llama.cpp` support from `b5092`, `--jinja` usage, and the note that the hard non-thinking switch is not exposed in `llama.cpp`.
- `.planning/PROJECT.md` - v1.6 scope and acceptance boundary.
- `tools/paritychecker/parity_main.cpp`, `tools/paritychecker/parity_runner.cpp`, and `tools/bench/generation_bench.cpp` - current maintained surfaces still lock generation to the Llama fixture and `format_raw`.
- `src/emel/text/formatter/format.hpp`, `src/emel/text/formatter/sm.hpp`, and `src/emel/text/conditioner/**` - existing formatter and conditioner seam available for Qwen prompt conditioning.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` and `build/paritychecker/_deps/reference_impl-src/src/llama-model.cpp` - local reference source shows `qwen3` is a distinct architecture, not a Llama alias.

---
*Stack research for: EMEL v1.6 Qwen3-0.6B parity and benchmark*
*Researched: 2026-03-27*
