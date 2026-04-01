# Pitfalls Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained LiquidAI
`LFM2.5-1.2B-Thinking-GGUF` ARM slice
**Researched:** 2026-03-31
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: False Architecture Readiness

**What goes wrong:**
The repo accepts the Liquid fixture by name while runtime/model code still assumes only
`llama`/`qwen3`.

**Why it happens:**
GGUF makes models look interchangeable, so it is tempting to widen an allow-list before the
execution path is real.

**How to avoid:**
Add explicit `lfm2` model and runtime requirements before parity or benchmark work starts.

**Warning signs:**
Architecture gates are widened, but generator/model code still only mentions existing families.

**Phase to address:**
Phase 34

---

### Pitfall 2: Template False Positive

**What goes wrong:**
The maintained path claims Liquid template support by reusing the old Qwen matcher even though the
official Liquid template includes `keep_past_thinking` and different tool markers.

**Why it happens:**
Both templates use ChatML-like framing, so superficial token similarity looks "close enough".

**How to avoid:**
Match one Liquid-specific maintained contract only: structured chat messages, `tools=none`,
`add_generation_prompt=true`, `keep_past_thinking=false`.

**Warning signs:**
Formatter-contract output still says `enable_thinking=false` on a path that never explicitly
resolved Liquid semantics.

**Phase to address:**
Phase 33

---

### Pitfall 3: Metadata Drift

**What goes wrong:**
Planning or docs use stale prose metadata instead of executable GGUF/config truth.

**Why it happens:**
The official model card prose advertises `32,768` context while config/GGUF metadata publish
`128000`, so casual reading produces conflicting assumptions.

**How to avoid:**
Treat GGUF/config metadata as the maintained truth source and document the discrepancy explicitly in
fixture/setup work.

**Warning signs:**
Docs, tests, and tool constants disagree on context length or architecture naming.

**Phase to address:**
Phase 33

---

### Pitfall 4: Silent Quant Scope Creep

**What goes wrong:**
The first Liquid milestone turns into broad quant-matrix work because the official repo publishes
multiple quants.

**Why it happens:**
Once the model page shows `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, and `F16`, it feels
natural to say the repo supports "the model" instead of one exact file.

**How to avoid:**
Pin one official maintained fixture only and state explicitly that sibling quants remain unsupported
until a later milestone proves them.

**Warning signs:**
Requirements or docs say `LFM2.5-1.2B-Thinking-GGUF` without naming one exact file.

**Phase to address:**
Phase 33

---

### Pitfall 5: Benchmark Claims Before Correctness

**What goes wrong:**
The repo publishes Liquid benchmark output before parity and regression prove the same slice.

**Why it happens:**
Benchmark rows are visible and easy to demo, while parity/runtime bring-up is slower.

**How to avoid:**
Keep the existing ordering discipline: runtime truth first, parity and regression second, benchmark
publication last.

**Warning signs:**
Bench case names or docs land before paritychecker accepts the exact same fixture and contract.

**Phase to address:**
Phase 36 and Phase 37

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Alias `lfm2` to `llama` or `qwen3` | Faster first compile | False readiness and brittle runtime behavior | Never |
| Reuse the old Qwen formatter matcher | Avoids adding a new maintained contract | Benchmark/parity claims no longer mean what docs say | Never |
| Say "Liquid support" without naming one file | Simpler docs language | Unbounded scope and unverifiable claims | Never |
| Use prose metadata over GGUF/config truth | Faster research | Docs/tests/tooling drift on context and architecture | Never |
| Skip regression protection for Llama/Qwen | Faster Liquid landing | Existing maintained anchors can silently break | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Liquid Hugging Face repo | Treat all official quants as equivalent scope | Pin one exact file and record provenance |
| Liquid chat template | Assume ChatML-like means Qwen-compatible | Match one Liquid-specific maintained subset only |
| llama.cpp reference | Assume a new reference pin is required | Start from the current repo pin; only upgrade if a concrete `lfm2` blocker appears |
| EMEL parity/bench tooling | Add Liquid as a benchmark-only row | Bring parity and regression up before publication |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Picking a maintained quant with broader runtime needs | New kernel/runtime blockers appear before architecture work is even started | Make the user-selected `Q4_K_M` scope explicit in the milestone and verify it early in Phase 35 | Immediately |
| Publishing a long-context claim from the wrong source | Reported limits drift between docs and runtime | Use GGUF/config metadata as truth | As soon as docs/tests are written |
| Broadening to generic template execution | Parity becomes hard to interpret and debug | Keep one fixed maintained contract | As soon as multiple templates appear |

## "Looks Done But Isn't" Checklist

- [ ] **Fixture support:** verify the milestone names one exact official file, not just a repo.
- [ ] **Architecture support:** verify runtime truth is explicit `lfm2`, not a widened allow-list.
- [ ] **Conditioning support:** verify the maintained formatter contract is Liquid-specific and
      published on setup output.
- [ ] **Parity support:** verify paritychecker uses the same fixture and contract as EMEL.
- [ ] **Benchmark support:** verify benchmark publication happens only after parity is green.
- [ ] **Regression support:** verify prior maintained Llama and Qwen anchors still pass.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| False architecture readiness | HIGH | Re-tighten architecture acceptance, add failing runtime tests, then re-land explicit `lfm2` support |
| Template false positive | MEDIUM | Revert matcher widening, add Liquid-specific contract tests, then re-publish formatter metadata |
| Metadata drift | LOW | Update fixture docs/constants to GGUF/config truth and add regression checks for architecture/context |
| Silent quant scope creep | MEDIUM | Narrow docs/tests back to one file and move sibling quants to future scope |
| Benchmark-first publication | MEDIUM | Remove published Liquid row, finish parity/regression, then re-publish from the proven slice |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| False architecture readiness | Phase 34 | Liquid fixture remains rejected until explicit `lfm2` runtime support exists |
| Template false positive | Phase 33 | Formatter contract output names one Liquid-specific maintained contract |
| Metadata drift | Phase 33 | Fixture docs/tool constants match GGUF/config truth |
| Silent quant scope creep | Phase 33 | Requirements and docs name exactly one maintained file |
| Benchmark claims before correctness | Phase 36 and Phase 37 | Bench row lands only after parity/regression pass |

## Sources

- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/config.json
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/tokenizer_config.json
- https://docs.liquid.ai/lfm/models/lfm25-1.2b-thinking
- https://docs.liquid.ai/deployment/on-device/llama-cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/.planning/PROJECT.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/model/data.cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_formatter_contract.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/parity_runner.cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/generation_bench.cpp

---
*Pitfalls research for: EMEL Liquid LFM2.5-1.2B-Thinking GGUF ARM slice*
*Researched: 2026-03-31*
