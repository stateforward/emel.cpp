# Pitfalls Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained Gemma 4 E2B text slice
**Researched:** 2026-04-02
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: False Architecture Readiness

**What goes wrong:**
The repo accepts the Gemma 4 fixture by name while runtime/model code still assumes current
maintained families only.

**Why it happens:**
GGUF makes models look interchangeable, so it is tempting to widen an allow-list before the
execution path is real.

**How to avoid:**
Add explicit `gemma4` model and runtime requirements before parity or benchmark work starts.

**Phase to address:**
Phase 39

---

### Pitfall 2: False Multimodal Claim

**What goes wrong:**
The repo claims Gemma 4 support in general even though the current milestone only proves
text-generation behavior and the official release ships a separate `mmproj` file.

**Why it happens:**
The upstream base model is any-to-any, so it is easy to blur "official file exists" into "repo
supports it."

**How to avoid:**
Keep `v1.11` text only, record the official `mmproj` file as source truth, and reject media/tool
request shapes explicitly.

**Phase to address:**
Phase 38 and Phase 39

---

### Pitfall 3: Template False Positive

**What goes wrong:**
The maintained path claims Gemma 4 template support by reusing a generic text lane even though the
official template includes tools and media surfaces.

**Why it happens:**
Structured chat templates can look "close enough" across model families when only simple prompts
are tested.

**How to avoid:**
Match one Gemma 4-specific maintained contract only: structured text messages,
`add_generation_prompt=true`, no media, no tools.

**Phase to address:**
Phase 38

---

### Pitfall 4: Reference Pin Drift

**What goes wrong:**
The repo gets through runtime bring-up and only then discovers that the pinned `llama.cpp`
reference lane cannot load Gemma 4.

**Why it happens:**
The current pinned reference commit appears not to contain `gemma4`, while upstream current master
does.

**How to avoid:**
Make reference-lane readiness explicit and verify or update the pin before relying on parity and
benchmark comparison.

**Phase to address:**
Phase 41

---

### Pitfall 5: Benchmark Claims Before Correctness

**What goes wrong:**
The repo publishes Gemma 4 benchmark output before reference readiness, parity, and regression
proof cover the same slice.

**Why it happens:**
Benchmark rows are visible and easy to demo, while reference/runtimе parity is slower.

**How to avoid:**
Keep the existing ordering discipline: runtime truth first, reference readiness and parity second,
benchmark publication last.

**Phase to address:**
Phase 41 and Phase 42

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Alias `gemma4` to an existing family | Faster first compile | False readiness and brittle runtime behavior | Never |
| Reuse a generic formatter matcher | Avoids adding a new maintained contract | Parity/benchmark claims no longer mean what docs say | Never |
| Say "Gemma 4 support" without naming one file and boundary | Simpler docs language | Unbounded scope and unverifiable claims | Never |
| Ignore the pinned reference commit until parity phase | Fewer early tasks | Late parity/bench blocker after runtime work is done | Never |
| Treat `mmproj` as supported because the file exists | Faster status update | False multimodal claims and misleading docs | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Official GGUF repo | Treat `Q8_0`, `F16`, and `mmproj` as one maintained scope | Pin one exact text file and record the others as source truth but out of scope |
| Gemma 4 template | Assume "structured chat" means existing contracts already fit | Add one Gemma 4-specific maintained subset only |
| llama.cpp reference lane | Assume the current pin is new enough | Audit the pinned commit explicitly and update if Gemma 4 is missing |
| EMEL parity/bench tooling | Add Gemma 4 as a benchmark-only row | Bring reference readiness and parity up before publication |

## "Looks Done But Isn't" Checklist

- [ ] **Fixture support:** verify the milestone names one exact official file, not just a repo.
- [ ] **Architecture support:** verify runtime truth is explicit `gemma4`, not a widened allow-list.
- [ ] **Conditioning support:** verify the maintained formatter contract is Gemma 4-specific and
      text only.
- [ ] **Scope honesty:** verify `mmproj`, media, and tool-use paths are explicitly out of scope or
      explicitly rejected.
- [ ] **Reference support:** verify the pinned `llama.cpp` lane is Gemma 4-capable.
- [ ] **Parity support:** verify paritychecker uses the same fixture and contract as EMEL.
- [ ] **Benchmark support:** verify benchmark publication happens only after parity is green.

## Sources

- https://huggingface.co/api/models/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/chat_template.jinja
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp
- `tools/paritychecker/reference_ref.txt`
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-model.cpp`
