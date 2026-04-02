# Pitfalls Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained Prism ML
`Bonsai-1.7B.gguf` 1-bit slice
**Researched:** 2026-04-02
**Confidence:** MEDIUM

Confidence is not HIGH because Prism's public materials clearly establish that `Q1_0_g128` is a
vendor-specific 1-bit path, but I did not directly inspect the live Bonsai binary's tensor enum
mapping inside the GGUF file.

## Critical Pitfalls

### Pitfall 1: Provenance Drift Between Repo State, Fixture, and Published Claims

**What goes wrong:**
EMEL lands "Bonsai support" against a moving target. Tests, parity output, benchmark docs, and
requirements stop referring to one frozen artifact and start referring to "the Prism repo" or
"Bonsai 1.7B" generically.

**Why it happens:**
The maintained slice is live on Hugging Face, stored through Xet, and already described across
multiple public surfaces. If the milestone does not freeze one exact file, checksum, download URL,
and upstream commit window, later reruns can silently exercise a different artifact.

**How to avoid:**
Create a fixture-truth phase that records all of the following in one place before runtime work:
- exact file: `Bonsai-1.7B.gguf`
- direct resolve URL
- SHA256 from the live file pointer page
- source repo URL and observed repo commit/date
- stable local path under `tests/models/`

Gate later phases on that exact fixture ID. If the upstream file changes, treat it as a new
milestone input, not an in-place refresh.

**Warning signs:**
- Requirements say `prism-ml/Bonsai-1.7B-gguf` without naming one file.
- Bench or parity output prints a repo slug but not a file identity.
- A refreshed download changes bytes without an explicit milestone decision.

**Phase to address:**
Phase 1: Fixture Provenance and Metadata Truth

---

### Pitfall 2: Filename and Model-Identity Mismatch

**What goes wrong:**
The codebase, docs, or scripts bind to the wrong identity because Prism's public surfaces do not
agree on the file name or quant naming. The Hugging Face Files tab exposes
`Bonsai-1.7B.gguf`, while the model-card quickstart shows
`Bonsai-1.7B-Q1_0_g128.gguf`, and the Hugging Face UI also says it cannot determine quantization
variants.

**Why it happens:**
GGUF naming is advisory, not perfectly machine-parsable, and vendor prose often mixes product
identity with quant identity. Developers reach for the most descriptive-looking filename and end up
encoding an assumption that is not true in the actual repo.

**How to avoid:**
Treat executable truth in this order:
1. live file entry in the Files tab
2. direct download target
3. GGUF metadata inside the maintained file
4. only then model-card prose

Add a milestone requirement that every EMEL-facing surface uses two separate fields:
- `fixture_file = Bonsai-1.7B.gguf`
- `weight_format = Q1_0_g128`

Do not infer one from the other.

**Warning signs:**
- Code constructs the fixture name by appending `-Q1_0_g128`.
- Docs or tests use `Q1_0_g128` as if it were the actual filename.
- Snapshot rows say "Bonsai 1.7B Q1_0_g128" without naming the on-disk file.

**Phase to address:**
Phase 1: Fixture Provenance and Metadata Truth

---

### Pitfall 3: Assuming Prism's `Q1_0_g128` Is Already an EMEL-Supported Quant Path

**What goes wrong:**
The milestone appears to "load" Bonsai, but the maintained hot path is not real. EMEL's current
native quantized contract only treats `q8_0`, `q2_k`, `q3_k`, `q4_k`, and `q6_k` as native
quantized execution types. Prism, meanwhile, documents Bonsai as a custom 1-bit `Q1_0_g128`
format and points users to Prism-specific kernels.

**Why it happens:**
There are two tempting false shortcuts:
- assume Prism's format is equivalent to an existing GGML/EMEL type because it is still "GGUF"
- assume a successful GGUF parse means the generator/runtime path is ready

Those are different questions. EMEL can know a tensor type numerically yet still have no truthful
native execution or audit path for it. In current EMEL code, GGUF type layout parsing already
knows about `TQ1_0`, and the kernel dtype enum also names `tq1_0`, but the maintained
`native_quantized` contract still only treats `q8_0`, `q2_k`, `q3_k`, `q4_k`, and `q6_k` as
native quantized execution types.

**How to avoid:**
Make runtime bring-up explicitly answer three separate questions before parity work starts:
1. What exact GGUF tensor type/layout does the Bonsai file use in practice?
2. Does EMEL map that type to a truthful native execution path?
3. If not, is there an explicitly approved interim mode, or must the milestone stop?

Add a failing runtime/proof test before implementation that asserts Bonsai either:
- produces native-quantized evidence on the maintained path, or
- is rejected with an explicit unsupported-path error.

Do not ship a silent whole-tensor dequantize-to-f32 fallback unless the user approves that
contract as interim.

**Warning signs:**
- Loaders succeed, but `native_quantized` evidence stays at zero or becomes `explicit_no_claim`.
- Code starts widening `is_native_quantized_dtype(...)` without corresponding kernels/tests.
- Developers cite the kernel enum or GGUF type layout as proof of runtime support.

**Phase to address:**
Phase 3: Runtime and Quant Path

---

### Pitfall 4: Reference Comparison Against the Wrong `llama.cpp`

**What goes wrong:**
Parity claims compare EMEL against upstream `ggml-org/llama.cpp`, even though Prism's own docs say
the required 1-bit kernels are not yet in upstream and point to `PrismML-Eng/llama.cpp` instead.
That makes any "mismatch" ambiguous: it may reflect EMEL, the wrong reference lane, or both.

**Why it happens:**
EMEL's current bench/parity tooling is already wired around upstream `llama.cpp`, so reusing that
surface is the path of least resistance.

**How to avoid:**
Create an explicit reference-lane decision in the roadmap before parity implementation:
- either switch the Bonsai reference lane to the designated Prism fork for this milestone
- or state that parity is blocked until a compatible reference lane exists

Also keep the existing repo rule intact: reference code stays quarantined to
`tools/paritychecker` and `tools/bench`, and EMEL's own lane must never bootstrap from reference
objects.

**Warning signs:**
- The parity pin still points only at upstream `ggml-org/llama.cpp`.
- A Bonsai compare run fails before model execution because the reference binary cannot run the
  file truthfully.
- Review comments talk about "Bonsai mismatch" without proving the reference lane is Bonsai-capable.

**Phase to address:**
Phase 4: Parity and Regression

---

### Pitfall 5: Qwen-Like Request Shape Reused Without Bonsai-Specific Conditioning Truth

**What goes wrong:**
EMEL reuses the current Qwen formatter contract because Bonsai is described as Qwen3-based, but
the maintained prompt surface does not actually match Prism's shipped request path. The official
demo scripts force `--reasoning-budget 0`, `--reasoning-format none`, and
`--chat-template-kwargs '{"enable_thinking": false}'`.

**Why it happens:**
Bonsai is close enough to Qwen3 that engineers will assume "existing Qwen path plus one new model"
is sufficient. That is exactly where silent prompt drift starts.

**How to avoid:**
Add a conditioning phase that derives one Bonsai contract from the maintained file and Prism's own
demo behavior. Publish it as a first-class milestone artifact with exact fields such as:
- message shape
- allowed roles
- `add_generation_prompt`
- `enable_thinking`
- whether assistant turns are permitted in the maintained slice

Make parity and benchmark code consume that same contract string, not ad hoc request assembly.

**Warning signs:**
- Bonsai support is described as "uses the Qwen formatter" with no Bonsai contract doc.
- Request builders do not carry `enable_thinking=false` explicitly.
- EMEL and the Prism demo produce different pre-tokenized prompts for the same user message.

**Phase to address:**
Phase 2: Conditioning Contract

---

### Pitfall 6: Template-Matcher Drift Causes Unsupported or Misclassified Bonsai Binding

**What goes wrong:**
EMEL's current maintained formatter binding only accepts a narrow set of known template markers and
rejects any model exposing named chat templates. A Bonsai GGUF that is "basically Qwen" but not
marker-identical can fall into `unsupported_template`, or worse, be force-fit into the wrong
contract by an over-broad matcher.

**Why it happens:**
The current matcher is intentionally conservative. That is correct for maintained truth, but it
means Bonsai bring-up will fail unless the team explicitly decides whether Bonsai maps to the
current Qwen contract or needs its own supported matcher.

**How to avoid:**
Treat template binding as a first-class acceptance item:
- inspect the maintained Bonsai chat template markers
- compare them to EMEL's current supported Qwen contract
- if they differ materially, add a Bonsai-specific maintained contract instead of widening the
  existing Qwen matcher casually

If the file exposes named templates, fail closed and record that as unsupported until the milestone
decides how to handle them.

**Warning signs:**
- `formatter_contract` prints `unsupported_template`.
- Review discussion focuses on token outputs while ignoring the bound template contract.
- A change "fixes" Bonsai by broadening marker matching with no new contract tests.

**Phase to address:**
Phase 2: Conditioning Contract

---

### Pitfall 7: Architecture Reuse Hides Bonsai-Specific Runtime Assumptions

**What goes wrong:**
The milestone widens acceptance to "Qwen3 works" and stops checking Bonsai's actual maintained
shape. Bonsai is advertised as Qwen3-1.7B dense with 28 decoder blocks, GQA 16/8, RoPE, RMSNorm,
and 32,768 context, but the existing maintained Qwen anchor proves a different size and weight
format.

**Why it happens:**
Architecture-family support and maintained-slice support are easy to conflate in a brownfield
runtime.

**How to avoid:**
Keep two separate acceptance questions:
- architecture family recognized: `qwen3`
- maintained Bonsai slice proven: exact file, exact tensor path, exact conditioning contract

Add fixture-specific runtime assertions for Bonsai's declared context length, head/KV layout, and
block count so the milestone cannot pass on a family-level allow-list alone.

**Warning signs:**
- Code review says "already supported because Bonsai is Qwen3-based."
- Tests cover only generic `qwen3` load/shape behavior, not the Bonsai fixture.
- Docs claim Bonsai support without naming Bonsai-specific topology/contract checks.

**Phase to address:**
Phase 3: Runtime and Quant Path

---

### Pitfall 8: Benchmark and Publication Overclaims

**What goes wrong:**
EMEL publishes Bonsai benchmark rows that look impressive but do not mean what readers think.
Prism's model card throughput numbers come from Prism's fork/backends, while EMEL's quality gate
currently soft-fails benchmark regressions and can therefore publish performance evidence before
the Bonsai path is parity-backed.

**Why it happens:**
Bench output is easy to demo, and Bonsai's headline value proposition is speed/size. That creates
pressure to publish numbers before the maintained slice is technically honest.

**How to avoid:**
Make benchmark publication the last phase and require all of the following first:
- exact fixture provenance
- exact conditioning contract
- designated reference lane
- parity green on the same slice
- explicit quantized-evidence statement for EMEL's own lane

Never reuse Prism's published tok/s numbers as EMEL evidence. Publish only EMEL-measured results
with explicit contract strings and quantized-path evidence.

**Warning signs:**
- A benchmark row lands before a Bonsai parity snapshot.
- Docs compare EMEL against Prism's published numbers instead of EMEL's own reproduced run.
- `generation_quantized_evidence` or `generation_formatter_contract` is missing from the bench
  publication path.

**Phase to address:**
Phase 5: Benchmark Publication

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Refer to the maintained slice by repo name only | Faster writing | Non-reproducible parity and benchmark claims | Never |
| Derive file identity from `Q1_0_g128` prose | Simpler naming | Wrong downloads, wrong fixture names, mismatched docs | Never |
| Reuse upstream `ggml-org/llama.cpp` as the Bonsai reference lane | No tool rework up front | Invalid parity evidence for a Prism-specific 1-bit model | Never |
| Alias Bonsai to the current Qwen formatter without a Bonsai contract | Faster first prompt | Silent request drift and meaningless parity | Never |
| Treat a parseable GGUF as a supported runtime path | Faster milestone narrative | False support claims and hidden dense fallback pressure | Never |
| Publish benches before parity/regression | Easy headline numbers | Benchmark overclaims and misleading docs | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Hugging Face Bonsai repo | Assume model-card prose is authoritative for file identity | Freeze the Files-tab entry, resolve URL, and SHA256 first |
| Hugging Face UI metadata | Trust the UI summary alone even when it says quant variants are undetermined | Separate file identity, architecture metadata, and weight-format claims explicitly |
| Prism demo scripts | Copy only prompt text and ignore runtime flags | Mirror the maintained conditioning flags, especially thinking-related settings |
| Prism `llama.cpp` fork | Treat it as optional even though Prism says upstream lacks the needed kernels | Make the reference-lane choice explicit before parity work |
| EMEL formatter binding | Broaden Qwen markers until Bonsai "works" | Add or reject one maintained Bonsai contract explicitly |
| EMEL bench/docs pipeline | Publish a Bonsai row without formatter and quantized-evidence metadata | Require the same evidence strings used by parity-backed maintained runs |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Whole-tensor dequantize-to-f32 fallback sold as Bonsai support | Functional output exists but native quant evidence disappears | Require explicit native-path proof or explicit interim approval | Immediately |
| Repacking or re-quantizing Bonsai weights per request | First-token latency spikes and allocations grow with prompt count | Build one persistent runtime layout per model load, not per dispatch | At the first real end-to-end run |
| Benchmarking against the Prism fork while EMEL uses a different operand class | EMEL looks slower or faster for reasons unrelated to implementation quality | Compare only same-fixture, same-conditioning, same effective operand class | As soon as numbers are published |
| Relying on the current warning-only benchmark gate | Performance drift is visible but not blocking | Add Bonsai publication criteria above the soft bench gate | At publication time |

## "Looks Done But Isn't" Checklist

- [ ] **Fixture truth:** verify `tests/models/` contains one exact `Bonsai-1.7B.gguf` with frozen
      SHA256 and direct download URL.
- [ ] **Identity split:** verify docs and code carry both `fixture_file` and `weight_format`
      separately.
- [ ] **Conditioning contract:** verify parity and bench print one Bonsai formatter contract, not a
      generic Qwen label.
- [ ] **Thinking flags:** verify the maintained path explicitly carries the Bonsai `enable_thinking`
      behavior and related request flags.
- [ ] **Runtime honesty:** verify Bonsai either produces native quantized evidence on EMEL's lane
      or is explicitly marked unsupported/interim.
- [ ] **Reference honesty:** verify the parity reference lane is Bonsai-capable before interpreting
      mismatches.
- [ ] **Regression coverage:** verify maintained Llama and Qwen anchors still pass after Bonsai
      work lands.
- [ ] **Benchmark honesty:** verify published Bonsai numbers come only after parity on the same
      fixture and contract.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Provenance drift | MEDIUM | Re-freeze the artifact, update fixture docs and snapshots, and invalidate any old compare/bench evidence tied to the wrong file |
| Filename/identity mismatch | LOW | Normalize every surface to `fixture_file` plus `weight_format`, then rerun fixture resolution tests |
| Wrong quant-path assumption | HIGH | Add a failing runtime test, remove unsupported claims, then implement or explicitly defer the real operand path |
| Wrong reference lane | HIGH | Re-pin parity tooling to the designated Bonsai-capable reference and discard prior mismatched evidence |
| Conditioning mismatch | MEDIUM | Capture pre-tokenized prompt diffs, codify one maintained contract, and rerun parity on the same request surface |
| Template-matcher drift | MEDIUM | Revert broad matcher changes, add Bonsai-specific contract tests, and fail closed on unsupported templates |
| Architecture reuse overreach | MEDIUM | Restore fixture-specific runtime checks and tighten docs back to one maintained slice |
| Benchmark overclaims | MEDIUM | Remove published Bonsai rows, finish parity/regression, then republish with explicit evidence metadata |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Provenance drift between repo state, fixture, and claims | Phase 1: Fixture Provenance and Metadata Truth | The maintained slice is identified by exact file, SHA256, resolve URL, and stable local path |
| Filename and model-identity mismatch | Phase 1: Fixture Provenance and Metadata Truth | All requirements/docs separate `fixture_file` from `weight_format` |
| Assuming `Q1_0_g128` is already an EMEL-supported quant path | Phase 3: Runtime and Quant Path | Bonsai either emits truthful native-quantized evidence or fails explicitly |
| Reference comparison against the wrong `llama.cpp` | Phase 4: Parity and Regression | The parity lane runs on the designated Bonsai-capable reference path |
| Qwen-like request shape reused without Bonsai truth | Phase 2: Conditioning Contract | EMEL and the reference lane serialize the same maintained Bonsai prompt contract |
| Template-matcher drift causes unsupported or misclassified binding | Phase 2: Conditioning Contract | `formatter_contract` is supported and Bonsai-specific, not generic or unsupported |
| Architecture reuse hides Bonsai-specific runtime assumptions | Phase 3: Runtime and Quant Path | Fixture-specific topology/context assertions pass on the Bonsai file |
| Benchmark and publication overclaims | Phase 5: Benchmark Publication | Published rows include the exact fixture, formatter contract, and quantized-evidence metadata after parity passes |

## Sources

- https://huggingface.co/prism-ml/Bonsai-1.7B-gguf
- https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/tree/main
- https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/blob/main/Bonsai-1.7B.gguf
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/README.md
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/common.sh
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/run_llama.sh
- https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/start_llama_server.sh
- https://github.com/PrismML-Eng/llama.cpp/blob/prism/README.md
- https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/.planning/PROJECT.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/generation_formatter_contract.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/kernel/events.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/kernel/detail.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/model/data.cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/generator/detail.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/.planning/codebase/CONCERNS.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/.planning/codebase/INTEGRATIONS.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/.planning/codebase/TESTING.md

---
*Pitfalls research for: EMEL v2.0 Bonsai 1.7B 1-bit bring-up*
*Researched: 2026-04-02*
