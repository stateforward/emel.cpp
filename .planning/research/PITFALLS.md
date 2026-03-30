# Pitfalls Research

**Domain:** Adding one maintained Qwen3-0.6B slice to an existing Llama-shaped parity and benchmark
stack
**Researched:** 2026-03-27
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Updating the fixture name while the repo still rejects `qwen3`

**What goes wrong:**
The planning docs say Qwen3 is the next milestone, but maintained generation surfaces still hard
reject any model whose architecture is not `"llama"`.

**Why it happens:**
The fixture constant is visible and easy to change, while the real architecture gates are spread
across paritychecker, bench, and EMEL runtime helpers.

**How to avoid:**
Treat fixture identity, architecture validation, and runtime support as one unit of work. If EMEL
cannot yet run one canonical Qwen3 slice, keep the tool surfaces rejecting it explicitly until that
support lands.

**Warning signs:**
- `tests/models/README.md` adds a Qwen fixture, but `run_emel_validate_architecture()` still
  returns success only for `"llama"`.
- Tool usage text mentions Qwen while generated output still depends on the Llama fixture.
- Only slug/path constants changed.

**Phase to address:**
Phase 26: fixture and architecture gate setup.

---

### Pitfall 2: Treating Qwen3 as a Llama alias

**What goes wrong:**
The tool accepts `qwen3`, but runtime code still assumes the Llama tensor family and fails later or
produces misleading results.

**Why it happens:**
The architecture string looks like the obvious difference, but local reference source shows Qwen3
has distinct attention-normalization tensors such as `attn_q_norm` and `attn_k_norm`.

**How to avoid:**
Make runtime support explicit at the model/execution-view boundary. Do not equate "accepted the
architecture string" with "supports the topology."

**Warning signs:**
- `src/emel/model/data.cpp` or `generator/detail.hpp` still depends entirely on
  `emel::model::llama::detail::*`.
- Qwen3 passes loader validation but fails when building the execution view.
- Debug output shows extra norm tensors present in the model, but EMEL has nowhere to bind them.

**Phase to address:**
Phase 27: runtime architecture bring-up.

---

### Pitfall 3: Shipping a misleading prompt contract

**What goes wrong:**
EMEL benchmarks or parity-checks a Qwen3 instruct model with raw prompt bytes and argmax selection,
then presents that as the canonical Qwen slice.

**Why it happens:**
The current maintained tools already inject `format_raw`, and changing prompt conditioning feels
optional compared with runtime bring-up.

**How to avoid:**
Define one explicit request-conditioning contract up front. If the slice uses chat-template
conditioning, wire that through the formatter/conditioner seam. If it uses a narrower interim
contract, label it clearly and keep reference and EMEL aligned.

**Warning signs:**
- `format_raw` remains the only formatter path in the maintained Qwen flow.
- Official Qwen docs are cited, but the request contract does not use their documented local-use
  flow.
- Generated output includes empty or unstable `<think>` blocks under argmax.

**Phase to address:**
Phase 26: conditioning contract decision.

---

### Pitfall 4: Letting thinking-mode behavior leak into a greedy benchmark path

**What goes wrong:**
Parity or benchmark runs become unstable, repetitive, or hard to interpret because the canonical
request still allows default thinking-mode behavior while the maintained tools select argmax.

**Why it happens:**
Official Qwen docs note that thinking mode is the default and also warn against greedy decoding.
The repo's current maintained generation flow is deterministic and argmax-oriented.

**How to avoid:**
Choose a deterministic non-ambiguous request contract for the maintained slice and apply the same
contract in both EMEL and `llama.cpp`.

**Warning signs:**
- Long or repetitive generations appear on small prompts.
- Bench numbers vary because prompt expansion is inconsistent.
- Operators cannot explain whether the canonical request was thinking or non-thinking.

**Phase to address:**
Phase 26: conditioning contract, then Phase 28: parity validation.

---

### Pitfall 5: Publishing benchmark numbers before parity is real

**What goes wrong:**
The repo lands benchmark rows and docs for a Qwen slice whose correctness, topology support, or
prompt contract is still unresolved.

**Why it happens:**
Benchmark publication is a visible milestone artifact, so it is tempting to refresh it before the
runtime story is complete.

**How to avoid:**
Keep the same discipline as earlier milestones: runtime truth first, parity second, publication
last.

**Warning signs:**
- Compare output changes before `tools/paritychecker --generation` passes on the same fixture.
- Benchmark docs require a long explanation about why they should not yet be read as parity-backed.
- The Qwen row exists, but there is no matching parity regression.

**Phase to address:**
Phase 29: benchmark publication only after Phase 28 is complete.

---

### Pitfall 6: Scope explosion into the rest of the Qwen family

**What goes wrong:**
The milestone grows from one Qwen3-0.6B fixture into Qwen3.5, Qwen3Next, alternative quants, or
MoE variants before the first maintained slice is stable.

**Why it happens:**
Qwen support looks like a family problem, and the reference implementation enumerates many related
architectures nearby.

**How to avoid:**
Use one official Qwen3-0.6B fixture as the only v1.6 truth anchor. Treat every additional model or
quant as future scope unless explicitly promoted later.

**Warning signs:**
- Multiple Qwen fixtures appear in planning before the first one is parity-backed.
- Bench or parity slugs multiply before the first canonical row is stable.
- Requirements start talking about "Qwen support" instead of "one canonical Qwen3-0.6B slice."

**Phase to address:**
Every phase; especially roadmap definition.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Accept `qwen3` at the tool gate but keep Llama runtime assumptions underneath | Makes early CLI smoke tests look better | Produces false readiness and late runtime failures | Never for shipped milestone state |
| Keep `format_raw` temporarily for Qwen bring-up | Avoids deciding prompt conditioning immediately | Makes parity and benchmark claims ambiguous | Only as a clearly labeled local probe, not as the milestone's final contract |
| Use a community quant or local conversion as the canonical fixture | Smaller download and faster iteration | Weakens provenance and makes later comparisons harder to trust | Acceptable only for private experiments |
| Refresh benchmark publication before parity proof | Gives visible progress fast | Creates ungrounded performance claims | Never for milestone completion |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `tests/models/README.md` ↔ tool constants | Documenting one file but benchmarking another | Keep fixture name, README entry, and tool slug/path constants aligned |
| Formatter/conditioner ↔ parity/bench reference path | EMEL and `llama.cpp` use different prompt-conditioning contracts | Choose one canonical contract and apply it in both paths |
| Tool architecture validation ↔ runtime execution view | Letting the tool accept `qwen3` before runtime support exists | Keep the gate narrow and explicit until runtime support is real |
| Qwen fixture bring-up ↔ existing Llama anchor | Breaking the prior canonical slice while widening support | Keep Llama regressions in place while Qwen is added |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Measuring load/provenance work instead of generation | Compare rows reflect setup churn, not runtime behavior | Keep the existing preloaded benchmark discipline and compare the maintained request path only | Immediately when the fixture is much larger than Llama-68M |
| Different prompt token counts between EMEL and `llama.cpp` | Parity fails or bench becomes incomparable | Make prompt conditioning explicit and shared | Immediately once chat-template behavior diverges |
| Publishing numbers from the wrong quant artifact | Apparent speedups or slowdowns are really fixture changes | Lock the official file name and checksum into the canonical slice | As soon as a second GGUF file appears locally |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting an undocumented model download | Operators cannot reproduce or verify the benchmark slice | Record source repo, file name, checksum, and URL in `tests/models/README.md` |
| Treating local-only fixture substitutions as shipped truth | Published results become unreproducible | Keep the official fixture identity explicit in planning and docs |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| "Qwen supported" language without naming the exact slice | Operators assume broader support than the repo actually has | Name the exact fixture and conditioning contract everywhere |
| Benchmark rows that do not make the fixture obvious | Reviewers cannot tell which slice the numbers represent | Use explicit Qwen fixture naming in compare/docs output |

## "Looks Done But Isn't" Checklist

- [ ] **Fixture support:** `tests/models/README.md` names the official Qwen3 file and checksum, not just a local path.
- [ ] **Architecture support:** maintained tools no longer reject `qwen3`, and runtime has real topology support behind that gate.
- [ ] **Conditioning contract:** parity and bench use the same documented request contract, not implicit raw formatting.
- [ ] **Parity proof:** `tools/paritychecker --generation` passes on the canonical Qwen3 slice before any benchmark publication is refreshed.
- [ ] **Publication:** compare/docs output names the same Qwen fixture that parity already proved.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Tool gate updated without runtime support | MEDIUM | Re-tighten the architecture gate, add failing tests, then re-open only the supported slice |
| Wrong prompt contract chosen | MEDIUM | Document the failure mode, pick one new canonical contract, and refresh parity before bench |
| Wrong fixture published | LOW | Correct README provenance, purge the incorrect publication row, and regenerate from the official artifact |
| Scope explosion | LOW | Move extra Qwen work back to v2/out-of-scope docs and re-anchor on the canonical 0.6B slice |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Tool gate still rejects `qwen3` | Phase 26 | Maintained tool usage/help and validation paths accept only the new canonical slice intentionally |
| Qwen3 treated as a Llama alias | Phase 27 | Runtime tests prove the Qwen execution view works and extra norm tensors are handled explicitly |
| Misleading prompt contract | Phase 26 | Planning docs and parity tests name one explicit conditioning contract |
| Thinking-mode leakage into argmax | Phase 26 and Phase 28 | Parity output is stable and the request contract is documented |
| Bench before parity | Phase 29 | No Qwen compare/docs refresh lands before Qwen parity is green |
| Scope explosion | Roadmap definition | Requirements and roadmap stay locked to one official Qwen3-0.6B fixture |

## Sources

- `.planning/PROJECT.md` - milestone scope and narrow acceptance boundary.
- `AGENTS.md` - rules against overstated parity claims and silent fallback.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` - current Llama-only gates and raw formatting path.
- `src/emel/model/data.cpp` and `src/emel/generator/detail.hpp` - current runtime assumptions.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` and `build/paritychecker/_deps/reference_impl-src/src/llama-model.cpp` - local reference evidence that Qwen3 is a distinct architecture.
- https://huggingface.co/Qwen/Qwen3-0.6B - official model card.
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF - official GGUF model card.
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html - official Qwen `llama.cpp` guidance and thinking-mode caveats.

---
*Pitfalls research for: EMEL v1.6 Qwen3-0.6B parity and benchmark*
*Researched: 2026-03-27*
