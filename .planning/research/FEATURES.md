# Feature Research

**Domain:** One maintained Prism ML `Bonsai-1.7B.gguf` 1-bit slice on EMEL's existing generator,
paritychecker, and benchmark surfaces
**Researched:** 2026-04-02
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features EMEL needs if it claims one truthful maintained Bonsai slice.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One documented official Bonsai fixture with provenance | EMEL already treats each maintained model family as one exact `tests/models/` truth anchor, and the live HF repo publishes exactly one GGUF file | LOW | Pin `Bonsai-1.7B.gguf` with stable maintained path, SHA256 `0ae245fc08236af7cb64caff164937e53a8d54af611b0f398cc992c0a5ba70c4`, source repo, and direct download URL. Do not use the model-card quickstart filename as truth; the live file tree disagrees and the file actually published is `Bonsai-1.7B.gguf`. |
| One explicit Bonsai conditioning contract derived from the embedded chat template | The GGUF ships its own `tokenizer.chat_template`, and Bonsai is published as a conversational model, not a raw-prompt-only completion artifact | MEDIUM | The embedded template is Qwen-style: `<|im_start|>...<|im_end|>`, supports `system`, `user`, and `assistant`, and expects `add_generation_prompt`. First maintained scope should use one structured chat-message contract with roles `system,user,assistant`, `tools=none`, and `enable_thinking=false`. |
| Explicit refusal semantics for unsupported Bonsai request shapes | The same embedded template also exposes `tools`, `tool` messages, `<tool_call>`, `<tool_response>`, and `<think>` branches | MEDIUM | Truthful first-slice support requires rejecting unsupported Bonsai features explicitly instead of silently dropping tool calls, replaying thought tags incorrectly, or accepting raw fallback prompts that bypass the maintained contract. |
| Truthful native `Q1_0_g128` runtime support for the exact fixture | Bonsai's differentiator is the custom 1-bit `Q1_0_g128` GGUF operand path; the upstream demo and model card both tie support to Prism's forked runtime kernels | HIGH | This milestone is not honest if EMEL only recognizes the model metadata but cannot execute the maintained file natively. The new feature is the exact 1-bit operand path on the shipped generator lane, not generic Qwen3 loading. |
| Maintained parity proof against the designated reference path | In this repo, correctness claims are maintained only when `tools/paritychecker` proves EMEL against a reference on the same artifact and formatter contract | HIGH | Use the same `Bonsai-1.7B.gguf` fixture and the same Bonsai chat contract on both sides. Because upstream `llama.cpp` does not yet expose the required kernels, the practical reference path is Prism's `llama.cpp` fork, not stock upstream. |
| Regression protection plus benchmark publication for the same slice | Operators expect new maintained slices to preserve old truth anchors and to publish performance only after correctness is proven | MEDIUM | Protect existing Llama/Qwen maintained flows while Bonsai lands, then add one Bonsai benchmark/compare/docs row backed by the same parity-proven fixture and conditioning contract. |

### Differentiators (Competitive Advantage)

Helpful additions that make Bonsai support more auditable and easier to operate, but are not
strictly required to call the first maintained slice complete.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Published formatter-contract metadata on parity and benchmark outputs | Lets reviewers see the exact Bonsai request contract used, instead of guessing whether a run used tool branches, think tags, or raw text | LOW | Reuse the repo's existing formatter-contract publication seam and make the Bonsai row explicit about `source=tokenizer.chat_template`, `tools=none`, and `enable_thinking=false`. |
| Bonsai-specific recommended decode preset | Operators need one documented, reproducible starting point for demos, parity triage, and benchmark interpretation | LOW | The model card recommends `temperature=0.5`, `top-k=20`, `top-p=0.85-0.95`, and `repetition_penalty=1.0`. Publishing one default preset keeps the first maintained slice reproducible without widening the public API. |
| Fixture drift detection against live Hugging Face truth | Bonsai's model card, quickstart snippets, and file tree already diverge on filename, so silent drift is a real risk | MEDIUM | Add checks or documentation that follow the live repo plus executable GGUF metadata, not stale prose. The first concrete example is catching `Bonsai-1.7B.gguf` vs `Bonsai-1.7B-Q1_0_g128.gguf`. |
| Stable three-anchor compare readability | Once Bonsai joins Llama and Qwen, compare output needs to stay human-auditable instead of collapsing into ambiguous "GGUF text model" rows | LOW | Use fixture-specific slugs and published naming that keep Bonsai visually distinct from the existing maintained anchors. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Generic "Bonsai support" or generic "1-bit GGUF support" | It sounds more impressive to say EMEL supports Bonsai or 1-bit models broadly | The live truth source is one file, one family member, and one custom quant path. Broad wording would imply support for sibling sizes, other 1-bit formats, or third-party exports that the milestone does not prove | Keep v1.10 fixed to one maintained `Bonsai-1.7B.gguf` slice only |
| Full embedded-template coverage on day one | The GGUF template visibly supports tools, tool responses, assistant history, and think tags, so implementing "everything in the template" is tempting | That turns a single-slice runtime bring-up into a request-surface expansion milestone and makes parity harder to interpret | Support one narrow structured subset first: `system,user,assistant`, `tools=none`, `enable_thinking=false`; reject the rest explicitly |
| Raw prompt fallback or silent reuse of an unrelated existing formatter | It looks like the fastest path to getting tokens out | It breaks the milestone's "derived from live Bonsai metadata" requirement and can mask real template incompatibilities | Bind Bonsai to one explicit formatter contract sourced from its embedded chat template |
| Claiming stock upstream `llama.cpp` compatibility | Upstream `llama.cpp` is the familiar default reference for GGUF operators | Prism's own demo README states the required Bonsai kernels are not yet in upstream `llama.cpp` or MLX, so claiming generic upstream compatibility would be false today | Tie reference and operator expectations to Prism's `llama.cpp` fork until upstream support actually exists |
| New Bonsai-specific server, OpenAI, tool-use, or agent workflow surfaces in the first slice | The upstream demo shows `llama-server`, browser UI, and tool-aware template branches, so expanding EMEL's API looks adjacent | This widens repo scope far beyond the existing maintained generator/parity/bench acceptance surface | Keep v1.10 on the existing EMEL seams only; defer server/API/tool workflow work to later milestones |
| Publishing benchmark or marketing claims before native `Q1_0_g128` and parity are real | Throughput numbers are easy to demo | Speed claims without the correct operand path and the exact same formatter contract are not truthful support | Land runtime truth and parity first, then publish one benchmark row for the same slice |

## Feature Dependencies

```text
[Official Bonsai fixture with provenance]
    ├──requires──> [Explicit Bonsai conditioning contract]
    │                  └──requires──> [Explicit refusal of unsupported template features]
    └──requires──> [Native Q1_0_g128 runtime support]

[Explicit Bonsai conditioning contract]
    └──requires──> [Maintained parity proof]

[Native Q1_0_g128 runtime support]
    └──requires──> [Maintained parity proof]

[Maintained parity proof]
    └──requires──> [Benchmark compare/docs publication]

[Generic 1-bit or broad Bonsai claims] ──conflicts──> [One maintained Bonsai slice]
```

### Dependency Notes

- **Official Bonsai fixture requires the explicit contract:** the repo cannot define the maintained
  request surface until the exact shipped artifact and its embedded template are pinned.
- **Explicit refusal enhances the contract:** Bonsai's template contains tool and thinking branches,
  so rejecting unsupported request shapes is part of the contract, not optional polish.
- **Native `Q1_0_g128` runtime support is a hard prerequisite for parity:** parity against a
  different operand pipeline would not be truthful Bonsai support.
- **Parity precedes benchmark publication:** EMEL's benchmark claims are only meaningful when they
  describe the same parity-backed fixture and formatter contract.
- **Broad 1-bit claims conflict with the maintained-slice strategy:** widening the promise before
  widening the proof would blur milestone acceptance and operator expectations.

## MVP Definition

### Launch With (v1.10)

These are the features that belong in the Bonsai milestone itself.

- [ ] One official `tests/models/Bonsai-1.7B.gguf` fixture is documented with source, checksum,
      stable maintained path, and direct download URL
      Essential because the milestone needs one reproducible truth anchor and the live HF file tree
      already disagrees with quickstart prose on the filename.
- [ ] One explicit Bonsai structured chat-message contract is documented and used in both EMEL and
      the reference path
      Essential because the GGUF embeds a real chat template and the maintained slice should follow
      that template's supported subset rather than use raw prompting.
- [ ] The maintained contract supports `system,user,assistant` history with
      `add_generation_prompt=true`, `tools=none`, and `enable_thinking=false`
      Essential because this is the smallest operator-facing contract that still matches Bonsai's
      conversational template shape.
- [ ] Unsupported Bonsai request shapes are rejected explicitly
      Essential because the embedded template exposes tool and thinking branches that the first
      maintained slice should not silently half-support.
- [ ] EMEL truthfully accepts, loads, and generates on `Bonsai-1.7B.gguf` through the shipped
      generator path using a native `Q1_0_g128` operand path
      Essential because the milestone is a 1-bit bring-up, not just metadata recognition.
- [ ] `tools/paritychecker --generation` proves EMEL against Prism's reference path for the same
      fixture and the same formatter contract
      Essential because parity is the repo's maintained correctness gate.
- [ ] Regression coverage keeps the shipped Llama and Qwen maintained slices green while Bonsai is
      added, and `tools/bench` publishes one Bonsai compare/docs row only after parity is real
      Essential because a brownfield maintained repo cannot trade existing truth anchors for the
      new one.

### Add After Validation (v2.x)

- [ ] More Bonsai conversation-history coverage and more decode-length cases on the same narrow
      formatter contract
      Add after the first slice is stable and the initial contract is no longer moving.
- [ ] Stronger formatter-drift checks against the live GGUF metadata
      Add after the repo has one stable Bonsai fixture and can afford stricter CI around template
      changes or republished artifacts.
- [ ] More explicit operator presets and docs examples for the Bonsai row
      Add after the parity-backed contract is fixed and benchmark publication is readable.

### Future Consideration (Later Milestones)

- [ ] Sibling Bonsai checkpoints such as `4B` or `8B`
      Defer because each adds a new maintained identity, more fixtures, and more benchmark rows.
- [ ] Tool calling / function calling support from Bonsai's embedded template
      Defer because it expands request/response semantics and validation far beyond first-slice
      runtime truth.
- [ ] Thinking preservation or replay support based on `<think>` / `reasoning_content`
      Defer because it changes formatter semantics, output interpretation, and parity expectations.
- [ ] New Bonsai-specific server, OpenAI-compatible API, or agent workflow surfaces
      Defer because the milestone is scoped to the existing EMEL generator/parity/bench seams.
- [ ] Generic `Q1_0_g128` or arbitrary third-party 1-bit GGUF support
      Defer because one truthful Prism Bonsai slice should be proven before any family-level
      generalization.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Official `Bonsai-1.7B.gguf` fixture with provenance | HIGH | LOW | P1 |
| Explicit Bonsai conditioning contract from embedded metadata | HIGH | MEDIUM | P1 |
| Explicit refusal of unsupported tool/thinking/raw request shapes | HIGH | MEDIUM | P1 |
| Native `Q1_0_g128` runtime support on the shipped generator path | HIGH | HIGH | P1 |
| Maintained parity proof against Prism's reference path | HIGH | HIGH | P1 |
| Regression protection plus benchmark publication for the same slice | HIGH | MEDIUM | P1 |
| Formatter-contract metadata publication | MEDIUM | LOW | P2 |
| Bonsai-specific recommended decode preset | MEDIUM | LOW | P2 |
| Fixture drift detection against live HF truth | MEDIUM | MEDIUM | P2 |
| Stable three-anchor compare readability | MEDIUM | LOW | P2 |
| Broader conversation-history coverage | MEDIUM | MEDIUM | P3 |
| Tool calling / function calling | LOW | HIGH | P3 |
| Thinking replay / preservation | LOW | HIGH | P3 |
| Sibling Bonsai sizes or generic 1-bit support | LOW | HIGH | P3 |
| New server/API surfaces for Bonsai workflows | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Sources

### Official Bonsai Sources

- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf`
- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/tree/main`
- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf`
  Downloaded and SHA256-verified on `2026-04-02`; embedded `tokenizer.chat_template` strings were
  inspected directly from the official GGUF.
- `https://github.com/PrismML-Eng/Bonsai-demo/blob/main/README.md`
- `https://github.com/PrismML-Eng/llama.cpp`

### Repo Sources

- `.planning/PROJECT.md`
- `tools/generation_formatter_contract.hpp`
- `tools/generation_fixture_registry.hpp`
- `tests/models/README.md`

---
*Feature research for: EMEL v1.10 Bonsai `Bonsai-1.7B.gguf` maintained slice*
*Researched: 2026-04-02*
