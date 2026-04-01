# Stack Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice
**Researched:** 2026-03-31
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | `lastModified=2026-03-30T12:55:23Z` | Official maintained GGUF source | This is the official GGUF distribution for the exact milestone target and explicitly points back to `LiquidAI/LFM2.5-1.2B-Thinking` as its `base_model`. |
| `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` | current file in official GGUF repo | Single maintained ARM fixture | The user explicitly wants the milestone anchored on the docs-recommended `Q4_K_M` quant. This turns some additional quant-runtime work into part of the maintained acceptance boundary rather than a later expansion. |
| `ggml-org/llama.cpp` reference pin | `ecbcb7ea9d3303097519723b264a8b5f1e977028` | Parity and benchmark reference | The repo's current pin already contains `LLM_ARCH_LFM2`, `Lfm2ForCausalLM` GGUF conversion support, and `tokenizer_pre == "lfm2"` handling. No reference dependency upgrade is required for this milestone. |
| EMEL internal `lfm2` slice | new repo-local support | Native execution of Liquid hybrid blocks | The official GGUF metadata advertises `general.architecture=lfm2`, not `llama` or `qwen3`. A truthful maintained slice therefore needs a new internal architecture path rather than reusing existing llama/qwen assumptions. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Existing EMEL quantized runtime surfaces plus new `Q4_K_M` bring-up work | current repo + new repo-local support | Native quantized execution for the maintained fixture | The maintained fixture is now `Q4_K_M`, so v1.9 must truthfully include whatever extra quantized runtime support this path requires. |
| Existing EMEL benchmark/parity FetchContent wiring to official `llama.cpp` | current repo | Reference execution and compare publication | Reuse as-is. Only add new Liquid fixture/case names and architecture acceptance. |
| Existing EMEL tokenizer BPE pre-processing mapped through official `llama.cpp` pre-type | current repo | Tokenization compatibility | Reuse the current LLAMA3-equivalent pre-tokenizer path unless parity proves a mismatch. Official `llama.cpp` maps `tokenizer_pre == "lfm2"` onto its LLAMA3 pre-tokenizer branch. |
| New Liquid-specific formatter binding in [`tools/generation_formatter_contract.hpp`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_formatter_contract.hpp) | repo-local addition | Truthful prompt formatting for the maintained slice | Add one explicit binding for the official Liquid primary template subset: BOS `<|startoftext|>`, ChatML framing, `add_generation_prompt=true`, `tools=none`, `keep_past_thinking=false`. Do not widen into generic Jinja/template execution. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| [`tools/paritychecker`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/CMakeLists.txt) | Stored parity against official `llama.cpp` | Keep the existing workflow. Add one Liquid fixture slug/path and architecture acceptance only. |
| [`tools/bench`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/CMakeLists.txt) | Benchmark compare/docs publication | Keep the existing compare/docs pipeline. Add one Liquid maintained case family; do not change benchmark-gate policy. |
| [`tests/models/README.md`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tests/models/README.md) | Fixture provenance ledger | Add the official Liquid repo URL, exact file name, license, download URL, size, and SHA256 after first download. |

## Required Additions In EMEL

| Integration Point | Change Needed | Why It Matters |
|-------------------|---------------|----------------|
| [`src/emel/model/data.cpp`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/model/data.cpp) | Add `lfm2` to supported execution architectures and define an `lfm2` tensor/metadata contract | Current code only accepts `llama` and `qwen3`. The official Liquid GGUF reports `architecture=lfm2`. |
| `src/emel/model/...` and generator compute path | Add a dedicated `lfm2` execution view/runtime slice | LFM2 is a hybrid architecture with mixed conv and full-attention layers, `shortconv_l_cache=3`, tied embeddings, and an output norm tensor named `token_embd_norm`. That is not a llama/qwen drop-in. |
| [`tools/generation_formatter_contract.hpp`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_formatter_contract.hpp) | Add one Liquid formatter contract string and matcher | The current supported contract is Qwen-specific and requires markers such as `enable_thinking` that are not part of the Liquid template. |
| `tools/paritychecker/parity_runner.cpp` | Add one official Liquid maintained fixture path/slug and `lfm2` architecture validation | This keeps parity truthful on one explicit asset instead of broadening to generic Liquid support. |
| `tools/bench/generation_bench.cpp` and [`tools/bench/bench_cases.hpp`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/bench_cases.hpp) | Add one Liquid benchmark case family using the same maintained fixture | This preserves the existing compare/docs workflow with one new case set instead of a new toolchain. |
| [`tests/models/README.md`](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tests/models/README.md) | Record official model provenance for the maintained Liquid fixture | The milestone should have the same provenance discipline as Qwen and Llama. |

## Model Provenance And Fixture Choice

### Canonical maintained asset

Use:

```text
Repo: LiquidAI/LFM2.5-1.2B-Thinking-GGUF
File: LFM2.5-1.2B-Thinking-Q4_K_M.gguf
Stable path: tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
Base model: LiquidAI/LFM2.5-1.2B-Thinking
License: lfm1.0
```

Why this file:

- It is an official LiquidAI GGUF artifact, not a third-party conversion.
- The user explicitly wants the milestone anchored on the docs-recommended quant.
- It keeps the repo honest about the exact operator-facing artifact users are likely to reach for first.
- It makes any additional quant-runtime work part of the stated milestone instead of hidden follow-up debt.

Why not `Q4_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, or `F16` as the maintained fixture:

- `Q4_0` is still out because it is a different quantized contract than the user-selected anchor.
- `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, and `F16` remain sibling variants that should not be implied by proving `Q4_K_M`.

## Architecture Metadata To Treat As Source Of Truth

Use the official GGUF/config metadata, not the prose summary table on the model card, as the implementation source of truth.

| Metadata | Official Value | Implementation Note |
|----------|----------------|---------------------|
| `general.architecture` | `lfm2` | New EMEL architecture path required. |
| Context length | `128000` in official GGUF metadata and `max_position_embeddings=128000` in `config.json` | The model card prose also says `32,768`; treat that prose as stale and prefer GGUF/config metadata. |
| Hidden size | `2048` | Required for runtime sizing. |
| Layers | `16` | Matches the official Thinking card. |
| Layer pattern | `10` conv-style layers and `6` full-attention layers via `layer_types` | This is the critical architectural difference from llama/qwen. |
| Heads | `32` attention heads, `8` KV heads on attention layers | Official llama.cpp conversion sets per-layer KV heads to `0` for conv layers. |
| Rope base | `1000000.0` | Must come from `lfm2.rope.freq_base`. |
| FFN | `block_ff_dim=12288` with auto-adjusted exported feed-forward length | Do not assume llama/qwen FFN metadata naming. |
| Short conv cache | `conv_L_cache=3` | Required for the recurrent/conv side of the hybrid model. |
| Tied embedding | `true` | Output handling must allow the tied-output case. |
| Output norm tensor | `token_embd_norm` naming in official llama.cpp tensor map | Current EMEL output-norm lookup assumes `output_norm.weight`; LFM2 needs an architecture-specific mapping. |

## Tokenizer And Chat-Template Handling

### What to support

- Support the official primary Liquid chat template subset used by the maintained parity and benchmark surfaces:
  - prepend BOS `<|startoftext|>`
  - render ChatML-style messages with `<|im_start|>` / `<|im_end|>`
  - require `add_generation_prompt=true`
  - keep `tools=none`
  - keep `keep_past_thinking=false`
- Treat the formatter contract as model-specific and explicit, the same way the Qwen slice is explicit today.

### What this means for the existing repo

- The current formatter contract is not sufficient for Liquid because it is keyed to Qwen-specific markers and omits the BOS token.
- The new formatter binding should stay narrow and support the maintained surface only: single-turn structured chat messages, no tools, no named templates, no generic template engine.
- Do not add a new tokenizer regex/pre-tokenizer family just for this milestone. Official `llama.cpp` already maps `tokenizer_pre == "lfm2"` to its LLAMA3 pre-tokenizer branch.

## Installation

```bash
# Official maintained fixture
curl -L \
  https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/LFM2.5-1.2B-Thinking-Q4_K_M.gguf \
  -o tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf

# Record provenance after download
shasum -a 256 tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
```

No new external C++ library should be added for this milestone.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Official `Q4_K_M` Liquid fixture | Official `Q6_K` Liquid fixture | Only after the first maintained `lfm2` slice is proven and you want a second quantized acceptance surface. |
| New explicit Liquid formatter binding | Generic Jinja/template execution against arbitrary `tokenizer.chat_template` | Only in a later milestone that explicitly broadens the conditioning surface beyond one maintained Liquid slice. |
| Current pinned official `llama.cpp` reference | Bumping the reference pin | Only if implementation discovers a concrete LFM2 bug missing from `ecbcb7ea9d3303097519723b264a8b5f1e977028`. Current pin already has the needed `lfm2` support. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `LFM2.5-1.2B-Thinking-Q4_0.gguf` as the maintained slice | EMEL does not currently have a native `q4_0` hot path, so this would silently turn the milestone into new kernel work | `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` |
| Generic Liquid-family support (`Base`, `Instruct`, `JP`, `VL`, `Audio`, `MoE`) | This milestone is about one truthful maintained Thinking GGUF ARM slice only | Support only `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` |
| Full tool-use support | The official template supports tools, but the maintained repo surface does not need them for parity/benchmark proof | Keep `tools=none` for this milestone |
| Full multi-turn thinking-history support | The template has `keep_past_thinking`, but the maintained workflow is single-turn and does not need history replay logic | Keep `keep_past_thinking=false` only |
| New MLX, ONNX, Transformers, or vLLM integration | Those are separate deployment stacks and add no value to the existing EMEL parity/benchmark acceptance surface | Reuse the current EMEL runtime plus official `llama.cpp` reference |
| New benchmark tooling or policy | The current compare/docs workflow is already validated | Add one Liquid case family to the existing tooling |

## Stack Patterns by Variant

**If the goal is the maintained milestone slice:**

- Use the official `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` repo.
- Use `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.
- Because this proves the new `lfm2` architecture without broadening quantization scope.

**If the loaded GGUF reports `general.architecture=lfm2`:**

- Route to a new `lfm2` architecture contract, not the existing llama/qwen paths.
- Because official metadata says this model is a hybrid LFM2 family model with conv and full-attention layers.

**If the prompt surface is parity/benchmark generation:**

- Use one explicit Liquid formatter contract based on the official primary template subset.
- Because the current Qwen formatter contract would be a false positive for Liquid.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` | `LiquidAI/LFM2.5-1.2B-Thinking` | Official GGUF repo declares the native Thinking model as `base_model`. |
| `ggml-org/llama.cpp@ecbcb7ea9d3303097519723b264a8b5f1e977028` | Liquid `lfm2` GGUF | This pin already contains `LLM_ARCH_LFM2`, `Lfm2ForCausalLM`, and `tokenizer_pre == "lfm2"` handling. |
| EMEL v1.9 maintained runtime work | Liquid `Q4_K_M` fixture | Required milestone target after user rescope. |
| EMEL current pre-v1.9 native path | Liquid `Q8_0` fixture | No longer the maintained truth anchor for this milestone. |

## Sources

- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
  - Verified official GGUF model identity, license family, and base-model linkage.
- https://huggingface.co/api/models/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
  - Verified `lastModified`, `gguf.architecture=lfm2`, `gguf.context_length=128000`, official chat template, BOS/EOS, and official sibling file list.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/config.json
  - Verified native architecture/config metadata: `model_type=lfm2`, `max_position_embeddings=128000`, `layer_types`, `conv_L_cache`, `rope_theta`, tied embeddings, head counts.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/README.md
  - Verified official model family, Thinking variant identity, ChatML-like template description, tool-use description, and the stale `32,768` prose context claim that conflicts with config/GGUF metadata.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/tokenizer_config.json
  - Verified special tokens: `<|startoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|tool_call_start|>`, `<|tool_response_start|>`.
- https://docs.liquid.ai/lfm/key-concepts/chat-template
  - Verified official ChatML-like formatting and BOS usage.
- https://docs.liquid.ai/lfm/key-concepts/tool-use
  - Verified official tool-use token framing and that tools are optional for the template.
- https://docs.liquid.ai/deployment/on-device/llama-cpp
  - Verified official Liquid guidance that GGUF is the supported llama.cpp deployment format for LFM2.5.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-arch.cpp
  - Verified the repo's current pin already knows `LLM_ARCH_LFM2`.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-vocab.cpp
  - Verified the repo's current pin maps `tokenizer_pre == "lfm2"` onto its LLAMA3 pre-tokenizer path.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/convert_hf_to_gguf.py
  - Verified the repo's current pin already has `Lfm2ForCausalLM` GGUF conversion support and LFM2 GGUF parameter export.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/gguf-py/gguf/constants.py
  - Verified official LFM2 tensor families and naming, including `token_embd_norm` and `shortconv` tensors.

---
*Stack research for: EMEL Liquid LFM2.5-1.2B-Thinking GGUF ARM slice*
*Researched: 2026-03-31*
