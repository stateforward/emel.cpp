# Stack Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained Gemma 4 E2B text slice
**Researched:** 2026-04-02
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `ggml-org/gemma-4-E2B-it-GGUF` | `sha=4b90c7b785141802608550fc3cd3c715201532e2`, `lastModified=2026-04-02T15:59:21Z` | Official GGUF source | This is the official GGUF distribution for the exact milestone target and points back to `google/gemma-4-E2B-it` as its base model. |
| `gemma-4-e2b-it-Q8_0.gguf` | current file in official GGUF repo | Single maintained text fixture | The official GGUF repo currently ships `Q8_0`, `F16`, and separate `mmproj`. `Q8_0` is the one quantized text fixture that fits the repo's current maintained runtime shape best. |
| Gemma 4 base model config | `model_type=gemma4`, `max_position_embeddings=131072` | Executable model truth | The upstream base model is explicitly `Gemma4ForConditionalGeneration` with multimodal token ids and separate text, vision, and audio configs. |
| EMEL internal `gemma4` slice | new repo-local support | Native execution of Gemma 4 text layers | The official GGUF metadata advertises `architecture=gemma4`, not `llama`, `qwen3`, or `lfm2`, so a truthful maintained slice needs a new internal architecture path. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Existing EMEL `q8_0` runtime surfaces | current repo | Native quantized execution for the maintained fixture | Reuse the current maintained quantized path wherever Gemma 4 topology permits; do not broaden into `F16` or new quant matrices in this milestone. |
| Existing EMEL benchmark/parity FetchContent wiring to official `llama.cpp` | current repo | Reference execution and compare publication | Reuse the existing lane, but promote reference-pin readiness to an explicit requirement because the current pinned commit appears not to contain `gemma4`. |
| Existing EMEL text formatter/conditioner/tokenizer seams | current repo | Text request preparation | Reuse the current maintained structured-chat surfaces and add one explicit Gemma 4 text-only contract derived from the official `chat_template`. |
| Separate `mmproj-gemma-4-e2b-it-f16.gguf` companion file | current official GGUF repo | Future multimodal encoder path | Record it as official source truth, but do not use it in `v1.11`. The current milestone is text only. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `tools/paritychecker` | Stored parity against official `llama.cpp` | Keep the existing workflow. Add one Gemma 4 fixture slug/path and make reference-pin readiness explicit. |
| `tools/bench` | Benchmark compare/docs publication | Keep the existing compare/docs pipeline. Add one Gemma 4 maintained case family only after parity is green. |
| `tests/models/README.md` | Fixture provenance ledger | Add the official GGUF repo URL, exact file name, base model, size, and SHA256 after first download. |

## Required Additions In EMEL

| Integration Point | Change Needed | Why It Matters |
|-------------------|---------------|----------------|
| `src/emel/model/data.*` | Add `gemma4` to supported execution architectures and define a Gemma 4 text contract | Current code explicitly supports maintained families such as `llama`, `qwen3`, and `lfm2`; the new fixture reports `architecture=gemma4`. |
| `src/emel/generator/...` | Add a dedicated `gemma4` execution/runtime path | Gemma 4 has alternating sliding/full attention, shared-KV layers, tied embeddings, and different rope behavior. That is not a drop-in alias of existing families. |
| `tools/generation_formatter_contract.hpp` | Add one Gemma 4 text-only formatter contract string and matcher | The official template supports tools and media. The maintained text slice needs one explicit, auditable subset. |
| `tools/paritychecker/reference_ref.txt` plus parity/bench CMake consumers | Upgrade or confirm the pinned reference lane | The current pinned commit `ecbcb7ea9d3303097519723b264a8b5f1e977028` appears not to contain `gemma4`, while current upstream `llama.cpp` master does. |
| `tools/paritychecker/parity_runner.cpp` | Add one official Gemma 4 maintained fixture path/slug and `gemma4` architecture validation | This keeps parity truthful on one explicit asset instead of broadening to generic Gemma-family support. |
| `tools/bench/generation_bench.cpp` and `tools/bench/bench_cases.hpp` | Add one Gemma 4 benchmark case family using the same maintained fixture | This preserves the existing compare/docs workflow with one new case set instead of a new toolchain. |

## Model Provenance And Fixture Choice

### Canonical maintained asset

Use:

```text
Repo: ggml-org/gemma-4-E2B-it-GGUF
File: gemma-4-e2b-it-Q8_0.gguf
Stable path: tests/models/gemma-4-e2b-it-Q8_0.gguf
Base model: google/gemma-4-E2B-it
Sibling files in official repo: gemma-4-e2b-it-f16.gguf, mmproj-gemma-4-e2b-it-f16.gguf
```

Why this file:

- It is an official `ggml-org` GGUF artifact, not a third-party conversion.
- It is the only quantized text-model file currently exposed in the official GGUF repo.
- It stays close to the repo's existing maintained quantized runtime surface.
- It keeps the milestone honest about the exact operator-facing artifact users can load today.

Why not `F16` or `mmproj` as the maintained fixture:

- `gemma-4-e2b-it-f16.gguf` would widen the milestone into a mixed precision/runtime story that the
  repo does not need for first-slice truth.
- `mmproj-gemma-4-e2b-it-f16.gguf` is a separate multimodal asset and should not be implied by a
  text-generation milestone.

## Architecture Metadata To Treat As Source Of Truth

Use the official GGUF/config metadata, not marketing prose, as the implementation source of truth.

| Metadata | Official Value | Implementation Note |
|----------|----------------|---------------------|
| `general.architecture` | `gemma4` | New EMEL architecture path required. |
| Context length | `131072` | Comes directly from the official GGUF API and base model config. |
| Base model type | `Gemma4ForConditionalGeneration` / `model_type=gemma4` | Upstream model is multimodal; milestone must stay explicit about text-only scope. |
| Text layers | `35` | Required for model/runtime sizing. |
| Layer pattern | Alternating `sliding_attention` with periodic `full_attention` | This is the critical topology difference from the current maintained families. |
| Sliding window | `512` | Needed for correct attention-mask/runtime behavior. |
| Heads | `8` attention heads, `1` KV head, `20` KV-shared layers | Must be represented explicitly in the Gemma 4 model contract. |
| Rope | `10000` for sliding attention, `1000000` proportional rope for full attention | Do not assume a single-family rope contract. |
| Tied embeddings | `true` | Output handling must allow the tied-output case. |
| Media token ids | `image=258880`, `audio=258881`, `video=258884` | Record as source truth so the text-only milestone can reject them explicitly instead of ignoring them accidentally. |

## Tokenizer And Chat-Template Handling

### What to support

- Support one maintained subset of the official Gemma 4 template used by the text slice:
  - structured text chat messages
  - `add_generation_prompt=true`
  - no implicit raw fallback
  - no tool calls
  - no image/audio/video placeholders
- Treat the formatter contract as model-specific and explicit, the same way the maintained Qwen
  slice uses an explicit contract today.

### What this means for the existing repo

- The current formatter contracts are not sufficient for Gemma 4 because the official template
  supports tools and media tokens that the repo does not yet execute.
- The new formatter binding should stay narrow and support the maintained surface only: text
  messages, no tools, no media, no generic Jinja/template execution.
- The milestone should record media token ids and the separate `mmproj` file as official metadata,
  but reject those request shapes on the maintained path.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Official `gemma-4-e2b-it-Q8_0.gguf` | Official `gemma-4-e2b-it-f16.gguf` | Only after the first maintained text slice is proven and you explicitly want a broader precision contract. |
| New explicit Gemma 4 text-only formatter binding | Generic Jinja/template execution against arbitrary `chat_template` | Only in a later milestone that explicitly broadens conditioning and tool/media scope. |
| Explicit reference-pin readiness requirement | Assume current pinned `llama.cpp` ref is enough | Only if the pinned commit is positively verified to load Gemma 4. Current evidence says otherwise. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `mmproj-gemma-4-e2b-it-f16.gguf` in `v1.11` | It implies image/media execution that the repo cannot currently prove | Keep `v1.11` text only |
| Broad "Gemma 4 support" language | It sounds stronger but hides the multimodal boundary | Name one exact maintained text fixture |
| Tool use or media placeholders on the maintained path | The official template allows them, but EMEL does not yet have maintained support | Reject those request shapes explicitly |
| Assuming the pinned `llama.cpp` reference lane already supports Gemma 4 | The current pinned commit appears not to contain `gemma4` | Make ref-pin readiness explicit in the milestone |

## Sources

- https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/processor_config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/chat_template.jinja
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp
- Local ref pin: `tools/paritychecker/reference_ref.txt`
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-model.cpp`
