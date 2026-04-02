# Stack Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained Prism ML `Bonsai-1.7B.gguf` 1-bit slice
**Researched:** 2026-04-02
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `prism-ml/Bonsai-1.7B-gguf` | `sha=c89c1b5578286827264c4217f40edee617f4f904` | Official maintained model source | This is the published Hugging Face repo for the exact milestone target. It exposes one GGUF sibling file, `Bonsai-1.7B.gguf`, and its API reports `gguf.architecture=qwen3`. |
| `Bonsai-1.7B.gguf` | current published file, `248302272` bytes | Single maintained fixture | The repo tree and direct resolve URL agree on this filename even though the README quickstart still uses `Bonsai-1.7B-Q1_0_g128.gguf`. Use the live artifact, not stale prose, as truth. |
| Existing EMEL GGUF loader | current repo | Container and metadata parsing | Bonsai is still a GGUF model. Do not add a Prism-specific container format or alternate model loader; extend the existing GGUF/tensor-type handling only where the new quant type demands it. |
| Existing EMEL `qwen3` model topology | current repo + Bonsai-specific validation | Architecture/runtime contract | The published Bonsai artifact is not a new execution architecture. It is a `qwen3` dense model with a new weight encoding. Reuse the existing Qwen3 topology path instead of inventing a `bonsai` or `prismml` architecture family. |
| Native EMEL `Q1_0_g128` operand path | new repo-local support | Truthful execution of Bonsai weights | The official Bonsai model card says embeddings, attention projections, MLP projections, and LM head are stored in GGUF `Q1_0_g128`. Supporting Bonsai honestly therefore requires a native EMEL tensor type, storage layout, and hot-path kernels for that operand class. |
| `PrismML-Eng/llama.cpp` reference lane | default branch `prism`; pin exact commit for milestone | Parity and benchmark reference | The official Bonsai README points users to Prism's `llama.cpp` fork for `Q1_0_g128` kernels. Upstream `ggml-org/llama.cpp` does not advertise or expose `Q1_0_g128` in its README or `ggml-quants` sources, so it is not the truthful Bonsai reference today. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Existing EMEL `qwen3` loader/runtime surfaces | current repo | Reuse Bonsai's architecture contract | Use for model metadata, tensor naming, block count, RoPE, RMSNorm, and tokenizer family handling because the artifact reports `general.architecture=qwen3`. |
| New Bonsai formatter binding in `tools/generation_formatter_contract.hpp` | repo-local addition | Explicit request-conditioning contract | Use when the loaded artifact is the maintained Bonsai fixture. The published chat template is Qwen-like, but it is not identical to the repo's current supported Qwen contract matcher and includes an assistant `<think>` preamble on generation. |
| Existing parity/bench FetchContent wiring retargeted to Prism fork | current repo + new pin | Truthful reference execution | Use in `tools/paritychecker` and `tools/bench` only. Keep EMEL runtime native; the fork is for comparison truth, not for product execution. |
| `curl` + `shasum -a 256` or `sha256sum` | system tools, no new dependency | Download and provenance capture | Use for fixture fetch and checksum recording. This milestone does not require adding `huggingface_hub`, Python tooling, or a new asset downloader. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `curl -I -L` | Pre-download provenance check | Use the official resolve URL to record `x-repo-commit`, `x-linked-size`, and the final served filename before first download. |
| `curl -L` | Fixture download | Download the single maintained asset directly from Hugging Face into `tests/models/Bonsai-1.7B.gguf`. |
| `shasum -a 256` / `sha256sum` | Post-download provenance | Record the actual local file checksum in `tests/models/README.md`. Treat the local SHA256 as the repo's checksum source of truth. |
| Existing `FetchContent` reference wiring | Reference fork pinning | Retarget `tools/paritychecker` and `tools/bench` from upstream `ggml-org/llama.cpp` to `PrismML-Eng/llama.cpp`, then pin an exact commit before claiming parity or benchmark truth. |

## Required Additions

| Integration Point | Change Needed | Why It Matters |
|-------------------|---------------|----------------|
| `src/emel/kernel/detail.hpp` | Add a native tensor dtype and block layout for `Q1_0_g128` | EMEL currently defines quant layouts for `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q6_k`, and related prepared/packed variants. Bonsai needs a project-owned `Q1_0_g128` representation before any truthful runtime support exists. |
| `src/emel/kernel/aarch64/actions.hpp` and associated kernel dispatch surfaces | Add native hot-path kernels for `Q1_0_g128` weights against EMEL's activation/input format | The milestone target is a maintained ARM/runtime slice. A Bonsai claim is not honest if EMEL falls back to whole-tensor dequantize-to-f32 in the hot path. |
| `src/emel/generator/detail.hpp` | Add Bonsai quantized-route preparation, workspace sizing, and dispatch accounting | Existing generator quant routing is built around `q8_0`, `q4_k`, and `q6_k` families. Bonsai needs the same route-selection, packing/preparation, and audit coverage for `Q1_0_g128`. |
| `src/emel/model/llama/detail.hpp` and parity audit reporting | Extend quantized-path auditing to classify Bonsai stages truthfully | The Bonsai README says 1-bit coverage includes token embeddings and LM head, not only inner projection weights. Audit output must reflect native support or explicit non-support stage by stage. |
| `tools/generation_formatter_contract.hpp` | Add a Bonsai-specific supported contract matcher and formatter | The published Bonsai `tokenizer.chat_template` is not a drop-in match for the current supported Qwen contract markers. It needs its own explicit maintained contract rather than a silent fallback or generic template engine. |
| `tools/generation_fixture_registry.hpp` | Add or replace the current publication fixture with `Bonsai-1.7B.gguf` | The milestone is one maintained Bonsai slice. The maintained fixture registry must point tools at the exact official file and slug. |
| `tests/models/README.md` | Add Bonsai provenance entry with source, stable path, direct download URL, byte size, and SHA256 after download | The repo already treats `tests/models/README.md` as the fixture ledger. Bonsai should follow the same provenance contract as Qwen and LFM2. |
| `tools/paritychecker/CMakeLists.txt`, `tools/bench/CMakeLists.txt`, `tools/paritychecker/reference_ref.txt`, `tools/bench/reference_ref.txt` | Retarget the reference lane to `PrismML-Eng/llama.cpp` and pin an exact ref | Upstream `ggml-org/llama.cpp` is not the truthful Bonsai comparison target while `Q1_0_g128` remains fork-only. |
| `tools/paritychecker/parity_runner.cpp` and bench case selection | Keep Bonsai on the existing `qwen3` architecture path while switching the fixture and formatter contract | Bonsai should ride the existing Qwen3 architecture acceptance surface. The new work is the quant path and prompt contract, not a new architecture family. |

## Fixture Provenance Contract

Use this exact maintained identity:

```text
Repo: prism-ml/Bonsai-1.7B-gguf
File: Bonsai-1.7B.gguf
Stable path: tests/models/Bonsai-1.7B.gguf
Download URL: https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf
License: Apache-2.0
Published repo commit at research time: c89c1b5578286827264c4217f40edee617f4f904
Published file size at research time: 248302272 bytes
```

Why this matters:

- The live repo publishes exactly one GGUF file, so there is no reason to widen the milestone into multi-fixture support.
- The README quickstart examples use a stale filename. The resolve URL and file tree are the executable truth and should drive the maintained fixture path.
- Hugging Face response headers already expose useful provenance before the download finishes, but the repo should still record a local SHA256 after the first download.

## Optional Additions

These may help, but they are not required for the milestone.

| Optional Item | Why Optional | When to Use |
|---------------|--------------|-------------|
| `hf` CLI / authenticated Hugging Face token flow | Direct `curl` works for the maintained asset already | Only add if CI or repeated large-file fetches start hitting rate limits or resume problems. |
| Small provenance helper script around `curl -I -L` + checksum capture | Nice for operator ergonomics, not a runtime dependency | Only if the team wants to automate fixture ledger updates after the first manual proof. |

## Installation

```bash
# Inspect remote provenance before download
curl -I -L \
  https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf

# Download the maintained fixture
curl -L \
  https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf \
  -o tests/models/Bonsai-1.7B.gguf

# Record the local checksum
shasum -a 256 tests/models/Bonsai-1.7B.gguf

# Linux spelling if preferred
sha256sum tests/models/Bonsai-1.7B.gguf
```

No new external C++ library should be added for this milestone. Keep the existing Zig-based build/toolchain and existing EMEL GGUF infrastructure.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `PrismML-Eng/llama.cpp` in `tools/` only | `ggml-org/llama.cpp` as Bonsai reference | Only after upstream officially lands `Q1_0_g128` support and the maintained Bonsai artifact runs there without Prism-specific patches. |
| Existing EMEL `qwen3` architecture path | New `bonsai` or `prismml` model architecture family | Only if a future published Bonsai artifact stops reporting `general.architecture=qwen3`. |
| Explicit Bonsai formatter contract | Generic Jinja/template execution against arbitrary `tokenizer.chat_template` | Only in a later milestone that explicitly widens conditioning beyond one maintained Bonsai slice. |
| Native `Q1_0_g128` hot path | Whole-tensor dequantize-to-f32 fallback | Only with explicit user approval as an interim milestone, clearly labeled as interim and non-parity-complete. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `ggml-org/llama.cpp` as the maintained Bonsai reference today | Upstream README and `ggml-quants` sources do not expose `Q1_0_g128`, while Prism's official Bonsai docs explicitly point to Prism's fork | `PrismML-Eng/llama.cpp` in `tools/paritychecker` and `tools/bench` only |
| A new `bonsai` or `prismml` execution architecture in EMEL | The published GGUF metadata says `general.architecture=qwen3` | Reuse the existing Qwen3 topology and loader path |
| Generic 1-bit or arbitrary third-party GGUF support | That widens the milestone beyond one truthful maintained fixture | Support only the official `Bonsai-1.7B.gguf` slice |
| Generic Jinja/template execution or tool-calling implementation | The published Bonsai template contains tools support, but the milestone acceptance surface does not require tools and should stay narrow | One explicit Bonsai formatter contract with `tools=none` |
| MLX, Swift, Android, ONNX, vLLM, or Transformers integration | Those are separate deployment stacks and add no value to EMEL's current generator/parity/bench acceptance boundary | Existing EMEL runtime plus a Prism-fork reference lane |
| Whole-tensor dequantize-to-f32 hot-path fallback | It breaks the milestone's performance and parity honesty contract | Native `Q1_0_g128` kernels or an explicitly approved interim milestone |
| Broad new downloader/provenance framework | The maintained asset can be fetched and verified with existing shell tooling | Direct Hugging Face resolve URL plus checksum capture in `tests/models/README.md` |

## Stack Patterns by Variant

**If the loaded GGUF reports `general.architecture=qwen3`:**

- Reuse the existing EMEL Qwen3 architecture path.
- Because the Bonsai artifact's novelty is the quant format, not a new model family.

**If the loaded weights are `Q1_0_g128`:**

- Route to a new native Bonsai quantized operand path.
- Because standard GGUF parsing is not enough; EMEL must understand the Prism-specific block layout and kernel contract.

**If the task is parity or benchmark publication:**

- Use `PrismML-Eng/llama.cpp` in `tools/` only.
- Because the official Bonsai docs point to Prism's fork and upstream is not the truthful comparator yet.

**If the task is prompt formatting:**

- Use one explicit Bonsai contract derived from the published `tokenizer.chat_template`: structured chat messages, `add_generation_prompt=true`, `tools=none`, no named templates, and the published assistant `<think>` preamble.
- Because the current repo's supported Qwen contract matcher is too narrow for the shipped Bonsai template.

**If prose and executable artifact disagree:**

- Trust Hugging Face repo tree, API metadata, and resolve headers.
- Because the Bonsai README quickstart filename is stale relative to the actual published artifact.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `prism-ml/Bonsai-1.7B-gguf@c89c1b5578286827264c4217f40edee617f4f904` | EMEL `qwen3` path + new native `Q1_0_g128` runtime | The artifact already fits the repo's Qwen3 architecture seam once the quant path exists. |
| `PrismML-Eng/llama.cpp@prism` | `Bonsai-1.7B.gguf` reference lane | Pin an exact commit before parity or benchmark claims. The repo's default branch is `prism`; the fork also exposes a `q1-cpu` branch, but the official Bonsai model card points to the fork generally, not that branch specifically. |
| `ggml-org/llama.cpp@ecbcb7ea9d3303097519723b264a8b5f1e977028` | Existing maintained Llama/Qwen/LFM2 slices | Not sufficient for Bonsai `Q1_0_g128` reference truth as published on 2026-04-02. |
| Direct Hugging Face resolve URL | `curl` + `shasum` provenance flow | Pre-download headers expose repo commit and byte size; post-download SHA256 should still be recorded locally. |

## Sources

- https://huggingface.co/prism-ml/Bonsai-1.7B-gguf
  - Verified the official maintained GGUF repo, published file list, README claims, and official reference-fork guidance.
- https://huggingface.co/api/models/prism-ml/Bonsai-1.7B-gguf
  - Verified `lastModified`, `sha`, `gguf.architecture=qwen3`, `gguf.context_length=32768`, the published chat template, and the single GGUF sibling file.
- https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf
  - Verified the direct download URL, final served filename, repo commit header, and published byte size.
- https://api.github.com/repos/PrismML-Eng/llama.cpp
  - Verified this is a public fork of `ggml-org/llama.cpp` and that its default branch is `prism`.
- https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/src/ggml.c
  - Verified Prism's fork contains `Q1_0_g128` references.
- https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/src/ggml-quants.c
  - Verified Prism's fork defines `quantize_row_q1_0_g128_ref` and related `Q1_0_g128` logic.
- https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/src/ggml-common.h
  - Verified Prism's fork defines `QK1_0_g128` and `block_q1_0_g128`.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/README.md
  - Verified upstream `llama.cpp` does not advertise `Q1_0_g128` support in its public README.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.c
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-quants.h
  - Verified upstream `ggml-org/llama.cpp` does not expose `Q1_0_g128` in the inspected quant sources.
- Local repo inspection:
  - `src/emel/kernel/detail.hpp`
  - `src/emel/kernel/aarch64/actions.hpp`
  - `src/emel/generator/detail.hpp`
  - `src/emel/model/llama/detail.hpp`
  - `tools/generation_formatter_contract.hpp`
  - `tools/generation_fixture_registry.hpp`
  - `tools/paritychecker/CMakeLists.txt`
  - `tools/bench/CMakeLists.txt`
  - `tools/paritychecker/reference_ref.txt`
  - `tests/models/README.md`

---
*Stack research for: EMEL Bonsai-1.7B 1-bit GGUF bring-up*
*Researched: 2026-04-02*
