#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace emel::tools::generation_fixture_registry {

inline constexpr std::string_view k_generation_parity_contract_append_only_baseline =
    "append_only_baseline";
inline constexpr std::string_view k_generation_parity_contract_live_reference_generation =
    "live_reference_generation";

struct maintained_fixture {
  std::string_view name = {};
  std::string_view slug = {};
  std::string_view fixture_rel = {};
  std::string_view reference_engine = {};
  std::string_view reference_repository = {};
  std::string_view reference_ref = {};
  std::string_view generation_parity_contract = {};
  std::string_view source_repo = {};
  std::string_view download_url = {};
  std::string_view sha256 = {};
  std::string_view architecture = {};
  std::string_view tokenizer_model = {};
  std::string_view tokenizer_pre = {};
  std::string_view weight_format = {};
  std::uint64_t size_bytes = 0u;
  std::int32_t context_length = 0;
  std::int32_t block_count = 0;
  std::int32_t embedding_length = 0;
  std::int32_t attention_head_count = 0;
  std::int32_t attention_head_count_kv = 0;
  bool generation_supported = false;
  bool current_publication = false;
};

inline constexpr maintained_fixture k_qwen3_generation_fixture = {
    .name = "Qwen3-0.6B-Q8_0.gguf",
    .slug = "qwen3_0_6b_q8_0",
    .fixture_rel = "tests/models/Qwen3-0.6B-Q8_0.gguf",
    .reference_engine = "llama.cpp",
    .reference_repository = "https://github.com/ggml-org/llama.cpp.git",
    .reference_ref = {},
    .generation_parity_contract = k_generation_parity_contract_live_reference_generation,
    .source_repo = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
    .download_url = "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/"
                    "Qwen3-0.6B-Q8_0.gguf",
    .sha256 = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
    .weight_format = "Q8_0",
    .size_bytes = 639446688u,
    .generation_supported = true,
    .current_publication = false,
};

inline constexpr maintained_fixture k_bonsai_generation_fixture = {
    .name = "Bonsai-1.7B.gguf",
    .slug = "bonsai_1_7b",
    .fixture_rel = "tests/models/Bonsai-1.7B.gguf",
    .reference_engine = "llama.cpp",
    .reference_repository = "https://github.com/PrismML-Eng/llama.cpp.git",
    .reference_ref = "f5dda7207ed5837f1c83c2f52f851ad9b933d2fd",
    .generation_parity_contract = k_generation_parity_contract_live_reference_generation,
    .source_repo = "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf",
    .download_url = "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/"
                    "Bonsai-1.7B.gguf",
    .sha256 = "0ae245fc08236af7cb64caff164937e53a8d54af611b0f398cc992c0a5ba70c4",
    .architecture = "qwen3",
    .tokenizer_model = "gpt2",
    .tokenizer_pre = "qwen2",
    .weight_format = "Q1_0_g128",
    .size_bytes = 248302272u,
    .context_length = 32768,
    .block_count = 28,
    .embedding_length = 2048,
    .attention_head_count = 16,
    .attention_head_count_kv = 8,
    .generation_supported = true,
    .current_publication = false,
};

inline constexpr maintained_fixture k_lfm2_generation_fixture = {
    .name = "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    .slug = "lfm2_5_1_2b_thinking_q4_k_m",
    .fixture_rel = "tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    .reference_engine = "llama.cpp",
    .reference_repository = "https://github.com/ggml-org/llama.cpp.git",
    .reference_ref = {},
    .generation_parity_contract = k_generation_parity_contract_append_only_baseline,
    .source_repo = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
    .download_url = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/resolve/main/"
                    "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    .sha256 = "7223a2202405b02e8e1e6c5baa543c43dc98c1d9741a5c2a0ee1583212e1231b",
    .architecture = "lfm2",
    .weight_format = "Q4_K_M",
    .size_bytes = 730895360u,
    .context_length = 128000,
    .generation_supported = true,
    .current_publication = true,
};

inline constexpr std::array<maintained_fixture, 3> k_maintained_generation_fixtures = {
    k_qwen3_generation_fixture,
    k_bonsai_generation_fixture,
    k_lfm2_generation_fixture,
};

inline constexpr std::array<maintained_fixture, 3> k_supported_generation_parity_fixtures = {
    k_qwen3_generation_fixture,
    k_bonsai_generation_fixture,
    k_lfm2_generation_fixture,
};

inline constexpr std::array<maintained_fixture, 2> k_supported_generation_bench_fixtures = {
    k_qwen3_generation_fixture,
    k_lfm2_generation_fixture,
};

}  // namespace emel::tools::generation_fixture_registry
