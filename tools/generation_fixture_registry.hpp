#pragma once

#include <array>
#include <string_view>

namespace emel::tools::generation_fixture_registry {

struct maintained_fixture {
  std::string_view name = {};
  std::string_view slug = {};
  std::string_view fixture_rel = {};
  bool current_publication = false;
};

inline constexpr maintained_fixture k_qwen3_generation_fixture = {
    .name = "Qwen3-0.6B-Q8_0.gguf",
    .slug = "qwen3_0_6b_q8_0",
    .fixture_rel = "tests/models/Qwen3-0.6B-Q8_0.gguf",
    .current_publication = false,
};

inline constexpr maintained_fixture k_lfm2_generation_fixture = {
    .name = "LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    .slug = "lfm2_5_1_2b_thinking_q4_k_m",
    .fixture_rel = "tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    .current_publication = true,
};

inline constexpr std::array<maintained_fixture, 2> k_maintained_generation_fixtures = {
    k_qwen3_generation_fixture,
    k_lfm2_generation_fixture,
};

}  // namespace emel::tools::generation_fixture_registry
