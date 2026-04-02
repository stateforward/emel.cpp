#pragma once

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace emel::bench {

enum class generation_lane_mode {
  emel,
  reference,
  compare,
};

struct config {
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
  std::uint64_t warmup_iterations = 0;
  std::size_t warmup_runs = 0;
};

struct result {
  std::string name;
  double ns_per_op = 0.0;
  std::uint64_t iterations = 0;
  std::size_t runs = 0;
};

struct generation_stage_probe {
  std::string name = {};
  std::string emel_prefill_contract = {};
  int32_t emel_prompt_tokens = 0;
  int32_t emel_prefill_step_size = 0;
  std::uint64_t emel_total_ns = 0;
  std::uint64_t emel_conditioning_ns = 0;
  std::uint64_t emel_prefill_ns = 0;
  std::uint64_t emel_first_decode_ns = 0;
  std::uint64_t emel_steady_decode_ns = 0;
  std::uint64_t emel_unattributed_ns = 0;
  std::uint64_t emel_prefill_linear_probe_ns = 0;
  std::uint64_t emel_prefill_attention_probe_ns = 0;
  std::uint64_t emel_prefill_misc_probe_ns = 0;
  std::uint64_t emel_prefill_misc_attention_norm_ns = 0;
  std::uint64_t emel_prefill_misc_qk_norm_ns = 0;
  std::uint64_t emel_prefill_misc_rope_ns = 0;
  std::uint64_t emel_prefill_misc_kv_store_ns = 0;
  std::uint64_t emel_prefill_misc_ctx_copy_ns = 0;
  std::uint64_t emel_prefill_misc_shortconv_ns = 0;
  std::uint64_t emel_prefill_shortconv_in_proj_ns = 0;
  std::uint64_t emel_prefill_shortconv_in_proj_prepare_ns = 0;
  std::uint64_t emel_prefill_shortconv_conv_ns = 0;
  std::uint64_t emel_prefill_shortconv_state_shift_ns = 0;
  std::uint64_t emel_prefill_shortconv_out_proj_ns = 0;
  std::uint64_t emel_prefill_shortconv_out_proj_prepare_ns = 0;
  std::uint64_t emel_prefill_misc_ffn_norm_ns = 0;
  std::uint64_t emel_prefill_misc_silu_ns = 0;
  std::uint64_t reference_total_ns = 0;
  std::uint64_t reference_conditioning_ns = 0;
  std::uint64_t reference_prefill_ns = 0;
  std::uint64_t reference_first_decode_ns = 0;
  std::uint64_t reference_steady_decode_ns = 0;
  std::uint64_t reference_unattributed_ns = 0;
  std::uint64_t reference_prefill_linear_probe_ns = 0;
  std::uint64_t reference_prefill_attention_probe_ns = 0;
  std::uint64_t reference_prefill_misc_probe_ns = 0;
};

void set_generation_lane_mode(generation_lane_mode mode) noexcept;
generation_lane_mode generation_lane_mode_current() noexcept;

template <class fn_type>
result measure_case(const char * name, const config & cfg, fn_type && fn) {
  std::vector<double> samples;
  samples.reserve(cfg.runs);

  for (std::size_t run = 0; run < cfg.warmup_runs; ++run) {
    for (std::uint64_t i = 0; i < cfg.warmup_iterations; ++i) {
      fn();
    }
  }

  for (std::size_t run = 0; run < cfg.runs; ++run) {
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < cfg.iterations; ++i) {
      fn();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    samples.push_back(static_cast<double>(duration_ns) / static_cast<double>(cfg.iterations));
  }

  std::sort(samples.begin(), samples.end());
  const double median = samples[samples.size() / 2];

  result out;
  out.name = name;
  out.ns_per_op = median;
  out.iterations = cfg.iterations;
  out.runs = cfg.runs;
  return out;
}

}  // namespace emel::bench
