#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>

#include "emel/kernel/attention/sm.hpp"
#include "emel/kernel/matmul/sm.hpp"
#include "emel/kernel/sm.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"

namespace emel::speech::predictor::moshi::executor::action {

struct kv_views {
  detail::temporal_kv_view temporal = {};
  detail::depformer_kv_view depformer = {};
  emel::memory::streaming::sm *temporal_positions = nullptr;
  emel::memory::streaming::sm *depformer_positions = nullptr;
};

struct policies {
  float rms_norm_epsilon = 0.0f;
  uint32_t zero_seed_state = 0u;
  uint32_t sampling_modulus = 0u;
  int32_t token_zero = 0;
};

struct capacities {
  uint64_t hidden_dim = 0u;
  uint64_t temporal_context = 0u;
  uint64_t depformer_context = 0u;
  uint64_t sampling_card = 0u;
  uint64_t sampling_top_k = 0u;
};

inline constexpr std::size_t k_max_attention_lanes = 8u;

struct attention_lane_storage {
  std::array<emel::kernel::attention::sm, k_max_attention_lanes> actors = {};
};

struct dependencies {
  kv_views kv = {};
  emel::kernel::sm &kernel;
  emel::kernel::matmul::sm &matmul;
  emel::kernel::matmul::lane_mode matmul_lane_mode =
      emel::kernel::matmul::lane_mode::serial;
  emel::kernel::matmul::lane_pool *attention_lanes = nullptr;
  std::size_t active_attention_lanes = 1u;
  emel::logits::sampler::sm *sampler = nullptr;
  policies policy = {};
  capacities capacity = {};
};

struct runtime {
  const emel::model::data *model = nullptr;
  emel::model::moshi::detail::execution_contract contract = {};
  int32_t codebook_count = 0;
  int32_t dep_q = 0;
  int32_t text_card = 0;
  int32_t audio_card = 0;
  int32_t hidden_dim = 0;
  std::array<const emel::model::data::tensor_record *,
             static_cast<std::size_t>(
                 emel::model::data::moshi_lm_hparams::k_max_delays)>
      temporal_input_projections = {};
  std::array<std::array<const emel::model::data::tensor_record *,
                        static_cast<std::size_t>(
                            emel::model::data::moshi_lm_hparams::k_max_delays)>,
             static_cast<std::size_t>(
                 emel::model::data::moshi_lm_hparams::k_max_delays)>
      depformer_input_projections = {};
};

struct sampling_config {
  bool enabled = false;
  bool consume_forced_text = false;
  float audio_temperature = 0.0f;
  float text_temperature = 0.0f;
  int32_t audio_top_k = 0;
  int32_t text_top_k = 0;
  uint32_t random_state = 1u;
};

struct context {
  explicit context(const dependencies &deps)
      : temporal_kv(deps.kv.temporal), depformer_kv(deps.kv.depformer),
        temporal_positions(deps.kv.temporal_positions),
        depformer_positions(deps.kv.depformer_positions), kernel(deps.kernel),
        matmul(deps.matmul), matmul_lane_mode(deps.matmul_lane_mode),
        attention_lanes(deps.attention_lanes),
        active_attention_lanes(deps.active_attention_lanes),
        attention_actors(new (std::nothrow) attention_lane_storage{}),
        sampler(deps.sampler), policy(deps.policy), capacity(deps.capacity) {
    // Attention actors own substantial reusable scratch. One construction-time
    // allocation avoids oversized parent and thread stacks; same-RTC inference
    // dispatches reuse it without allocation.
    if (attention_actors == nullptr) {
      std::terminate();
    }
  }

  runtime session = {};
  sampling_config sampling = {};
  const detail::temporal_kv_view temporal_kv;
  const detail::depformer_kv_view depformer_kv;
  emel::memory::streaming::sm *temporal_positions = nullptr;
  emel::memory::streaming::sm *depformer_positions = nullptr;
  emel::kernel::sm &kernel;
  emel::kernel::matmul::sm &matmul;
  const emel::kernel::matmul::lane_mode matmul_lane_mode;
  emel::kernel::matmul::lane_pool *attention_lanes = nullptr;
  const std::size_t active_attention_lanes;
  std::unique_ptr<attention_lane_storage> attention_actors = {};
  emel::logits::sampler::sm *sampler = nullptr;
  const policies policy;
  const capacities capacity;
};

} // namespace emel::speech::predictor::moshi::executor::action
