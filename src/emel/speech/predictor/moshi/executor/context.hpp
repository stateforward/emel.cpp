#pragma once

#include <cstddef>
#include <cstdint>

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

struct dependencies {
  kv_views kv = {};
  emel::kernel::sm &kernel;
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
        sampler(deps.sampler), policy(deps.policy), capacity(deps.capacity) {}

  runtime session = {};
  sampling_config sampling = {};
  const detail::temporal_kv_view temporal_kv;
  const detail::depformer_kv_view depformer_kv;
  emel::memory::streaming::sm *temporal_positions = nullptr;
  emel::memory::streaming::sm *depformer_positions = nullptr;
  emel::kernel::sm &kernel;
  emel::logits::sampler::sm *sampler = nullptr;
  const policies policy;
  const capacities capacity;
};

} // namespace emel::speech::predictor::moshi::executor::action
