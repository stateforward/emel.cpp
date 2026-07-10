#pragma once

#include <cstdint>

#include "emel/kernel/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/executor/detail.hpp"

namespace emel::speech::predictor::moshi::executor::action {

using temporal_kv_bind_fn = bool(void *, const emel::model::data &,
                                 const emel::memory::view::snapshot &, int32_t,
                                 detail::temporal_kv_view &);

using depformer_kv_bind_fn = bool(void *, const emel::model::data &,
                                  const emel::memory::view::snapshot &, int32_t,
                                  detail::depformer_kv_view &);

struct temporal_kv_binding {
  void *cache = nullptr;
  temporal_kv_bind_fn *bind = nullptr;
};

struct depformer_kv_binding {
  void *cache = nullptr;
  depformer_kv_bind_fn *bind = nullptr;
};

struct kv_bindings {
  temporal_kv_binding temporal = {};
  depformer_kv_binding depformer = {};
  emel::memory::streaming::sm *temporal_positions = nullptr;
  emel::memory::streaming::sm *depformer_positions = nullptr;
};

struct runtime {
  const emel::model::data *model = nullptr;
  emel::model::moshi::detail::execution_contract contract = {};
  int32_t codebook_count = 0;
  int32_t dep_q = 0;
  int32_t text_card = 0;
  int32_t audio_card = 0;
  int32_t hidden_dim = 0;
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
  context() = default;
  explicit context(const temporal_kv_binding &kv_binding)
      : temporal_kv(kv_binding) {}
  explicit context(const kv_bindings &kv_binding_set)
      : temporal_kv(kv_binding_set.temporal),
        depformer_kv(kv_binding_set.depformer),
        temporal_positions(kv_binding_set.temporal_positions),
        depformer_positions(kv_binding_set.depformer_positions) {}

  runtime session = {};
  sampling_config sampling = {};
  temporal_kv_binding temporal_kv = {};
  depformer_kv_binding depformer_kv = {};
  emel::memory::streaming::sm *temporal_positions = nullptr;
  emel::memory::streaming::sm *depformer_positions = nullptr;
  emel::kernel::sm kernel = {};
};

inline temporal_kv_binding
bind_temporal_kv_cache(void *cache, temporal_kv_bind_fn &bind) noexcept {
  return temporal_kv_binding{.cache = cache, .bind = &bind};
}

inline depformer_kv_binding
bind_depformer_kv_cache(void *cache, depformer_kv_bind_fn &bind) noexcept {
  return depformer_kv_binding{.cache = cache, .bind = &bind};
}

inline kv_bindings
bind_kv_caches(const temporal_kv_binding &temporal,
               const depformer_kv_binding &depformer) noexcept {
  return kv_bindings{.temporal = temporal, .depformer = depformer};
}

inline kv_bindings
bind_kv_caches(const temporal_kv_binding &temporal,
               const depformer_kv_binding &depformer,
               emel::memory::streaming::sm &temporal_positions,
               emel::memory::streaming::sm &depformer_positions) noexcept {
  return kv_bindings{
      .temporal = temporal,
      .depformer = depformer,
      .temporal_positions = &temporal_positions,
      .depformer_positions = &depformer_positions,
  };
}

} // namespace emel::speech::predictor::moshi::executor::action
