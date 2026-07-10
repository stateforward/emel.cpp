#pragma once

#include "emel/speech/generator/moshi/executor/sm.hpp"

namespace emel::speech::generator::moshi::executor {

inline bool dispatch_graph_step(
    void *executor_ptr,
    const emel::speech::generator::moshi::event::graph_step &ev) noexcept {
  return static_cast<sm *>(executor_ptr)->process_event(ev);
}

inline emel::speech::generator::moshi::action::graph_binding
bind_graph_executor(sm &executor) noexcept {
  return emel::speech::generator::moshi::action::bind_graph_executor(
      &executor, dispatch_graph_step);
}

using temporal_kv_binding = action::temporal_kv_binding;
using depformer_kv_binding = action::depformer_kv_binding;
using kv_bindings = action::kv_bindings;

inline temporal_kv_binding
bind_temporal_kv_cache(void *cache,
                       action::temporal_kv_bind_fn &bind) noexcept {
  return action::bind_temporal_kv_cache(cache, bind);
}

inline depformer_kv_binding
bind_depformer_kv_cache(void *cache,
                        action::depformer_kv_bind_fn &bind) noexcept {
  return action::bind_depformer_kv_cache(cache, bind);
}

inline kv_bindings
bind_kv_caches(const temporal_kv_binding &temporal,
               const depformer_kv_binding &depformer) noexcept {
  return action::bind_kv_caches(temporal, depformer);
}

inline kv_bindings
bind_kv_caches(const temporal_kv_binding &temporal,
               const depformer_kv_binding &depformer,
               emel::memory::streaming::sm &temporal_positions,
               emel::memory::streaming::sm &depformer_positions) noexcept {
  return action::bind_kv_caches(temporal, depformer, temporal_positions,
                                depformer_positions);
}

using MoshiExecutor = sm;

} // namespace emel::speech::generator::moshi::executor
