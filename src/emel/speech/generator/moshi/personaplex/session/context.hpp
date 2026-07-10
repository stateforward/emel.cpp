#pragma once

#include "emel/memory/hybrid/context.hpp"
#include "emel/speech/codec/mimi/any.hpp"
#include "emel/speech/generator/moshi/any.hpp"
#include "emel/speech/generator/moshi/executor/any.hpp"

namespace emel::speech::generator::moshi::personaplex::session::action {

namespace mimi = emel::speech::codec::mimi;
namespace moshi = emel::speech::generator::moshi;
namespace executor = emel::speech::generator::moshi::executor;

struct dependencies {
  executor::detail::temporal_kv_view temporal_kv = {};
  executor::detail::depformer_kv_view depformer_kv = {};
  emel::memory::hybrid::kv_binding generator_memory = {};
};

struct context {
  explicit context(const dependencies &deps)
      : temporal_kv(deps.temporal_kv), depformer_kv(deps.depformer_kv),
        graph_executor(executor::bind_kv_caches(
            executor::bind_temporal_kv_cache(this, bind_temporal_cache),
            executor::bind_depformer_kv_cache(this, bind_depformer_cache))),
        generator(deps.generator_memory,
                  moshi::action::bind_graph_executor(
                      &graph_executor, executor::dispatch_graph_step)) {}

  static bool
  bind_temporal_cache(void *context_ptr, const emel::model::data &,
                      const emel::memory::view::snapshot &, int32_t,
                      executor::detail::temporal_kv_view &view) noexcept {
    view = static_cast<context *>(context_ptr)->temporal_kv;
    return true;
  }

  static bool
  bind_depformer_cache(void *context_ptr, const emel::model::data &,
                       const emel::memory::view::snapshot &, int32_t,
                       executor::detail::depformer_kv_view &view) noexcept {
    view = static_cast<context *>(context_ptr)->depformer_kv;
    return true;
  }

  executor::detail::temporal_kv_view temporal_kv = {};
  executor::detail::depformer_kv_view depformer_kv = {};
  int32_t frame_samples = 0;
  int32_t public_n_q = 0;
  mimi::sm encoder = {};
  mimi::sm decoder = {};
  executor::sm graph_executor;
  moshi::sm generator;
};

} // namespace emel::speech::generator::moshi::personaplex::session::action
