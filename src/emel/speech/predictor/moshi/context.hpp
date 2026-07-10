#pragma once

#include <array>
#include <cstdint>

#include "emel/memory/hybrid/context.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/events.hpp"

namespace emel::speech::predictor::moshi::action {

inline constexpr int32_t k_max_codebooks =
    static_cast<int32_t>(event::k_max_codebooks);
inline constexpr int32_t k_max_delay_rows = 128;
inline constexpr int32_t k_token_zero = -1;
inline constexpr int32_t k_token_ungenerated = -2;

using graph_step_dispatch_fn = bool(void *, const event::graph_step &);

struct graph_binding {
  void *executor = nullptr;
  graph_step_dispatch_fn *dispatch_step = nullptr;
};

struct lmgen_state {
  int32_t codebook_count = 0;
  int32_t generated_dep_q = 0;
  int32_t delayed_dep_q = 0;
  int32_t needed_tokens = 0;
  int32_t max_delay = 0;
  int32_t delay_steps = 0;
  int32_t cache_row_count = 0;
  int64_t offset = 0;
  int32_t skip = 0;
  std::array<int32_t, k_max_codebooks> delays = {};
  std::array<int32_t, k_max_codebooks> initial = {};
  std::array<int32_t, k_max_delay_rows * k_max_codebooks> cache = {};
};

struct runtime {
  const emel::model::data *model = nullptr;
  emel::model::moshi::detail::execution_contract contract = {};
  int32_t n_q = 0;
  int32_t dep_q = 0;
  int32_t text_card = 0;
  int32_t audio_card = 0;
  int32_t sequence_id = 0;
};

struct voice_prompt_state {
  const emel::model::data *model = nullptr;
  emel::model::moshi::detail::execution_contract contract = {};
  int32_t embedding_dim = 0;
  int32_t prompt_frame_count = 0;
  int32_t prompt_frame_index = 0;
  int32_t cache_row_count = 0;
  int32_t cache_column_count = 0;
  int32_t pre_text_silence_remaining = 0;
  int32_t text_tokens_remaining = 0;
  int32_t post_text_silence_remaining = 0;
  bool loaded = false;
  bool ready = false;
  bool prompt_started = false;
  bool prompt_ready = false;
};

struct context {
  context() = default;
  explicit context(const emel::memory::hybrid::kv_binding &kv_cache)
      : memory(kv_cache) {}
  context(const emel::memory::hybrid::kv_binding &kv_cache,
          const graph_binding &graph_executor)
      : graph(graph_executor), memory(kv_cache) {}

  runtime session = {};
  voice_prompt_state voice = {};
  graph_binding graph = {};
  lmgen_state lmgen = {};
  emel::memory::hybrid::sm memory = {};
};

inline graph_binding
bind_graph_executor(void *executor,
                    graph_step_dispatch_fn &dispatch_step) noexcept {
  return graph_binding{.executor = executor, .dispatch_step = &dispatch_step};
}

} // namespace emel::speech::predictor::moshi::action
