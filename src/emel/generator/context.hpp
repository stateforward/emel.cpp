#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "emel/batch/planner/sm.hpp"
#include "emel/generator/detail.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/sm.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/renderer/context.hpp"
#include "emel/text/renderer/sm.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::generator::action {

inline constexpr int32_t MAX_GENERATION_STEPS = 4096;
inline constexpr int32_t k_sequence_id = 0;
inline constexpr int32_t k_sequence_mask_words = 1;

struct tokenizer_binding {
  void * actor = nullptr;
  bool (*dispatch_bind)(void * tokenizer_sm,
                        const emel::text::tokenizer::event::bind &) = nullptr;
  bool (*dispatch_tokenize)(void * tokenizer_sm,
                            const emel::text::tokenizer::event::tokenize &) = nullptr;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder =
      emel::text::encoders::encoder_kind::fallback;
  bool add_special = true;
  bool parse_special = false;
};

struct graph_binding {
  emel::generator::detail::native_backend backend = {};
  emel::model::llama::detail::topology model_topology = {};
  emel::model::llama::detail::step_plan prefill_plan = {};
  emel::model::llama::detail::step_plan decode_plan = {};
  bool backend_ready = false;
};

struct session_limits {
  int32_t prompt_capacity = 0;
  int32_t decode_capacity = 0;
  int32_t block_capacity = 0;
  int32_t block_tokens = 0;
};

struct session_buffers {
  std::array<int32_t, MAX_GENERATION_STEPS> prompt_tokens = {};
  std::array<int32_t, MAX_GENERATION_STEPS> positions = {};
  std::array<uint64_t, k_sequence_mask_words> seq_masks = {1u};
  std::array<int32_t, 1> seq_primary_ids = {k_sequence_id};
  std::unique_ptr<float[]> logits = {};
  std::unique_ptr<int32_t[]> candidate_ids = {};
  std::unique_ptr<float[]> candidate_scores = {};
  int32_t candidate_capacity = 0;
  int32_t vocab_size = 0;
};

struct session_state {
  bool sequence_live = false;
  emel::graph::event::reserve_output graph_reservation = {};
  emel::memory::view::snapshot memory_snapshot = {};
};

struct renderer_session {
  bool strip_leading_space = false;
  size_t stop_sequence_used = 0;
  std::array<std::array<char, emel::text::renderer::action::k_max_stop_length>,
             emel::text::renderer::action::k_max_stop_sequences>
      stop_sequence_bytes = {};
  std::array<size_t, emel::text::renderer::action::k_max_stop_sequences>
      stop_sequence_lengths = {};
};

struct context {
  const emel::model::data * model = nullptr;
  emel::text::conditioner::sm * conditioner = nullptr;
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt =
      emel::text::formatter::format_raw;

  emel::text::renderer::sm renderer = {};
  emel::batch::planner::sm planner = {};
  emel::memory::hybrid::sm memory = {};
  emel::graph::sm graph = {};
  emel::logits::sampler::sm sampler = {};

  tokenizer_binding conditioning = {};
  graph_binding compute = {};
  session_limits limits = {};
  session_buffers buffers = {};
  session_state state = {};
  renderer_session renderer_session = {};
};

}  // namespace emel::generator::action
