#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/text/detokenizer/events.hpp"
#include "emel/text/renderer/events.hpp"

namespace emel::text::renderer::action {

inline constexpr size_t k_max_sequences = 64;
inline constexpr size_t k_max_pending_bytes = 4;
inline constexpr size_t k_max_stop_sequences = 8;
inline constexpr size_t k_max_stop_storage = 256;
inline constexpr size_t k_max_stop_length = 32;
inline constexpr size_t k_max_holdback_bytes = k_max_stop_length - 1;

struct stop_sequence_entry {
  uint16_t offset = 0;
  uint16_t length = 0;
};

struct sequence_state {
  std::array<uint8_t, k_max_pending_bytes> pending_bytes = {};
  size_t pending_length = 0;

  std::array<char, k_max_holdback_bytes> holdback = {};
  size_t holdback_length = 0;

  bool strip_leading_space = false;
  bool stop_matched = false;
};

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  void * detokenizer_sm = nullptr;
  bool (*dispatch_detokenizer_bind)(
      void * detokenizer_sm,
      const emel::text::detokenizer::event::bind &) = nullptr;
  bool (*dispatch_detokenizer_detokenize)(
      void * detokenizer_sm,
      const emel::text::detokenizer::event::detokenize &) = nullptr;

  bool strip_leading_space_default = false;
  std::array<stop_sequence_entry, k_max_stop_sequences> stop_sequences = {};
  size_t stop_sequence_count = 0;
  std::array<char, k_max_stop_storage> stop_storage = {};
  size_t stop_storage_used = 0;
  size_t stop_max_length = 0;

  std::array<sequence_state, k_max_sequences> sequences = {};
  bool is_bound = false;

  int32_t token_id = -1;
  int32_t sequence_id = 0;
  bool emit_special = false;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t output_length = 0;
  sequence_status status = sequence_status::running;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::text::renderer::action
