#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/text/detokenizer/sm.hpp"

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
  emel::text::detokenizer::sm detokenizer = {};

  bool strip_leading_space_default = false;
  std::array<stop_sequence_entry, k_max_stop_sequences> stop_sequences = {};
  size_t stop_sequence_count = 0;
  std::array<char, k_max_stop_storage> stop_storage = {};
  size_t stop_storage_used = 0;
  size_t stop_max_length = 0;

  std::array<sequence_state, k_max_sequences> sequences = {};
};

}  // namespace emel::text::renderer::action
