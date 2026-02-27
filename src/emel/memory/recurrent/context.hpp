#pragma once

#include <array>
#include <cstdint>

#include "emel/memory/recurrent/detail.hpp"

namespace emel::memory::recurrent::action {

struct context {
  int32_t max_sequences = detail::max_sequences;
  int32_t max_slots = detail::max_sequences;

  detail::slot_pool slots = {};
  std::array<int32_t, detail::max_sequences> free_stack = {};
  int32_t free_count = 0;
  std::array<int32_t, detail::max_sequences> seq_to_slot = {};
  std::array<int32_t, detail::max_sequences> slot_owner_seq = {};
  std::array<int32_t, detail::max_sequences> sequence_length = {};
};

}  // namespace emel::memory::recurrent::action
