#pragma once

#include <array>
#include <cstdint>

#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>

#include "emel/emel.h"

namespace emel::memory::kv::action {

inline constexpr int32_t DEFAULT_BLOCK_TOKENS = 16;
inline constexpr int32_t MAX_SEQUENCES = 256;
inline constexpr int32_t MAX_BLOCKS = 32768;
inline constexpr int32_t MAX_BLOCKS_PER_SEQUENCE = 4096;
inline constexpr int32_t INVALID_INDEX = -1;

struct block_link {};
struct block_unlink {};
struct router_ready {};

struct block_storage {
  std::array<uint16_t, MAX_BLOCKS> refs = {};

  void reset() noexcept { refs.fill(0); }
};

struct block_router_model {
  auto operator()() const {
    namespace sml = boost::sml;

    const auto valid_link = [](const block_storage & storage,
                               const sml::utility::indexed_event<block_link> & ev) {
      return ev.id < storage.refs.size() && storage.refs[ev.id] < UINT16_MAX;
    };

    const auto do_link = [](block_storage & storage,
                            const sml::utility::indexed_event<block_link> & ev) {
      ++storage.refs[ev.id];
    };

    const auto valid_unlink = [](const block_storage & storage,
                                 const sml::utility::indexed_event<block_unlink> & ev) {
      return ev.id < storage.refs.size() && storage.refs[ev.id] > 0;
    };

    const auto do_unlink = [](block_storage & storage,
                              const sml::utility::indexed_event<block_unlink> & ev) {
      --storage.refs[ev.id];
    };

    // clang-format off
    return sml::make_transition_table(
      *sml::state<router_ready> + sml::event<sml::utility::indexed_event<block_link>>[valid_link] / do_link,
       sml::state<router_ready> + sml::event<sml::utility::indexed_event<block_unlink>>[valid_unlink] / do_unlink
    );
    // clang-format on
  }
};

using block_pool = boost::sml::utility::sm_pool<block_storage, block_router_model>;

struct context {
  int32_t max_sequences = MAX_SEQUENCES;
  int32_t max_blocks = MAX_BLOCKS;
  int32_t block_tokens = DEFAULT_BLOCK_TOKENS;

  block_pool block_refs = {};
  std::array<bool, MAX_SEQUENCES> sequence_active = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_length = {};
  std::array<int32_t, MAX_SEQUENCES> sequence_block_count = {};
  std::array<std::array<uint16_t, MAX_BLOCKS_PER_SEQUENCE>, MAX_SEQUENCES> seq_to_blocks = {};

  std::array<uint16_t, MAX_BLOCKS> free_stack = {};
  int32_t free_count = 0;

  int32_t phase_error = EMEL_OK;
  bool phase_out_of_memory = false;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::memory::kv::action
