#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <boost/sml.hpp>
#include <boost/sml/utility/sm_pool.hpp>

#include "emel/emel.h"
#include "emel/error/error.hpp"

namespace emel::memory::kv::detail {

inline constexpr int32_t default_block_tokens = 16;
inline constexpr int32_t max_sequences = 256;
inline constexpr int32_t max_blocks = 32768;
inline constexpr int32_t max_blocks_per_sequence = 4096;
inline constexpr int32_t invalid_index = -1;

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

inline emel::error::type cast_api_error(const emel::error::type err) noexcept {
  return err;
}

inline bool valid_sequence_id(const int32_t max_sequence_count, const int32_t seq_id) noexcept {
  return seq_id >= 0 && seq_id < max_sequence_count;
}

struct block_link {};
struct block_unlink {};
struct router_ready {};

struct block_storage {
  std::array<uint16_t, max_blocks> refs = {};

  void reset() noexcept { refs.fill(0); }
};

struct block_router_model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct valid_link {
      bool operator()(const block_storage & storage,
                      const sml::utility::indexed_event<block_link> & ev) const noexcept {
        return ev.id < storage.refs.size() && storage.refs[ev.id] < UINT16_MAX;
      }
    };

    struct do_link {
      void operator()(block_storage & storage,
                      const sml::utility::indexed_event<block_link> & ev) const noexcept {
        ++storage.refs[ev.id];
      }
    };

    struct valid_unlink {
      bool operator()(const block_storage & storage,
                      const sml::utility::indexed_event<block_unlink> & ev) const noexcept {
        return ev.id < storage.refs.size() && storage.refs[ev.id] > 0;
      }
    };

    struct do_unlink {
      void operator()(block_storage & storage,
                      const sml::utility::indexed_event<block_unlink> & ev) const noexcept {
        --storage.refs[ev.id];
      }
    };

    // clang-format off
    return sml::make_transition_table(
        sml::state<router_ready> <= *sml::state<router_ready>
            + sml::event<sml::utility::indexed_event<block_link>>
            [ valid_link{} ] / do_link{}
      , sml::state<router_ready> <= sml::state<router_ready>
            + sml::event<sml::utility::indexed_event<block_unlink>>
            [ valid_unlink{} ] / do_unlink{}
    );
    // clang-format on
  }
};

using block_pool = boost::sml::utility::sm_pool<block_storage, block_router_model>;

}  // namespace emel::memory::kv::detail
