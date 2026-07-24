#pragma once

// Public facade for the Mimi streaming codec (Kyutai Moshi / NVIDIA
// PersonaPlex): initialize once against an enriched mimi-component GGUF,
// then stream 80 ms frames through encode_frame / decode_frame.
#include "emel/speech/codec/mimi/sm.hpp"

namespace emel::speech::codec::mimi {

// Compute immutable validation facts before entering the actor's RTC
// boundary. The state machine still owns route selection through guards over
// these facts; no behavior is selected by this preflight.
inline event::bind_contract
make_bind_contract(const emel::model::data &model) noexcept {
  return guard::make_bind_contract(model);
}

// Public arena-sizing contract for event::initialize: the caller owns the
// four arenas, sizes them here before dispatching initialize, and keeps them
// alive for the codec's lifetime (the machine never allocates). A result of
// zero means the model does not satisfy the mimi contract; initialize would
// answer with error::bind_failed. External callers (benchmarks, parity
// harnesses, integrators) use these - never the component detail helpers.
inline uint64_t prepared_arena_floats(const emel::model::data &model) noexcept {
  return detail::required_prepared_floats(model);
}

inline uint64_t state_arena_floats(const emel::model::data &model) noexcept {
  return detail::required_state_floats(model);
}

inline uint64_t
workspace_arena_floats(const emel::model::data &model) noexcept {
  return detail::required_workspace_floats(model);
}

inline uint64_t frame_arena_floats(const emel::model::data &model) noexcept {
  return detail::required_frame_floats(model);
}

} // namespace emel::speech::codec::mimi
