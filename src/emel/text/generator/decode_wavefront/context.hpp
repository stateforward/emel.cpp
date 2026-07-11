#pragma once

#include "emel/sm.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::action {

using lane_pool =
    emel::policy::fork_join_lane_pool<event::k_max_lanes, 128u, 1048576u>;

struct context {
  lane_pool *pool = nullptr;
};

} // namespace emel::text::generator::decode_wavefront::action
