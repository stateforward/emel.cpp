#pragma once

#include "emel/sm.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::action {

using lane_pool =
    emel::policy::thread_pool_scheduler<event::k_max_lanes, 16u, 128u>;

struct context {
  lane_pool * pool = nullptr;
};

}  // namespace emel::text::generator::decode_wavefront::action
