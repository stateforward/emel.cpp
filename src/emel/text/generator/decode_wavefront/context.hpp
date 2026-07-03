#pragma once

#include "emel/sm.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::action {

using lane_pool =
    emel::policy::thread_pool_scheduler<event::k_max_lanes, 16u, 128u>;
using lane_scheduler = emel::policy::thread_pool_scheduler_ref<lane_pool>;

struct context {
  lane_pool * pool = nullptr;
};

}  // namespace emel::text::generator::decode_wavefront::action
