#pragma once

#include "emel/sm.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::action {

using worker_pool =
    emel::policy::thread_pool_scheduler<event::k_max_lanes, 8u, 128u>;

struct context {
  worker_pool *pool = nullptr;
};

} // namespace emel::text::generator::decode_wavefront::action
