#pragma once

#include "emel/kernel/mhc/actions.hpp"

namespace emel::kernel::mhc::guard {

struct guard_execute_pre_mix {
  bool operator()(const event::execute_pre_mix &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t lane_total =
        static_cast<uint64_t>(request.lane_count) * request.dim;
    return request.lane_count > 0u &&
           request.lane_count <= event::k_max_lanes && request.dim > 0u &&
           request.lane_index < request.lane_count &&
           request.lanes.size() >= lane_total &&
           request.phi_dots.size() >= request.lane_count &&
           request.a.size() >= 2u &&
           request.b.size() >= static_cast<uint64_t>(request.lane_count) * 2u &&
           request.output.size() >= request.dim;
  }
};

struct guard_execute_post_mix {
  bool operator()(const event::execute_post_mix &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    const uint64_t lane_total =
        static_cast<uint64_t>(request.lane_count) * request.dim;
    const uint64_t square =
        static_cast<uint64_t>(request.lane_count) * request.lane_count;
    return request.lane_count > 0u &&
           request.lane_count <= event::k_max_lanes && request.dim > 0u &&
           request.lane_index < request.lane_count &&
           request.lanes.size() >= lane_total &&
           request.block_out.size() >= request.dim &&
           request.u.size() >= request.dim &&
           request.post_dots.size() >= request.lane_count &&
           request.res_dots.size() >= square && request.a_post.size() >= 2u &&
           request.a_res.size() >= 2u &&
           request.b_post.size() >=
               static_cast<uint64_t>(request.lane_count) * 2u &&
           request.b_res.size() >= square * 2u &&
           request.output.size() >= lane_total;
  }
};

struct guard_execute_mean_lanes {
  bool operator()(const event::execute_mean_lanes &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    return request.lane_count > 0u && request.dim > 0u &&
           request.lanes.size() >=
               static_cast<uint64_t>(request.lane_count) * request.dim &&
           request.output.size() >= request.dim;
  }
};

} // namespace emel::kernel::mhc::guard
