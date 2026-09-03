#pragma once

#include <cstddef>

#include "emel/graph/sm.hpp"
#include "emel/text/generator/decode_wavefront/context.hpp"
#include "emel/text/generator/decode_wavefront/events.hpp"

namespace emel::text::generator::decode_wavefront::action {

struct effect_begin_run {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out = {};
    ev.out.err = emel::error::cast(error::none);
  }
};

struct effect_mark_single_lane {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.grouped = false;
  }
};

struct effect_mark_grouped_lanes {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.grouped = true;
  }
};

struct effect_reject_invalid_request {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_reject_incompatible_lanes {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.err = emel::error::cast(error::incompatible_lanes);
  }
};

struct effect_reject_parallel_scheduler {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.err = emel::error::cast(error::backend);
    ev.out.failed_lane = event::k_no_failed_lane;
  }
};

template <size_t lane_index>
struct effect_dispatch_lane {
  void operator()(const event::run & ev, context &) const noexcept {
    auto & lane = ev.lanes[lane_index];
    const emel::graph::event::compute_reserved reserved_compute{lane.compute};
    lane.accepted = lane.graph.process_event(reserved_compute);
    ev.out.dispatched_lanes = static_cast<int32_t>(lane_index + 1u);
  }
};

struct effect_dispatch_parallel_lanes {
  void operator()(const event::run & ev, context & ctx) const noexcept {
    for (auto & lane : ev.lanes) {
      lane.accepted = false;
    }

    // Fork/join over the already-selected lane group. The upstream batch API
    // either publishes every lane or rejects the whole group before moving a
    // callable, so post-join guards keep routing scheduler failures explicitly.
    worker_pool::join_group group{};
    emel::policy::fork_join_start_gate gate{};
    auto make_task = [&gate](event::lane &lane) noexcept {
      auto *lane_ptr = &lane;
      return [lane_ptr, &gate]() noexcept {
        gate.arrive_and_wait();
        auto &current_lane = *lane_ptr;
        const emel::graph::event::compute_reserved reserved_compute{
            current_lane.compute};
        current_lane.accepted =
            current_lane.graph.process_event(reserved_compute);
      };
    };
    size_t submitted_lanes = 0u;
    switch (ev.lanes.size()) {
    case 2u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]));
      break;
    case 3u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]));
      break;
    case 4u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]), make_task(ev.lanes[3]));
      break;
    case 5u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]), make_task(ev.lanes[3]),
          make_task(ev.lanes[4]));
      break;
    case 6u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]), make_task(ev.lanes[3]),
          make_task(ev.lanes[4]), make_task(ev.lanes[5]));
      break;
    case 7u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]), make_task(ev.lanes[3]),
          make_task(ev.lanes[4]), make_task(ev.lanes[5]),
          make_task(ev.lanes[6]));
      break;
    case 8u:
      submitted_lanes = ctx.pool->try_submit_batch(
          group, make_task(ev.lanes[0]), make_task(ev.lanes[1]),
          make_task(ev.lanes[2]), make_task(ev.lanes[3]),
          make_task(ev.lanes[4]), make_task(ev.lanes[5]),
          make_task(ev.lanes[6]), make_task(ev.lanes[7]));
      break;
    default:
      break;
    }
    gate.open_after_arrivals(submitted_lanes);
    ev.out.all_submitted = submitted_lanes == ev.lanes.size();
    ev.out.joined = group.wait();
    ev.out.dispatched_lanes = static_cast<int32_t>(submitted_lanes);
  }
};

template <size_t lane_index>
struct effect_mark_lane_rejected {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.err = emel::error::cast(error::lane_rejected);
    ev.out.failed_lane = static_cast<int32_t>(lane_index);
  }
};

struct effect_commit_done {
  void operator()(const event::run & ev, context &) const noexcept {
    ev.out.err = emel::error::cast(error::none);
    ev.out.failed_lane = event::k_no_failed_lane;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type & ev, context &) const noexcept {
    if constexpr (requires { ev.out.err; }) {
      ev.out.err = emel::error::cast(error::backend);
    }
  }
};

inline constexpr effect_begin_run effect_begin_run{};
inline constexpr effect_mark_single_lane effect_mark_single_lane{};
inline constexpr effect_mark_grouped_lanes effect_mark_grouped_lanes{};
inline constexpr effect_reject_invalid_request effect_reject_invalid_request{};
inline constexpr effect_reject_incompatible_lanes effect_reject_incompatible_lanes{};
inline constexpr effect_reject_parallel_scheduler effect_reject_parallel_scheduler{};
inline constexpr effect_dispatch_parallel_lanes effect_dispatch_parallel_lanes{};
inline constexpr effect_commit_done effect_commit_done{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

}  // namespace emel::text::generator::decode_wavefront::action
