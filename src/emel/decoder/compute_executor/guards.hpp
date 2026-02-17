#pragma once

#include "emel/decoder/compute_executor/actions.hpp"

namespace emel::decoder::compute_executor::guard {

inline constexpr auto graph_reused = [](const events::prepare_graph_done & ev) {
  return ev.reused;
};

inline constexpr auto graph_needs_allocation = [](const events::prepare_graph_done & ev) {
  return !ev.reused;
};

}  // namespace emel::decoder::compute_executor::guard
