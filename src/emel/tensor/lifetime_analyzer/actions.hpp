#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/tensor/lifetime_analyzer/events.hpp"

namespace emel::tensor::lifetime_analyzer::action {

inline constexpr int32_t k_max_tensors = 2048;

struct context {
  int32_t tensor_count = 0;

  uint32_t step = 0;

  std::array<int32_t, k_max_tensors> tensor_ids = {};
  std::array<int32_t, k_max_tensors> first_use = {};
  std::array<int32_t, k_max_tensors> last_use = {};
  std::array<int32_t, k_max_tensors> n_children = {};
  std::array<int32_t, k_max_tensors> n_views = {};
  std::array<int32_t, k_max_tensors> view_src_indices = {};
  std::array<bool, k_max_tensors> tensor_is_view = {};
  std::array<bool, k_max_tensors> tensor_is_exec_node = {};
  std::array<bool, k_max_tensors> tensor_is_control_dep = {};
};

namespace detail {

inline int32_t normalize_error(const int32_t err, const int32_t fallback) noexcept {
  if (err != 0) return err;
  if (fallback != 0) return fallback;
  return EMEL_ERR_BACKEND;
}

inline int32_t index_of_tensor(const context & c, const int32_t tensor_id) noexcept {
  for (int32_t i = 0; i < c.tensor_count; ++i) {
    if (c.tensor_ids[i] == tensor_id) return i;
  }
  return -1;
}

inline bool has_duplicate_ids(const context & c, const int32_t upto_inclusive) noexcept {
  if (upto_inclusive < 0) return false;
  for (int32_t i = 0; i <= upto_inclusive; ++i) {
    for (int32_t j = i + 1; j <= upto_inclusive; ++j) {
      if (c.tensor_ids[i] >= 0 && c.tensor_ids[i] == c.tensor_ids[j]) return true;
    }
  }
  return false;
}

}  // namespace detail

struct begin_analyze {
  void operator()(const event::analyze & ev, context & c) const noexcept {
    c = {};
    c.tensor_count = ev.tensor_count;
    c.first_use.fill(-1);
    c.last_use.fill(-1);
    c.tensor_ids.fill(-1);
    c.n_children.fill(0);
    c.n_views.fill(0);
    c.view_src_indices.fill(-1);
    c.tensor_is_view.fill(false);
    c.tensor_is_exec_node.fill(false);
    c.tensor_is_control_dep.fill(false);
    c.step += 1;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (ev.tensor_count < 0 || ev.tensor_count > k_max_tensors) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (ev.tensor_count > 0 && ev.tensors == nullptr) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (ev.ranges_out_count < 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (
        (ev.first_use_out == nullptr || ev.last_use_out == nullptr) && ev.ranges_out_count != 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (
        ev.first_use_out != nullptr && ev.last_use_out != nullptr &&
        ev.ranges_out_count < ev.tensor_count) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (err == EMEL_OK) {
      c.tensor_count = ev.tensor_count;
    }
    c.step += 1;
  }
};

struct run_collect_ranges {
  void operator()(const event::collect_ranges & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    c.first_use.fill(-1);
    c.last_use.fill(-1);
    c.tensor_ids.fill(-1);
    c.n_children.fill(0);
    c.n_views.fill(0);
    c.view_src_indices.fill(-1);
    c.tensor_is_view.fill(false);
    c.tensor_is_exec_node.fill(false);
    c.tensor_is_control_dep.fill(false);

    // register tensors
    for (int32_t i = 0; i < c.tensor_count; ++i) {
      const auto & t = ev.tensors[i];
      if (t.tensor_id < 0) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        break;
      }
      c.tensor_ids[i] = t.tensor_id;
      c.first_use[i] = i;
      c.last_use[i] = i;
      c.tensor_is_view[i] = t.is_view;
      c.tensor_is_exec_node[i] = t.is_exec_node;
      c.tensor_is_control_dep[i] = t.is_control_dep;
      if (detail::has_duplicate_ids(c, i)) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        break;
      }
      if (t.is_view && t.view_src_id < 0) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        break;
      }
    }

    // count children and views (ggml_gallocr_alloc_graph_impl parity model)
    for (int32_t i = 0; i < c.tensor_count && err == EMEL_OK; ++i) {
      const auto & t = ev.tensors[i];
      if (!t.is_exec_node) {
        continue;
      }
      if (t.is_view && !t.is_control_dep) {
        const int32_t view_src_idx = detail::index_of_tensor(c, t.view_src_id);
        if (view_src_idx < 0) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          break;
        }
        c.view_src_indices[i] = view_src_idx;
        c.n_views[view_src_idx] += 1;
      }

      for (int32_t s = 0; s < event::k_max_sources; ++s) {
        const int32_t src_id = t.src_ids[s];
        if (src_id < 0) continue;
        const int32_t src_idx = detail::index_of_tensor(c, src_id);
        if (src_idx < 0) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          break;
        }
        c.n_children[src_idx] += 1;
      }
    }

    // simulate graph execution and infer release points
    for (int32_t i = 0; i < c.tensor_count && err == EMEL_OK; ++i) {
      const auto & t = ev.tensors[i];
      if (!t.is_exec_node) {
        continue;
      }
      for (int32_t s = 0; s < event::k_max_sources; ++s) {
        const int32_t src_id = t.src_ids[s];
        if (src_id < 0) {
          continue;
        }
        const int32_t parent_idx = detail::index_of_tensor(c, src_id);
        if (parent_idx < 0) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          break;
        }

        c.n_children[parent_idx] -= 1;
        if (c.n_children[parent_idx] < 0) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          break;
        }

        if (c.n_children[parent_idx] == 0 && c.n_views[parent_idx] == 0) {
          c.last_use[parent_idx] = i;
          if (c.tensor_is_view[parent_idx] && !c.tensor_is_control_dep[parent_idx]) {
            const int32_t view_src_idx = c.view_src_indices[parent_idx];
            if (view_src_idx < 0) {
              err = EMEL_ERR_INVALID_ARGUMENT;
              break;
            }

            c.n_views[view_src_idx] -= 1;
            if (c.n_views[view_src_idx] < 0) {
              err = EMEL_ERR_INVALID_ARGUMENT;
              break;
            }

            if (c.n_views[view_src_idx] == 0 && c.n_children[view_src_idx] == 0) {
              c.last_use[view_src_idx] = i;
            }
          }
        }
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_publish {
  void operator()(const event::publish & ev, context & c) const noexcept {
    if (ev.first_use_out != nullptr && ev.last_use_out != nullptr) {
      for (int32_t i = 0; i < c.tensor_count; ++i) {
        ev.first_use_out[i] = c.first_use[i];
        ev.last_use_out[i] = c.last_use[i];
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }
};

struct begin_reset {
  void operator()(const event::reset & ev, context & c) const noexcept {
    (void)ev;
    c.step += 1;
  }
};

struct on_analyze_done {
  void operator()(const events::analyze_done &, context & c) const noexcept {
    c.step += 1;
  }
};

struct on_analyze_error {
  void operator()(const events::analyze_error & ev, context & c) const noexcept {
    (void)ev;
    c.step += 1;
  }
};

struct on_reset_done {
  void operator()(const events::reset_done &, context & c) const noexcept {
    c = {};
    c.step = 1;
  }
};

struct on_reset_error {
  void operator()(const events::reset_error & ev, context & c) const noexcept {
    (void)ev;
    c.step += 1;
  }
};

struct record_phase_error {
  template <class ErrorEvent>
  void operator()(const ErrorEvent & ev, context & c) const noexcept {
    (void)ev;
    c.step += 1;
  }
};

inline constexpr begin_analyze begin_analyze{};
inline constexpr run_validate run_validate{};
inline constexpr run_collect_ranges run_collect_ranges{};
inline constexpr run_publish run_publish{};
inline constexpr begin_reset begin_reset{};
inline constexpr on_analyze_done on_analyze_done{};
inline constexpr on_analyze_error on_analyze_error{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_reset_error on_reset_error{};
inline constexpr record_phase_error record_phase_error{};

}  // namespace emel::tensor::lifetime_analyzer::action
