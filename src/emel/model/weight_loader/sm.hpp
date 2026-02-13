#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"
#include "emel/buffer_allocator/sm.hpp"

namespace emel::model::weight_loader {

namespace test {
struct sm_test_peer;
}

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct loading_mmap {};
    struct loading_streamed {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::load_weights>[guard::use_mmap] /
          action::map_weights_mmap =
          sml::state<loading_mmap>,
      sml::state<initialized> + sml::event<event::load_weights>[guard::not_use_mmap] /
          action::load_weights_streamed =
          sml::state<loading_streamed>,

      sml::state<loading_mmap> + sml::event<event::weights_loaded>[guard::no_error] =
          sml::state<done>,
      sml::state<loading_mmap> + sml::event<event::weights_loaded>[guard::has_error] =
          sml::state<errored>,

      sml::state<loading_streamed> + sml::event<event::weights_loaded>[guard::no_error] =
          sml::state<done>,
      sml::state<loading_streamed> + sml::event<event::weights_loaded>[guard::has_error] =
          sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  sm() : emel::sm<model>(*this) {}

  bool load(const event::load_weights & load_weights) {
    if (!process_event(load_weights)) {
      return false;
    }

    begin_load(load_weights, guard::use_mmap(load_weights));
    return emit_weights_loaded();
  }

 private:
  friend struct test::sm_test_peer;

  void dispatch_loading_done_to_owner(const event::weights_loaded & loaded) {
    if (model_loader_sm_ == nullptr) {
      return;
    }

    model_loader_sm_->process_event(
      events::loading_done{
        .status_code = loaded.status_code,
        .bytes_total = loaded.bytes_total,
        .bytes_done = loaded.bytes_done
      }
    );
  }

  void dispatch_loading_error_to_owner(const event::weights_loaded & loaded) {
    if (model_loader_sm_ == nullptr) {
      return;
    }

    model_loader_sm_->process_event(
      events::loading_error{
        .status_code = loaded.status_code,
        .used_mmap = loaded.used_mmap
      }
    );
  }

  void begin_load(const event::load_weights & load_weights, const bool use_mmap) {
    model_loader_sm_ = load_weights.model_loader_sm;
    buffer_allocator_sm_ = load_weights.buffer_allocator_sm;
    status_code_ = EMEL_OK;
    n_weight_files_ = 1;
    bytes_total_ = 0;
    bytes_done_ = 0;

    use_mmap_ = use_mmap;
    use_direct_io_ = load_weights.request_direct_io && load_weights.direct_io_supported;
    if (use_mmap_ && use_direct_io_) {
      use_direct_io_ = false;
    }

    check_tensors_ = load_weights.check_tensors;
    no_alloc_ = load_weights.no_alloc;

    if (use_mmap_ && !load_weights.mmap_supported) {
      status_code_ = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    if (model_loader_sm_ == nullptr) {
      status_code_ = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    if (use_mmap_) {
      return;
    }
  }

  bool emit_weights_loaded() {
    const event::weights_loaded loaded{
      .success = status_code_ == EMEL_OK,
      .status_code = status_code_,
      .used_mmap = use_mmap_,
      .bytes_total = bytes_total_,
      .bytes_done = bytes_done_
    };
    const bool handled = process_event(loaded);
    if (!handled) {
      return false;
    }

    if (loaded.success) {
      dispatch_loading_done_to_owner(loaded);
      return true;
    }

    dispatch_loading_error_to_owner(loaded);
    return false;
  }

  int32_t status_code_ = 0;
  int32_t n_weight_files_ = 0;
  uint64_t bytes_total_ = 0;
  uint64_t bytes_done_ = 0;
  bool use_mmap_ = false;
  bool use_direct_io_ = false;
  bool check_tensors_ = true;
  bool no_alloc_ = false;
  emel::model::loader::sm * model_loader_sm_ = nullptr;
  emel::buffer_allocator::sm * buffer_allocator_sm_ = nullptr;
};

namespace test {

struct sm_test_peer {
  static void begin_load(
      sm & machine, const event::load_weights & load_weights, const bool use_mmap) {
    machine.begin_load(load_weights, use_mmap);
  }

  static bool emit_weights_loaded(sm & machine) {
    return machine.emit_weights_loaded();
  }

  static void dispatch_loading_done_to_owner(
      sm & machine, const event::weights_loaded & loaded) {
    machine.dispatch_loading_done_to_owner(loaded);
  }

  static void dispatch_loading_error_to_owner(
      sm & machine, const event::weights_loaded & loaded) {
    machine.dispatch_loading_error_to_owner(loaded);
  }
};

}  // namespace test

}  // namespace emel::model::weight_loader
