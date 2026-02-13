#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

emel::model::loader::sm make_loader_ready_for_weight_load() {
  emel::model::loader::sm loader_sm{};
  emel::model::data model_data{};

  CHECK(
      emel::model::loader::load(loader_sm, emel::model::loader::event::load{
                                               .model_data = model_data,
                                               .model_path = "mock_model.gguf",
                                               .request_mmap = true,
                                           }));
  CHECK(loader_sm.process_event(
      emel::model::loader::event::mapping_parser_done{}));
  CHECK(loader_sm.process_event(emel::model::parser::events::parsing_done{}));

  return loader_sm;
}

bool dispatch_reached_owner(
    emel::model::loader::sm &owner,
    const emel::model::weight_loader::event::load_weights &ev) {
  emel::model::weight_loader::sm weight_loader{};
  CHECK(weight_loader.load(ev));
  return owner.process_event(emel::model::loader::event::layers_mapped{});
}

} // namespace

TEST_CASE("weight_loader_starts_in_initialized") {
  emel::model::weight_loader::sm machine{};
  int state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("weight_loader_uses_mmap_when_available") {
  auto owner = make_loader_ready_for_weight_load();
  CHECK(dispatch_reached_owner(owner,
                               emel::model::weight_loader::event::load_weights{
                                   .request_mmap = true,
                                   .mmap_supported = true,
                                   .buffer_allocator_sm = nullptr,
                                   .model_loader_sm = &owner}));
}

TEST_CASE("weight_loader_falls_back_to_streamed_when_mmap_not_supported") {
  auto owner = make_loader_ready_for_weight_load();
  CHECK(dispatch_reached_owner(owner,
                               emel::model::weight_loader::event::load_weights{
                                   .request_mmap = true,
                                   .mmap_supported = false,
                                   .buffer_allocator_sm = nullptr,
                                   .model_loader_sm = &owner}));
}

TEST_CASE(
    "weight_loader_forces_streamed_when_direct_io_supersedes_mmap_request") {
  auto owner = make_loader_ready_for_weight_load();
  CHECK(dispatch_reached_owner(owner,
                               emel::model::weight_loader::event::load_weights{
                                   .request_mmap = true,
                                   .request_direct_io = true,
                                   .mmap_supported = true,
                                   .direct_io_supported = true,
                                   .buffer_allocator_sm = nullptr,
                                   .model_loader_sm = &owner}));
}

TEST_CASE("weight_loader_reports_error_when_owner_machine_missing") {
  auto owner = make_loader_ready_for_weight_load();
  emel::model::weight_loader::sm weight_loader{};

  CHECK_FALSE(weight_loader.load(emel::model::weight_loader::event::load_weights{
      .request_mmap = true,
      .mmap_supported = true,
      .buffer_allocator_sm = nullptr,
      .model_loader_sm = nullptr}));
  CHECK_FALSE(owner.process_event(emel::model::loader::event::layers_mapped{}));
}

TEST_CASE("weight_loader_load_fails_when_event_is_unhandled") {
  auto owner = make_loader_ready_for_weight_load();
  emel::model::weight_loader::sm weight_loader{};

  const emel::model::weight_loader::event::load_weights load_ev{
    .request_mmap = true,
    .mmap_supported = true,
    .buffer_allocator_sm = nullptr,
    .model_loader_sm = &owner
  };

  CHECK(weight_loader.load(load_ev));
  CHECK_FALSE(weight_loader.load(load_ev));
}

TEST_CASE("weight_loader_test_peer_exercises_private_branches") {
  using emel::model::weight_loader::test::sm_test_peer;
  using emel::model::weight_loader::event::load_weights;
  using emel::model::weight_loader::event::weights_loaded;

  auto owner = make_loader_ready_for_weight_load();

  emel::model::weight_loader::sm no_state_machine{};
  CHECK_FALSE(sm_test_peer::emit_weights_loaded(no_state_machine));

  emel::model::weight_loader::sm no_owner_machine{};
  CHECK(
    no_owner_machine.process_event(load_weights{
      .request_mmap = true,
      .mmap_supported = true,
      .buffer_allocator_sm = nullptr,
      .model_loader_sm = nullptr
    }));
  CHECK(sm_test_peer::emit_weights_loaded(no_owner_machine));

  emel::model::weight_loader::sm branch_machine{};
  CHECK(
    branch_machine.process_event(load_weights{
      .request_mmap = true,
      .mmap_supported = true,
      .buffer_allocator_sm = nullptr,
      .model_loader_sm = &owner
    }));

  sm_test_peer::begin_load(
    branch_machine,
    load_weights{
      .request_mmap = true,
      .request_direct_io = true,
      .mmap_supported = true,
      .direct_io_supported = true,
      .model_loader_sm = &owner
    },
    true);

  sm_test_peer::begin_load(
    branch_machine,
    load_weights{
      .request_mmap = true,
      .mmap_supported = false,
      .model_loader_sm = &owner
    },
    true);

  sm_test_peer::dispatch_loading_done_to_owner(
    branch_machine,
    weights_loaded{
      .success = true,
      .status_code = EMEL_OK,
      .used_mmap = true,
      .bytes_total = 256,
      .bytes_done = 128
    });

  sm_test_peer::dispatch_loading_error_to_owner(
    branch_machine,
    weights_loaded{
      .success = false,
      .status_code = EMEL_ERR_INVALID_ARGUMENT,
      .used_mmap = false,
      .bytes_total = 0,
      .bytes_done = 0
    });

  sm_test_peer::dispatch_loading_error_to_owner(
    no_owner_machine,
    weights_loaded{
      .success = false,
      .status_code = EMEL_ERR_INVALID_ARGUMENT,
      .used_mmap = false,
      .bytes_total = 0,
      .bytes_done = 0
    });
}

TEST_CASE("weight_loader_guards_model_status") {
  using namespace emel::model::weight_loader;
  emel::model::weight_loader::event::load_weights mmap_ev{
      .request_mmap = true, .mmap_supported = true};
  emel::model::weight_loader::event::load_weights conflict_ev{
      .request_mmap = true,
      .request_direct_io = true,
      .mmap_supported = true,
      .direct_io_supported = true};
  emel::model::weight_loader::event::weights_loaded ok{.success = true,
                                                       .status_code = EMEL_OK};
  emel::model::weight_loader::event::weights_loaded bad{
      .success = false, .status_code = EMEL_ERR_IO};

  CHECK(guard::use_mmap(mmap_ev));
  CHECK_FALSE(guard::use_mmap(conflict_ev));
  CHECK_FALSE(guard::not_use_mmap(mmap_ev));
  CHECK(guard::not_use_mmap(conflict_ev));
  CHECK(guard::no_error(ok));
  CHECK_FALSE(guard::no_error(bad));
  CHECK(guard::has_error(bad));
  CHECK_FALSE(guard::has_error(ok));
}
