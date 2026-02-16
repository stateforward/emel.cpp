#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::model::loader::events::loading_done done_event{};
  emel::model::loader::events::loading_error error_event{};
};

bool dispatch_done(void * owner_sm, const emel::model::loader::events::loading_done & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->done = true;
  owner->done_event = ev;
  return true;
}

bool dispatch_error(void * owner_sm, const emel::model::loader::events::loading_error & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->error = true;
  owner->error_event = ev;
  return true;
}

bool map_mmap_ok(const emel::model::weight_loader::event::load_weights &,
                 uint64_t * bytes_done,
                 uint64_t * bytes_total,
                 int32_t * err_out) {
  if (bytes_done != nullptr) {
    *bytes_done = 64;
  }
  if (bytes_total != nullptr) {
    *bytes_total = 128;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("weight loader dispatches loading done for mmap") {
  owner_state owner{};
  emel::model::weight_loader::sm machine{};

  emel::model::weight_loader::event::load_weights request{};
  request.request_mmap = true;
  request.map_mmap = map_mmap_ok;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK(!owner.error);
  CHECK(owner.done_event.bytes_done == 64);
  CHECK(owner.done_event.bytes_total == 128);
  CHECK(owner.done_event.used_mmap);
}

TEST_CASE("weight loader dispatches loading errors") {
  owner_state owner{};
  emel::model::weight_loader::sm machine{};

  emel::model::weight_loader::event::load_weights request{};
  request.request_mmap = true;
  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &,
                        uint64_t *,
                        uint64_t *,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  };
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  CHECK(machine.process_event(request));
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_BACKEND);
}
