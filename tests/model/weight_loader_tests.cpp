#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/sm.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

struct sink_base {
  template <class Event>
  void process_event(const Event &) {}
};

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

TEST_CASE("weight loader guard selection covers branches") {
  emel::model::weight_loader::guard::use_mmap_selected use_mmap{};
  emel::model::weight_loader::guard::use_stream_selected use_stream{};

  emel::model::weight_loader::events::strategy_selected ev{};
  ev.use_mmap = true;
  CHECK(use_mmap(ev));
  CHECK(!use_stream(ev));

  ev.use_mmap = false;
  CHECK(!use_mmap(ev));
  CHECK(use_stream(ev));
}

TEST_CASE("weight loader guard error predicates") {
  emel::model::weight_loader::guard::no_error no_error{};
  emel::model::weight_loader::guard::has_error has_error{};
  emel::model::weight_loader::events::weights_loaded ok{.err = EMEL_OK};
  emel::model::weight_loader::events::weights_loaded bad{.err = EMEL_ERR_BACKEND};

  CHECK(no_error(ok));
  CHECK(!has_error(ok));
  CHECK(!no_error(bad));
  CHECK(has_error(bad));
}

TEST_CASE("weight loader select_strategy emits selection") {
  emel::model::weight_loader::event::load_weights request{};
  request.request_mmap = true;
  request.mmap_supported = true;
  request.request_direct_io = false;
  request.direct_io_supported = true;

  struct owner : sink_base {
    using sink_base::process_event;
    bool called = false;
    emel::model::weight_loader::events::strategy_selected captured{};
    void process_event(const emel::model::weight_loader::events::strategy_selected & ev) {
      called = true;
      captured = ev;
    }
  };
  owner sink{};
  emel::detail::process_support<owner, emel::model::weight_loader::process_t> support{&sink};

  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::action::select_strategy{}(request, ctx, support.process_);
  CHECK(sink.called);
  CHECK(sink.captured.use_mmap);
  CHECK(!sink.captured.use_direct_io);
}

TEST_CASE("weight loader init_mappings handles failures") {
  emel::model::weight_loader::event::load_weights request{};
  struct owner : sink_base {
    using sink_base::process_event;
    bool called = false;
    emel::model::weight_loader::events::mappings_ready captured{};
    void process_event(const emel::model::weight_loader::events::mappings_ready & ev) {
      called = true;
      captured = ev;
    }
  };
  owner sink{};
  emel::detail::process_support<owner, emel::model::weight_loader::process_t> support{&sink};

  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::events::strategy_selected selection{&request, true, false, EMEL_OK};
  CHECK(emel::model::weight_loader::guard::use_mmap_no_error_skip_init_mappings{}(selection));
  emel::model::weight_loader::action::skip_init_mappings{}(selection, ctx, support.process_);
  CHECK(sink.called);
  CHECK(sink.captured.err == EMEL_OK);

  sink.called = false;
  request.init_mappings = [](const emel::model::weight_loader::event::load_weights &,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::weight_loader::guard::use_mmap_no_error_can_init_mappings{}(selection));
  emel::model::weight_loader::action::init_mappings{}(selection, ctx, support.process_);
  CHECK(sink.called);
  CHECK(sink.captured.err == EMEL_ERR_BACKEND);
}

TEST_CASE("weight loader validate and cleaning_up handle failures") {
  emel::model::weight_loader::event::load_weights request{};
  request.check_tensors = true;
  request.validate = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  request.clean_up = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };

  struct vowner : sink_base {
    using sink_base::process_event;
    bool called = false;
    emel::model::weight_loader::events::validation_done captured{};
    void process_event(const emel::model::weight_loader::events::validation_done & ev) {
      called = true;
      captured = ev;
    }
  };
  vowner vsink{};
  emel::detail::process_support<vowner, emel::model::weight_loader::process_t> vsupport{&vsink};

  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::events::weights_loaded loaded{&request, EMEL_OK, false, 0, 0};
  emel::model::weight_loader::action::validate{}(loaded, ctx, vsupport.process_);
  CHECK(vsink.called);
  CHECK(vsink.captured.err == EMEL_ERR_MODEL_INVALID);

  struct cowner : sink_base {
    using sink_base::process_event;
    bool called = false;
    emel::model::weight_loader::events::cleaning_up_done captured{};
    void process_event(const emel::model::weight_loader::events::cleaning_up_done & ev) {
      called = true;
      captured = ev;
    }
  };
  cowner csink{};
  emel::detail::process_support<cowner, emel::model::weight_loader::process_t> csupport{&csink};
  ctx.used_mmap = true;
  emel::model::weight_loader::action::cleaning_up{}(
    emel::model::weight_loader::events::validation_done{&request, EMEL_OK},
    ctx,
    csupport.process_);
  CHECK(csink.called);
  CHECK(csink.captured.err == EMEL_ERR_BACKEND);
}
