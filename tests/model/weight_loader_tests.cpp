#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/guards.hpp"
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

TEST_CASE("weight loader guard selection covers branches") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::guard::use_mmap_selected use_mmap{};
  emel::model::weight_loader::guard::use_stream_selected use_stream{};

  ctx.use_mmap = true;
  CHECK(use_mmap(ctx));
  CHECK(!use_stream(ctx));

  ctx.use_mmap = false;
  CHECK(!use_mmap(ctx));
  CHECK(use_stream(ctx));
}

TEST_CASE("weight loader guard error predicates") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::guard::phase_ok phase_ok{};
  emel::model::weight_loader::guard::phase_failed phase_failed{};

  ctx.phase_error = EMEL_OK;
  CHECK(phase_ok(ctx));
  CHECK(!phase_failed(ctx));

  ctx.phase_error = EMEL_ERR_BACKEND;
  CHECK(!phase_ok(ctx));
  CHECK(phase_failed(ctx));
}

TEST_CASE("weight loader guards cover capability paths") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::event::load_weights request{};
  ctx.request = &request;

  CHECK(emel::model::weight_loader::guard::has_request{}(ctx));
  CHECK(!emel::model::weight_loader::guard::can_init_mappings{}(ctx));
  CHECK(emel::model::weight_loader::guard::skip_init_mappings{}(ctx));

  request.init_mappings = [](const emel::model::weight_loader::event::load_weights &,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::can_init_mappings{}(ctx));
  CHECK(!emel::model::weight_loader::guard::skip_init_mappings{}(ctx));

  request.load_streamed = nullptr;
  CHECK(emel::model::weight_loader::guard::cannot_load_streamed{}(ctx));

  request.load_streamed = [](const emel::model::weight_loader::event::load_weights &,
                             uint64_t *,
                             uint64_t *,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::can_load_streamed{}(ctx));

  request.map_mmap = nullptr;
  CHECK(emel::model::weight_loader::guard::cannot_load_mmap{}(ctx));
  request.map_mmap = map_mmap_ok;
  CHECK(emel::model::weight_loader::guard::can_load_mmap{}(ctx));

  request.check_tensors = true;
  request.validate = nullptr;
  CHECK(emel::model::weight_loader::guard::skip_validate{}(ctx));
  CHECK(!emel::model::weight_loader::guard::can_validate{}(ctx));

  request.validate = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::can_validate{}(ctx));

  request.clean_up = nullptr;
  ctx.used_mmap = true;
  CHECK(emel::model::weight_loader::guard::skip_clean_up{}(ctx));

  request.clean_up = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::can_clean_up{}(ctx));
}

TEST_CASE("weight loader strategy selection handles mmap and direct io") {
  emel::model::weight_loader::event::load_weights request{};
  request.request_mmap = true;
  request.mmap_supported = true;
  request.request_direct_io = false;
  request.direct_io_supported = true;

  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::action::begin_load(request, ctx);
  emel::model::weight_loader::action::select_strategy(ctx);
  CHECK(ctx.use_mmap);
  CHECK(!ctx.use_direct_io);

  request.request_direct_io = true;
  emel::model::weight_loader::action::begin_load(request, ctx);
  emel::model::weight_loader::action::select_strategy(ctx);
  CHECK(!ctx.use_mmap);
  CHECK(ctx.use_direct_io);
}

TEST_CASE("weight loader actions handle init and load failures") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::event::load_weights request{};
  ctx.request = &request;

  request.init_mappings = [](const emel::model::weight_loader::event::load_weights &,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::weight_loader::action::run_init_mappings(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.init_mappings = [](const emel::model::weight_loader::event::load_weights &,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::weight_loader::action::run_init_mappings(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &,
                        uint64_t *,
                        uint64_t *,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::weight_loader::action::run_load_mmap(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &,
                        uint64_t * done,
                        uint64_t * total,
                        int32_t * err_out) {
    if (done != nullptr) {
      *done = 2;
    }
    if (total != nullptr) {
      *total = 4;
    }
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::weight_loader::action::run_load_mmap(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
  CHECK(ctx.bytes_done == 2);
  CHECK(ctx.bytes_total == 4);
  CHECK(ctx.used_mmap);

  request.load_streamed = [](const emel::model::weight_loader::event::load_weights &,
                             uint64_t *,
                             uint64_t *,
                             int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::weight_loader::action::run_load_streamed(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("weight loader validate and cleanup handle failures") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::event::load_weights request{};
  ctx.request = &request;

  request.validate = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::weight_loader::action::run_validate(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  request.clean_up = [](const emel::model::weight_loader::event::load_weights &,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::weight_loader::action::run_clean_up(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("weight loader publish_done and publish_error notify owners") {
  owner_state owner{};
  emel::model::weight_loader::event::load_weights request{};
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  emel::model::weight_loader::action::context ctx{};
  ctx.request = &request;
  ctx.bytes_total = 10;
  ctx.bytes_done = 5;
  ctx.used_mmap = true;

  emel::model::weight_loader::action::publish_done(ctx);
  CHECK(owner.done);
  CHECK(owner.done_event.bytes_total == 10);
  CHECK(owner.done_event.bytes_done == 5);
  CHECK(owner.done_event.used_mmap);
  CHECK(ctx.request == nullptr);

  ctx.request = &request;
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  emel::model::weight_loader::action::publish_error(ctx);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_BACKEND);
  CHECK(ctx.request == nullptr);
}

TEST_CASE("weight loader on_unexpected sets backend error") {
  emel::model::weight_loader::action::context ctx{};
  emel::model::weight_loader::action::on_unexpected(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}
