#include <array>
#include <span>

#include "doctest/doctest.h"

#include "emel/gguf/loader/guards.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"

namespace {

void on_probe_done(const emel::gguf::loader::events::probe_done &) {}
void on_probe_error(const emel::gguf::loader::events::probe_error &) {}
void on_bind_done(const emel::gguf::loader::events::bind_done &) {}
void on_bind_error(const emel::gguf::loader::events::bind_error &) {}
void on_parse_done(const emel::gguf::loader::events::parse_done &) {}
void on_parse_error(const emel::gguf::loader::events::parse_error &) {}

}  // namespace

TEST_CASE("gguf loader probe bind parse lifecycle") {
  emel::gguf::loader::sm machine{};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const emel::gguf::loader::event::bind_done_fn bind_done_cb =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const emel::gguf::loader::event::bind_error_fn bind_error_cb =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const emel::gguf::loader::event::parse_done_fn parse_done_cb =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const emel::gguf::loader::event::parse_error_fn parse_error_cb =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  std::array<uint8_t, 4> file_bytes = {};
  emel::gguf::loader::requirements req = {};
  const emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{file_bytes},
    req,
    probe_done_cb,
    probe_error_cb,
  };
  CHECK(machine.process_event(probe));

  std::array<uint8_t, 8> kv_arena = {};
  std::array<emel::gguf::loader::kv_entry, 1> kv_entries = {};
  std::array<emel::model::data::tensor_record, 1> tensors = {};
  const emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{kv_arena},
    std::span<emel::gguf::loader::kv_entry>{kv_entries},
    std::span<emel::model::data::tensor_record>{tensors},
    bind_done_cb,
    bind_error_cb,
  };
  CHECK(machine.process_event(bind));

  const emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{file_bytes},
    parse_done_cb,
    parse_error_cb,
  };
  CHECK(machine.process_event(parse));
}

TEST_CASE("gguf loader probe rejects invalid inputs") {
  emel::gguf::loader::sm machine{};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();

  emel::gguf::loader::requirements req = {};
  const emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{},
    req,
    probe_done_cb,
    probe_error_cb,
  };
  CHECK_FALSE(machine.process_event(probe));
}

TEST_CASE("gguf loader explicit error guard classification") {
  emel::gguf::loader::action::context ctx = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const emel::gguf::loader::event::bind_done_fn bind_done_cb =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const emel::gguf::loader::event::bind_error_fn bind_error_cb =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const emel::gguf::loader::event::parse_done_fn parse_done_cb =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const emel::gguf::loader::event::parse_error_fn parse_error_cb =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  std::array<uint8_t, 4> file_bytes = {};
  emel::gguf::loader::requirements req = {};
  emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{file_bytes},
    req,
    probe_done_cb,
    probe_error_cb,
  };
  emel::gguf::loader::event::probe_ctx probe_ctx = {};
  emel::gguf::loader::event::probe_runtime probe_runtime{probe, probe_ctx};

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::probe_error_none{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::probe_error_invalid_request{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::probe_error_model_invalid{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::probe_error_capacity{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::probe_error_parse_failed{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::probe_error_internal_error{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::probe_error_untracked{}(probe_runtime, ctx));

  probe_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::probe_error_unknown{}(probe_runtime, ctx));

  std::array<uint8_t, 8> kv_arena = {};
  std::array<emel::gguf::loader::kv_entry, 1> kv_entries = {};
  std::array<emel::model::data::tensor_record, 1> tensors = {};
  emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{kv_arena},
    std::span<emel::gguf::loader::kv_entry>{kv_entries},
    std::span<emel::model::data::tensor_record>{tensors},
    bind_done_cb,
    bind_error_cb,
  };
  emel::gguf::loader::event::bind_ctx bind_ctx = {};
  emel::gguf::loader::event::bind_runtime bind_runtime{bind, bind_ctx};

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::bind_error_none{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::bind_error_invalid_request{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::bind_error_model_invalid{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::bind_error_capacity{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::bind_error_parse_failed{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::bind_error_internal_error{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::bind_error_untracked{}(bind_runtime, ctx));

  bind_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::bind_error_unknown{}(bind_runtime, ctx));

  emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{file_bytes},
    parse_done_cb,
    parse_error_cb,
  };
  emel::gguf::loader::event::parse_ctx parse_ctx = {};
  emel::gguf::loader::event::parse_runtime parse_runtime{parse, parse_ctx};

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::parse_error_none{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::parse_error_invalid_request{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::parse_error_model_invalid{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::parse_error_capacity{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::parse_error_parse_failed{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::parse_error_internal_error{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::parse_error_untracked{}(parse_runtime, ctx));

  parse_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::parse_error_unknown{}(parse_runtime, ctx));
}
