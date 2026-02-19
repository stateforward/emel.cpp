#include <cstring>

#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/parser/actions.hpp"
#include "emel/parser/dispatch.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/parser/guards.hpp"
#include "emel/parser/map.hpp"
#include "emel/parser/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::model::loader::events::parsing_done done_event{};
  emel::model::loader::events::parsing_error error_event{};
};

bool dispatch_done(void * owner_sm, const emel::model::loader::events::parsing_done & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->done = true;
  owner->done_event = ev;
  return true;
}

bool dispatch_error(void * owner_sm, const emel::model::loader::events::parsing_error & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->error = true;
  owner->error_event = ev;
  return true;
}

bool can_handle_true(const emel::model::loader::event::load &) {
  return true;
}

bool can_handle_false(const emel::model::loader::event::load &) {
  return false;
}

}  // namespace

TEST_CASE("parser completes happy path") {
  emel::model::data model_data{};
  owner_state owner{};
  emel::parser::gguf::sm machine{};
  emel::parser::gguf::context gguf_ctx{};

  std::memcpy(gguf_ctx.architecture.data(), "test", 4);
  gguf_ctx.architecture_len = 4;
  gguf_ctx.block_count = 1;
  model_data.params.n_ctx = 1;
  model_data.params.n_embd = 1;
  model_data.vocab_data.n_tokens = 1;
  model_data.vocab_data.entries[0].text_length = 1;
  model_data.n_tensors = 1;

  emel::parser::event::parse_model request{};
  request.model = &model_data;
  request.format_ctx = &gguf_ctx;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK(!owner.error);
}

TEST_CASE("parser reports parsing errors") {
  emel::model::data model_data{};
  owner_state owner{};
  emel::parser::gguf::sm machine{};
  emel::parser::gguf::context gguf_ctx{};

  emel::parser::event::parse_model request{};
  request.model = &model_data;
  request.format_ctx = &gguf_ctx;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  CHECK_FALSE(machine.process_event(request));
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("parser actions and guards handle state transitions") {
  emel::parser::action::context ctx{};
  emel::parser::event::parse_model request{};

  CHECK(emel::parser::guard::invalid_parse_request{}(request));
  CHECK(!emel::parser::guard::valid_parse_request{}(request));

  emel::model::data model_data{};
  emel::parser::gguf::context gguf_ctx{};
  request.model = &model_data;
  request.format_ctx = &gguf_ctx;
  request.map_tensors = false;

  CHECK(emel::parser::guard::valid_parse_request{}(request));
  emel::parser::action::begin_parse(request, ctx);
  CHECK(ctx.request.model == &model_data);
  CHECK(ctx.request.owner_sm == nullptr);
  CHECK(ctx.request.dispatch_done == nullptr);
  CHECK(ctx.request.dispatch_error == nullptr);

  CHECK(emel::parser::guard::skip_map_tensors{}(ctx));
  CHECK(!emel::parser::guard::should_map_tensors{}(ctx));
  CHECK(emel::parser::guard::phase_ok_and_skip_map_tensors{}(ctx));
  CHECK(!emel::parser::guard::phase_ok_and_map_tensors{}(ctx));

  emel::parser::action::set_invalid_argument(ctx);
  CHECK(emel::parser::guard::phase_failed{}(ctx));
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::parser::action::set_backend_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.phase_error = EMEL_ERR_PARSE_FAILED;
  ctx.last_error = EMEL_OK;
  emel::parser::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_PARSE_FAILED);

  emel::parser::action::mark_done(ctx);
  CHECK(emel::parser::guard::phase_ok{}(ctx));

  ctx.request.map_tensors = true;
  CHECK(emel::parser::guard::should_map_tensors{}(ctx));
  CHECK(emel::parser::guard::phase_ok_and_map_tensors{}(ctx));

  emel::parser::action::on_unexpected(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  emel::parser::action::clear_request_action(ctx);
  CHECK(ctx.request.model == nullptr);
}

TEST_CASE("parser map selection skips invalid entries") {
  emel::model::data model_data{};
  emel::model::loader::event::load request{model_data};

  emel::parser::entry entries[3] = {};
  entries[0].kind_id = emel::parser::kind::count;
  entries[0].can_handle = can_handle_true;
  entries[1].kind_id = emel::parser::kind::gguf;
  entries[1].can_handle = nullptr;
  entries[2].kind_id = emel::parser::kind::gguf;
  entries[2].can_handle = can_handle_true;

  emel::parser::map parser_map{entries, 3};
  emel::parser::selection selection = emel::parser::select(&parser_map, request);
  CHECK(selection.entry == &entries[2]);
  CHECK(selection.kind_id == emel::parser::kind::gguf);

  selection = emel::parser::select(nullptr, request);
  CHECK(selection.entry == nullptr);

  entries[2].can_handle = can_handle_false;
  selection = emel::parser::select(&parser_map, request);
  CHECK(selection.entry == nullptr);
}

TEST_CASE("parser dispatch table covers kind lookup") {
  emel::model::data model_data{};
  emel::parser::gguf::context gguf_ctx{};
  emel::parser::gguf::sm machine{};

  std::memcpy(gguf_ctx.architecture.data(), "test", 4);
  gguf_ctx.architecture_len = 4;
  gguf_ctx.block_count = 1;
  model_data.params.n_ctx = 1;
  model_data.params.n_embd = 1;
  model_data.vocab_data.n_tokens = 1;
  model_data.vocab_data.entries[0].text_length = 1;
  model_data.n_tensors = 1;

  emel::parser::event::parse_model request{};
  request.model = &model_data;
  request.format_ctx = &gguf_ctx;

  const auto dispatch = emel::parser::dispatch_for_kind(emel::parser::kind::gguf);
  CHECK(dispatch != nullptr);
  CHECK(!dispatch(nullptr, request));
  CHECK(dispatch(&machine, request));

  CHECK(emel::parser::dispatch_for_kind(emel::parser::kind::count) == nullptr);
}
