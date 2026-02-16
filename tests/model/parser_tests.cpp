#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/model/parser/sm.hpp"

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

bool parse_architecture_ok(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool map_architecture_ok(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool parse_hparams_ok(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool parse_vocab_ok(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool map_tensors_ok(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("parser completes happy path") {
  emel::model::data model_data{};
  owner_state owner{};
  emel::model::parser::sm machine{};

  emel::model::parser::event::parse_model request{};
  request.model = &model_data;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors_impl = map_tensors_ok;
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
  emel::model::parser::sm machine{};

  emel::model::parser::event::parse_model request{};
  request.model = &model_data;
  request.parse_architecture = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_PARSE_FAILED;
    }
    return false;
  };
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors_impl = map_tensors_ok;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;

  CHECK(machine.process_event(request));
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_PARSE_FAILED);
}
