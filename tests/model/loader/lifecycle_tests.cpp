#include "doctest/doctest.h"

#include <memory>

#include "emel/error/error.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

void on_done(void * object, const emel::model::loader::events::load_done & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->bytes_total = ev.bytes_total;
  owner->bytes_done = ev.bytes_done;
  owner->used_mmap = ev.used_mmap;
}

void on_error(void * object, const emel::model::loader::events::load_error & ev) noexcept {
  auto * owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
}

emel::error::type parse_ok(void *, const emel::model::loader::event::load & req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type parse_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::parse_failed);
}

emel::error::type load_weights_ok(void *,
                                  const emel::model::loader::event::load & req,
                                  uint64_t & bytes_total,
                                  uint64_t & bytes_done,
                                  bool & used_mmap) noexcept {
  static_cast<void>(req);
  bytes_total = 4096;
  bytes_done = 4096;
  used_mmap = true;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type map_layers_ok(void *, const emel::model::loader::event::load & req) noexcept {
  req.model_data.n_layers = 2;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_structure_ok(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_architecture_ok(void *,
                                           const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace

TEST_CASE("model loader lifecycle succeeds on full load path") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_ok};
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 4096);
  CHECK(owner.bytes_done == 4096);
  CHECK(owner.used_mmap);
}

TEST_CASE("model loader rejects missing source payload") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  emel::model::loader::event::load request{*model, parse_model};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader allows vocab-only parse without weight and map callbacks") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.vocab_only = true;
  request.check_tensors = false;
  request.validate_architecture = false;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 0);
  CHECK(owner.bytes_done == 0);
  CHECK_FALSE(owner.used_mmap);
}

TEST_CASE("model loader propagates parse failure") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_fail};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == emel::error::cast(emel::model::loader::error::parse_failed));
}
