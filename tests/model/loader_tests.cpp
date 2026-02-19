#include <cstring>

#include "doctest/doctest.h"

#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/parser/actions.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/parser/guards.hpp"
#include "emel/parser/map.hpp"
#include "emel/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::model::loader::events::load_done done_event{};
  emel::model::loader::events::load_error error_event{};
};

bool * map_mmap_called_flag = nullptr;

bool dispatch_done(void * owner_sm, const emel::model::loader::events::load_done & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->done = true;
  owner->done_event = ev;
  return true;
}

bool dispatch_error(void * owner_sm, const emel::model::loader::events::load_error & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->error = true;
  owner->error_event = ev;
  return true;
}

bool map_parser_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool can_handle_any(const emel::model::loader::event::load &) {
  return true;
}

struct parser_map_builder {
  emel::parser::entry entry{};
  emel::parser::map map{};

  parser_map_builder(void * parser_sm,
                     emel::parser::map_parser_fn map_parser,
                     emel::parser::dispatch_parse_fn dispatch_parse,
                     emel::parser::can_handle_fn can_handle = can_handle_any) {
    entry.kind_id = emel::parser::kind::gguf;
    entry.parser_sm = parser_sm;
    entry.map_parser = map_parser;
    entry.can_handle = can_handle;
    entry.dispatch_parse = dispatch_parse;
    map.entries = &entry;
    map.count = 1;
  }
};

bool dispatch_parse_done(void *, const emel::parser::event::parse_model & ev) {
  if (ev.dispatch_done == nullptr) {
    return false;
  }
  return ev.dispatch_done(ev.owner_sm,
                          emel::model::loader::events::parsing_done{ev.loader_request});
}

bool dispatch_parse_error_parse_failed(void *, const emel::parser::event::parse_model & ev) {
  if (ev.dispatch_error == nullptr) {
    return false;
  }
  return ev.dispatch_error(ev.owner_sm,
                           emel::model::loader::events::parsing_error{
                             ev.loader_request,
                             EMEL_ERR_PARSE_FAILED,
                           });
}

bool dispatch_parse_error_backend(void *, const emel::parser::event::parse_model & ev) {
  if (ev.dispatch_error == nullptr) {
    return false;
  }
  return ev.dispatch_error(ev.owner_sm,
                           emel::model::loader::events::parsing_error{
                             ev.loader_request,
                             EMEL_ERR_BACKEND,
                           });
}

bool dispatch_parse_fail(void *, const emel::parser::event::parse_model &) {
  return false;
}

bool map_mmap_ok(const emel::model::weight_loader::event::load_weights &,
                 uint64_t * bytes_done,
                 uint64_t * bytes_total,
                 int32_t * err_out) {
  if (bytes_done != nullptr) {
    *bytes_done = 256;
  }
  if (bytes_total != nullptr) {
    *bytes_total = 512;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool dispatch_load_weights_fail(void *, const emel::model::weight_loader::event::load_weights &) {
  return false;
}

bool map_mmap_flagged(const emel::model::weight_loader::event::load_weights &,
                      uint64_t *,
                      uint64_t *,
                      int32_t * err_out) {
  if (map_mmap_called_flag != nullptr) {
    *map_mmap_called_flag = true;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_BACKEND;
  }
  return false;
}

bool load_streamed_ok(const emel::model::weight_loader::event::load_weights &,
                      uint64_t * bytes_done,
                      uint64_t * bytes_total,
                      int32_t * err_out) {
  if (bytes_done != nullptr) {
    *bytes_done = 128;
  }
  if (bytes_total != nullptr) {
    *bytes_total = 256;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool map_layers_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool validate_structure_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool validate_architecture_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("model loader guards follow request flags") {
  emel::model::data model_data{};
  emel::model::loader::event::load request{model_data};
  emel::model::loader::action::context ctx{};
  ctx.request = &request;

  request.validate_architecture = true;
  CHECK(emel::model::loader::guard::has_arch_validate{}(ctx));
  CHECK(!emel::model::loader::guard::no_arch_validate{}(ctx));

  request.validate_architecture = false;
  CHECK(!emel::model::loader::guard::has_arch_validate{}(ctx));
  CHECK(emel::model::loader::guard::no_arch_validate{}(ctx));

  request.vocab_only = false;
  CHECK(emel::model::loader::guard::should_load_weights{}(ctx));
  CHECK(!emel::model::loader::guard::skip_weights{}(ctx));

  request.vocab_only = true;
  CHECK(!emel::model::loader::guard::should_load_weights{}(ctx));
  CHECK(emel::model::loader::guard::skip_weights{}(ctx));

  request.check_tensors = false;
  CHECK(emel::model::loader::guard::skip_validate_structure{}(ctx));
  CHECK(!emel::model::loader::guard::can_validate_structure{}(ctx));

  request.check_tensors = true;
  request.validate_structure = validate_structure_ok;
  CHECK(!emel::model::loader::guard::skip_validate_structure{}(ctx));
  CHECK(emel::model::loader::guard::can_validate_structure{}(ctx));
}

TEST_CASE("model loader guards cover parser and weight paths") {
  emel::model::data model_data{};
  emel::model::loader::event::load request{model_data};
  emel::model::loader::action::context ctx{};

  CHECK(!emel::model::loader::guard::has_request{}(ctx));
  ctx.request = &request;
  CHECK(emel::model::loader::guard::has_request{}(ctx));

  CHECK(!emel::model::loader::guard::can_parse{}(ctx));
  CHECK(emel::model::loader::guard::cannot_parse{}(ctx));

  ctx.parser_sm = &ctx;
  ctx.parser_dispatch = dispatch_parse_done;
  CHECK(emel::model::loader::guard::can_parse{}(ctx));
  CHECK(!emel::model::loader::guard::cannot_parse{}(ctx));

  emel::parser::map empty_map{};
  request.parser_map = nullptr;
  CHECK(!emel::model::loader::guard::can_map_parser{}(request));
  CHECK(emel::model::loader::guard::cannot_map_parser{}(request));

  request.parser_map = &empty_map;
  CHECK(!emel::model::loader::guard::can_map_parser{}(request));
  CHECK(emel::model::loader::guard::cannot_map_parser{}(request));

  parser_map_builder parser_map(&ctx, map_parser_ok, dispatch_parse_done);
  request.parser_map = &parser_map.map;
  CHECK(emel::model::loader::guard::can_map_parser{}(request));
  CHECK(!emel::model::loader::guard::cannot_map_parser{}(request));

  request.dispatch_load_weights = nullptr;
  request.weight_loader_sm = nullptr;
  CHECK(!emel::model::loader::guard::can_load_weights{}(ctx));
  CHECK(emel::model::loader::guard::cannot_load_weights{}(ctx));

  emel::model::weight_loader::sm weight_machine{};
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.weight_loader_sm = &weight_machine;
  CHECK(emel::model::loader::guard::can_load_weights{}(ctx));
  CHECK(!emel::model::loader::guard::cannot_load_weights{}(ctx));

  request.map_layers = map_layers_ok;
  CHECK(emel::model::loader::guard::can_map_layers{}(ctx));
  CHECK(!emel::model::loader::guard::cannot_map_layers{}(ctx));

  request.map_layers = nullptr;
  CHECK(!emel::model::loader::guard::can_map_layers{}(ctx));
  CHECK(emel::model::loader::guard::cannot_map_layers{}(ctx));

  request.check_tensors = true;
  request.validate_structure = nullptr;
  CHECK(emel::model::loader::guard::cannot_validate_structure{}(ctx));

  request.validate_architecture = true;
  request.validate_architecture_impl = nullptr;
  CHECK(emel::model::loader::guard::has_arch_validate_and_cannot_validate_architecture{}(ctx));

  ctx.phase_error = EMEL_OK;
  CHECK(emel::model::loader::guard::phase_ok_and_has_arch_validate_and_cannot_validate_architecture{}(ctx));
}

TEST_CASE("model loader completes happy path with mmap") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok, dispatch_parse_done);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.parser_map = &parser_map.map;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(loader_machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(owner.done);
  CHECK(!owner.error);
  CHECK(owner.done_event.bytes_done == 256);
  CHECK(owner.done_event.bytes_total == 512);
  CHECK(owner.done_event.used_mmap);
}

TEST_CASE("model loader reports parsing errors") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_OK;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok,
                                dispatch_parse_error_parse_failed);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.parser_map = &parser_map.map;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(!loader_machine.process_event(request));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("model loader reports loading errors") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_OK;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok, dispatch_parse_done);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.parser_map = &parser_map.map;
  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &,
                        uint64_t *,
                        uint64_t *,
                        int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  };
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(!loader_machine.process_event(request));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_BACKEND);
}

TEST_CASE("model loader reports structure validation errors") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_OK;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok, dispatch_parse_done);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.parser_map = &parser_map.map;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  };
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(!loader_machine.process_event(request));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("model loader reports architecture validation errors") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_OK;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok, dispatch_parse_done);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.parser_map = &parser_map.map;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  };
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(!loader_machine.process_event(request));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  CHECK(!owner.done);
  CHECK(owner.error);
  CHECK(owner.error_event.err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("model loader skips weight loading when vocab_only") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  bool load_called = false;

  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  parser_map_builder parser_map(&loader_machine, map_parser_ok, dispatch_parse_done);

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.vocab_only = true;
  request.parser_map = &parser_map.map;
  map_mmap_called_flag = &load_called;
  request.map_mmap = map_mmap_flagged;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;

  CHECK(loader_machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(owner.done);
  CHECK(!owner.error);
  CHECK(!load_called);
  map_mmap_called_flag = nullptr;
}

namespace {

struct owner_probe {
  int done_calls = 0;
  int error_calls = 0;
  emel::model::loader::events::load_done last_done{};
  emel::model::loader::events::load_error last_error{};

  bool dispatch_done(const emel::model::loader::events::load_done & ev) {
    last_done = ev;
    done_calls++;
    return true;
  }
  bool dispatch_error(const emel::model::loader::events::load_error & ev) {
    last_error = ev;
    error_calls++;
    return true;
  }
};

bool dispatch_load_weights_error(void *, const emel::model::weight_loader::event::load_weights & ev) {
  if (ev.dispatch_error != nullptr) {
    ev.dispatch_error(ev.owner_sm,
                      emel::model::loader::events::loading_error{ev.loader_request, EMEL_ERR_BACKEND});
  }
  return true;
}

bool dispatch_load_weights_done(void *, const emel::model::weight_loader::event::load_weights & ev) {
  if (ev.dispatch_done != nullptr) {
    ev.dispatch_done(ev.owner_sm,
                     emel::model::loader::events::loading_done{ev.loader_request, 64, 32, true});
  }
  return true;
}

}  // namespace

TEST_CASE("loader actions map_parser and parse update phase_error") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  emel::model::loader::action::begin_load(request, ctx);

  parser_map_builder parser_fail(&ctx,
                                 [](const emel::model::loader::event::load &, int32_t * err_out) {
                                   if (err_out != nullptr) {
                                     *err_out = EMEL_OK;
                                   }
                                   return false;
                                 },
                                 dispatch_parse_done);
  request.parser_map = &parser_fail.map;
  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  parser_map_builder parser_ok(&ctx, map_parser_ok, dispatch_parse_error_backend);
  request.parser_map = &parser_ok.map;
  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::model::loader::action::run_parse(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.parser_dispatch = dispatch_parse_done;
  emel::model::loader::action::run_parse(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  ctx.parser_sm = nullptr;
  ctx.parser_dispatch = nullptr;
  emel::model::loader::action::run_parse(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("loader actions handle missing callbacks and requests") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};

  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.request = &request;
  emel::parser::map empty_map{};
  request.parser_map = &empty_map;
  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_FORMAT_UNSUPPORTED);

  emel::parser::entry bad_entry{};
  bad_entry.kind_id = emel::parser::kind::gguf;
  bad_entry.parser_sm = &ctx;
  bad_entry.map_parser = nullptr;
  bad_entry.can_handle = can_handle_any;
  emel::parser::map bad_map{&bad_entry, 1};
  request.parser_map = &bad_map;
  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  parser_map_builder parser_map(&ctx, map_parser_ok, nullptr);
  request.parser_map = &parser_map.map;
  emel::model::loader::action::run_map_parser(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
  CHECK(ctx.parser_dispatch != nullptr);

  ctx.parser_sm = &ctx;
  ctx.parser_dispatch = dispatch_parse_fail;
  emel::model::loader::action::run_parse(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.dispatch_load_weights = nullptr;
  request.weight_loader_sm = nullptr;
  emel::model::loader::action::run_load_weights(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  request.dispatch_load_weights = dispatch_load_weights_fail;
  request.weight_loader_sm = &ctx;
  emel::model::loader::action::run_load_weights(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.map_layers = nullptr;
  emel::model::loader::action::run_map_layers(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  request.validate_structure = nullptr;
  emel::model::loader::action::run_validate_structure(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  request.validate_architecture_impl = nullptr;
  emel::model::loader::action::run_validate_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.request = nullptr;
  emel::model::loader::action::publish_done(ctx);
  emel::model::loader::action::publish_error(ctx);
}

TEST_CASE("loader actions handle null callback owners") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::events::parsing_done parsing_done{&request};
  emel::model::loader::events::parsing_error parsing_error{&request, EMEL_ERR_BACKEND};
  emel::model::loader::events::loading_done loading_done{&request, 0, 0, false};
  emel::model::loader::events::loading_error loading_error{&request, EMEL_ERR_BACKEND};

  CHECK(!emel::model::loader::action::store_parsing_done(nullptr, parsing_done));
  CHECK(!emel::model::loader::action::store_parsing_error(nullptr, parsing_error));
  CHECK(!emel::model::loader::action::store_loading_done(nullptr, loading_done));
  CHECK(!emel::model::loader::action::store_loading_error(nullptr, loading_error));
}

TEST_CASE("loader actions load_weights and map_layers update phase_error") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  ctx.request = &request;

  request.weight_loader_sm = &ctx;
  request.dispatch_load_weights = dispatch_load_weights_error;
  emel::model::loader::action::run_load_weights(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.dispatch_load_weights = dispatch_load_weights_done;
  emel::model::loader::action::run_load_weights(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
  CHECK(ctx.bytes_total == 64);
  CHECK(ctx.bytes_done == 32);
  CHECK(ctx.used_mmap);

  request.map_layers = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::loader::action::run_map_layers(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  request.map_layers = map_layers_ok;
  emel::model::loader::action::run_map_layers(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("loader actions validate structure and architecture update errors") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  ctx.request = &request;

  request.validate_structure = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::loader::action::run_validate_structure(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  request.validate_structure = validate_structure_ok;
  emel::model::loader::action::run_validate_structure(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::model::loader::action::skip_validate_structure(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  request.validate_architecture_impl = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::loader::action::run_validate_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  request.validate_architecture_impl = validate_architecture_ok;
  emel::model::loader::action::run_validate_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("loader actions publish_done and publish_error notify owners") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  owner_probe owner{};
  int32_t err = EMEL_OK;

  request.owner_sm = &owner;
  request.dispatch_done = [](void * owner_sm, const emel::model::loader::events::load_done & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_done(ev);
  };
  request.dispatch_error = [](void * owner_sm, const emel::model::loader::events::load_error & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_error(ev);
  };
  request.error_out = &err;
  ctx.request = &request;
  ctx.bytes_total = 10;
  ctx.bytes_done = 5;
  ctx.used_mmap = true;

  emel::model::loader::action::publish_done(ctx);
  CHECK(owner.done_calls == 1);
  CHECK(err == EMEL_OK);
  CHECK(owner.last_done.bytes_total == 10);
  CHECK(owner.last_done.bytes_done == 5);
  CHECK(owner.last_done.used_mmap);
  CHECK(ctx.request == nullptr);

  ctx.request = &request;
  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  emel::model::loader::action::publish_error(ctx);
  CHECK(owner.error_calls == 1);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(owner.last_error.err == EMEL_ERR_BACKEND);
  CHECK(ctx.request == nullptr);
}

TEST_CASE("parser actions cover error and success branches") {
  emel::parser::action::context ctx{};
  emel::parser::event::parse_model request{};
  CHECK(emel::parser::guard::invalid_parse_request{}(request));

  emel::parser::gguf::context gguf_ctx{};
  emel::model::data model{};
  request.model = &model;
  request.format_ctx = &gguf_ctx;
  emel::parser::action::begin_parse(request, ctx);

  emel::parser::gguf::action::run_parse_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  std::memcpy(gguf_ctx.architecture.data(), "test", 4);
  gguf_ctx.architecture_len = 4;
  emel::parser::gguf::action::run_parse_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  const char * mismatch_list[] = {"other"};
  ctx.request.architectures = mismatch_list;
  ctx.request.n_architectures = 1;
  emel::parser::gguf::action::run_map_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  const char * match_list[] = {ctx.request.model->architecture_name.data()};
  ctx.request.architectures = match_list;
  ctx.request.n_architectures = 1;
  emel::parser::gguf::action::run_map_architecture(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  gguf_ctx.block_count = 0;
  ctx.request.model->params.n_ctx = 1;
  ctx.request.model->params.n_embd = 1;
  emel::parser::gguf::action::run_parse_hparams(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  gguf_ctx.block_count = 2;
  ctx.request.model->params.n_ctx = 1;
  ctx.request.model->params.n_embd = 1;
  emel::parser::gguf::action::run_parse_hparams(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  ctx.request.model->vocab_data.n_tokens = 0;
  emel::parser::gguf::action::run_parse_vocab(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  ctx.request.model->vocab_data.n_tokens = 1;
  ctx.request.model->vocab_data.entries[0].text_length = 1;
  emel::parser::gguf::action::run_parse_vocab(ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  ctx.request.map_tensors = false;
  CHECK(emel::parser::guard::skip_map_tensors{}(ctx));

  ctx.request.map_tensors = true;
  CHECK(emel::parser::guard::should_map_tensors{}(ctx));
  ctx.request.model->n_tensors = 0;
  emel::parser::gguf::action::run_map_tensors(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_MODEL_INVALID);

  ctx.request.model->n_tensors = 1;
  emel::parser::gguf::action::run_map_tensors(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("loader on_unexpected actions dispatch error when possible") {
  emel::model::loader::action::context ctx{};
  emel::model::loader::action::on_unexpected(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  emel::parser::action::context parser_ctx{};
  emel::parser::action::on_unexpected(parser_ctx);
  CHECK(parser_ctx.last_error == EMEL_ERR_BACKEND);
}
