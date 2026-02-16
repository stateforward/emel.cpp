#include "doctest/doctest.h"

#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/sm.hpp"
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

bool dispatch_parsing_done(void * loader_sm, const emel::model::loader::events::parsing_done & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_parsing_error(void * loader_sm, const emel::model::loader::events::parsing_error & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_loading_done(void * loader_sm, const emel::model::loader::events::loading_done & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_loading_error(void * loader_sm, const emel::model::loader::events::loading_error & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool map_parser_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
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
  request.validate_architecture = true;

  emel::model::loader::events::structure_validated validated{&request};
  CHECK(emel::model::loader::guard::has_arch_validate{}(validated));
  CHECK(!emel::model::loader::guard::no_arch_validate{}(validated));

  request.validate_architecture = false;
  CHECK(!emel::model::loader::guard::has_arch_validate{}(validated));
  CHECK(emel::model::loader::guard::no_arch_validate{}(validated));
}

TEST_CASE("model loader completes happy path with mmap") {
  emel::model::data model_data{};
  owner_state owner{};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.map_parser = map_parser_ok;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

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

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.map_parser = map_parser_ok;
  request.parse_architecture = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_PARSE_FAILED;
    }
    return false;
  };
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
  request.map_mmap = map_mmap_ok;
  request.load_streamed = load_streamed_ok;
  request.map_layers = map_layers_ok;
  request.validate_structure = validate_structure_ok;
  request.validate_architecture_impl = validate_architecture_ok;
  request.error_out = &err;
  request.owner_sm = &owner;
  request.dispatch_done = dispatch_done;
  request.dispatch_error = dispatch_error;
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

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

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.map_parser = map_parser_ok;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
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
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

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

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.map_parser = map_parser_ok;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
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
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

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

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.map_parser = map_parser_ok;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
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
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

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

  emel::model::parser::sm parser_machine{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};

  emel::model::loader::event::load request{model_data};
  request.model_path = "model.gguf";
  request.vocab_only = true;
  request.map_parser = map_parser_ok;
  request.parse_architecture = parse_architecture_ok;
  request.map_architecture = map_architecture_ok;
  request.parse_hparams = parse_hparams_ok;
  request.parse_vocab = parse_vocab_ok;
  request.map_tensors = map_tensors_ok;
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
  request.parser_sm = &parser_machine;
  request.dispatch_parse_model = emel::model::parser::dispatch_parse_model;
  request.weight_loader_sm = &weight_machine;
  request.dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;

  CHECK(loader_machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(owner.done);
  CHECK(!owner.error);
  CHECK(!load_called);
  map_mmap_called_flag = nullptr;
}
