#include "doctest/doctest.h"

#include "emel/model/loader/actions.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/guards.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/actions.hpp"
#include "emel/model/parser/sm.hpp"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/sm.hpp"

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

namespace {

struct loader_sink {
  int mapping_done = 0;
  int mapping_error = 0;
  int parsing_error = 0;
  int loading_error = 0;
  int layers_error = 0;
  int layers_done = 0;
  int structure_error = 0;
  int structure_done = 0;
  int architecture_error = 0;
  int architecture_done = 0;
  int32_t last_err = EMEL_OK;

  bool process_event(const emel::model::loader::events::mapping_parser_done &) {
    mapping_done++;
    return true;
  }
  bool process_event(const emel::model::loader::events::mapping_parser_error & ev) {
    mapping_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::parsing_error & ev) {
    parsing_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::loading_error & ev) {
    loading_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::layers_map_error & ev) {
    layers_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::layers_mapped &) {
    layers_done++;
    return true;
  }
  bool process_event(const emel::model::loader::events::structure_error & ev) {
    structure_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::structure_validated &) {
    structure_done++;
    return true;
  }
  bool process_event(const emel::model::loader::events::architecture_error & ev) {
    architecture_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::loader::events::architecture_validated &) {
    architecture_done++;
    return true;
  }

  template <class Event>
  bool process_event(const Event &) { return true; }
};

struct parser_sink {
  int parse_arch_error = 0;
  int parse_arch_done = 0;
  int map_arch_error = 0;
  int map_arch_done = 0;
  int parse_hparams_error = 0;
  int parse_hparams_done = 0;
  int parse_vocab_error = 0;
  int parse_vocab_done = 0;
  int map_tensors_error = 0;
  int map_tensors_done = 0;
  int32_t last_err = EMEL_OK;

  bool process_event(const emel::model::parser::events::parse_architecture_error & ev) {
    parse_arch_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::parser::events::parse_architecture_done &) {
    parse_arch_done++;
    return true;
  }
  bool process_event(const emel::model::parser::events::map_architecture_error & ev) {
    map_arch_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::parser::events::map_architecture_done &) {
    map_arch_done++;
    return true;
  }
  bool process_event(const emel::model::parser::events::parse_hparams_error & ev) {
    parse_hparams_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::parser::events::parse_hparams_done &) {
    parse_hparams_done++;
    return true;
  }
  bool process_event(const emel::model::parser::events::parse_vocab_error & ev) {
    parse_vocab_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::parser::events::parse_vocab_done &) {
    parse_vocab_done++;
    return true;
  }
  bool process_event(const emel::model::parser::events::map_tensors_error & ev) {
    map_tensors_error++;
    last_err = ev.err;
    return true;
  }
  bool process_event(const emel::model::parser::events::map_tensors_done &) {
    map_tensors_done++;
    return true;
  }

  template <class Event>
  bool process_event(const Event &) { return true; }
};

struct weight_sink {
  int weights_loaded = 0;
  int32_t last_err = EMEL_OK;
  bool used_mmap = false;

  bool process_event(const emel::model::weight_loader::events::weights_loaded & ev) {
    weights_loaded++;
    last_err = ev.err;
    used_mmap = ev.used_mmap;
    return true;
  }

  template <class Event>
  bool process_event(const Event &) { return true; }
};

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

}  // namespace

TEST_CASE("loader actions map_parser and parse handle failures") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  emel::model::parser::sm parser_machine{};
  emel::model::loader::sm loader_machine{};
  loader_sink sink{};
  emel::detail::process_support<loader_sink, emel::model::loader::process_t> support{&sink};
  auto & process = support.process_;

  CHECK(!emel::model::loader::guard::can_map_parser{}(request));

  request.map_parser = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::loader::action::map_parser{}(request, ctx, process);
  CHECK(sink.mapping_error == 1);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  request.map_parser = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::loader::action::map_parser{}(request, ctx, process);
  CHECK(sink.mapping_done == 1);

  emel::model::loader::events::mapping_parser_done done{&request};
  CHECK(!emel::model::loader::guard::can_parse{}(done));

  request.parser_sm = &parser_machine;
  request.loader_sm = &loader_machine;
  request.dispatch_parsing_done = dispatch_parsing_done;
  request.dispatch_parsing_error = dispatch_parsing_error;
  request.dispatch_parse_model = [](void *, const emel::model::parser::event::parse_model &) {
    return false;
  };
  CHECK(emel::model::loader::guard::can_parse{}(done));
  emel::model::loader::action::parse{}(done, ctx, process);
  CHECK(sink.parsing_error == 1);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);
}

TEST_CASE("loader actions load_weights and map_layers cover branches") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  emel::model::weight_loader::sm weight_machine{};
  emel::model::loader::sm loader_machine{};
  loader_sink sink{};
  emel::detail::process_support<loader_sink, emel::model::loader::process_t> support{&sink};
  auto & process = support.process_;

  emel::model::loader::events::parsing_done parsed{&request};
  CHECK(!emel::model::loader::guard::can_load_weights{}(parsed));

  request.weight_loader_sm = &weight_machine;
  request.loader_sm = &loader_machine;
  request.dispatch_loading_done = dispatch_loading_done;
  request.dispatch_loading_error = dispatch_loading_error;
  request.dispatch_load_weights = [](void *, const emel::model::weight_loader::event::load_weights &) {
    return false;
  };
  CHECK(emel::model::loader::guard::can_load_weights{}(parsed));
  emel::model::loader::action::load_weights{}(parsed, ctx, process);
  CHECK(sink.loading_error == 1);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  emel::model::loader::events::loading_done loaded{&request};
  CHECK(!emel::model::loader::guard::can_map_layers{}(loaded));

  request.map_layers = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::loader::action::store_and_map_layers{}(loaded, ctx, process);
  CHECK(sink.layers_error == 1);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  request.map_layers = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::loader::guard::can_map_layers{}(loaded));
  emel::model::loader::action::store_and_map_layers{}(loaded, ctx, process);
  CHECK(sink.layers_done == 1);
}

TEST_CASE("loader actions validate_structure and validate_architecture") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  loader_sink sink{};
  emel::detail::process_support<loader_sink, emel::model::loader::process_t> support{&sink};
  auto & process = support.process_;

  emel::model::loader::events::layers_mapped mapped{&request};
  CHECK(!emel::model::loader::guard::can_validate_structure{}(mapped));

  request.check_tensors = false;
  CHECK(emel::model::loader::guard::can_validate_structure{}(mapped));
  emel::model::loader::action::validate_structure{}(mapped, ctx, process);
  CHECK(sink.structure_done == 1);

  request.check_tensors = true;
  CHECK(!emel::model::loader::guard::can_validate_structure{}(mapped));

  request.validate_structure = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::loader::guard::can_validate_structure{}(mapped));
  emel::model::loader::action::validate_structure{}(mapped, ctx, process);
  CHECK(sink.structure_error == 1);
  CHECK(sink.last_err == EMEL_ERR_MODEL_INVALID);

  request.validate_structure = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::loader::action::validate_structure{}(mapped, ctx, process);
  CHECK(sink.structure_done == 2);

  request.validate_architecture = true;
  emel::model::loader::events::structure_validated validated{&request};
  CHECK(!emel::model::loader::guard::can_validate_architecture{}(validated));

  request.validate_architecture_impl = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::loader::guard::can_validate_architecture{}(validated));
  emel::model::loader::action::validate_architecture{}(validated, ctx, process);
  CHECK(sink.architecture_error == 1);
  CHECK(sink.last_err == EMEL_ERR_MODEL_INVALID);

  request.validate_architecture_impl = [](const emel::model::loader::event::load &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::loader::action::validate_architecture{}(validated, ctx, process);
  CHECK(sink.architecture_done == 1);
}

TEST_CASE("loader actions dispatch_done and dispatch_error") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  emel::model::loader::action::context ctx{};
  emel::detail::process_support<loader_sink, emel::model::loader::process_t> dummy_support{nullptr};
  auto & dummy_process = dummy_support.process_;

  owner_probe owner{};
  request.owner_sm = &owner;
  request.dispatch_done = [](void * owner_sm, const emel::model::loader::events::load_done & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_done(ev);
  };
  request.dispatch_error = [](void * owner_sm, const emel::model::loader::events::load_error & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_error(ev);
  };

  ctx.bytes_total = 10;
  ctx.bytes_done = 5;
  ctx.used_mmap = true;

  emel::model::loader::events::architecture_validated done{&request};
  emel::model::loader::action::dispatch_done{}(done, ctx, dummy_process);
  CHECK(owner.done_calls == 1);
  CHECK(owner.last_done.bytes_total == 10);
  CHECK(owner.last_done.bytes_done == 5);
  CHECK(owner.last_done.used_mmap);

  emel::model::loader::events::loading_error err{&request, EMEL_ERR_BACKEND};
  emel::model::loader::action::dispatch_error{}(err, ctx, dummy_process);
  CHECK(owner.error_calls == 1);
  CHECK(owner.last_error.err == EMEL_ERR_BACKEND);

  request.owner_sm = nullptr;
  emel::model::loader::action::dispatch_done{}(done, ctx, dummy_process);
  emel::model::loader::action::dispatch_error{}(err, ctx, dummy_process);
}

TEST_CASE("parser actions cover error and success branches") {
  parser_sink sink{};
  emel::detail::process_support<parser_sink, emel::model::parser::process_t> support{&sink};
  auto & process = support.process_;

  emel::model::parser::event::parse_model request{};
  CHECK(!emel::model::parser::guard::can_parse_architecture{}(request));

  request.parse_architecture = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  emel::model::parser::action::parse_architecture{}(request, process);
  CHECK(sink.parse_arch_error == 1);
  CHECK(sink.last_err == EMEL_ERR_PARSE_FAILED);

  request.parse_architecture = parse_architecture_ok;
  emel::model::parser::action::parse_architecture{}(request, process);
  CHECK(sink.parse_arch_done == 1);

  emel::model::parser::events::parse_architecture_done parsed{&request};
  CHECK(!emel::model::parser::guard::can_map_architecture{}(parsed));

  request.map_architecture = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::parser::guard::can_map_architecture{}(parsed));
  emel::model::parser::action::map_architecture{}(parsed, process);
  CHECK(sink.map_arch_error == 1);
  CHECK(sink.last_err == EMEL_ERR_MODEL_INVALID);

  request.map_architecture = map_architecture_ok;
  emel::model::parser::action::map_architecture{}(parsed, process);
  CHECK(sink.map_arch_done == 1);

  emel::model::parser::events::map_architecture_done mapped{&request};
  CHECK(!emel::model::parser::guard::can_parse_hparams{}(mapped));

  request.parse_hparams = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::parser::guard::can_parse_hparams{}(mapped));
  emel::model::parser::action::parse_hparams{}(mapped, process);
  CHECK(sink.parse_hparams_error == 1);
  CHECK(sink.last_err == EMEL_ERR_PARSE_FAILED);

  request.parse_hparams = parse_hparams_ok;
  emel::model::parser::action::parse_hparams{}(mapped, process);
  CHECK(sink.parse_hparams_done == 1);

  emel::model::parser::events::parse_hparams_done hp{&request};
  CHECK(!emel::model::parser::guard::can_parse_vocab{}(hp));

  request.parse_vocab = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::parser::guard::can_parse_vocab{}(hp));
  emel::model::parser::action::parse_vocab{}(hp, process);
  CHECK(sink.parse_vocab_error == 1);
  CHECK(sink.last_err == EMEL_ERR_PARSE_FAILED);

  request.parse_vocab = parse_vocab_ok;
  emel::model::parser::action::parse_vocab{}(hp, process);
  CHECK(sink.parse_vocab_done == 1);

  emel::model::parser::events::parse_vocab_done vocab{&request};
  request.map_tensors = false;
  CHECK(emel::model::parser::guard::skip_map_tensors{}(vocab));

  request.map_tensors = true;
  CHECK(!emel::model::parser::guard::can_map_tensors{}(vocab));
  CHECK(emel::model::parser::guard::cannot_map_tensors{}(vocab));

  request.map_tensors_impl = [](const emel::model::parser::event::parse_model &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::parser::guard::can_map_tensors{}(vocab));
  emel::model::parser::action::map_tensors{}(vocab, process);
  CHECK(sink.map_tensors_error == 1);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  request.map_tensors_impl = map_tensors_ok;
  emel::model::parser::action::map_tensors{}(vocab, process);
  CHECK(sink.map_tensors_done == 1);
}

TEST_CASE("weight loader actions cover branches") {
  weight_sink sink{};
  emel::detail::process_support<weight_sink, emel::model::weight_loader::process_t> support{&sink};
  auto & process = support.process_;
  emel::model::weight_loader::action::context ctx{};

  emel::model::weight_loader::event::load_weights request{};
  emel::model::weight_loader::events::mappings_ready ready{&request, EMEL_OK};
  CHECK(emel::model::weight_loader::guard::mappings_ready_no_error_cannot_load_mmap{}(ready));
  emel::model::weight_loader::action::reject_invalid_mmap{}(ready, ctx, process);
  CHECK(sink.weights_loaded == 1);
  CHECK(sink.last_err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(sink.used_mmap);

  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &, uint64_t *, uint64_t *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  CHECK(emel::model::weight_loader::guard::mappings_ready_no_error_can_load_mmap{}(ready));
  emel::model::weight_loader::action::load_mmap{}(ready, ctx, process);
  CHECK(sink.weights_loaded == 2);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &, uint64_t * done, uint64_t * total, int32_t * err_out) {
    if (done != nullptr) {
      *done = 2;
    }
    if (total != nullptr) {
      *total = 4;
    }
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::mappings_ready_no_error_can_load_mmap{}(ready));
  emel::model::weight_loader::action::load_mmap{}(ready, ctx, process);
  CHECK(sink.weights_loaded == 3);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  request.map_mmap = [](const emel::model::weight_loader::event::load_weights &, uint64_t * done, uint64_t * total, int32_t * err_out) {
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
  CHECK(emel::model::weight_loader::guard::mappings_ready_no_error_can_load_mmap{}(ready));
  emel::model::weight_loader::action::load_mmap{}(ready, ctx, process);
  CHECK(sink.weights_loaded == 4);
  CHECK(sink.last_err == EMEL_OK);

  emel::model::weight_loader::event::load_weights stream_request{};
  emel::model::weight_loader::events::strategy_selected stream_selected{&stream_request, false, false, EMEL_OK};
  CHECK(emel::model::weight_loader::guard::use_stream_no_error_cannot_load_streamed{}(stream_selected));
  emel::model::weight_loader::action::reject_invalid_streamed{}(stream_selected, ctx, process);
  CHECK(sink.weights_loaded == 5);
  CHECK(sink.used_mmap == false);
  CHECK(sink.last_err == EMEL_ERR_INVALID_ARGUMENT);

  stream_request.load_streamed = [](const emel::model::weight_loader::event::load_weights &, uint64_t *, uint64_t *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return false;
  };
  stream_selected.request = &stream_request;
  CHECK(emel::model::weight_loader::guard::use_stream_no_error_can_load_streamed{}(stream_selected));
  emel::model::weight_loader::action::load_streamed{}(stream_selected, ctx, process);
  CHECK(sink.weights_loaded == 6);
  CHECK(sink.last_err == EMEL_ERR_BACKEND);

  stream_request.load_streamed = [](const emel::model::weight_loader::event::load_weights &, uint64_t * done, uint64_t * total, int32_t * err_out) {
    if (done != nullptr) {
      *done = 1;
    }
    if (total != nullptr) {
      *total = 2;
    }
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  CHECK(emel::model::weight_loader::guard::use_stream_no_error_can_load_streamed{}(stream_selected));
  emel::model::weight_loader::action::load_streamed{}(stream_selected, ctx, process);
  CHECK(sink.weights_loaded == 7);
  CHECK(sink.last_err == EMEL_OK);

  stream_request.validate = [](const emel::model::weight_loader::event::load_weights &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  stream_request.clean_up = [](const emel::model::weight_loader::event::load_weights &, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::model::weight_loader::events::weights_loaded done{&stream_request, EMEL_OK, true, 2, 1};
  emel::model::weight_loader::action::store_and_validate{}(done, ctx, process);
  emel::model::weight_loader::action::cleaning_up{}(
    emel::model::weight_loader::events::validation_done{&stream_request, EMEL_OK},
    ctx,
    process);
  emel::model::weight_loader::action::dispatch_done{}(
    emel::model::weight_loader::events::cleaning_up_done{&stream_request, EMEL_OK},
    ctx,
    process);
  emel::model::weight_loader::action::store_and_dispatch_error{}(
    emel::model::weight_loader::events::weights_loaded{&stream_request, EMEL_ERR_BACKEND, false, 0, 0},
    ctx,
    process);

  stream_request.owner_sm = nullptr;
  emel::model::weight_loader::action::dispatch_done{}(
    emel::model::weight_loader::events::cleaning_up_done{&stream_request, EMEL_OK},
    ctx,
    process);
  emel::model::weight_loader::action::dispatch_error{}(done, process);
}

TEST_CASE("loader on_unexpected actions dispatch error when possible") {
  emel::model::data model{};
  owner_probe owner{};
  emel::model::loader::event::load request{model};
  request.owner_sm = &owner;
  request.dispatch_error = [](void * owner_sm, const emel::model::loader::events::load_error & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_error(ev);
  };

  emel::model::loader::action::context ctx{};
  emel::detail::process_support<loader_sink, emel::model::loader::process_t> dummy_support{nullptr};

  emel::model::loader::events::mapping_parser_error ev{&request, EMEL_ERR_BACKEND};
  emel::model::loader::action::on_unexpected{}(ev, ctx, dummy_support.process_);
  CHECK(owner.error_calls == 1);

  emel::model::parser::event::parse_model parse_request{};
  parse_request.loader_request = &request;
  parse_request.owner_sm = &owner;
  parse_request.dispatch_error = [](void * owner_sm,
                                    const emel::model::loader::events::parsing_error & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_error(
      emel::model::loader::events::load_error{ev.request, ev.err});
  };
  emel::model::parser::process_t parser_process{};
  emel::model::parser::action::on_unexpected{}(parse_request, parser_process);

  emel::model::weight_loader::event::load_weights load_request{};
  load_request.loader_request = &request;
  load_request.owner_sm = &owner;
  load_request.dispatch_error = [](void * owner_sm,
                                   const emel::model::loader::events::loading_error & ev) {
    return static_cast<owner_probe *>(owner_sm)->dispatch_error(
      emel::model::loader::events::load_error{ev.request, ev.err});
  };
  emel::model::weight_loader::action::context wl_ctx{};
  emel::model::weight_loader::process_t wl_process{};
  emel::model::weight_loader::action::on_unexpected{}(load_request, wl_ctx, wl_process);
}
