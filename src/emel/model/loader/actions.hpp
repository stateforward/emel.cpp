#pragma once

#include <cstdint>
#include <type_traits>

#include "boost/sml.hpp"
#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::loader {

using process_t = boost::sml::back::process<
  events::mapping_parser_done,
  events::mapping_parser_error,
  events::parsing_done,
  events::parsing_error,
  events::loading_done,
  events::loading_error,
  events::layers_mapped,
  events::layers_map_error,
  events::structure_validated,
  events::structure_error,
  events::architecture_validated,
  events::architecture_error>;

}  // namespace emel::model::loader

namespace emel::model::loader::action {

struct context {
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct reset {
  void operator()(const event::load & ev, context & ctx) const {
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct map_parser {
  void operator()(const event::load & ev, context &, process_t & process) const {
    if (ev.map_parser == nullptr) {
      process(events::mapping_parser_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = ev.map_parser(ev, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::mapping_parser_error{&ev, err});
      return;
    }
    process(events::mapping_parser_done{&ev});
  }
};

struct start_map_parser {
  void operator()(const event::load & ev, context & ctx, process_t & process) const {
    reset{}(ev, ctx);
    map_parser{}(ev, ctx, process);
  }
};

struct parse {
  void operator()(const events::mapping_parser_done & ev, context &, process_t & process) const {
    const event::load * request = ev.request;
    if (request == nullptr || request->dispatch_parse_model == nullptr || request->parser_sm == nullptr
        || request->loader_sm == nullptr || request->dispatch_parsing_done == nullptr
        || request->dispatch_parsing_error == nullptr) {
      process(events::parsing_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    emel::model::parser::event::parse_model parse_request{
      .model = &request->model_data,
      .model_path = request->model_path,
      .architectures = request->architectures,
      .n_architectures = request->n_architectures,
      .file_handle = request->file_handle,
      .format_ctx = request->format_ctx,
      .map_tensors = !request->vocab_only,
      .parse_architecture = request->parse_architecture,
      .map_architecture = request->map_architecture,
      .parse_hparams = request->parse_hparams,
      .parse_vocab = request->parse_vocab,
      .map_tensors_impl = request->map_tensors,
      .loader_request = request,
      .owner_sm = request->loader_sm,
      .dispatch_done = request->dispatch_parsing_done,
      .dispatch_error = request->dispatch_parsing_error
    };
    const bool ok = request->dispatch_parse_model(request->parser_sm, parse_request);
    if (!ok) {
      process(events::parsing_error{request, EMEL_ERR_BACKEND});
    }
  }
};

struct load_weights {
  void operator()(const events::parsing_done & ev, context &, process_t & process) const {
    const event::load * request = ev.request;
    if (request == nullptr || request->dispatch_load_weights == nullptr || request->weight_loader_sm == nullptr
        || request->loader_sm == nullptr || request->dispatch_loading_done == nullptr
        || request->dispatch_loading_error == nullptr) {
      process(events::loading_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    emel::model::weight_loader::event::load_weights load_request{
      .request_mmap = request->request_mmap,
      .request_direct_io = request->request_direct_io,
      .check_tensors = request->check_tensors,
      .no_alloc = request->no_alloc,
      .mmap_supported = request->mmap_supported,
      .direct_io_supported = request->direct_io_supported,
      .buffer_allocator_sm = request->buffer_allocator_sm,
      .map_mmap = request->map_mmap,
      .load_streamed = request->load_streamed,
      .progress_callback = request->progress_callback,
      .progress_user_data = request->progress_user_data,
      .loader_request = request,
      .owner_sm = request->loader_sm,
      .dispatch_done = request->dispatch_loading_done,
      .dispatch_error = request->dispatch_loading_error
    };
    const bool ok = request->dispatch_load_weights(request->weight_loader_sm, load_request);
    if (!ok) {
      process(events::loading_error{request, EMEL_ERR_BACKEND});
    }
  }
};

struct store_and_map_layers {
  void operator()(const events::loading_done & ev, context & ctx, process_t & process) const {
    ctx.bytes_total = ev.bytes_total;
    ctx.bytes_done = ev.bytes_done;
    ctx.used_mmap = ev.used_mmap;
    const event::load * request = ev.request;
    if (request == nullptr || request->map_layers == nullptr) {
      process(events::layers_map_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->map_layers(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::layers_map_error{request, err});
      return;
    }
    process(events::layers_mapped{request});
  }
};

struct validate_structure {
  template <class Event>
  void operator()(const Event & ev, context &, process_t & process) const {
    const event::load * request = nullptr;
    if constexpr (std::is_same_v<Event, events::layers_mapped>) {
      request = ev.request;
    } else if constexpr (std::is_same_v<Event, events::parsing_done>) {
      request = ev.request;
    } else {
      return;
    }
    if (request == nullptr) {
      process(events::structure_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    if (!request->check_tensors) {
      process(events::structure_validated{request});
      return;
    }
    if (request->validate_structure == nullptr) {
      process(events::structure_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->validate_structure(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      process(events::structure_error{request, err});
      return;
    }
    process(events::structure_validated{request});
  }
};

struct validate_architecture {
  void operator()(const events::structure_validated & ev, context &, process_t & process) const {
    const event::load * request = ev.request;
    int32_t err = EMEL_OK;
    if (request == nullptr || request->validate_architecture_impl == nullptr) {
      process(events::architecture_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    const bool ok = request->validate_architecture_impl(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      process(events::architecture_error{request, err});
      return;
    }
    process(events::architecture_validated{request});
  }
};

struct dispatch_done {
  template <class Event>
  void operator()(const Event & ev, context & ctx, process_t &) const {
    const event::load * request = ev.request;
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_OK;
    }
    if (request->dispatch_done != nullptr && request->owner_sm != nullptr) {
      request->dispatch_done(request->owner_sm, events::load_done{
        request,
        ctx.bytes_total,
        ctx.bytes_done,
        ctx.used_mmap
      });
    }
  }
};

struct dispatch_error {
  template <class Event>
  void operator()(const Event & ev, context &, process_t &) const {
    const event::load * request = ev.request;
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = ev.err;
    }
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(request->owner_sm, events::load_error{request, ev.err});
    }
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context &, process_t &) const {
    const event::load * request = nullptr;
    if constexpr (requires { ev.request; }) {
      request = ev.request;
    } else if constexpr (std::is_same_v<Event, event::load>) {
      request = &ev;
    }
    if (request == nullptr) {
      return;
    }
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_ERR_BACKEND;
    }
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(request->owner_sm, events::load_error{request, EMEL_ERR_BACKEND});
    }
  }
};

}  // namespace emel::model::loader::action
