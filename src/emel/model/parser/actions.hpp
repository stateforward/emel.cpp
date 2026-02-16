#pragma once

#include "boost/sml.hpp"
#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/parser/events.hpp"
#include <type_traits>

namespace emel::model::parser {

using process_t = boost::sml::back::process<
  events::parse_architecture_done,
  events::parse_architecture_error,
  events::map_architecture_done,
  events::map_architecture_error,
  events::parse_hparams_done,
  events::parse_hparams_error,
  events::parse_vocab_done,
  events::parse_vocab_error,
  events::map_tensors_done,
  events::map_tensors_error>;

}  // namespace emel::model::parser

namespace emel::model::parser::action {

struct parse_architecture {
  void operator()(const event::parse_model & ev, process_t & process) const {
    if (ev.parse_architecture == nullptr) {
      process(events::parse_architecture_error{&ev, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = ev.parse_architecture(ev, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      process(events::parse_architecture_error{&ev, err});
      return;
    }
    process(events::parse_architecture_done{&ev});
  }
};

struct map_architecture {
  void operator()(const events::parse_architecture_done & ev, process_t & process) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr || request->map_architecture == nullptr) {
      process(events::map_architecture_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->map_architecture(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      process(events::map_architecture_error{request, err});
      return;
    }
    process(events::map_architecture_done{request});
  }
};

struct parse_hparams {
  void operator()(const events::map_architecture_done & ev, process_t & process) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr || request->parse_hparams == nullptr) {
      process(events::parse_hparams_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->parse_hparams(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      process(events::parse_hparams_error{request, err});
      return;
    }
    process(events::parse_hparams_done{request});
  }
};

struct parse_vocab {
  void operator()(const events::parse_hparams_done & ev, process_t & process) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr || request->parse_vocab == nullptr) {
      process(events::parse_vocab_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->parse_vocab(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      process(events::parse_vocab_error{request, err});
      return;
    }
    process(events::parse_vocab_done{request});
  }
};

struct map_tensors {
  void operator()(const events::parse_vocab_done & ev, process_t & process) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr) {
      process(events::map_tensors_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    if (!request->map_tensors) {
      process(events::map_tensors_done{request});
      return;
    }
    if (request->map_tensors_impl == nullptr) {
      process(events::map_tensors_error{request, EMEL_ERR_INVALID_ARGUMENT});
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->map_tensors_impl(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::map_tensors_error{request, err});
      return;
    }
    process(events::map_tensors_done{request});
  }
};

struct dispatch_done {
  template <class Event>
  void operator()(const Event & ev, process_t &) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr || request->dispatch_done == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_done(request->owner_sm, emel::model::loader::events::parsing_done{
      request->loader_request
    });
  }
};

struct dispatch_error {
  template <class Event>
  void operator()(const Event & ev, process_t &) const {
    const event::parse_model * request = ev.request;
    if (request == nullptr || request->dispatch_error == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_error(request->owner_sm, emel::model::loader::events::parsing_error{
      request->loader_request,
      ev.err
    });
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, process_t &) const {
    const event::parse_model * request = nullptr;
    if constexpr (requires { ev.request; }) {
      request = ev.request;
    } else if constexpr (std::is_same_v<Event, event::parse_model>) {
      request = &ev;
    }
    if (request == nullptr || request->dispatch_error == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_error(request->owner_sm, emel::model::loader::events::parsing_error{
      request->loader_request,
      EMEL_ERR_BACKEND
    });
  }
};

}  // namespace emel::model::parser::action
