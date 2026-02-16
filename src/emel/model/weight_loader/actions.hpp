#pragma once

#include "boost/sml.hpp"
#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/events.hpp"
#include <type_traits>

namespace emel::model::weight_loader {

using process_t = boost::sml::back::process<events::weights_loaded>;

}  // namespace emel::model::weight_loader

namespace emel::model::weight_loader::action {

struct context {
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct load_mmap {
  void operator()(const event::load_weights & ev, context &, process_t & process) const {
    if (ev.map_mmap == nullptr) {
      process(events::weights_loaded{&ev, EMEL_ERR_INVALID_ARGUMENT, true});
      return;
    }
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = ev.map_mmap(ev, &bytes_done, &bytes_total, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::weights_loaded{&ev, err, true, bytes_total, bytes_done});
      return;
    }
    process(events::weights_loaded{&ev, EMEL_OK, true, bytes_total, bytes_done});
  }
};

struct load_streamed {
  void operator()(const event::load_weights & ev, context &, process_t & process) const {
    if (ev.load_streamed == nullptr) {
      process(events::weights_loaded{&ev, EMEL_ERR_INVALID_ARGUMENT, false});
      return;
    }
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = ev.load_streamed(ev, &bytes_done, &bytes_total, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::weights_loaded{&ev, err, false, bytes_total, bytes_done});
      return;
    }
    process(events::weights_loaded{&ev, EMEL_OK, false, bytes_total, bytes_done});
  }
};

struct store_result {
  void operator()(const events::weights_loaded & ev, context & ctx) const {
    ctx.bytes_total = ev.bytes_total;
    ctx.bytes_done = ev.bytes_done;
    ctx.used_mmap = ev.used_mmap;
  }
};

struct dispatch_done {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t &) const {
    const event::load_weights * request = ev.request;
    if (request == nullptr || request->dispatch_done == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_done(request->owner_sm, emel::model::loader::events::loading_done{
      request->loader_request,
      ctx.bytes_total,
      ctx.bytes_done,
      ctx.used_mmap
    });
  }
};

struct dispatch_error {
  void operator()(const events::weights_loaded & ev, process_t &) const {
    const event::load_weights * request = ev.request;
    if (request == nullptr || request->dispatch_error == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_error(request->owner_sm, emel::model::loader::events::loading_error{
      request->loader_request,
      ev.err
    });
  }
};

struct store_and_dispatch_done {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t & process) const {
    store_result{}(ev, ctx);
    dispatch_done{}(ev, ctx, process);
  }
};

struct store_and_dispatch_error {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t & process) const {
    store_result{}(ev, ctx);
    dispatch_error{}(ev, process);
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context &, process_t &) const {
    const event::load_weights * request = nullptr;
    if constexpr (requires { ev.request; }) {
      request = ev.request;
    } else if constexpr (std::is_same_v<Event, event::load_weights>) {
      request = &ev;
    }
    if (request == nullptr || request->dispatch_error == nullptr || request->owner_sm == nullptr) {
      return;
    }
    request->dispatch_error(request->owner_sm, emel::model::loader::events::loading_error{
      request->loader_request,
      EMEL_ERR_BACKEND
    });
  }
};

}  // namespace emel::model::weight_loader::action
