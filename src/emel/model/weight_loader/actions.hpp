#pragma once

#include "boost/sml.hpp"
#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/events.hpp"
#include <type_traits>

namespace emel::model::weight_loader {

using process_t = boost::sml::back::process<
  events::strategy_selected,
  events::mappings_ready,
  events::weights_loaded,
  events::validation_done,
  events::cleaning_up_done>;

}  // namespace emel::model::weight_loader

namespace emel::model::weight_loader::action {

struct context {
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

struct select_strategy {
  void operator()(const event::load_weights & ev, context &, process_t & process) const {
    bool use_direct_io = ev.request_direct_io && ev.direct_io_supported;
    bool use_mmap = false;
    if (!ev.no_alloc) {
      if (ev.request_mmap && ev.mmap_supported && !use_direct_io) {
        use_mmap = true;
      }
    }
    process(events::strategy_selected{&ev, use_mmap, use_direct_io, EMEL_OK});
  }
};

struct init_mappings {
  void operator()(const events::strategy_selected & ev, context &, process_t & process) const {
    const event::load_weights * request = ev.request;
    int32_t err = EMEL_OK;
    const bool ok = request->init_mappings(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::mappings_ready{request, err});
      return;
    }
    process(events::mappings_ready{request, EMEL_OK});
  }
};

struct load_mmap {
  void operator()(const events::mappings_ready & ev, context &, process_t & process) const {
    const event::load_weights * request = ev.request;
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = request->map_mmap(*request, &bytes_done, &bytes_total, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::weights_loaded{request, err, true, bytes_total, bytes_done});
      return;
    }
    process(events::weights_loaded{request, EMEL_OK, true, bytes_total, bytes_done});
  }
};

struct load_streamed {
  void operator()(const events::strategy_selected & ev, context &, process_t & process) const {
    const event::load_weights * request = ev.request;
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = request->load_streamed(*request, &bytes_done, &bytes_total, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::weights_loaded{request, err, false, bytes_total, bytes_done});
      return;
    }
    process(events::weights_loaded{request, EMEL_OK, false, bytes_total, bytes_done});
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
  void operator()(const events::cleaning_up_done & ev, context & ctx, process_t &) const {
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
  template <class Event>
  void operator()(const Event & ev, process_t &) const {
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

struct validate {
  void operator()(const events::weights_loaded & ev, context &, process_t & process) const {
    const event::load_weights * request = ev.request;
    int32_t err = EMEL_OK;
    const bool ok = request->validate(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      process(events::validation_done{request, err});
      return;
    }
    process(events::validation_done{request, EMEL_OK});
  }
};

struct cleaning_up {
  void operator()(const events::validation_done & ev, context &, process_t & process) const {
    const event::load_weights * request = ev.request;
    int32_t err = EMEL_OK;
    const bool ok = request->clean_up(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      process(events::cleaning_up_done{request, err});
      return;
    }
    process(events::cleaning_up_done{request, EMEL_OK});
  }
};

struct store_and_validate {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t & process) const {
    store_result{}(ev, ctx);
    validate{}(ev, ctx, process);
  }
};

struct skip_init_mappings {
  void operator()(const events::strategy_selected & ev, context &, process_t & process) const {
    process(events::mappings_ready{ev.request, EMEL_OK});
  }
};

struct reject_invalid_mappings {
  void operator()(const events::strategy_selected & ev, context &, process_t & process) const {
    process(events::mappings_ready{ev.request, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct reject_invalid_mmap {
  void operator()(const events::mappings_ready & ev, context &, process_t & process) const {
    process(events::weights_loaded{ev.request, EMEL_ERR_INVALID_ARGUMENT, true});
  }
};

struct reject_invalid_streamed {
  void operator()(const events::strategy_selected & ev, context &, process_t & process) const {
    process(events::weights_loaded{ev.request, EMEL_ERR_INVALID_ARGUMENT, false});
  }
};

struct skip_validate {
  void operator()(const events::weights_loaded & ev, context &, process_t & process) const {
    process(events::validation_done{ev.request, EMEL_OK});
  }
};

struct reject_invalid_validate {
  void operator()(const events::weights_loaded & ev, context &, process_t & process) const {
    process(events::validation_done{ev.request, EMEL_ERR_INVALID_ARGUMENT});
  }
};

struct store_and_skip_validate {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t & process) const {
    store_result{}(ev, ctx);
    skip_validate{}(ev, ctx, process);
  }
};

struct store_and_reject_validate {
  void operator()(const events::weights_loaded & ev, context & ctx, process_t & process) const {
    store_result{}(ev, ctx);
    reject_invalid_validate{}(ev, ctx, process);
  }
};

struct skip_cleaning_up {
  void operator()(const events::validation_done & ev, context &, process_t & process) const {
    process(events::cleaning_up_done{ev.request, EMEL_OK});
  }
};

struct reject_invalid_cleaning {
  void operator()(const events::validation_done & ev, context &, process_t & process) const {
    process(events::cleaning_up_done{ev.request, EMEL_ERR_INVALID_ARGUMENT});
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
