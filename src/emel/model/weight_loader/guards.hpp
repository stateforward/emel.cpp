#pragma once

#include "emel/emel.h"
#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

struct use_mmap_selected {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.use_mmap;
  }
};

struct use_stream_selected {
  bool operator()(const events::strategy_selected & ev) const {
    return !ev.use_mmap;
  }
};

struct no_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err == EMEL_OK;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err != EMEL_OK;
  }
};

struct can_init_mappings {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.request != nullptr && ev.request->init_mappings != nullptr;
  }
};

struct skip_init_mappings {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.request != nullptr && ev.request->init_mappings == nullptr;
  }
};

struct can_load_streamed {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.request != nullptr && ev.request->load_streamed != nullptr;
  }
};

struct cannot_load_streamed {
  bool operator()(const events::strategy_selected & ev) const {
    return ev.request == nullptr || ev.request->load_streamed == nullptr;
  }
};

struct can_load_mmap {
  bool operator()(const events::mappings_ready & ev) const {
    return ev.request != nullptr && ev.request->map_mmap != nullptr;
  }
};

struct cannot_load_mmap {
  bool operator()(const events::mappings_ready & ev) const {
    return ev.request == nullptr || ev.request->map_mmap == nullptr;
  }
};

struct can_validate {
  bool operator()(const events::weights_loaded & ev) const {
    return ev.request != nullptr && ev.request->check_tensors && ev.request->validate != nullptr;
  }
};

struct skip_validate {
  bool operator()(const events::weights_loaded & ev) const {
    return ev.request != nullptr && (!ev.request->check_tensors || ev.request->validate == nullptr);
  }
};

struct cannot_validate {
  bool operator()(const events::weights_loaded & ev) const { return ev.request == nullptr; }
};

struct can_clean_up {
  bool operator()(const events::validation_done & ev, const action::context & ctx) const {
    return ev.request != nullptr && ctx.used_mmap && ev.request->clean_up != nullptr;
  }
};

struct skip_clean_up {
  bool operator()(const events::validation_done & ev, const action::context & ctx) const {
    return ev.request != nullptr && (!ctx.used_mmap || ev.request->clean_up == nullptr);
  }
};

struct cannot_clean_up {
  bool operator()(const events::validation_done & ev) const { return ev.request == nullptr; }
};

struct use_mmap_no_error_can_init_mappings {
  bool operator()(const events::strategy_selected & ev) const {
    return use_mmap_selected{}(ev) && no_error{}(ev) && can_init_mappings{}(ev);
  }
};

struct use_mmap_no_error_skip_init_mappings {
  bool operator()(const events::strategy_selected & ev) const {
    return use_mmap_selected{}(ev) && no_error{}(ev) && skip_init_mappings{}(ev);
  }
};

struct use_mmap_no_error_cannot_init_mappings {
  bool operator()(const events::strategy_selected & ev) const {
    return use_mmap_selected{}(ev) && no_error{}(ev) && !skip_init_mappings{}(ev) &&
           !can_init_mappings{}(ev);
  }
};

struct use_stream_no_error_can_load_streamed {
  bool operator()(const events::strategy_selected & ev) const {
    return use_stream_selected{}(ev) && no_error{}(ev) && can_load_streamed{}(ev);
  }
};

struct use_stream_no_error_cannot_load_streamed {
  bool operator()(const events::strategy_selected & ev) const {
    return use_stream_selected{}(ev) && no_error{}(ev) && cannot_load_streamed{}(ev);
  }
};

struct mappings_ready_no_error_can_load_mmap {
  bool operator()(const events::mappings_ready & ev) const {
    return no_error{}(ev) && can_load_mmap{}(ev);
  }
};

struct mappings_ready_no_error_cannot_load_mmap {
  bool operator()(const events::mappings_ready & ev) const {
    return no_error{}(ev) && cannot_load_mmap{}(ev);
  }
};

struct weights_loaded_no_error_can_validate {
  bool operator()(const events::weights_loaded & ev) const {
    return no_error{}(ev) && can_validate{}(ev);
  }
};

struct weights_loaded_no_error_skip_validate {
  bool operator()(const events::weights_loaded & ev) const {
    return no_error{}(ev) && skip_validate{}(ev);
  }
};

struct weights_loaded_no_error_cannot_validate {
  bool operator()(const events::weights_loaded & ev) const {
    return no_error{}(ev) && cannot_validate{}(ev);
  }
};

struct validation_done_no_error_can_clean_up {
  bool operator()(const events::validation_done & ev, const action::context & ctx) const {
    return no_error{}(ev) && can_clean_up{}(ev, ctx);
  }
};

struct validation_done_no_error_skip_clean_up {
  bool operator()(const events::validation_done & ev, const action::context & ctx) const {
    return no_error{}(ev) && skip_clean_up{}(ev, ctx);
  }
};

struct validation_done_no_error_cannot_clean_up {
  bool operator()(const events::validation_done & ev) const {
    return no_error{}(ev) && cannot_clean_up{}(ev);
  }
};

}  // namespace emel::model::weight_loader::guard
