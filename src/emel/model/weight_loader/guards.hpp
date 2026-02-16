#pragma once

#include "emel/emel.h"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

struct use_mmap {
  bool operator()(const event::load_weights & ev) const {
    if (ev.no_alloc) {
      return false;
    }
    const bool direct_io = ev.request_direct_io && ev.direct_io_supported;
    return ev.request_mmap && ev.mmap_supported && !direct_io;
  }
};

struct use_stream {
  bool operator()(const event::load_weights & ev) const {
    if (ev.no_alloc) {
      return true;
    }
    const bool direct_io = ev.request_direct_io && ev.direct_io_supported;
    if (direct_io) {
      return true;
    }
    if (!ev.request_mmap) {
      return true;
    }
    return !ev.mmap_supported;
  }
};

struct no_error {
  bool operator()(const events::weights_loaded & ev) const {
    return ev.err == EMEL_OK;
  }
};

struct has_error {
  bool operator()(const events::weights_loaded & ev) const {
    return ev.err != EMEL_OK;
  }
};

}  // namespace emel::model::weight_loader::guard
